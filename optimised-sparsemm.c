#include "utils.h"
#include "stdlib.h"
#include <stdio.h>
#include <string.h>

void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);


void spgemm(const CSR A, const CSR B, CSR C){
    // get C dimensions
    int m = C->m;
    int n = C->n;
    int nnz_counter = 0;

    // temp accumulates a row of the product
    double * temp = calloc(n, sizeof(double));

    // next keeps track of which columns have entries in - init to -1
    int * next = malloc(n * sizeof(int));
    memset(next, -1, n*sizeof(int));

    //first in row pointer is always 0
    C->row_start[0] = 0;

    for (int i = 0; i < m; i++){
        int pos = -2;
        int nnz_row = 0;

        // loop over columns + data in row of A
        for (int n = A->row_start[i]; n < A->row_start[i+1]; n++){
            int j = A->col_indices[n];
            double x = A->data[n];

            // loop over each row of B that line up with the column j
            for (int k = B->row_start[j]; k < B->row_start[j+1]; k++) {
                int k_col = B->col_indices[k];
                temp[k_col] += x * B->data[k];

                //if product doesnt have element in column, add it
                if (next[k_col] == -1){
                    next[k_col] = pos;
                    pos = k_col;
                    nnz_row++;
                }
            }
        }

        // if we underestimated the number of non zeros, allocate twice the memory
        if (nnz_counter + nnz_row > C->NZ){
            printf("re estimating NZ in product\n");
            int NZ_estimate = 2 * C->NZ;
            int * realloc_col_indices = realloc(C->col_indices, NZ_estimate * sizeof(int));
            double * realloc_data = realloc(C->data, NZ_estimate * sizeof(double));

            //check successfully allocated, else throw an error and exit.
            if (realloc_col_indices && realloc_data){
                C->NZ = NZ_estimate;
                C->col_indices = realloc_col_indices;
                C->data = realloc_data;
            }
            else{
                fprintf(stderr, "FAILED RAN OUT OF MEMORY");
                exit(-1);
            }
        }

        // accumulate - follow next through to get the right values in temp!
        for (int cj = 0; cj < nnz_row; cj++){
            if(temp[pos] != 0){
                C->col_indices[nnz_counter] = pos;
                C->data[nnz_counter] = temp[pos];
                nnz_counter++;
            }

            // reset next and temp for each row
            int t = pos;
            pos = next[pos];
            next[t] = -1;
            temp[t] = 0;
        }

        // update the row pointer
        C->row_start[i+1] = nnz_counter;

    }

    // strip off the excess from the estimation. No need to do a realloc.
    C->NZ = nnz_counter;
}

void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C){
    // get dimensions of C
    int C_m = A->m;
    int C_n = B->n;
    // estimate number of NZ
    int C_nnz = 2*(A->NZ + B->NZ);
    alloc_sparse_CSR(C_m, C_n, C_nnz, C);

    spgemm(A, B, *C);
}

/* Computes C = A*B.
 * C should be allocated by this routine.
 */
void optimised_sparsemm(const COO A, const COO B, COO *C)
{
    // convert input to csr
    CSR A_csr, B_csr, C_csr;
    coo_to_csr(A, &A_csr);
    coo_to_csr(B, &B_csr);

    // get dimensions of C
    int C_m = A->m;
    int C_n = B->n;

    // estimate number of NZ (2 times the sum of nz input)
    int C_nnz = 2*(A->NZ + B->NZ);
    alloc_sparse_CSR(C_m, C_n, C_nnz, &C_csr);

    spgemm(A_csr, B_csr, C_csr);

    //convert product back to COO, free CSRs
    csr_to_coo(C_csr, C);
    free_CSR(&A_csr);
    free_CSR(&B_csr);
    free_CSR(&C_csr);
}


void sum(const CSR mat_1, const CSR mat_2, const CSR mat_3, CSR sum){
    int m = sum->m;
    int n = sum->n;
    int nnz_counter = 0;


    double * temp = calloc(n, sizeof(double));
    int * next = malloc(n * sizeof(int));
    memset(next, -1, n * sizeof(int));
    sum->row_start[0] = 0;

    // loop over each row
    for (int i = 0; i < m; i++){

        int pos = -2;
        int row_nz = 0;

        // similar premise to multiplication, but instead we are adding to temp instead of +=
        // one loop for each matrix adding to temp
        for (int col_ind = mat_1->row_start[i]; col_ind < mat_1->row_start[i+1]; col_ind++){
            int col = mat_1->col_indices[col_ind];
            temp[col] = mat_1->data[col_ind];
            if (next[col] == -1){
                next[col] = pos;
                pos = col;
                row_nz++;
            }
        }

        for (int col_ind = mat_2->row_start[i]; col_ind < mat_2->row_start[i+1]; col_ind++){
            int col = mat_2->col_indices[col_ind];
            temp[col] += mat_2->data[col_ind];
            if (next[col] == -1){
                next[col] = pos;
                pos = col;
                row_nz++;
            }
        }

        for (int col_ind = mat_3->row_start[i]; col_ind < mat_3->row_start[i+1]; col_ind++){
            int col = mat_3->col_indices[col_ind];
            temp[col] += mat_3->data[col_ind];
            if (next[col] == -1){
                next[col] = pos;
                pos = col;
                row_nz++;
            }
        }

        // row is accumulated in same way as multiplication
        for (int cj = 0; cj < row_nz; cj++){
            if(temp[pos] != 0){
                sum->col_indices[nnz_counter] = pos;
                sum->data[nnz_counter] = temp[pos];
                nnz_counter++;
            }

            int t = pos;
            pos = next[pos];
            next[t] = -1;
            temp[t] = 0;
        }

        sum->row_start[i+1] = nnz_counter;

    }

    sum->NZ = nnz_counter;
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{
    // convert input to csr
    CSR A_csr, B_csr, C_csr, D_csr, E_csr, F_csr, ABC, DEF, ret;
    coo_to_csr(A, &A_csr);
    coo_to_csr(B, &B_csr);
    coo_to_csr(C, &C_csr);
    coo_to_csr(D, &D_csr);
    coo_to_csr(E, &E_csr);
    coo_to_csr(F, &F_csr);

    // estimate num non zeros - maximum can be sum of input NZ
    int nnz_ABC = A_csr->NZ + B_csr->NZ + C_csr->NZ;
    int nnz_DEF = D_csr->NZ + E_csr->NZ + F_csr->NZ;

    alloc_sparse_CSR(A_csr->m, A_csr->n, nnz_ABC, &ABC);
    alloc_sparse_CSR(D_csr->m, D_csr->n, nnz_DEF, &DEF);

    sum(A_csr, B_csr, C_csr, ABC);
    sum(D_csr, E_csr, F_csr, DEF);

    optimised_sparsemm_CSR(ABC, DEF, &ret);

    // convert product back to COO. Free CSRs
    csr_to_coo(ret, O);

    free_CSR(&A_csr);
    free_CSR(&B_csr);
    free_CSR(&C_csr);
    free_CSR(&D_csr);
    free_CSR(&E_csr);
    free_CSR(&F_csr);
    free_CSR(&ret);
}
