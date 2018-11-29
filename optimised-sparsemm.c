#include "utils.h"
#include "stdlib.h"
#include <stdio.h>
#include <string.h>

void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);

/* Computes C = A*B.
 * C should be allocated by this routine.
 */
void optimised_sparsemm(const COO A, const COO B, COO *C){
    return basic_sparsemm(A, B, C);
}

void spgemm(const CSR A, const CSR B, CSR C){
    // get C dimensions
    int m = C->m;
    int n = C->n;
    int nnz_counter = 0;

    // temp accumulates a column of the product
    double * temp = calloc(n, sizeof(double));

    // next keeps track of where we are in the column comp. - init to -1.
    int * next = malloc(n * sizeof(int));
    memset(next, -1, n*sizeof(int));

    C->row_start[0] = 0;

    for (int i = 0; i < m; i++){
        int col_start = -2;
        int length = 0;

        for (int n = A->row_start[i]; n < A->row_start[i+1]; n++){
            int j = A->col_indices[n];
            double x = A->data[n];

            for (int k = B->row_start[j]; k < B->row_start[j+1]; k++) {
                int k_col = B->col_indices[k];
                temp[k_col] += x * B->data[k];

                if (next[k_col] == -1){
                    next[k_col] = col_start;
                    col_start = k_col;
                    length++;
                }
            }
        }

        // if we underestimated, allocate twice the memory
        if (nnz_counter + length > C->NZ){
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

        for (int cj = 0; cj < length; cj++){
            if(temp[col_start] != 0){
                C->col_indices[nnz_counter] = col_start;
                C->data[nnz_counter] = temp[col_start];
                nnz_counter++;
            }

            int t = col_start;
            col_start = next[col_start];
            next[t] = -1;
            temp[t] = 0;
        }

        C->row_start[i+1] = nnz_counter;

    }

    // strip off the excess from the estimation. No need to do a realloc.
    C->NZ = nnz_counter;
}
/*
 * get the nnz so C can be allocated, then do the multiplication.
 */
void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C)
{
    // get dimensions of C
    int C_m = A->m;
    int C_n = B->n;
    // estimate number of NZ
    int C_nnz = 2*(A->NZ + B->NZ);
    alloc_sparse_CSR(C_m, C_n, C_nnz, C);

    spgemm(A, B, *C);
}

void sum(const CSR mat_1, const CSR mat_2, CSR sum){
    int num_rows = sum->m;
    int nnz_counter = 0;

    // loop over each row
    for (int m = 0; m < num_rows; m++){

        // loop over row in both mat1 and 2, don't stop until both complete
        int mat_1_index = mat_1->row_start[m];
        int mat_2_index = mat_2->row_start[m];
        while((mat_1_index < mat_1->row_start[m+1]) && (mat_2_index < mat_2->row_start[m+1])){

            int mat_1_col = mat_1->col_indices[mat_1_index];
            int mat_2_col = mat_2->col_indices[mat_2_index];

            if(mat_1_col == mat_2_col){
                sum->col_indices[nnz_counter] = mat_1_col;
                sum->data[nnz_counter] = mat_1->data[mat_1_col] + mat_2->data[mat_2_col];
                mat_1_index++;
                mat_2_index++;
            }
            else if(mat_2_col > mat_1_col){
                sum->col_indices[nnz_counter] = mat_2_col;
                sum->data[nnz_counter] = mat_2->data[mat_2_col];
                mat_2_index++;
            }
            else{
                sum->col_indices[nnz_counter] = mat_1_col;
                sum->data[nnz_counter] = mat_1->data[mat_1_col];
                mat_1_index++;
            }
            nnz_counter++;
        }

        while(mat_1_index < mat_1->row_start[m+1]){
            int mat_1_col = mat_1->col_indices[mat_1_index];
            sum->col_indices[nnz_counter] = mat_1_col;
            sum->data[nnz_counter] = mat_1->data[mat_1_col];
            mat_1_index++;
            nnz_counter++;
        }

        while(mat_2_index < mat_2->row_start[m+1]){
            int mat_2_col = mat_2->col_indices[mat_2_index];
            sum->col_indices[nnz_counter] = mat_2_col;
            sum->data[nnz_counter] = mat_2->data[mat_2_col];
            mat_2_index++;
            nnz_counter++;
        }

        sum->row_start[m+1] = nnz_counter;
    }

    sum->NZ = nnz_counter;
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const CSR A, const CSR B, const CSR C,
                            const CSR D, const CSR E, const CSR F,
                            CSR *R)
{
    CSR AB, DE, ABC, DEF;
    int nnz_AB = A->NZ + B->NZ;
    int nnz_DE = D->NZ + E->NZ;
    int nnz_ABC = A->NZ + B->NZ + C->NZ;
    int nnz_DEF = D->NZ + E->NZ + F->NZ;

    alloc_sparse_CSR(A->m, A->n, nnz_AB, &AB);
    alloc_sparse_CSR(D->m, D->n, nnz_DE, &DE);
    alloc_sparse_CSR(A->m, A->n, nnz_ABC, &ABC);
    alloc_sparse_CSR(D->m, D->n, nnz_DEF, &DEF);

    sum(A, B, AB);
    sum(AB, C, ABC);
    sum(D, E, DE);
    sum(DE, F, DEF);

    optimised_sparsemm_CSR(ABC, DEF, R);

    free_CSR(&AB);
    free_CSR(&ABC);
    free_CSR(&DE);
    free_CSR(&DEF);
}
