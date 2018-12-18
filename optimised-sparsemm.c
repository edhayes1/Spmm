#include "utils.h"
#include "stdlib.h"
#include <stdio.h>
#include <string.h>

void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);


/* gets the number of non zeroes for C.
 *  Arp = A_row_pointer, Acp = A_column_pointer
 *  In the future, want to estimate nnz not spend time calculating it directly
 */
void get_nnz(const int num_rows, const int num_cols, const int *Arp, const int *Acp, const int *Brp, const int *Bcp, int *ret){
    int * index;
    int nz = 0;

    #pragma acc parallel firstprivate (index[0:num_cols])
    {
        index = malloc(num_cols * sizeof(int));
        memset(index, -1, num_cols*sizeof(int));

        ret[0] = 0;
        #pragma acc loop independent
        for (int i = 0; i < num_rows; i++){
            nz = 0;
            for (int j = Arp[i]; j < Arp[i+1]; j++){
                int A_col_index = Acp[j];

                for (int k = Brp[A_col_index]; k < Brp[A_col_index+1]; k++){
                    int B_col_index = Bcp[k];

                    if(index[B_col_index] != i){
                        nz++;
                        index[B_col_index] = i;
                    }
                }
            }
            // add to row pointer
            ret[i+1] = nz;
        }
        free(index);
    }
    // cumulate at end
    for (int i = 0; i < num_rows; i++){
        ret[i+1] = ret[i] + ret[i+1];
    }
}

void spgemm(const CSR A, const CSR B, CSR C) {
    // get C dimensions
    int m = C->m;
    int n = C->n;
    int nnz_cum = 0;

    // temp accumulates a row of the product
    double *temp;
    int *index;
    C->row_start[0] = 0;

    #pragma acc parallel firstprivate(temp[0:n], index[0:n])
    {
        temp = calloc(n, sizeof(double));
        // next keeps track of which columns have entries in - init to -1
        index = malloc(n * sizeof(int));
        memset(index, -1, n * sizeof(int));

        #pragma acc loop independent
        for (int i = 0; i < m; i++) {
            int pos = -2;
            int nnz_row = 0;

            // loop over columns + data in row of A
            for (int n = A->row_start[i]; n < A->row_start[i + 1]; n++) {
                int j = A->col_indices[n];
                double x = A->data[n];

                // loop over each row of B that line up with the column j
                for (int k = B->row_start[j]; k < B->row_start[j + 1]; k++) {
                    int k_col = B->col_indices[k];
                    temp[k_col] += x * B->data[k];

                    //if product doesnt have element in column, add it
                    if (index[k_col] == -1) {
                        index[k_col] = pos;
                        pos = k_col;
                        nnz_row++;
                    }
                }
            }

            #if !defined(multi)
            // if we underestimated the number of non zeros, allocate twice the memory
            if (nnz_cum + nnz_row > C->NZ) {
                printf("re estimating NZ in product\n");
                int NZ_estimate = 2 * C->NZ;
                int *realloc_col_indices = realloc(C->col_indices, NZ_estimate * sizeof(int));
                double *realloc_data = realloc(C->data, NZ_estimate * sizeof(double));

                //check successfully allocated, else throw an error and exit.
                if (realloc_col_indices && realloc_data) {
                    C->NZ = NZ_estimate;
                    C->col_indices = realloc_col_indices;
                    C->data = realloc_data;
                } else {
                    fprintf(stderr, "FAILED RAN OUT OF MEMORY");
                    exit(-1);
                }
            }
            #endif

            int col_index = C->row_start[i];

            for (int cj = 0; cj < nnz_row; cj++) {
                C->col_indices[col_index + cj] = pos;
                C->data[col_index + cj] = temp[pos];

                // reset next and temp for each row
                int t = pos;
                pos = index[pos];
                index[t] = -1;
                temp[t] = 0;
            }

        #if !defined(multi)
            nnz_cum += nnz_row;

            // update the row pointer
            C->row_start[i + 1] = nnz_cum;
        #endif
        }

        // strip off the excess from the estimation. No need to do a realloc.
    #if !defined(multi)
        C->NZ = nnz_cum;
    #endif
    }
}

void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C){
    // get dimensions of C
    int C_m = A->m;
    int C_n = B->n;

    #if defined(multi)
        int * Ccp  = malloc((C_m+1) * sizeof(int));
        get_nnz(C_m, C_n, A->row_start, A->col_indices, B->row_start, B->col_indices, Ccp);
        alloc_sparse_CSR_with_rp(C_m, C_n, Ccp, C);
    #else
        // estimate number of NZ
        int C_nnz = 5*(A->NZ + B->NZ);
        alloc_sparse_CSR(C_m, C_n, C_nnz, C);
    #endif

    spgemm(A, B, *C);
}

/* Computes C = A*B.
 * C should be allocated by this routine.
 */
void optimised_sparsemm(const COO A, const COO B, COO *C)
{
    if (A->n != B->m) {
        fprintf(stderr, "Invalid matrix sizes, got %d x %d and %d x %d\n",
                A->m, A->n, B->m, B->n);
        free(A);
        free(B);
        exit(1);
    }
    // convert input to csr
    CSR A_csr, B_csr, C_csr;
    coo_to_csr(A, &A_csr);
    coo_to_csr(B, &B_csr);

    optimised_sparsemm_CSR(A_csr, B_csr, &C_csr);

    //convert product back to COO, free CSRs
    csr_to_coo(C_csr, C);
    free_CSR(&A_csr);
    free_CSR(&B_csr);
    free_CSR(&C_csr);
}

void sum_getnnz(const CSR mat_1, const CSR mat_2, const CSR mat_3, int * sumRp){
    int m = mat_1->m;
    int n = mat_1->n;

    int * index;

    #pragma acc parallel firstprivate (index[0:n])
    {
        index = malloc(n * sizeof(int));
        memset(index, -1, n * sizeof(int));
        sumRp[0] = 0;

        #pragma acc loop independent
        for (int i = 0; i < m; i++) {
            int nz = 0;

            for (int j = mat_1->row_start[i]; j < mat_1->row_start[i + 1]; j++) {
                int col = mat_1->col_indices[j];
                if (index[col] != i) {
                    index[col] = i;
                    nz++;
                }
            }

            for (int j = mat_2->row_start[i]; j < mat_2->row_start[i + 1]; j++) {
                int col = mat_2->col_indices[j];
                if (index[col] != i) {
                    index[col] = i;
                    nz++;
                }
            }

            for (int j = mat_3->row_start[i]; j < mat_3->row_start[i + 1]; j++) {
                int col = mat_3->col_indices[j];
                if (index[col] != i) {
                    index[col] = i;
                    nz++;
                }
            }

            sumRp[i + 1] = nz;
        }
        free(index);
    }

    for (int i = 0; i < m; i++){
        sumRp[i+1] = sumRp[i] + sumRp[i+1];
    }
}

void sum(const CSR mat_1, const CSR mat_2, const CSR mat_3, CSR sum){
    int num_rows = sum->m;
    int num_cols = sum->n;
    int nnz_cum = 0;

    double * temp;
    int * next;
    sum->row_start[0] = 0;

    // loop over each row
    #pragma acc parallel firstprivate(temp[0:num_cols], next[0:num_cols])
    {
        temp = calloc(num_cols, sizeof(double));
        // next keeps track of where we are in the column comp. - init to -1.
        next = malloc(num_cols * sizeof(int));
        memset(next, -1, num_cols*sizeof(int));

        #pragma acc loop independent
        for (int i = 0; i < num_rows; i++) {
            int pos = -2;
            int row_nz = 0;

            // similar premise to multiplication, but instead we are adding to temp instead of +=
            // one loop for each matrix adding to temp
            for (int col_ind = mat_1->row_start[i]; col_ind < mat_1->row_start[i + 1]; col_ind++) {
                int col = mat_1->col_indices[col_ind];
                temp[col] = mat_1->data[col_ind];
                if (next[col] == -1) {
                    next[col] = pos;
                    pos = col;
                    row_nz++;
                }
            }

            for (int col_ind = mat_2->row_start[i]; col_ind < mat_2->row_start[i + 1]; col_ind++) {
                int col = mat_2->col_indices[col_ind];
                temp[col] += mat_2->data[col_ind];
                if (next[col] == -1) {
                    next[col] = pos;
                    pos = col;
                    row_nz++;
                }
            }

            for (int col_ind = mat_3->row_start[i]; col_ind < mat_3->row_start[i + 1]; col_ind++) {
                int col = mat_3->col_indices[col_ind];
                temp[col] += mat_3->data[col_ind];
                if (next[col] == -1) {
                    next[col] = pos;
                    pos = col;
                    row_nz++;
                }
            }

            int col_index = sum->row_start[i];

            for (int cj = 0; cj < row_nz; cj++) {
                sum->col_indices[col_index + cj] = pos;
                sum->data[col_index + cj] = temp[pos];

                int t = pos;
                pos = next[pos];
                next[t] = -1;
                temp[t] = 0;
            }

            #if !defined(multi)
                nnz_cum += row_nz;
                sum->row_start[i + 1] = nnz_cum;
            #endif
        }
    }

    #if !defined(multi)
        sum->NZ = nnz_cum;
    #endif
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{
    if ((A->m != B->m || A->n != B->n) || (A->m != C->m || A->n != C->n) || (D->m != E->m || D->n != E->n) || (D->m != F->m || D->n != F->n) || (A->n != D->m)){
        fprintf(stderr, "invalid matrix sizes");
        exit(1);
    }

    // convert input to csr
    CSR A_csr, B_csr, C_csr, D_csr, E_csr, F_csr, ABC, DEF, ret;
    coo_to_csr(A, &A_csr);
    coo_to_csr(B, &B_csr);
    coo_to_csr(C, &C_csr);
    coo_to_csr(D, &D_csr);
    coo_to_csr(E, &E_csr);
    coo_to_csr(F, &F_csr);

    #if defined(multi)

        int * ABCRp = malloc((A->m + 1) * sizeof(int));
        int * DEFRp = malloc((D->m + 1) * sizeof(int));

        sum_getnnz(A_csr, B_csr, C_csr, ABCRp);
        sum_getnnz(D_csr, E_csr, F_csr, DEFRp);

        alloc_sparse_CSR_with_rp(A_csr->m, A_csr->n, ABCRp, &ABC);
        alloc_sparse_CSR_with_rp(D_csr->m, D_csr->n, DEFRp, &DEF);

    #else
        // estimate num non zeros - maximum can be sum of input NZ
        int nnz_ABC = A_csr->NZ + B_csr->NZ + C_csr->NZ;
        int nnz_DEF = D_csr->NZ + E_csr->NZ + F_csr->NZ;

        alloc_sparse_CSR(A_csr->m, A_csr->n, nnz_ABC, &ABC);
        alloc_sparse_CSR(D_csr->m, D_csr->n, nnz_DEF, &DEF);

    #endif

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
