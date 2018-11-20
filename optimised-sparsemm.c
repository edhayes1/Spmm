/*
 * TODO pass by ref to spgemm
 * TODO guess nnz for c, create the row_pointer in advance
 * TODO try to get rid of ifs, might be possible once done above
 * TODO transpose
 */

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
    int m = C->m;
    int n = C->n;
    int nnz_counter = 0;
    double * temp = calloc(n, sizeof(double));
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

        if (nnz_counter + length > C->NZ){
            printf("re estimating NZ in product\n");
            int NZ_estimate = C->NZ;
            NZ_estimate *= 2;
            int * realloc_col_indices = realloc(C->col_indices, NZ_estimate * sizeof(int));
            double * realloc_data = realloc(C->data, NZ_estimate * sizeof(double));
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

    C->NZ = nnz_counter;
}
/*
 * get the nnz so C can be allocated, then do the multiplication.
 */
void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C)
{
    int C_m = A->m;
    int C_n = B->n;
    int C_nnz = 2*(A->NZ + B->NZ);
    alloc_sparse_CSR(C_m, C_n, C_nnz, C);

    spgemm(A, B, *C);
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{
    return basic_sparsemm_sum(A, B, C, D, E, F, O);
}
