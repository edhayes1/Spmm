/*
 * TODO pass by ref to spgemm
 * TODO guess nnz for c, create the row_pointer in advance
 * TODO try to get rid of ifs, might be possible once done above
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

/* gets the number of non zeroes for C.
 *  Arp = A_row_pointer, Acp = A_column_pointer
 *  In the future, want to estimate nnz not spend time calculating it directly
 */
int get_nnz(const int m, const int n, const int *Arp, const int *Acp, const int *Brp, const int *Bcp){
    int nz = 0;
    int * index = malloc(n * sizeof(int));
    memset(index, -1, n*sizeof(int));

    for (int i = 0; i < m; i++){
        for (int j = Arp[i]; j < Arp[i+1]; j++){
            int A_col_index = Acp[j];

            for (int k = Brp[A_col_index]; k < Brp[A_col_index+1]; k++){
                int B_col_index = Bcp[k];

                if(index[B_col_index] != i){
                    index[B_col_index] = i;
                    nz++;
                }
            }
        }
    }

    free(index);
    return nz;
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
}
/*
 * get the nnz so C can be allocated, then do the multiplication.
 */
void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C)
{
    int C_m = A->m;
    int C_n = B->n;
    int C_nnz = get_nnz(C_m, C_n, A->row_start, A->col_indices, B->row_start, B->col_indices);
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
