#include "utils.h"
#include "stdlib.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

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
void get_nnz(const int num_rows, const int num_cols, const int *Arp, const int *Acp, const int *Brp, const int *Bcp, int *Ccp){
    int * index;
    int nz = 0;

    #pragma acc parallel firstprivate (index[0:num_cols])
    {
        index = malloc(num_cols * sizeof(int));
        memset(index, -1, num_cols*sizeof(int));

        Ccp[0] = 0;
        #pragma acc loop
        for (int i = 0; i < num_rows; i++){
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
            Ccp[i+1] = nz;
            nz = 0;
        }
        free(index);
    }

    for (int i = 0; i < num_rows; i++){
        Ccp[i+1] = Ccp[i] + Ccp[i+1];
    }
}

void spgemm(const CSR A, const CSR B, CSR C){
    // get C dimensions
    int m = C->m;
    int n = C->n;

    // temp accumulates a column of the product
    double * temp;
    int * next;
    C->row_start[0] = 0;

    #pragma acc parallel firstprivate(temp[0:n], next[0:n])
    {
        temp = calloc(n, sizeof(double));
        // next keeps track of where we are in the column comp. - init to -1.
        next = malloc(n * sizeof(int));
        memset(next, -1, n*sizeof(int));

	    #pragma acc loop
        for (int i = 0; i < m; i++){

            int nnz_counter = 0;
            int col_start = -2;
            int length = 0;

            #pragma acc loop
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

            int col_index = C->row_start[i];

            for (int cj = 0; cj < length; cj++){
                C->col_indices[col_index+nnz_counter] = col_start;
                C->data[col_index+nnz_counter] = temp[col_start];
                nnz_counter++;

                int t = col_start;
                col_start = next[col_start];
                next[t] = -1;
                temp[t] = 0;
            }
        }
    }
}

/*
 * get the nnz so C can be allocated, then do the multiplication.
 */
void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C)
{
    struct timespec start, stop;
    double accum;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 10; i++) {
        if (A->n != B->m) {
            fprintf(stderr, "Invalid matrix sizes");
        }

        int C_m = A->m;
        int C_n = B->n;
        int *Ccp = malloc((C_m+1) * sizeof(int));
        get_nnz(C_m, C_n, A->row_start, A->col_indices, B->row_start, B->col_indices, Ccp);
        int C_nnz = Ccp[C_m];
        alloc_sparse_CSR(C_m, C_n, C_nnz, C);
        (*C)->row_start = Ccp;
        spgemm(A, B, *C);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    accum = ( stop.tv_sec - start.tv_sec )
            + ( stop.tv_nsec - start.tv_nsec )
              / 1000000000.0;
    printf("time: %lf\n", (accum/10));
}


void sum(const CSR mat_1, const CSR mat_2, const CSR mat_3, CSR sum){
    int num_rows = sum->m;
    int num_cols = sum->n;
    int nnz_counter = 0;

    double * temp = calloc(num_cols, sizeof(double));
    int * next = malloc(num_cols * sizeof(int));
    memset(next, -1, num_cols * sizeof(int));
    sum->row_start[0] = 0;

    // loop over each row
    #pragma acc parallel loop
    for (int m = 0; m < num_rows; m++){

        int col_start = -2;
        int length = 0;

        for (int col_ind = mat_1->row_start[m]; col_ind < mat_1->row_start[m+1]; col_ind++){
            int col = mat_1->col_indices[col_ind];
            temp[col] = mat_1->data[col_ind];
            if (next[col] == -1){
                next[col] = col_start;
                col_start = col;
                length++;
            }
        }

        for (int col_ind = mat_2->row_start[m]; col_ind < mat_2->row_start[m+1]; col_ind++){
            int col = mat_2->col_indices[col_ind];
            temp[col] += mat_2->data[col_ind];
            if (next[col] == -1){
                next[col] = col_start;
                col_start = col;
                length++;
            }
        }

        for (int col_ind = mat_3->row_start[m]; col_ind < mat_3->row_start[m+1]; col_ind++){
            int col = mat_3->col_indices[col_ind];
            temp[col] += mat_3->data[col_ind];
            if (next[col] == -1){
                next[col] = col_start;
                col_start = col;
                length++;
            }
        }

        for (int cj = 0; cj < length; cj++){
            if(temp[col_start] != 0){
                sum->col_indices[nnz_counter] = col_start;
                sum->data[nnz_counter] = temp[col_start];
                nnz_counter++;
            }

            int t = col_start;
            col_start = next[col_start];
            next[t] = -1;
            temp[t] = 0;
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
    CSR ABC, DEF;
    int nnz_ABC = A->NZ + B->NZ + C->NZ;
    int nnz_DEF = D->NZ + E->NZ + F->NZ;

    alloc_sparse_CSR(A->m, A->n, nnz_ABC, &ABC);
    alloc_sparse_CSR(D->m, D->n, nnz_DEF, &DEF);

    sum(A, B, C, ABC);
    sum(D, E, F, DEF);

    optimised_sparsemm_CSR(ABC, DEF, R);
}
