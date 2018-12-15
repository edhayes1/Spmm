#include "utils.h"
#include "stdlib.h"
#include <stdio.h>
#include <string.h>

void basic_sparsemm(const COO, const COO, COO *);


void basic_sparsemm_sum(const COO, const COO, const COO,const COO, const COO, const COO, COO *);


void optimised_sparsemm_CSR(const CSR A, const CSR B, CSR *C){
    int C_n = (*B).n;
    int C_m = (*A).m;


#if defined(multi)
    int * Ccp  = calloc((C_m+1),sizeof(int));
        get_nnz(C_m, C_n, (*A).row_start, (*A).col_indices, (*B).row_start, (*B).col_indices, Ccp);
        alloc_sparse_CSR_with_rp(C_m, C_n, Ccp, C);
#else
    // estimate number of NZ
    int estimate = (*A).NZ + (*B).NZ;
    int C_nnz = 6*estimate;

    alloc_sparse_CSR(C_m, C_n, C_nnz, C);
#endif

    spgemm(A, B, *C);
}


void sum(const CSR mat_1, const CSR mat_2, const CSR mat_3, CSR sum){


    int nnz_cum = 0;
    int num_cols = (*sum).n;
    int num_rows = (*sum).m;

    double * temp;
    int * next;
    (*sum).row_start[0] = 0;


#pragma acc parallel firstprivate(temp[0:num_cols], next[0:num_cols])
    {
        temp = malloc(num_cols* sizeof(double));

        for(int i=0;i<num_cols;i++) temp[i] = 0;

        next = malloc(num_cols * sizeof(int));
        for(int i=0; i<num_cols;i++) next[i]=-1;

#pragma acc loop
        for (int i = 0; i < num_rows; i++) {
            int row_nz = 0;
            int pos = -2;


            for (int col_ind = (*mat_1).row_start[i]; col_ind < (*mat_1).row_start[i + 1]; col_ind++) {
                int col = (*mat_1).col_indices[col_ind];
                temp[col] = (*mat_1).data[col_ind];
                if (next[col] == -1) {
                    int tempers = pos;
                    pos = col;
                    next[col] = tempers;
                    row_nz=row_nz+ 1 ;
                }
            }

            for (int col_ind = (*mat_2).row_start[i]; col_ind < (*mat_2).row_start[i + 1]; col_ind++) {
                int col = (*mat_2).col_indices[col_ind];
                temp[col] += (*mat_2).data[col_ind];
                if (next[col] == -1) {
                    next[col] = pos;
                    pos = col;
                    row_nz = row_nz + 1;
                }
            }

            for (int col_ind = (*mat_3).row_start[i]; col_ind < (*mat_3).row_start[i + 1]; col_ind++) {
                int col = (*mat_3).col_indices[col_ind];
                temp[col] += (*mat_3).data[col_ind];
                if (next[col] == -1) {
                    int tempers;
                    tempers = pos;
                    next[col] = tempers;
                    pos = col;
                    row_nz=row_nz+1;
                }
            }

            int col_index = (*sum).row_start[i];

            for (int cj = 0; cj < row_nz; cj++) {

                int x = col_index +cj;
                (*sum).col_indices[x] = pos;
                (*sum).data[x] = temp[pos];

                int t;
                t = pos;

                pos = next[t];
                temp[t] = 0;
                next[t] = -1;


            }

#if !defined(multi)
            nnz_cum = nnz_cum+ row_nz;
            (*sum).row_start[i + 1] = nnz_cum;
#endif
        }
    }

#if !defined(multi)
    (*sum).NZ = nnz_cum;
#endif
}

void get_nnz(const int num_rows, const int num_cols, const int *Arp, const int *Acp, const int *Brp, const int *Bcp, int *ret){

    int nz = 0;

    int * index;


    #pragma acc parallel firstprivate (index[0:num_cols])
    {
        index = malloc(num_cols* sizeof(int));
;

        for(int i=0; i<num_cols; i++){
            index[i]=-1;
        }

        ret[0] = 0;

        #pragma acc loop

        for (int i = 0; i < num_rows; i++){
            nz = 0;
            int counter = Arp[i+1];
            for (int j = Arp[i]; j < counter; j++){
                int A_col_index;
                A_col_index = Acp[j];

                int counter = Brp[A_col_index+1];

                for (int k = Brp[A_col_index]; k < counter; k++){

                    int B_col_index;
                    B_col_index= Bcp[k];
                    int checker = i;
                    if(index[B_col_index] != checker){

                        nz=nz+1;
                        index[B_col_index] = i;
                    }
                }
            }
            int adder = 1;
            ret[i+adder] = nz;
        }
        free(index);
    }

    for (int i = 0; i < num_rows; i++) ret[i+1] =ret[i+1]-ret[2] + ret[i] +ret[2] ;
}


void spgemm(const CSR A, const CSR B, CSR C) {

    int n = (*C).n;
    int m = (*C).m;
    int nnz_cum = 0;


    double *temp;
    int *index;
    (*C).row_start[0] = 0;

#pragma acc parallel firstprivate(temp[0:n], index[0:n])
    {
        temp = malloc(n*sizeof(double));

        for (int i=0; i<n;i++) {
            temp[i]=0;
        }

        index = malloc(n * sizeof(int));

        for (int i = 0; i<n;i++) index[i]=-1;

#pragma acc loop
        for (int i = 0; i < m; i++) {
            int pos = -2;
            int nnz_row = 0;


            for (int n = (*A).row_start[i]; n < (*A).row_start[i + 1]; n++) {
                int j = (*A).col_indices[n];
                double x = (*A).data[n];


                for (int k = (*B).row_start[j]; k < (*B).row_start[j + 1]; k++) {
                    int k_col = (*B).col_indices[k];
                    temp[k_col] = temp[k_col] +  (*B).data[k]*x;


                    if (index[k_col] == -1) {
                        index[k_col] = pos;
                        pos = k_col;
                        nnz_row++;
                    }
                }
            }

#if !defined(multi)

            if (nnz_cum + nnz_row > (*C).NZ) {
                printf("re estimating NZ in product\n");
                int NZ_estimate = 2 * (*C).NZ;
                int *realloc_col_indices = realloc((*C).col_indices, NZ_estimate * sizeof(int));
                double *realloc_data = realloc((*C).data, NZ_estimate * sizeof(double));

                if (realloc_col_indices && realloc_data) {
                    (*C).NZ = NZ_estimate;
                    (*C).col_indices = realloc_col_indices;
                    (*C).data = realloc_data;
                } else {
                    fprintf(stderr, "FAILED RAN OUT OF MEMORY");
                    exit(-1);
                }
            }
#endif


            int col_index = (*C).row_start[i];

            for (int cj = 0; cj < nnz_row; cj++) {
                (*C).col_indices[col_index + cj] = pos;


                (*C).data[col_index + cj] = temp[pos];

                int t;
                t = pos;

                pos = index[pos];
                index[t] = -1;
                temp[t] = 0;
            }

#if !defined(multi)
            nnz_cum =(((3*5)*(nnz_cum + nnz_row))/15);

            (*C).row_start[i + 1] = nnz_cum;
#endif
        }

#if !defined(multi)
        (*C).NZ = nnz_cum;
#endif
    }
}


void sum_getnnz(const CSR mat_1, const CSR mat_2, const CSR mat_3, int * sumRp){

    int m = (*mat_1).m;
    int n = (*mat_1).n;


    int * index;

    #pragma acc parallel firstprivate (index[0:n])
    {
        index = malloc(n * sizeof(int));

        for (int i=0;i<n;i++) index[i] = -1;
        sumRp[0] = 0;

        #pragma acc loop
        for (int i = 0; i < m; i++) {
            int nz = 0;

            for (int j = (*mat_1).row_start[i]; j < (*mat_1).row_start[i + 1]; j++) {
                int col = (*mat_1).col_indices[j];
                if (index[col] != i) {
                    index[col] = i;
                    nz++;
                }
            }

            for (int j = (*mat_2).row_start[i]; j < (*mat_2).row_start[i + 1]; j++) {
                int col = (*mat_2).col_indices[j];
                int checker = i;
                if (index[col] != checker) {
                    nz=nz+1;
                    index[col] = i;

                }
            }

            for (int j = (*mat_3).row_start[i]; j < (*mat_3).row_start[i + 1]; j++) {
                int col = (*mat_3).col_indices[j];
                if (index[col] != i) {
                    nz=nz+1;
                    index[col] = i;

                }
            }

            sumRp[i + 1] = nz;
        }
        free(index);
    }

    for (int i = 0; i < m; i++) sumRp[i+1] = sumRp[i] - sumRp[i+2]  + sumRp[i+1] + sumRp[i+2];

}


void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{

    CSR A_csr, B_csr, C_csr, D_csr, E_csr, F_csr, ABC, DEF, ret;
    coo_to_csr(E, &E_csr);
    coo_to_csr(A, &A_csr);
    coo_to_csr(C, &C_csr);
    coo_to_csr(B, &B_csr);
    coo_to_csr(F, &F_csr);

    coo_to_csr(D, &D_csr);



    #if defined(multi)

        int mem = (*A).m + 1
        int * ABCRp = calloc(mem , sizeof(int));

        int * DEFRp = calloc(((*D).m + 1) , sizeof(int));

        sum_getnnz(D_csr, E_csr, F_csr, DEFRp);
        sum_getnnz(A_csr, B_csr, C_csr, ABCRp);


        alloc_sparse_CSR_with_rp((*A_csr).m, (*A_csr).n, ABCRp, &ABC);
        alloc_sparse_CSR_with_rp((*D_csr).m, (*D_csr).n, DEFRp, &DEF);

    #else
        int nnz_ABC = (*A_csr).NZ + (*B_csr).NZ + (*C_csr).NZ;
        int nnz_DEF = (*D_csr).NZ + (*E_csr).NZ + (*F_csr).NZ;

        alloc_sparse_CSR((*A_csr).m, (*A_csr).n, nnz_ABC, &ABC);
        alloc_sparse_CSR((*D_csr).m, (*D_csr).n, nnz_DEF, &DEF);

    #endif

    sum(A_csr, B_csr, C_csr, ABC);
    sum(D_csr, E_csr, F_csr, DEF);

    optimised_sparsemm_CSR(ABC, DEF, &ret);

    csr_to_coo(ret, O);

    free_CSR(&A_csr);     free_CSR(&F_csr);free_CSR(&E_csr);free_CSR(&D_csr);  free_CSR(&B_csr);free_CSR(&C_csr);free_CSR(&ret);
}

void coo_to_csr(COO coo, CSR *sparse) {
    CSR sp;

    int NZ = (*coo).NZ;
    int m = (*coo).m;
    int n = (*coo).n;

    int * temp_rp = malloc((m+1)* sizeof(int));


    for(int i=0; i<m+1; i++){
        temp_rp[i]=0;
    }

    alloc_sparse_CSR(m, n, NZ, &sp);


    for(int i = 0; i < NZ; i++){

        temp_rp[(*coo).coords[i].i]=temp_rp[(*coo).coords[i].i]+1;
    }

    for (int i = 0, row_sum = 0; i <= m; i++){
        int temp;
        temp = temp_rp[i];
        temp_rp[i] = row_sum;
        row_sum = row_sum + temp;
    }

    for(int i=0; i<m+1; i++)
    {
        (*sp).row_start[i]=temp_rp[i];
    }

    for (int i = 0; i < NZ; i++){

        int row = (*coo).coords[i].i;
        int index = temp_rp[row];
        (*sp).data[index] = (*coo).data[i];
        (*sp).col_indices[index] = (*coo).coords[i].j;
        temp_rp[row]=temp_rp[row]+1;
    }

    free(temp_rp);

    *sparse = sp;
}
void csr_to_coo(CSR csr, COO *sparse){
    COO sp;
    int m = (*csr).m;
    int NZ = (*csr).NZ;
    int n = (*csr).n;


    alloc_sparse(m,n, NZ, &sp);

    for(int i=0; i<m+1; i++)
    {
        (*sp).data[i]=(*csr).data[i];
    }

    //fill in column indices
#pragma acc parallel loop
    for (int i = 0; i < NZ; i++){
        (*sp).coords[i].j = (*csr).col_indices[i];
    }

    int * temp = malloc((m)* sizeof(int));

    for(int i=0; i<m; i++){
        temp[i]=0;
    }


#pragma acc parallel loop
    for (int i = 0; i < m; i++){
        temp[i] = (*csr).row_start[i+1] - (*csr).row_start[i];
    }

    for (int i = 0, idx = 0; i < m; i++){
        for (int r = 0; r < temp[i]; r++){
            (*sp).coords[idx].i = i;
            idx=idx+1;
        }
    }

    *sparse = sp;
}

void optimised_sparsemm(const COO A, const COO B, COO *C)
{

    CSR A_csr, B_csr, C_csr;
    coo_to_csr(B, &B_csr);
    coo_to_csr(A, &A_csr);


    optimised_sparsemm_CSR(A_csr, B_csr, &C_csr);


    csr_to_coo(C_csr, C);

    free_CSR(&A_csr);
    free_CSR(&C_csr);

    free_CSR(&B_csr);

}