#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>

struct coord {
    int i, j;
};

struct _p_COO {
    int m, n, NZ;
    int * row_indices;
    int * col_indices;
    double *data;
};

struct _p_CSR {
    int m, n, NZ;
    double *data;   //just all the data.
    int *col_indices; //column indices of the values
    int *row_start; //index to the start of each row in data
};

typedef struct _p_CSR *CSR;
typedef struct _p_COO *COO;

void alloc_sparse(int, int, int, COO*);
void alloc_sparse_CSR(int, int, int, CSR*);
void free_sparse(COO*);
void free_CSR(CSR*);
void alloc_dense(int, int, double **);
void free_dense(double **);
void zero_dense(int, int, double *);

void convert_sparse_to_dense(const COO, double **);
void convert_dense_to_sparse(const double *, int, int, COO *);
void coo_to_csr(const COO, CSR *);
void csr_to_coo(const CSR, COO *);

void read_sparse(const char *, COO *);
void write_sparse(FILE *, COO);
void print_sparse(COO);
void random_matrix(int, int, double, COO *);

#endif
