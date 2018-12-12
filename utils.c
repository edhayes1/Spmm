#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include "utils.h"


#ifdef _MSC_VER
double drand48()
{
  return (double)rand()/(RAND_MAX + 1);
}
#endif  /* _MSC_VER */

/*
 * Allocate a dense matrix
 * m - number of rows
 * n - number of columns
 * dense - newly allocated matrix.
 */
void alloc_dense(int m, int n, double **dense)
{
  *dense = malloc(m*n*sizeof(**dense));
}

/*
 * Free a dense matrix
 * dense - dense matrix, may be NULL
 */
void free_dense(double **dense)
{
    if (!*dense) {
        return;
    }
    free(*dense);
    *dense = NULL;
}

/*
 * Zero a dense matrix
 * m - number of rows
 * n - number of columns
 * dense - matrix to zero.
 */
void zero_dense(int m, int n, double *dense)
{
    int i, j;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            dense[j*m + i] = 0;
        }
    }
}

/*
 * Allocate a sparse matrix in coordinate format.
 * m - number of rows
 * n - number of columns
 * NZ - number of nonzeros
 * sparse - newly allocated matrix.
 */
void alloc_sparse(int m, int n, int NZ, COO *sparse)
{
    COO sp = calloc(1, sizeof(struct _p_COO));
    sp->m = m;
    sp->n = n;
    sp->NZ = NZ;
    sp->coords = calloc(NZ, sizeof(struct coord));
    sp->data = calloc(NZ, sizeof(double));
    *sparse = sp;
}

// TODO change col_indeces and row start
void alloc_sparse_CSR(int m, int n, int NZ, CSR *sparse)
{
    CSR sp = calloc(1, sizeof(struct _p_CSR));
    sp->m = m;
    sp->n = n;
    sp->NZ = NZ;
    sp->data = calloc(NZ, sizeof(double));
    sp->col_indices = calloc(NZ, sizeof(int));
    sp->row_start = calloc(m+1, sizeof(int));
    *sparse = sp;
}

// TODO ignore for now
void alloc_sparse_CSR_with_rp(int m, int n, int * rp, CSR *sparse)
{
    CSR sp = calloc(1, sizeof(struct _p_CSR));
    int NZ = rp[m];
    sp->m = m;
    sp->n = n;
    sp->NZ = NZ;
    sp->data = calloc(NZ, sizeof(double));
    sp->col_indices = calloc(NZ, sizeof(int));
    sp->row_start = rp;
    *sparse = sp;
}


/*
 * Free a sparse matrix.
 * sparse - sparse matrix, may be NULL
 */
void free_sparse(COO *sparse)
{
    COO sp = *sparse;
    if (!sp) {
        return;
    }
    free(sp->coords);
    free(sp->data);
    free(sp);
    *sparse = NULL;
}

//TODO vairables can change
void free_CSR(CSR *sparse)
{
    CSR sp = *sparse;
    if (!sp) {
        return;
    }
    free(sp->data);
    free(sp->row_start);
    free(sp->col_indices);
    free(sp);
    *sparse = NULL;
}

/*
 * Convert a sparse matrix to dense format in column major format.
 *
 * sparse - The sparse matrix to convert
 * dense - pointer to output dense matrix (will be allocated)
 */
void convert_sparse_to_dense(const COO sparse, double **dense)
{
    int n;
    int i, j;
    alloc_dense(sparse->m, sparse->n, dense);
    zero_dense(sparse->m, sparse->n, *dense);
    for (n = 0; n < sparse->NZ; n++) {
        i = sparse->coords[n].i;
        j = sparse->coords[n].j;
        (*dense)[j * sparse->m + i] = sparse->data[n];
    }
}

/*
 * Convert a dense matrix in column major format to sparse.
 * Entries with absolute value < 1e-15 are flushed to zero and not
 * stored in the sparse format.
 *
 * dense - the dense array
 * m - number of rows
 * n - number of columns
 * sparse - output sparse matrix (allocated by this routine)
 */
void convert_dense_to_sparse(const double *dense, int m, int n,
                             COO *sparse)
{
    int i, j, NZ;
    COO sp;
    NZ = 0;
    /* Figure out how many nonzeros we're going to have. */
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            double val = dense[j*m + i];
            if (fabs(val) > 1e-15) {
                NZ++;
            }
        }
    }
    alloc_sparse(m, n, NZ, &sp);

    NZ = 0;
    /* Fill up the sparse matrix */
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double val = dense[j*m + i];
            if (fabs(val) > 1e-15) {
                sp->coords[NZ].i = i;
                sp->coords[NZ].j = j;
                sp->data[NZ] = val;
                NZ++;
            }
        }
    }
    *sparse = sp;
}

/*
 * Create a random sparse matrix
 *
 * m - number of rows
 * n - number of columns
 * frac - fraction of entries that should be nonzero
 * sparse - newly allocated random matrix.
 */
void random_matrix(int m, int n, double frac, COO *sparse)
{
    int i, j;
    double *d;
    alloc_dense(m, n, &d);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            if (drand48() < frac) {
                d[j*m + i] = drand48();
            } else {
              d[j*m + i] = 0.0;
            }
        }
    }
    convert_dense_to_sparse(d, m, n, sparse);
    free_dense(&d);
}

/*
 * Read a sparse matrix from a file.
 *
 * file - The filename to read
 * sparse - The newly read sparse matrix (allocated here)
 */
void read_sparse(const char *file, COO *sparse)
{
    COO sp;
    int i, j, k, m, n, NZ;
    double val;
    int c;
    FILE *f = fopen(file, "r");
    if (!f) {
        fprintf(stderr, "Unable to open %s for reading.\n", file);
        exit(1);
    }
    c = fscanf(f, "%d %d %d\n", &m, &n, &NZ);
    if (c != 3) {
        fprintf(stderr, "File format incorrect on line 1, expecting 3 integers, got %d\n", c);
        fclose(f);
        exit(1);
    }
    if (NZ > (uint64_t)m*n) {
        fprintf(stderr, "More nonzeros (%d) than matrix entries (%d x %d)!\n", NZ, m, n);
        fclose(f);
        exit(1);
    }
    alloc_sparse(m, n, NZ, &sp);
    k = 0;
    while ((c = fscanf(f, "%d %d %lg\n", &i, &j, &val)) == 3) {
        if (k >= NZ) {
            fprintf(stderr, "File has nonzero lines than expected (%d)\n", NZ);
            fclose(f);
            free_sparse(&sp);
            exit(1);
        }
        if (i >= m || j >= n) {
            fprintf(stderr, "Entry on line %d incorrect, index (%d, %d) out of bounds for %d x %d matrix\n", k + 2, i, j, m, n);
            fclose(f);
            free_sparse(&sp);
            exit(1);
        }
        sp->coords[k].i = i;
        sp->coords[k].j = j;
        sp->data[k] = val;
        k++;
    }

    if (k != NZ) {
        fprintf(stderr, "File has fewer lines (%d) than expected (%d)\n",
                k, NZ);
        fclose(f);
        free_sparse(&sp);
        exit(1);
    }
    *sparse = sp;
    fclose(f);
}

// TODO ignore for now
//void sort(COO coo){
//
//    int nz = coo->NZ;
//    int i = 0;
//    int changes = 1;
//    int first = 0;
//
//    while(changes || !first){
//        first = i%2;
//        changes = 0;
//
//        #pragma acc parallel loop
//        for (int j = first; j < nz - i; j+=2){
//            if( coo->coords[j].i > coo->coords[j+1].i )
//            {
//                struct coord tmp_coord = coo->coords[j+1];
//                double tmp_data = coo->data[j+1];
//
//                coo->coords[j+1] = coo->coords[j];
//                coo->data[j+1] = coo-> data[j];
//
//                coo->coords[j] = tmp_coord;
//                coo->data[j] = tmp_data;
//                changes = 1;
//            }
//        }
//
//        i++;
//    }
//
//    for (int i = 0; i < coo->NZ; i++){
//        if( coo->coords[i].i > coo->coords[i+1].i )
//            printf("not sorted\n\n %d, %d, i: %d", coo->coords[i].i, coo->coords[i+1].i, i);
//    }
//
//}

//TODO rewrite
void coo_to_csr(COO coo, CSR *sparse) {
    CSR sp;
    int m = coo->m;
    int n = coo->n;
    int NZ = coo->NZ;
    int * temp_rp = calloc(m+1, sizeof(int));
    alloc_sparse_CSR(m, n, NZ, &sp);

    // bubble sort
//    #ifdef multi
        //sort(coo);
//    #endif

    //fill in non zeros
    memcpy(sp->data, coo->data, NZ * sizeof(double));

    //fill in column indices
    for (int i = 0; i < NZ; i++){
        sp->col_indices[i]= coo->coords[i].j;
    }

    //get num nz per row
    #pragma acc parallel loop
    for(int i = 0; i < NZ; i++){
        temp_rp[coo->coords[i].i]++;
    }

    for (int i = 0, row_sum = 0; i <= m; i++){
        int temp = temp_rp[i];
        temp_rp[i] = row_sum;
        row_sum += temp;
    }

    memcpy(sp->row_start, temp_rp, (m+1) * sizeof(int));

    #pragma acc parallel loop
    for (int i = 0; i < NZ; i++){
        int row = coo->coords[i].i;
        int index = temp_rp[row];
        sp->data[index] = coo->data[i];
        sp->col_indices[index] = coo->coords[i].j;
        #pragma acc atomic update
        temp_rp[row]++;
    }

    free(temp_rp);

    *sparse = sp;
}

// TODO rewrite
void csr_to_coo(CSR csr, COO *sparse){
    COO sp;
    int m = csr->m;
    int n = csr->n;
    int NZ = csr->NZ;
    alloc_sparse(m,n, NZ, &sp);

    //fill in non zeros
    memcpy(sp->data, csr->data, NZ * sizeof(double));

    //fill in column indices
    #pragma acc parallel loop
    for (int i = 0; i < NZ; i++){
        sp->coords[i].j = csr->col_indices[i];
    }

    //get num nz per row
    int * temp = calloc(m, sizeof(int));
    #pragma acc parallel loop
    for (int i = 0; i < m; i++){
        temp[i] = csr->row_start[i+1] - csr->row_start[i];
    }

    //fill row nums
    for (int i = 0, idx = 0; i < m; i++){
        for (int r = 0; r < temp[i]; r++){
            sp->coords[idx].i = i;
            idx++;
        }
    }

    *sparse = sp;
}

/*
 * Write a sparse matrix to a file.
 *
 * f - The file handle.
 * sp - The sparse matrix to write.
 */
void write_sparse(FILE *f, COO sp)
{
    int i;
    fprintf(f, "%d %d %d\n", sp->m, sp->n, sp->NZ);
    for (i = 0; i < sp->NZ; i++) {
        fprintf(f, "%d %d %.6g\n", sp->coords[i].i, sp->coords[i].j, sp->data[i]);
    }
}

/*
 * Print a sparse matrix to stdout
 *
 * sp - The sparse matrix to print.
 */
void print_sparse(COO sp)
{
    write_sparse(stdout, sp);
}

void read_sparse_binary(const char *file, COO *sparse)
{
    COO sp;
    int m, n, NZ;
    size_t nread;
    FILE *f = fopen(file, "r");
    if (!f) {
        fprintf(stderr, "Unable to open %s for reading.\n", file);
        exit(1);
    }
    nread = fread(&m, sizeof(m), 1, f);
    if (nread != 1) {
      fprintf(stderr, "Did not read rows from file\n");
      exit(1);
    }
    nread = fread(&n, sizeof(n), 1, f);
    if (nread != 1) {
      fprintf(stderr, "Did not read columns from file\n");
      exit(1);
    }
    nread = fread(&NZ, sizeof(NZ), 1, f);
    if (nread != 1) {
      fprintf(stderr, "Did not read number of nonzeros from file\n");
      exit(1);
    }
    alloc_sparse(m, n, NZ, &sp);
    nread = fread(sp->coords, sizeof(*sp->coords), NZ, f);
    if (nread != NZ) {
      fprintf(stderr, "Did not read nonzero locations from file\n");
      exit(1);
    }
    nread = fread(sp->data, sizeof(*sp->data), NZ, f);
    if (nread != NZ) {
      fprintf(stderr, "Did not read nonzero values from file\n");
      exit(1);
    }
    *sparse = sp;
    fclose(f);
}

void write_sparse_binary(FILE *f, COO sp)
{
  size_t nwrite;
  nwrite = fwrite(&(sp->m), sizeof(sp->m), 1, f);
  if (nwrite != 1) {
    fprintf(stderr, "Could not write rows to output file\n");
    exit(1);
  }

  nwrite = fwrite(&(sp->n), sizeof(sp->n), 1, f);
  if (nwrite != 1) {
    fprintf(stderr, "Could not write columns to output file\n");
    exit(1);
  }

  nwrite = fwrite(&(sp->NZ), sizeof(sp->NZ), 1, f);
  if (nwrite != 1) {
    fprintf(stderr, "Could not write number of nonzeros to output file\n");
    exit(1);
  }
  nwrite = fwrite(sp->coords, sizeof(*sp->coords), sp->NZ, f);
  if (nwrite != sp->NZ) {
    fprintf(stderr, "Could not write nonzero locations to output file\n");
    exit(1);
  }
  nwrite = fwrite(sp->data, sizeof(*sp->data), sp->NZ, f);
  if (nwrite != sp->NZ) {
    fprintf(stderr, "Could not write nonzero values to output file\n");
    exit(1);
  }
}
