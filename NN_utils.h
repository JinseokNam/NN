#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "cs.h"

#define MATRIX_IDX(n, i, j) j*n + i
#define MATRIX_ELEMENT(A, m, n, i, j) A[ MATRIX_IDX(m, i, j) ]

#ifdef FLOAT
typedef float element;
#else
typedef double element;
#endif

typedef enum {false,true} bool;
typedef enum {T,N} Transpose;
typedef enum {cosineDist,squaredError,crossEntropy} errFuncType;
typedef enum {sigmoid,rectified,_tanh} actFuncType;

typedef struct {
	element* data;
	size_t nrows;
	size_t ncols;
} NN_Matrix;

typedef struct {
	element l1_lambda;
	element l2_lambda;
	bool normalize;
} regularization_option; 

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);
void timeval_print(struct timeval *tv);

int 			NN_randi(int r_min, int r_max);
element 		NN_rand();
element 		NN_randn();
unsigned int* 	NN_randperm(size_t n);
void 			NN_irandperm(unsigned int* v,size_t n);

NN_Matrix* 	NN_rand_matrix(size_t m, size_t n);
NN_Matrix* 	NN_randn_matrix(size_t m, size_t n);
NN_Matrix* 	NN_randi_matrix(int r_min, int r_max, size_t m, size_t n);
NN_Matrix* 	NN_binornd_matrix(unsigned int n_trials, float p, size_t m, size_t n);
NN_Matrix* 	NN_zeros_matrix(size_t m, size_t n);
NN_Matrix* 	NN_fill_zeros(NN_Matrix* A);
void 		NN_fill_values(NN_Matrix* dst, NN_Matrix* src);
NN_Matrix* 	NN_ones_matrix(size_t m, size_t n);
NN_Matrix* 	NN_init_matrix(size_t m, size_t n);

bool 		NN_is_same_size(NN_Matrix* A, NN_Matrix* B);

NN_Matrix* 	NN_infinity_to_zero(NN_Matrix* A);
NN_Matrix* 	NN_negative_to_zero(NN_Matrix* A);

NN_Matrix* 	NN_lt(NN_Matrix* A, element s);
NN_Matrix* 	NN_lti(NN_Matrix* A, element s, NN_Matrix* B);
NN_Matrix* 	NN_gt(NN_Matrix* A, element s);
NN_Matrix* 	NN_gti(NN_Matrix* A, element s, NN_Matrix* B);
NN_Matrix* 	NN_eq(NN_Matrix* A, element s);
NN_Matrix* 	NN_eqi(NN_Matrix* A, element s, NN_Matrix* B);
NN_Matrix* 	NN_neq(NN_Matrix* A, element s);
NN_Matrix* 	NN_neqi(NN_Matrix* A, element s, NN_Matrix* B);
NN_Matrix*	NN_max_s(NN_Matrix* A, element s, NN_Matrix* Y);


bool 		NN_isvec(NN_Matrix* A);
size_t 		NN_maxdim(NN_Matrix* A);

NN_Matrix* 	NN_submatrix_sp(cs* A, int r_left, int r_right, int c_left, int c_right);
void 		NN_isubmatrix_sp(cs* A, int r_left, int r_right, int c_left, int c_right, NN_Matrix* ret);
NN_Matrix* 	NN_submatrix(NN_Matrix* A, int r_left, int r_right, int c_left, int c_right);
void 		NN_isubmatrix(NN_Matrix* A, int r_left, int r_right, int c_left, int c_right, NN_Matrix* ret);
NN_Matrix* 	NN_subColumns(NN_Matrix* A, unsigned int* colIndices, size_t nIndices);
void 		NN_isubColumns(NN_Matrix* A, unsigned int* colIndices, size_t nIndices, NN_Matrix* ret);
cs* 		NN_subColumns_sp(cs* A, int c_left, int c_right);
NN_Matrix* 	NN_subRows(NN_Matrix* A, unsigned int* rowIndices, size_t nIndices);
void 		NN_isubRows(NN_Matrix* A, unsigned int* rowIndices, size_t nIndices, NN_Matrix* ret);

void 		NN_sp_to_full(NN_Matrix* dst, cs* src);

FILE* 		NN_fopen(char* filepath,char* mode);
cs* 		NN_load_spdata(FILE* fp);

void 		NN_error(const char *err_msg);
NN_Matrix*	NN_free(NN_Matrix *ptr);
void 		NN_print_matrix(const char* msg, const NN_Matrix* A);
void 		NN_show_dim(const char* msg, const NN_Matrix* A);
element 	NN_get_value(const NN_Matrix* A, unsigned int i, unsigned int j);
element 	NN_get_value_sp(const cs* A, unsigned int i, unsigned int j);
void 		NN_set_value(NN_Matrix* A, unsigned int i, unsigned int j, element v);

void NN_current_time();

#endif
