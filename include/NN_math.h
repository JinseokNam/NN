#ifndef NN_MATH_H
#define NN_MATH_H

#ifdef USE_MKL_BLAS
#define USE_BLAS
#include <mkl.h>
#else
#define USE_BLAS
#include <cblas.h>
#endif

#include "NN_utils.h"

/********************************************

    Wrapper functions of BLAS interface

**********************************************/

/*
    Matrix-Matrix multiplication
    in-place
    C = \alpha x A x B + \beta x C
*/
NN_Matrix* mul_mmi(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B, Transpose TransB, element beta, NN_Matrix* C);

/*
    Matrix-Matrix multiplication
    C = \alpha x A x B 
*/
NN_Matrix* mul_mm(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B, Transpose TransB);

/*
    (Dense) Matrix-(Sparse)Matrix multiplication
    C = \alpha x dA x sB + \beta x C
*/
NN_Matrix* mul_dmsmi(const element alpha, NN_Matrix* A, cs* B, const element beta, NN_Matrix* C);

/*
    (Dense) Matrix-(Sparse)Matrix multiplication
    C = \alpha x dA x sB
*/
NN_Matrix* mul_dmsm(const element alpha, NN_Matrix* A, cs* B);

/*
    Matrix-vector multiplication
    C = \alpha x A x B  where A is a matrix and B is a vector
*/
NN_Matrix* mul_mv(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B);


/*
    Matrix-vector multiplication
    in-place
    C = \alpha x A x B  + \beta x C     where A is a matrix and both B and C are a vector
*/
NN_Matrix* mul_mvi(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B, element beta, NN_Matrix* C);

/*
    element-wise matrix multiplication 
    C = A .* B
*/
NN_Matrix* elemwise_mm      ( NN_Matrix* A, NN_Matrix* B );

/*
    element-wise matrix multiplication
    B = A .* B
*/
NN_Matrix* elemwise_mmi     ( NN_Matrix* A, NN_Matrix* B );

/*
    element-wise matrix division
    C = B ./ A
*/
NN_Matrix* elemwise_div     ( NN_Matrix* A, NN_Matrix* B );

/*
    element-wise matrix division
    B = B ./ A
*/
NN_Matrix* elemwise_divi    ( NN_Matrix* A, NN_Matrix* B );

/*
element* div_mv(const element alpha, const element* A, const enum CBLAS_TRANSPOSE TransA, const element* B, size_t m, size_t n);
*/

NN_Matrix* bsxplus_mv       ( NN_Matrix* A, NN_Matrix* B );
NN_Matrix* ibsxplus_mv      ( NN_Matrix* A, NN_Matrix* B );
NN_Matrix* bsxtimes_mv      ( NN_Matrix* A, NN_Matrix* B );
NN_Matrix* ibsxtimes_mv     ( NN_Matrix* A, NN_Matrix* B );
NN_Matrix* bsxrdivide_mv    ( NN_Matrix* A, NN_Matrix* B );
NN_Matrix* ibsxrdivide_mv   ( NN_Matrix* A, NN_Matrix* B );

NN_Matrix* add          ( element alpha, NN_Matrix* A, element beta, NN_Matrix* B );
NN_Matrix* addi         ( element alpha, NN_Matrix* A, element beta, NN_Matrix* B );

NN_Matrix* add_col_vec      ( element alpha, NN_Matrix* A, unsigned int* col_idx, size_t nIndices, NN_Matrix* B );
NN_Matrix* add_row_vec      ( element alpha, NN_Matrix* A, unsigned int* row_idx, size_t nIndices, NN_Matrix* B );
/*
    dot product of two vector A and B
*/
//element dot(element* A, element* B, size_t m);

/*
    Utility functions for NN
*/
NN_Matrix* NN_copy_matrix   ( NN_Matrix* A );

NN_Matrix* NN_sumi      (NN_Matrix* A, unsigned int axis, NN_Matrix* B);
NN_Matrix* NN_sum       (NN_Matrix* A, unsigned int axis);
NN_Matrix* NN_meani     (NN_Matrix* A, unsigned int axis, NN_Matrix* B);
NN_Matrix* NN_mean      (NN_Matrix* A, unsigned int axis);

NN_Matrix* NN_imins     (NN_Matrix* A, element s);
NN_Matrix* NN_mins      (NN_Matrix* A, element s);
NN_Matrix* NN_imin      (NN_Matrix* A, int axis, NN_Matrix* B);
NN_Matrix* NN_min       (NN_Matrix* A, int axis);

NN_Matrix* NN_muli      ( NN_Matrix* A, element s );
NN_Matrix* NN_mul       ( NN_Matrix* A, element s );
NN_Matrix* NN_rdivi     ( element s, NN_Matrix* A );
NN_Matrix* NN_rdiv      ( element s, NN_Matrix* A );
NN_Matrix* NN_negi      ( NN_Matrix* A );
NN_Matrix* NN_neg       ( NN_Matrix* A );
NN_Matrix* NN_addsi     ( element alpha, NN_Matrix* A, element s );
NN_Matrix* NN_adds      ( element alpha, NN_Matrix* A, element s );
NN_Matrix* NN_powi      ( NN_Matrix* A, element exponent );
NN_Matrix* NN_pow       ( NN_Matrix* A, element exponent );
NN_Matrix* NN_logi      ( NN_Matrix* A );
NN_Matrix* NN_log       ( NN_Matrix* A );
NN_Matrix* NN_expi      ( NN_Matrix* A );
NN_Matrix* NN_exp       ( NN_Matrix* A );
NN_Matrix* NN_logi      ( NN_Matrix* A );
NN_Matrix* NN_log       ( NN_Matrix* A );
NN_Matrix* NN_sqrti     ( NN_Matrix* A );
NN_Matrix* NN_sqrt      ( NN_Matrix* A );

#endif
