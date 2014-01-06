#ifndef AE_H
#define AE_H

#include "NN_utils.h"
#include "NN_math.h"
#include "NN_core.h"

typedef struct {
    NN_Matrix *A1;
    NN_Matrix *H;
    NN_Matrix *dH;

    NN_Matrix *A2;
    NN_Matrix *O;
    NN_Matrix *dO;

    NN_Matrix *J;
    NN_Matrix *dJ;
} AE_interim;

void AE(NN_Matrix* W, NN_Matrix* hb, NN_Matrix* vb,
        NN_Matrix* x, NN_Matrix* y,
        regularization_option reg_option, actFuncType _actFuncType, errFuncType _errFuncType,
        element *cost, NN_Matrix *dW, NN_Matrix *dhb, NN_Matrix *dvb, AE_interim *tmp_vars);

#endif
