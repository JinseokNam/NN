#ifndef NN_CORE_H
#define NN_CORE_H

#include "NN_utils.h"
struct _parameters {
    element **weights;
    element **biases;
};
typedef struct _parameters NN_params;

struct _gradients {
    element cost;
    element **dW;
    element **dB;
};
typedef struct _gradients NN_grads;

void tanh_layer(NN_Matrix* A, NN_Matrix* f, NN_Matrix* dFA);
void sigmoid_layer(NN_Matrix* A, NN_Matrix* f, NN_Matrix* dFA);
void relu_layer(NN_Matrix* A, NN_Matrix* f, NN_Matrix* dFA);
void linear_layer(NN_Matrix* W, Transpose trans, NN_Matrix* x, NN_Matrix* b, NN_Matrix* f);
//void linear_layer(const element* W, const char t, size_t W_m, size_t W_n, const element* x, size_t x_m, size_t x_n, const element* b, element** f);

void cosine_distance_loss(NN_Matrix* A, NN_Matrix* B, NN_Matrix* C, NN_Matrix* D);
void cross_entropy(NN_Matrix* A, NN_Matrix* B, NN_Matrix* C, NN_Matrix* D);
void squared_error(NN_Matrix* A, NN_Matrix* B, NN_Matrix* C, NN_Matrix* D);

#endif
