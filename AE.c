#include "AE.h"

void AE(NN_Matrix* W, NN_Matrix* hb, NN_Matrix* vb,
        NN_Matrix* x, NN_Matrix* y,
        regularization_option reg_option, actFuncType _actFuncType, errFuncType _errFuncType,
        element *cost, NN_Matrix *dW, NN_Matrix *dhb, NN_Matrix *dvb,
        AE_interim *tmp_vars)
{
    size_t sz_M; //,sz_D,sz_F;
    //sz_D = x->nrows;
    sz_M = x->ncols;
    //sz_F = W->nrows;

    // Define temporary variables
    NN_Matrix *A1   = tmp_vars->A1;
    NN_Matrix *H    = tmp_vars->H;
    NN_Matrix *dH   = tmp_vars->dH;

    NN_Matrix *A2   = tmp_vars->A2;
    NN_Matrix *O    = tmp_vars->O;
    NN_Matrix *dO   = tmp_vars->dO;

    NN_Matrix *J    = tmp_vars->J;
    NN_Matrix *dJ   = tmp_vars->dJ;

    NN_Matrix *err=NULL, *_cost=NULL;
    NN_Matrix *delta=NULL, *ndelta=NULL;

    element l1_lambda = reg_option.l1_lambda;
    element l2_lambda = reg_option.l2_lambda;
    bool normalize = reg_option.normalize;

    NN_Matrix* prev_norm_O = NULL;
    NN_Matrix* scale = NULL;


    /*
    *   Forward pass
    */

    // For hidden layer
    linear_layer(W,N,x,hb,A1);
    switch(_actFuncType) {
        case sigmoid:
            sigmoid_layer(A1, H, dH);       
            break;
        case rectified:
            relu_layer(A1, H, dH);      
            break;
        default:
            NN_error("Unsupported function!\n");
    }

    // For output layer
    linear_layer(W,T,H,vb,A2);
    sigmoid_layer(A2, O, dO);   

    if(normalize) {
        prev_norm_O = NN_copy_matrix(O);
        NN_Matrix* powO = NN_pow(O,2);
        scale = NN_sqrti(NN_sum(powO,1));   NN_free(powO);
        O = ibsxrdivide_mv(O,scale);
    }

    // Compute cost
    switch(_errFuncType) {
        case cosineDist:    
            cosine_distance_loss(y,O,J,dJ); 
            err = NN_copy_matrix(J);
            break;
        case crossEntropy:
            cross_entropy(y,O,J,dJ);    
            err = NN_sum(J,1);      
            break;
        case squaredError:
            squared_error(y,O,J,dJ);    
            err = NN_sum(J,1);          
            break;
        default:
            NN_error("Unsupported error function\n");
    }

    NN_Matrix* Hc = NN_copy_matrix(H);
    Hc = NN_logi(NN_addsi(1,NN_powi(Hc,2),1));
    NN_Matrix* sumHc = NN_sum(Hc,1);        NN_free(Hc);
    err = addi(l1_lambda, sumHc, 1.0, err);         NN_free(sumHc);
    _cost = NN_mean(err,0);             NN_free(err);

    NN_Matrix* W_square = NN_pow(W,2.0);
    NN_Matrix* l2_penalty_W = NN_sum(W_square,0);       NN_free(W_square);
    
    *cost = NN_get_value(_cost,0,0) + .5*l2_lambda*(NN_get_value(l2_penalty_W,0,0));
    NN_free(_cost);     NN_free(l2_penalty_W);      
    
    /*
    *   Backpropagation
    */
    if(dW!=NULL && dhb!=NULL && dvb!=NULL) {
        dJ = NN_muli(dJ,1/(element)sz_M);       /* devide delta by the number of instances */

        delta = elemwise_mm(dJ,dO); 

        if(normalize) {
            NN_Matrix* first = bsxrdivide_mv(delta,scale);
            NN_Matrix* temp = elemwise_mm(delta,prev_norm_O);
            NN_Matrix* temp2 = NN_sum(temp,1);      NN_free(temp);
            NN_Matrix* second = bsxtimes_mv(prev_norm_O,temp2); NN_free(temp2);
            NN_Matrix* scale3 = NN_pow(scale,3);
            second = ibsxrdivide_mv(second,scale3); NN_free(scale3);

            NN_free(delta); delta=NULL;
            delta = addi(1.0,first,-1,second);  NN_free(first); second=NULL;

            NN_free(prev_norm_O);   NN_free(scale);
        }

        dW = mul_mmi(1.0,H,N,delta,T,0,dW);     
        dW = addi(l2_lambda,W,1.0,dW);          // L2 regularization
        dvb = NN_sumi(delta,2,dvb); 

        NN_Matrix* l1_penalty = NN_muli(elemwise_mmi(H,NN_powi(NN_addsi(1,NN_pow(H,2),1),-1)),(l1_lambda/(element)sz_M)*2);
                                
        // update delta
        ndelta = addi(1.0,l1_penalty,1.0,mul_mm(1.0,W,N,delta,N));      NN_free(l1_penalty);
        NN_free(delta); delta=ndelta;   
        
        delta = elemwise_mm(delta,dH);      
    
        dW = mul_mmi(1.0,delta,N,x,T,1,dW);
        dhb = NN_sumi(delta,2,dhb); 

        // releasing resources
        NN_free(delta);
        NN_free(ndelta);
    }  else {
        NN_free(prev_norm_O);
        NN_free(scale);
    }
}
