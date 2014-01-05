#include "NN_core.h"
#include "NN_math.h"
#include <math.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

void tanh_layer(NN_Matrix* A, NN_Matrix* f, NN_Matrix* dFA)
{
	unsigned int i,size;
	element *Adp,*fdp,*dFAdp;
	Adp=A->data;
	fdp=f->data;
	dFAdp=dFA->data;
	
	size=A->nrows*A->ncols;

#pragma omp parallel for default(none) private(i) shared(size,fdp,dFAdp,Adp)
	for(i=0;i<size;i++) {
		fdp[i]=tanh(Adp[i]);
		dFAdp[i]=1-(fdp[i]*fdp[i]);
	}
}

void sigmoid_layer(NN_Matrix* A, NN_Matrix* f, NN_Matrix* dFA)
{
	/*
		f is an element-wise activation function which takes A as an argument.
		dFA is partial derivatives of the function f with respect to its input A
	 */
	unsigned int i,size;
	element *Adp,*fdp,*dFAdp;
	Adp=A->data;
	fdp=f->data;
	dFAdp=dFA->data;
	
	size=A->nrows*A->ncols;

#pragma omp parallel for default(none) private(i) shared(size,fdp,dFAdp,Adp)
	for(i=0;i<size;i++) {
		fdp[i]=1.0/(1+exp(-Adp[i]));
		dFAdp[i]=fdp[i]*(1-fdp[i]);
	}
}

void relu_layer(NN_Matrix* A, NN_Matrix* f, NN_Matrix* dFA)
{
	/*
		f is an element-wise activation function which takes A as an argument.
		dFA is partial derivatives of the function f with respect to its input A
	 */
	unsigned int i,size;
	element *Adp,*fdp,*dFAdp;
	Adp=A->data;
	fdp=f->data;
	dFAdp=dFA->data;
	
	size=A->nrows*A->ncols;
#pragma omp parallel for default(none) private(i) shared(size,fdp,dFAdp,Adp)
	for(i=0;i<size;i++) {
		fdp[i]=(Adp[i]>0)?Adp[i]:0;
		dFAdp[i]=(Adp[i]>0)?1:0;
	}
}
void linear_layer(NN_Matrix* W, Transpose trans, NN_Matrix* X, NN_Matrix* b, NN_Matrix* f)
{
	size_t Wn,Xm;
	Wn=(trans==N)?W->ncols:W->nrows;	Xm=X->nrows;	
	if(Wn!=Xm)		NN_error("linear_layer: Dimension mismatch");
	
	f=mul_mmi(1,W,trans,X,N,0,f);
	f=ibsxplus_mv(f,b);
}

/*
	A: target
	B: reconstruction
*/
void cosine_distance_loss(NN_Matrix* A, NN_Matrix* B, NN_Matrix* C, NN_Matrix* D)
{
	NN_Matrix *temp,*cosDist;
	element gamma_param = 20;
	temp = elemwise_mm(A,B);
	cosDist=NN_addsi(1.0,NN_sum(temp,1),-0.5);	NN_free(temp);	// cosine distance
	
	//NN_print_matrix("cosine distance", cosDist,1,n);
	// compute cost
	NN_fill_values(C,cosDist);
	C=NN_logi(NN_addsi(1.0,NN_expi(NN_muli(C,-gamma_param)),1));
	//NN_print_matrix("log loss", *C,1,n);

	// compute gradients
	temp=NN_mul(cosDist,gamma_param);
	temp=NN_expi(temp);
	temp=NN_addsi(1.0,temp,1);
	temp=NN_powi(temp,-1);
	temp=NN_muli(temp,-gamma_param);
	NN_fill_values(D,A);
	D=ibsxtimes_mv(D,temp);

	NN_free(temp);	NN_free(cosDist);
}

void cross_entropy(NN_Matrix* A, NN_Matrix* B, NN_Matrix* C, NN_Matrix* D)
{
	/*
		C = -(A*log(B)+(1-A)*log(1-B))	: function value
		D = -A/B + (1-A)/(1-B)			: derivative of the function
		
		A: target
		B: predicted
	 */

	unsigned int i,size;
	element *Adp,*Bdp,*Cdp,*Ddp;
	size=A->nrows*A->ncols;
	Adp=A->data;	Bdp=B->data;	Cdp=C->data;	Ddp=D->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp,Cdp,Ddp)
	for(i=0;i<size;i++) {
		Cdp[i]=-(Adp[i]*log(Bdp[i])+(1-Adp[i])*log(1-Bdp[i]));
		Ddp[i]=-Adp[i]/Bdp[i]+(1-Adp[i])/(1-Bdp[i]);
	}

}

void squared_error(NN_Matrix* A, NN_Matrix* B, NN_Matrix* C, NN_Matrix* D)
{
	/*
		C = 0.5*(A-B)^2	: function value
		D = B-A			: derivative of the function
		
		A: target
		B: predicted
	 */
	C=NN_muli(NN_powi(addi(-1,B,1,addi(1,A,0,C)),2),0.5);
	D=addi(-1,A,1,addi(1,B,0,D));
}
