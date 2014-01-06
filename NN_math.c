#include "NN_math.h"

#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

NN_Matrix* NN_copy_matrix(NN_Matrix* A)
{
	NN_Matrix* B = NULL;
	size_t nrows,ncols;
	// input check
	if(!A)	NN_error("NN_copy_matrix: Empty matrix");

	nrows = A->nrows;	ncols = A->ncols;
	B = NN_init_matrix(nrows,ncols);
	element* Adp = A->data;
	element* Bdp = B->data;
	memcpy(&Bdp[0],&Adp[0],nrows*ncols*sizeof(element));

	return B;
}


/*
	C = \alpha*A x B + \beta*C
*/
NN_Matrix* mul_mmi(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B, Transpose TransB, element beta, NN_Matrix* C)
{
	enum CBLAS_TRANSPOSE _TransA,_TransB;
	_TransA=(TransA==N)?CblasNoTrans:CblasTrans;
	_TransB=(TransB==N)?CblasNoTrans:CblasTrans;

	size_t Am,An,Bm,Bn,Cm,Cn,Atemp,Btemp;
	Am=A->nrows;	An=A->ncols;
	if(_TransA==CblasTrans) { Atemp = Am; Am=An; An=Atemp;}
	Bm=B->nrows;	Bn=B->ncols;
	if(_TransB==CblasTrans) { Btemp = Bm; Bm=Bn; Bn=Btemp;}
	Cm=C->nrows;	Cn=C->ncols;
	
	if((Am!=Cm) || (Bn!=Cn) || (An!=Bm))	NN_error("mul_mmi: Dimension mismatch");	
	element *Adp,*Bdp,*Cdp;
	unsigned int lda,ldb,ldc;
	size_t k;
	k = An;
	lda=(_TransA==CblasNoTrans)?Cm:k;
	ldb=(_TransB==CblasNoTrans)?k:Cn;
	ldc=Cm;
	Adp=A->data;	Bdp=B->data;	Cdp=C->data;
#ifdef USE_MKL_BLAS
	mkl_set_num_threads(omp_get_max_threads());
#else
	openblas_set_num_threads(omp_get_max_threads());
#endif

#ifdef FLOAT
	cblas_sgemm(CblasColMajor, _TransA, _TransB, Cm, Cn, k, alpha, &Adp[0], lda, &Bdp[0], ldb, beta, &Cdp[0], ldc);
#else
	cblas_dgemm(CblasColMajor, _TransA, _TransB, Cm, Cn, k, alpha, &Adp[0], lda, &Bdp[0], ldb, beta, &Cdp[0], ldc);
#endif

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(1);
#else
	openblas_set_num_threads(1);
#endif

	return C;
}

NN_Matrix* mul_mm(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B, Transpose TransB)
{
	NN_Matrix* C = NULL;
	size_t Am,An,Bm,Bn;
	size_t Atemp,Btemp;
	Am=A->nrows;	An=A->ncols;
	Bm=B->nrows;	Bn=B->ncols;
	if(TransA==T) { Atemp = Am; Am=An; An=Atemp;}
	if(TransB==T) { Btemp = Bm; Bm=Bn; Bn=Btemp;}

	if(An!=Bm)	NN_error("mul_mm: Dimension mismatch");

	C=NN_init_matrix(Am,Bn);
	return mul_mmi(alpha,A,TransA,B,TransB,0,C);
}

NN_Matrix* mul_dmsmi(const element alpha, NN_Matrix* A, cs* B, const element beta, NN_Matrix* C)
{
	if(!A || !B || !C)	NN_error("mul_dmsmi: input error; empty matrix");
	if(A->ncols != B->m)	NN_error("mul_dmsmi: Dimension mismatch");

	unsigned int i,j;
	size_t m,n; //,k;
	csi p,*Bp,*Bi;
	double *Bx;
	element *Adp,*Cdp;
	m=A->nrows;	n=B->n; //k=A->ncols;
	Bp=B->p;	Bi=B->i;	Bx=B->x;	Adp=A->data;	Cdp=C->data;

	for(j=0;j<n;j++) {
		for(i=0;i<m;i++) {
			//Cdp[j*m+i] += (beta*Cdp[j*m+i]);
			for(p=Bp[j];p<Bp[j+1];p++) {
				Cdp[j*m+i] = alpha*Adp[Bi[p]*m+i]*Bx[p] + beta*Cdp[j*m+i];
			}
		}
	}
	
	return C;
}

NN_Matrix* mul_dmsm(const element alpha, NN_Matrix* A, cs* B)
{
	if(!A || !B)	NN_error("mul_dmsm: input error; empty matrix");
	if(A->ncols != B->m)	NN_error("mul_dmsm: Dimension mismatch");
	size_t m,n;
	m=A->nrows;	n=B->n;
	NN_Matrix* C = NN_init_matrix(m,n);

	mul_dmsmi(alpha,A,B,0,C);

	return C;
}

/*
	Matrix-vector multiplication
*/
NN_Matrix* mul_mvi(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B, element beta, NN_Matrix* C)
{
	enum CBLAS_TRANSPOSE _TransA;
	_TransA=(TransA==N)?CblasNoTrans:CblasTrans;
	size_t Am,An,Bm,Bn,Cm,Cn;
	Am=A->nrows;	An=A->ncols;	Bm=B->nrows;	Bn=B->ncols;	Cm=C->nrows;	Cn=C->ncols;
	if(_TransA==CblasTrans) { size_t temp = Am; Am = An; An = temp;}

	if(!NN_isvec(B))	NN_error("mul_mvi: right operand should be a vector");
	if(An!=Bm) 			NN_error("mul_mvi: Dimension mismatch");
	if(Am!=Cm||Cn!=Bn)	NN_error("mul_mvi: Dimension mismatch");
	if(!NN_isvec(C))	NN_error("mul_mvi: result should be a vector");

	element *Adp,*Bdp,*Cdp;
	Adp=A->data;	Bdp=B->data;	Cdp=C->data;

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(omp_get_max_threads());
#else
	openblas_set_num_threads(omp_get_max_threads());
#endif

#ifdef FLOAT
	cblas_sgemv(CblasColMajor, _TransA, Am, An, alpha, &Adp[0], Am, &Bdp[0], 1, beta, &Cdp[0], 1);
#else
	cblas_dgemv(CblasColMajor, _TransA, Am, An, alpha, &Adp[0], Am, &Bdp[0], 1, beta, &Cdp[0], 1);
#endif

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(1);
#else
	openblas_set_num_threads(1);
#endif

	return C;
}

/*
	Matrix-vector multiplication
*/
NN_Matrix* mul_mv(const element alpha, NN_Matrix* A, Transpose TransA, NN_Matrix* B)
{
	size_t Am,An,Bm,Bn;
	Am=A->nrows;	An=A->ncols;	Bm=B->nrows;	Bn=B->ncols;
	if(TransA==T) { size_t temp = Am; Am = An; An = temp;}

	if(!NN_isvec(B))	NN_error("mul_mv: right operand should be a vector");
	if(An!=Bm)	NN_error("mul_mv: Dimension mismatch");
	NN_Matrix* C=NN_init_matrix(Am,Bn);

	return mul_mvi(alpha,A,TransA,B,0,C);
}

/*
	C = A.*B
*/
NN_Matrix* elemwise_mm(NN_Matrix* A, NN_Matrix* B)
{
	NN_Matrix* C = NN_copy_matrix(B);
	return elemwise_mmi(A,C);
}

/*
	B = A.*B	( The operator .* denotes element-wise matrix multiplication. )
	in-place operation
*/
NN_Matrix* elemwise_mmi(NN_Matrix* A, NN_Matrix* B)
{	
	unsigned int i;
	size_t Anrows,Ancols,Bnrows,Bncols;
	Anrows=A->nrows;	Ancols=A->ncols;	Bnrows=B->nrows;	Bncols=B->ncols;
	if((Anrows!=Bnrows) || (Ancols!=Bncols))	NN_error("Dimension mismatch");
	size_t size = Anrows*Ancols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp)
	for(i=0;i<size;i++) Bdp[i]*=Adp[i];

	return B;
}

/*
	C = B./A
*/
NN_Matrix* elemwise_div(NN_Matrix* A, NN_Matrix* B)
{
	NN_Matrix* C = NN_copy_matrix(B);
	elemwise_divi(A,C);
	return C;
}

/*
	B = B./A	( The operator .* denotes element-wise matrix multiplication. )
	in-place operation
*/
NN_Matrix* elemwise_divi(NN_Matrix* A, NN_Matrix* B)
{	
	unsigned int i;
	size_t Anrows,Ancols,Bnrows,Bncols;
	Anrows=A->nrows;	Ancols=A->ncols;	Bnrows=B->nrows;	Bncols=B->ncols;
	if((Anrows!=Bnrows) || (Ancols!=Bncols))	NN_error("Dimension mismatch");
	size_t size = Anrows*Ancols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp)
	for(i=0;i<size;i++) Bdp[i]/=Adp[i];

	return B;
}

NN_Matrix* bsxtimes_mv(NN_Matrix* A, NN_Matrix* B)
{
	if(!NN_isvec(B))				NN_error("bsxtimes_mv: right operand should be a vector");
	size_t Am,An,Bk;
	Am=A->nrows;	An=A->ncols;	Bk=NN_maxdim(B);
	if(!((Am==Bk) || (An==Bk)))		NN_error("bsxtimes_mv: Dimension mismatch");

	NN_Matrix* C = NN_copy_matrix(A);
	return ibsxtimes_mv(C,B);
}

NN_Matrix* ibsxtimes_mv(NN_Matrix* A, NN_Matrix* B)
{
	if(!NN_isvec(B))		NN_error("ibsxtimes_mv: right operand should be a vector");
	size_t Am,An,Bk;
	Am=A->nrows;	An=A->ncols;	Bk=NN_maxdim(B);
	if(Am!=Bk && An!=Bk)	NN_error("bsxtimes_mv: Dimension mismatch");
	
	unsigned int i,ii,jj,size;
	element *Adp,*Bdp;
	size=Am*An;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i,ii,jj) shared(size,Am,Bk,Adp,Bdp)
	for(i=0;i<size;i++) {
		ii=i%Am;
		jj=i/Am;
		Adp[jj*Am+ii]*=Bdp[(Bk==Am)?ii:jj];
	}
	return A;
}

NN_Matrix* bsxrdivide_mv(NN_Matrix* A, NN_Matrix* B)
{
	if(!NN_isvec(B))				NN_error("bsxtimes_mv: right operand should be a vector");
	size_t Am,An,Bk;
	Am=A->nrows;	An=A->ncols;	Bk=NN_maxdim(B);
	if(!((Am==Bk) || (An==Bk)))		NN_error("bsxtimes_mv: Dimension mismatch");

	NN_Matrix* C = NN_copy_matrix(A);
	return ibsxrdivide_mv(C,B);
}

NN_Matrix* ibsxrdivide_mv(NN_Matrix* A, NN_Matrix* B)
{
	if(!NN_isvec(B))		NN_error("ibsxtimes_mv: right operand should be a vector");
	size_t Am,An,Bk;
	Am=A->nrows;	An=A->ncols;	Bk=NN_maxdim(B);
	if(Am!=Bk && An!=Bk)	NN_error("bsxtimes_mv: Dimension mismatch");
	
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

	unsigned int i,ii,jj,size;
	size = Am*An;	
	
#pragma omp parallel for default(none) private(i,ii,jj) shared(size,Am,Bk,Adp,Bdp)
	for(i=0;i<size;i++) {
		ii = i%Am;
		jj = i/Am;
		Adp[jj*Am+ii]/=Bdp[(Bk==Am)?ii:jj];
	}

	return A;
}

NN_Matrix* bsxplus_mv(NN_Matrix* A, NN_Matrix* B)
{
	if(!NN_isvec(B))				NN_error("bsxplus_mv: right operand should be a vector");
	size_t Am,An,Bk;
	Am=A->nrows;	An=A->ncols;	Bk=NN_maxdim(B);
	if(!((Am==Bk) || (An==Bk)))		NN_error("bsxplus_mv: Dimension mismatch");

	NN_Matrix* C = NN_copy_matrix(A);
	return ibsxplus_mv(C,B);
}

NN_Matrix* ibsxplus_mv(NN_Matrix* A, NN_Matrix* B)
{
	unsigned int i,j;
	if(!NN_isvec(B))		NN_error("ibsxplus_mv: right operand should be a vector");
	size_t Am,An,Bk;
	Am=A->nrows;	An=A->ncols;	Bk=NN_maxdim(B);
	if(Am!=Bk && An!=Bk)	NN_error("bsxplus_mv: Dimension mismatch");
	
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(omp_get_max_threads());
#else
	openblas_set_num_threads(omp_get_max_threads());
#endif

	if(Am==Bk) {
		for(j=0;j<An;j++) {
		#ifdef FLOAT
			cblas_saxpy(Am,1.0,&Bdp[0],1,&Adp[j*Am],1);
		#else
			cblas_daxpy(Am,1.0,&Bdp[0],1,&Adp[j*Am],1);
		#endif
		}
	}
	else if(An==Bk) {
		for(i=0;i<Am;i++) {
		#ifdef FLOAT
			cblas_saxpy(An,1.0,&Bdp[0],1,&Adp[i],Am);
		#else
			cblas_daxpy(An,1.0,&Bdp[0],1,&Adp[i],Am);
		#endif
		}
	}

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(1);
#else
	openblas_set_num_threads(1);
#endif

	return A;
}

/*
	B = \alpha x A + \beta B
*/
NN_Matrix* addi(element alpha, NN_Matrix* A, element beta, NN_Matrix* B)
{
	size_t Anrows,Ancols,Bnrows,Bncols;
	Anrows=A->nrows;	Ancols=A->ncols;
	Bnrows=B->nrows;	Bncols=B->ncols;
	if((Anrows!=Bnrows) || (Ancols!=Bncols))	NN_error("addi: Dimension mismatch");

	size_t length = A->nrows*A->ncols;

	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(omp_get_max_threads());
#else
	openblas_set_num_threads(omp_get_max_threads());
#endif

	if(beta!=1.0) {
	#ifdef FLOAT
		cblas_sscal(length,beta,&Bdp[0],1);	
	#else
		cblas_dscal(length,beta,&Bdp[0],1);	
	#endif
	}

#ifdef FLOAT
	cblas_saxpy(length,alpha,&Adp[0],1,&Bdp[0],1);
#else
	cblas_daxpy(length,alpha,&Adp[0],1,&Bdp[0],1);
#endif

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(1);
#else
	openblas_set_num_threads(1);
#endif

	return B;
}

/*
	C = \alpha x A + \beta x B
*/
NN_Matrix* add(element alpha, NN_Matrix* A, element beta, NN_Matrix* B)
{
	NN_Matrix* C=NULL;
	size_t Anrows,Ancols,Bnrows,Bncols;
	Anrows=A->nrows;	Ancols=A->ncols;
	Bnrows=B->nrows;	Bncols=B->ncols;
	if((Anrows!=Bnrows) || (Ancols!=Bncols))	NN_error("add: Dimension mismatch");

	C=NN_copy_matrix(B);
	return addi(alpha,A,beta,C);
}

NN_Matrix* add_col_vec(element alpha, NN_Matrix* A, unsigned int* col_idx, size_t nIndices, NN_Matrix* B)
{
	/*
		A: d x M
		B: d x L
	*/
	unsigned int i;
	size_t Am,An,Bm,m; //,Bn;
	Am=A->nrows;	An=A->ncols;	Bm=B->nrows;//	Bn=B->ncols;
	if(Am!=Bm || An!=nIndices)		NN_error("add_col_vec: Dimension mismatch");
	m=Am;
	//if(col_idx<0 || col_idx>=An)	NN_error("add_col_vec: col_idx out of bounds");
	//if(!NN_isvec(B))				NN_error("right operand should be a vector");
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;
	
	for(i=0;i<nIndices;i++) {
#ifdef FLOAT
		cblas_saxpy(m,alpha,&Adp[i*m],1,&Bdp[col_idx[i]*m],1);
#else
		cblas_daxpy(m,alpha,&Adp[i*m],1,&Bdp[col_idx[i]*m],1);
#endif
	}

	return B;
}

NN_Matrix* add_row_vec(element alpha, NN_Matrix* A, unsigned int* row_idx, size_t nIndices, NN_Matrix* B)
{
	//size_t m,n;
	//m=A->nrows;	n=A->ncols;

	return NULL;	
}

NN_Matrix* NN_sumi(NN_Matrix* A, unsigned int axis, NN_Matrix* B)
{
	unsigned int i,k;
	unsigned int ii,jj;
	size_t m,n,size;
	m=A->nrows;	n=A->ncols;	size=m*n;
	element *Adp,*Bdp;
	element* partial_sums;
	Adp=A->data;	Bdp=NULL;
	unsigned int max_num_threads = 1;
	bool dim_match = false;
	switch(axis)
	{
		case 0:
			dim_match=(B->nrows==1 && B->ncols==1)?true:false;
			break;
		case 1:
			dim_match=(B->nrows==1 && B->ncols==n)?true:false;
			break;
		case 2:
			dim_match=(B->nrows==m && B->ncols==1)?true:false;
			break;
		default:
			NN_error("axis: either 0, 1, or 2");
			break;
	}

	if(!dim_match)		NN_error("NN_sumi: B: dimension mismatch");

	//NN_fill_zeros(B);

	Bdp=B->data;
#ifdef _OPENMP
	max_num_threads = omp_get_max_threads();
#endif
	switch(axis)
	{
		case 0:		// sum over all dimensions
			partial_sums = (element*)malloc(max_num_threads*sizeof(element));
		#pragma omp parallel num_threads(max_num_threads)
			{
				element *Adp_;
				int i, n_, blocksize, offset;
				int thread_num = omp_get_thread_num();
				int num_threads = omp_get_num_threads();
				element partial_sum=0.0;

				blocksize = size/num_threads;
				offset=thread_num*blocksize;
				if(thread_num==num_threads-1) {
					n_=size-offset;
				} else {
					n_=blocksize;
				}
				Adp_ = Adp+offset;
				for(i=0;i<n_;i++)	partial_sum += Adp_[i];
				partial_sums[thread_num] = partial_sum;
			}
			for(k=0;k<max_num_threads;k++)	Bdp[0]+=partial_sums[k];
			free(partial_sums);
			break;
		case 1:		// outputs a row vector
		case 2:		// outputs a column vector
			for(i=0;i<size;i++) {
				ii = (i)%m;
				jj = (i)/m;
				Bdp[(axis==1)?jj:ii]+=Adp[jj*m+ii];
			}
			break;
	}

	return B;
}

NN_Matrix* NN_sum(NN_Matrix* A, unsigned int axis)
{
	size_t m,n;
	m=A->nrows;	n=A->ncols;
	NN_Matrix* B = NULL;
	switch(axis)
	{
		case 0:
			B=NN_zeros_matrix(1,1);
			break;
		case 1:
			B=NN_zeros_matrix(1,n);
			break;
		case 2:
			B=NN_zeros_matrix(m,1);
			break;
		default:
			NN_error("axis: either 0, 1, or 2");
	}

	return NN_sumi(A,axis,B);
}

NN_Matrix* NN_meani(NN_Matrix* A, unsigned int axis, NN_Matrix* B)
{
	size_t m,n;
	m=A->nrows;	n=A->ncols;
	B = NN_sumi(A,axis,B);
	switch(axis)
	{
		case 0:		// sum all dimensions
			B=NN_muli(B,1/(element)(m*n));
			break;
		case 1:		// outputs a row vector
			B=NN_muli(B,1/(element)m);
			break;
		case 2:		// outputs a column vector
			B=NN_muli(B,1/(element)n);
			break;
		default:
			NN_error("axis: either 0, 1, or 2");
	}
	return B;
}

NN_Matrix* NN_mean(NN_Matrix* A, unsigned int axis)
{
	size_t m,n;
	m=A->nrows;	n=A->ncols;
	NN_Matrix* B = NULL;

	switch(axis)
	{
		case 0:
			B=NN_zeros_matrix(1,1);
			break;
		case 1:
			B=NN_zeros_matrix(1,n);
			break;
		case 2:
			B=NN_zeros_matrix(m,1);
			break;
		default:
			NN_error("axis: either 0, 1, or 2");
	}
	return NN_meani(A,axis,B);
}

NN_Matrix* NN_imins(NN_Matrix* A, element s)
{
	unsigned int size,i;
	element *Adp;
	
	size=A->nrows*A->ncols;
	Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,s)
	for(i=0;i<size;i++) {
		Adp[i]=(Adp[i]<s)?Adp[i]:s;
	}

	return A;
}

NN_Matrix* NN_mins(NN_Matrix* A, element s)
{
	NN_Matrix* B=NN_copy_matrix(A);
	return NN_imins(B,s);
}

NN_Matrix* NN_imin(NN_Matrix* A, int axis, NN_Matrix* B)
{
	NN_error("Not implemented yet");
	return NULL;
}

NN_Matrix* NN_min(NN_Matrix* A, int axis)
{
	NN_error("Not implemented yet");
	return NULL;
}

NN_Matrix* NN_muli(NN_Matrix* A, element s)
{
	size_t nrows,ncols;
	element* Adp=NULL;
	if(!A)	NN_error("Input error");
	nrows=A->nrows;	ncols=A->ncols;
	Adp=A->data;

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(omp_get_max_threads());
#else
	openblas_set_num_threads(omp_get_max_threads());
#endif

#ifdef FLOAT
	cblas_sscal(nrows*ncols,s,Adp,1.0);
#else
	cblas_dscal(nrows*ncols,s,Adp,1.0);
#endif

#ifdef USE_MKL_BLAS
	mkl_set_num_threads(1);
#else
	openblas_set_num_threads(1);
#endif
	return A;
}

NN_Matrix* NN_mul(NN_Matrix* A, element s)
{
	NN_Matrix* B = NN_copy_matrix(A);
	return NN_muli(B,s);	
}

NN_Matrix* NN_rdiv(element s, NN_Matrix* A )
{
	NN_Matrix* B = NN_copy_matrix(A);
	return NN_rdivi(s,B);
}

NN_Matrix* NN_rdivi(element s, NN_Matrix* A )
{
	element* Adp;
	Adp=A->data;
	unsigned int i;
	size_t size;
	size=A->nrows*A->ncols;	
	
#pragma omp parallel for default(none) private(i) shared(size,s,Adp)
	for(i=0;i<size;i++) {
		Adp[i] = s/Adp[i];
	}

	return A;
}

NN_Matrix* NN_neg(NN_Matrix* A)
{
	NN_Matrix* B = NN_copy_matrix(A);
	return NN_negi(B);
}

NN_Matrix* NN_negi(NN_Matrix* A)
{
	unsigned int i;
	size_t size=A->nrows*A->ncols;
	element* Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp)
	for(i=0;i<size;i++)	Adp[i] = -Adp[i];

	return A;
}

NN_Matrix* NN_addsi(element alpha, NN_Matrix* A, element scalar)
{
	unsigned int i;
	size_t size=A->nrows*A->ncols;
	element* Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,alpha,scalar)
	for(i=0;i<size;i++)	Adp[i]=alpha*Adp[i]+scalar;
	
	return A;
}

NN_Matrix* NN_adds(element alpha, NN_Matrix* A, element scalar)
{
	NN_Matrix* B=NN_copy_matrix(A);
	return NN_addsi(alpha,B,scalar);
}

NN_Matrix* NN_powi(NN_Matrix* A, element exponent)
{
	unsigned int i;
	size_t size=A->nrows*A->ncols;
	element* Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,exponent)
	for(i=0;i<size;i++) {
		if(exponent==2)	Adp[i]*=Adp[i];	
		else			Adp[i]=pow(Adp[i],exponent);
	}

	return A;
}

NN_Matrix* NN_pow(NN_Matrix* A, element exponent)
{
	NN_Matrix* B = NN_copy_matrix(A);
	return NN_powi(B,exponent);
}

NN_Matrix* NN_expi(NN_Matrix* A)
{
	unsigned int i;
	size_t size=A->nrows*A->ncols;
	element* Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp)
	for(i=0;i<size;i++) Adp[i]=exp(Adp[i]);

	return A;
}

NN_Matrix* NN_exp(NN_Matrix* A)
{
	NN_Matrix* B=NN_copy_matrix(A);
	return NN_expi(B);
}

NN_Matrix* NN_logi(NN_Matrix* A)
{
	unsigned int i;
	size_t size=A->nrows*A->ncols;
	element* Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp)
	for(i=0;i<size;i++) Adp[i]=log(Adp[i]);

	return A;
}

NN_Matrix* NN_log(NN_Matrix* A)
{
	NN_Matrix* B=NN_copy_matrix(A);
	return NN_logi(B);	
}

NN_Matrix* NN_sqrti(NN_Matrix* A)
{
	unsigned int i;
	size_t size=A->nrows*A->ncols;
	element* Adp=A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp)
	for(i=0;i<size;i++) Adp[i]=sqrt(Adp[i]);

	return A;
}

NN_Matrix* NN_sqrt(NN_Matrix* A)
{
	NN_Matrix* B=NN_copy_matrix(A);
	return NN_sqrti(B);
}
