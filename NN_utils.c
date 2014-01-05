#include "NN_utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

void timeval_print(struct timeval *tv)
{
    char buffer[30];
    time_t curtime;

    printf("%ld.%06ld", tv->tv_sec, tv->tv_usec);
    curtime = tv->tv_sec;
    strftime(buffer, 30, "%m-%d-%Y  %T", localtime(&curtime));
    printf(" = %s.%06ld\n", buffer, tv->tv_usec);
}

/*
	randi(min,max) returns random integers drawn from the uniform distribution on [min,max]
*/
int NN_randi(int r_min, int r_max)
{
	return rand()%(r_max-r_min+1)+r_min;
}

/*
	rand() generates random number between 0 and 1
*/
element NN_rand()
{
	return (element)rand()/RAND_MAX;
}

element NN_randn()
{
	static element V1, V2, S;
	static int phase = 0;
	element X;

	if(phase == 0) {
		do {
			element U1 = (element)rand() / RAND_MAX;
			element U2 = (element)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

void NN_irandperm(unsigned int* v, size_t n)
{
	unsigned int i,j,t;
	if(v==NULL)	NN_error("NN_randperm: Failed to allocate a space for randomized permutation");

//#pragma omp parallel for default(none) private(i) shared(v,n)
	for(i=0;i<n;i++) {
		v[i]=i;
	}

	for(i=n-1;i>0;i--) {
		//j=i+rand()/(RAND_MAX+(n-i)+1);
		j=rand()%(i+1);
		t=v[j];	v[j]=v[i];	v[i]=t;
	}

}

unsigned int* NN_randperm(size_t n)
{
	unsigned int* perm=NULL;
	perm = (unsigned int*) malloc(n*sizeof(unsigned int));

	NN_irandperm(perm,n);

	return perm;
}

NN_Matrix* NN_binornd_matrix(unsigned int n_trials, float p, size_t m, size_t n)
{
	unsigned int i,t,numel;
	numel = m*n;
	NN_Matrix* A = NN_zeros_matrix(m,n);
	element* Adp = A->data;
	for(i=0;i<numel;i++)
		for(t=0;t<n_trials;t++)
			if(p < ((element)rand() / RAND_MAX))
				Adp[i]+=1;
	return A;
}

NN_Matrix* NN_init_matrix(size_t m, size_t n)
{
	NN_Matrix* A = malloc(sizeof(NN_Matrix));
	if(!A) NN_error("Init:Memory Allocation Error!\n");
	A->data = malloc(m*n*sizeof(element));
	if(!(A->data)) NN_error("Init:Memory Allocation Error!\n");

	A->nrows = m;	A->ncols = n;
		
	return A;
}

NN_Matrix* NN_randn_matrix(size_t m, size_t n)
{
	unsigned int i;
	NN_Matrix* A = NN_init_matrix(m,n);
	element* Adp = A->data;
	for(i=0;i<m*n;i++)	Adp[i] = NN_randn();
	return A;
}

NN_Matrix* NN_rand_matrix(size_t m, size_t n)
{
	unsigned i;
	NN_Matrix* A=NN_init_matrix(m,n);
	element* Adp = A->data;
	for(i=0;i<m*n;i++)	Adp[i] = NN_rand();
	return A;
}

NN_Matrix* NN_randi_matrix(int r_min, int r_max, size_t m, size_t n)
{
	//unsigned int i;
	size_t max_num_threads=1;
	size_t size=m*n;
	NN_Matrix* A=NN_init_matrix(m,n);
	element* Adp = A->data;
#ifdef _OPENMP
	max_num_threads = omp_get_max_threads();
#endif
#pragma omp parallel num_threads(max_num_threads)
{
	element *Adp_;
	int i, n_, blocksize, offset;
	int thread_num = omp_get_thread_num();
	int num_threads = omp_get_num_threads();

	blocksize = size/num_threads;
	offset=thread_num*blocksize;
	if(thread_num==num_threads-1) {
		n_=size-offset;
	} else {
		n_=blocksize;
	}
	Adp_ = Adp+offset;
	for(i=0;i<n_;i++)	{
		int val = NN_randi(r_min,r_max);
		//printf("Thread %d: %d\n", thread_num,val);
		//Adp_[i] = NN_randi(r_min,r_max);
		Adp_[i] = val;
	}
}
	return A;
}
NN_Matrix* NN_zeros_matrix(size_t m, size_t n)
{
	NN_Matrix* A = NN_init_matrix(m,n);
	element* Adp = A->data;
	memset(Adp,0,m*n*sizeof(element));
	return A;
}

NN_Matrix* NN_fill_zeros(NN_Matrix* A)
{
	element* Adp;
	Adp=A->data;

	memset(&Adp[0],0,A->nrows*A->ncols*sizeof(element));

	return A;
}

bool NN_is_same_size(NN_Matrix* A, NN_Matrix* B)
{
	size_t Am,An,Bm,Bn;
	Am=A->nrows;	An=A->ncols;	Bm=B->nrows;	Bn=B->ncols;
	return ((Am==Bm) && (An==Bn))?true:false;
}

void NN_fill_values(NN_Matrix* dst, NN_Matrix* src)
{
	// ensure that two matrices have same size
	if(!NN_is_same_size(dst,src))	NN_error("NN_fill_values: Dimension mismatch");

	element *dstdp,*srcdp;
	dstdp=dst->data;	srcdp=src->data;

	unsigned int i;
	unsigned int size = dst->nrows*dst->ncols;

#pragma omp parallel for default(none) private(i) shared(size,dstdp,srcdp)
	for(i=0;i<size;i++)	dstdp[i]=srcdp[i];

}

NN_Matrix* NN_ones_matrix(size_t m, size_t n)
{
	unsigned i;
	NN_Matrix* A = NN_init_matrix(m,n);
	element* Adp = A->data;
	for(i=0;i<m*n;i++)	Adp[i] = 1;
	return A;
}

void NN_print_matrix(const char* msg, const NN_Matrix* A)
{
	unsigned int i, j, nrows, ncols;
	
	printf("%s\n\n", msg);
	if(!A)	{fprintf(stderr, "\tEmpty matrix\n\n"); return;}

	element* Adp = A->data;	
	nrows = A->nrows;
	ncols = A->ncols;	
	for(i=0;i<nrows;i++) {
		for(j=0;j<ncols;j++) {
			printf("\t%.4f\t", MATRIX_ELEMENT(Adp,nrows,ncols,i,j));
		}
		printf("\n");
	}
	printf("\n");
}

void NN_show_dim(const char* msg, const NN_Matrix* A)
{
	printf("%s\n\n", msg);
	printf("\t%zu\t%zu\n", A->nrows,A->ncols);
}

element NN_get_value(const NN_Matrix* A, unsigned int i, unsigned int j)
{
	size_t m,n;
	element* Adp=NULL;
	m=A->nrows;	n=A->ncols;

	if((i<0 || i>=m) || (j<0 || j>=n))	NN_error("NN_get_value: Indexing error");

	Adp = A->data;	

	return Adp[j*m+i];
}

element NN_get_value_sp(const cs* A, unsigned int i, unsigned int j)
{
	csi p,*Ap,*Ai;
	double *Ax;	
	element val = 0;
	Ap=A->p;	Ai=A->i;	Ax=A->x;
	for(p=Ap[j];p<Ap[j+1];p++) {
		if(Ai[p] == i) val = Ax[p];
	}
	
	return val;
}

void NN_set_value(NN_Matrix* A, unsigned int i, unsigned int j, element v)
{
	size_t m,n;
	element* Adp=NULL;
	m=A->nrows;	n=A->ncols;

	if((i<0 || i>=m) || (j<0 || j>=n))	NN_error("NN_get_value: Indexing error");

	Adp = A->data;	

	Adp[j*m+i]=v;
}

NN_Matrix* NN_infinity_to_zero(NN_Matrix* A)
{
	unsigned int i;
	unsigned int size = A->nrows*A->ncols;
	element* Adp = A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp)
	for(i=0;i<size;i++)	if(isinf(Adp[i])) Adp[i]=0;

	return A;
}

NN_Matrix* NN_negative_to_zero(NN_Matrix* A)
{
	unsigned int i;
	unsigned int size = A->nrows*A->ncols;
	element* Adp = A->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp)
	for(i=0;i<size;i++) {
		if(Adp[i]<0)	Adp[i]=0;
	}

	return A;
}

NN_Matrix* NN_lt(NN_Matrix* A, element s)
{
	NN_Matrix* B = NN_init_matrix(A->nrows,A->ncols);
	return NN_lti(A,s,B);
}

NN_Matrix* NN_lti(NN_Matrix* A, element s, NN_Matrix* B)
{
	if(!NN_is_same_size(A,B))	NN_error("NN_lti: dimension mismatch");

	unsigned int i;
	size_t size;
	size=A->nrows*A->ncols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp,s)
	for(i=0;i<size;i++) {
		Bdp[i] = (Adp[i]<s)?1:0;
	}

	return B;
}

NN_Matrix* NN_gt(NN_Matrix* A, element s)
{
	NN_Matrix* B = NN_init_matrix(A->nrows,A->ncols);
	return NN_gti(A,s,B);
}

NN_Matrix* NN_gti(NN_Matrix* A, element s, NN_Matrix* B)
{
	if(!NN_is_same_size(A,B))	NN_error("NN_gti: dimension mismatch");

	unsigned int i;
	size_t size;
	size=A->nrows*A->ncols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp,s)
	for(i=0;i<size;i++) {
		Bdp[i] = (Adp[i]>s)?1:0;
	}

	return B;
}

NN_Matrix* 	NN_eq(NN_Matrix* A, element s)
{
	NN_Matrix* B = NN_init_matrix(A->nrows,A->ncols);
	return NN_eqi(A,s,B);
}

NN_Matrix* 	NN_eqi(NN_Matrix* A, element s, NN_Matrix* B)
{
	if(!NN_is_same_size(A,B))	NN_error("NN_eqi: dimension mismatch");

	unsigned int i;
	size_t size;
	size=A->nrows*A->ncols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp,s)
	for(i=0;i<size;i++) {
		Bdp[i] = (Adp[i]==s)?1:0;
	}

	return B;
}

NN_Matrix* 	NN_neq(NN_Matrix* A, element s)
{
	NN_Matrix* B = NN_init_matrix(A->nrows,A->ncols);
	return NN_neqi(A,s,B);
}

NN_Matrix* 	NN_neqi(NN_Matrix* A, element s, NN_Matrix* B)
{
	if(!NN_is_same_size(A,B))	NN_error("NN_neqi: dimension mismatch");

	unsigned int i;
	size_t size;
	size=A->nrows*A->ncols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp,s)
	for(i=0;i<size;i++) {
		Bdp[i] = (Adp[i]!=s)?1:0;
	}

	return B;

}

NN_Matrix*	NN_max_s(NN_Matrix* A, element s, NN_Matrix* B)
{
	if(!NN_is_same_size(A,B))	NN_error("NN_max_s: dimension mismatch");

	unsigned int i;
	size_t size;
	size=A->nrows*A->ncols;
	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;

#pragma omp parallel for default(none) private(i) shared(size,Adp,Bdp,s)
	for(i=0;i<size;i++) {
		Bdp[i] = (Adp[i]>s)?Adp[i]:s;
	}

	return B;
}

bool NN_isvec(NN_Matrix* A)
{
	return ((A->nrows==1 || A->ncols==1))?true:false;
}

size_t NN_maxdim(NN_Matrix* A)
{
	return (A->nrows>=A->ncols)?A->nrows:A->ncols;
}

NN_Matrix* NN_submatrix_sp(cs* A, int r_left, int r_right, int c_left, int c_right)
{
	csi m,n,*Ap,*Ai;
	double *Ax;
	NN_Matrix* B=NULL;
	int newM, newN;
	m = A->m;	n = A->n;	Ap = A->p;	Ai = A->i;	Ax = A->x;

	newM = r_right-r_left+1;
	newN = c_right-c_left+1;
	if((r_left<0 || r_right>(m-1)) || (c_left<0 || c_right>(n-1))) { fprintf(stderr, "Index exceeds matrix dimensions or out of bounds\n"); return NULL;}
	if(newM <= 0 || newN <= 0)	{ fprintf(stderr, "\tEmpty matrix: %d-by-%d\n", newM, newN); return NULL; }
	
	B = NN_zeros_matrix(newM,newN);

	NN_isubmatrix_sp(A, r_left, r_right, c_left, c_right, B);

	return B;
}

void NN_isubmatrix_sp(cs* A, int r_left, int r_right, int c_left, int c_right, NN_Matrix* B)
{
	csi m,n,p,j,*Ap,*Ai;
	double *Ax;
	element* Bdp=NULL;
	int newM, newN;
	m = A->m;	n = A->n;	Ap = A->p;	Ai = A->i;	Ax = A->x;

	newM = r_right-r_left+1;
	newN = c_right-c_left+1;
	if(B->nrows!=newM || B->ncols!=newN)		NN_error("NN_isubmatrix_sp: sub-matrix != ret matrix");
	if((r_left<0 || r_right>(m-1)) || (c_left<0 || c_right>(n-1))) NN_error("Index exceeds matrix dimensions or out of bounds\n");
	if(newM <= 0 || newN <= 0)	{ fprintf(stderr, "\tEmpty matrix: %d-by-%d\n", newM, newN); return; }
	
	Bdp=B->data;
	for(j=c_left; j<=c_right; j++) {
		for(p=Ap[j]; p<Ap[j+1];p++) {
			if(Ai[p]>=r_left && Ai[p] <=r_right) {
				Bdp[(j-c_left)*newM+(Ai[p]-r_left)]=Ax[p];
			}
		}
	}
}

NN_Matrix* NN_submatrix(NN_Matrix* A, int r_left, int r_right, int c_left, int c_right)
{
	size_t m,n;
	m=A->nrows;	n=A->ncols;
	if((r_left<0 || r_right>=m) || (c_left<0 || c_right>=n))	NN_error("NN_submatrix: indexing out of bounds");

	size_t newM,newN,size;
	newM = r_right-r_left+1;
	newN = c_right-c_left+1;
	size=newM*newN;

	NN_Matrix* B = NN_init_matrix(newM,newN);

	NN_isubmatrix(A,r_left,r_right,c_left,c_right,B);

	return B;
}

void NN_isubmatrix(NN_Matrix* A, int r_left, int r_right, int c_left, int c_right, NN_Matrix* B)
{
	size_t m,n;
	m=A->nrows;	n=A->ncols;
	if((r_left<0 || r_right>=m) || (c_left<0 || c_right>=n))	NN_error("NN_isubmatrix: indexing out of bounds");

	size_t newM,newN,size;
	unsigned i,ii,jj;
	newM = r_right-r_left+1;
	newN = c_right-c_left+1;
	if(B->nrows!=newM || B->ncols!=newN)		NN_error("NN_isubmatrix: size of result matrix is wrong");
	size=newM*newN;

	element* Adp=A->data;
	element* Bdp=B->data;
	
//#pragma omp parallel for default(none) private(i,ii,jj) shared(size,Adp,Bdp,newM,c_left,m,r_left)
// DO NOT parallelize the following loop	
	for(i=0;i<size;i++) {
		ii = i%newM;	jj = i/newM;
		Bdp[jj*newM+ii]=Adp[(jj+c_left)*m+ii+r_left];
	}

}

NN_Matrix* NN_subColumns(NN_Matrix* A, unsigned int* colIndices, size_t nIndices)
{
	size_t m,n;
	unsigned int i;
	m=A->nrows;	n=A->ncols;
	// check
	for(i=0;i<nIndices;i++)	if(colIndices[i]<0 || colIndices[i]>=n)	NN_error("NN_subColumns: Indexing out of bounds");

	NN_Matrix* B;
	
	B=NN_init_matrix(m,nIndices);

	NN_isubColumns(A,colIndices,nIndices,B);

	return B;
}

void NN_isubColumns(NN_Matrix* A, unsigned int* colIndices, size_t nIndices, NN_Matrix* B)
{
	size_t m,n;
	unsigned int i;
	m=A->nrows;	n=A->ncols;
	// check
	if(B->nrows!=m || B->ncols!=nIndices)		NN_error("NN_isubColumns: size of result matrix is wrong");
	for(i=0;i<nIndices;i++)	if(colIndices[i]<0 || colIndices[i]>=n)	NN_error("NN_isubColumns: Indexing out of bounds");

	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;	

	for(i=0;i<nIndices;i++) {
		memcpy(&Bdp[i*m],&Adp[colIndices[i]*m],m*sizeof(element));
	}

}

cs* NN_subColumns_sp(cs* A, int c_left, int c_right)
{
	csi j,p,*Ap,*Ai;
	double *Ax;
    cs *T,*ret;

	Ap = A->p;	Ai = A->i;	Ax = A->x;
    if (!A) return (NULL) ;                             /* check inputs */
    T = cs_spalloc (0, 0, 1, 1, 1) ;                    /* allocate result */
	for(j=c_left;j<=c_right;j++) {
		for(p=Ap[j];p<Ap[j+1];p++) {
			if(!cs_entry(T, Ai[p], j-c_left, Ax[p])) return (cs_spfree(T));
		}
	}
	cs_entry(T,A->m-1,c_right-c_left,0.0);

	ret = cs_compress(T);	cs_spfree(T);
	return ret;
}

NN_Matrix* NN_subRows(NN_Matrix* A, unsigned int* rowIndices, size_t nIndices)
{
	size_t m,n;
	unsigned int i;
	m=A->nrows;	n=A->ncols;
	// check
	for(i=0;i<nIndices;i++)	if(rowIndices[i]<0 || rowIndices[i]>=m)	NN_error("NN_subRows: Indexing out of bounds");

	NN_Matrix* B;
	
	B=NN_init_matrix(nIndices,n);
	NN_isubRows(A,rowIndices,nIndices,B);
	return B;
}

void NN_isubRows(NN_Matrix* A, unsigned int* rowIndices, size_t nIndices, NN_Matrix* B)
{
	size_t m,n;
	unsigned int i,j;
	m=A->nrows;	n=A->ncols;
	// check
	if(B->nrows!=nIndices || B->ncols!=n)		NN_error("NN_isubColumns: size of result matrix is wrong");
	for(i=0;i<nIndices;i++)	if(rowIndices[i]<0 || rowIndices[i]>=m)	NN_error("NN_isubRows: Indexing out of bounds");

	element *Adp,*Bdp;
	Adp=A->data;	Bdp=B->data;	

#pragma omp parallel for private(i,j) shared(Adp,Bdp,nIndices,rowIndices)
	for(j=0;j<n;j++) {
		for(i=0;i<nIndices;i++) {
			Bdp[j*nIndices+i]=Adp[j*m+rowIndices[i]];
		}
	}

}

void NN_sp_to_full(NN_Matrix* dst, cs* src)
{
	unsigned int j;
	size_t m,n;
	csi p,*src_p,*src_i;
	double *src_x;

	if(!src)	NN_error("source matrix is empty");
	m=src->m;	n=src->n;
	if(dst->nrows!=m || dst->ncols!=n)	NN_error("size of full matrix and sparse matrix does not match");	

	NN_fill_zeros(dst);
	src_p=src->p;	src_i=src->i;	src_x=src->x;
	
	for(j=0;j<n;j++) {
		for(p=src_p[j];p<src_p[j+1];p++) {
			NN_set_value(dst,src_i[p],j,src_x[p]);
		}
	}
	
}

FILE* NN_fopen(char* filepath, char* mode)
{
	FILE* fp = fopen(filepath,mode);
	if(fp==NULL)	NN_error("NN_open_file: failed to open");
	
	return fp;
}

cs* NN_load_spdata(FILE* fp)
{
	cs *triplet,*compressed;
	triplet = cs_load(fp);
	compressed = cs_compress(triplet);		cs_spfree(triplet);
	
	return compressed;
}

void NN_error(const char *err_msg)
{
	fprintf(stdout, err_msg);
	fprintf(stdout, "\n");
	exit(-1);
}

NN_Matrix *NN_free(NN_Matrix *ptr)
{
	if(!ptr) return (NULL);
	free(ptr->data);
	free(ptr);

	return (NULL);
}

void NN_current_time()
{
	time_t t;
	struct tm* timeinfo;
	char buf[80];
	
	t = time(NULL);
	timeinfo = localtime(&t);

	strftime(buf,80,"[%Y-%m-%d %H:%M:%S] ", timeinfo);
	fprintf(stdout, "%s", buf);

	return;
}
