#include <stdio.h>
#include <string.h>
#include <time.h>

#include "NN_utils.h"
#include "NN_math.h"
#include "NN_core.h"
#include "AE.h"
#include "cs.h"		/* For sparse matrix */

typedef struct {
	size_t D;
	size_t M;
	size_t F;
} AE_arch;

typedef struct {
	element* W;
	element* hb;
	element* vb;
	element* adaGrad_W;
	element* adaGrad_hb;
	element* adaGrad_vb;
} AE_learned_params;

void AE_fwrite_results(AE_learned_params *params, size_t D, size_t M, size_t F, FILE* fp)
{
	fwrite(params->W, sizeof(element), F*D, fp);
	fwrite(params->hb, sizeof(element), F, fp);
	fwrite(params->vb, sizeof(element), D, fp);
	fwrite(params->adaGrad_W, sizeof(element), F*D, fp);
	fwrite(params->adaGrad_hb, sizeof(element), F, fp);
	fwrite(params->adaGrad_vb, sizeof(element), D, fp);
}

void print_help(char *prg_name)
{
	printf("usage: %s [options]\n", prg_name);
	printf("\n");
	printf("\t--help						show this message\n");
	printf("\t--trd-path					training data file path\n");
	printf("\t--vad-path					validation data file path\n");
	printf("\t--hiddenlayer-size			set the number of units in a hidden layer [500]\n");
	printf("\t--learn-rate					set learning rate of Stochastic Gradient Descent [0.1]\n");
	printf("\t--max-iter					set the maximum number of iteration in SGD [10]\n");
	printf("\t--normalize					set normalization to make reconstructions have unit length [false]\n");
	printf("\t--l1-lambda					set magnitude of L1 regularization on hidden activations [0]\n");
	printf("\t--l2-lambda					set magnitude of L2 regularization on weights [0]\n");
	printf("\t--minibatch-size				set the size of mini-batch [100]\n");
	printf("\t--activation-function			set the type of hidden activation function (sigmoid,rectified, and _tanh) [sigmoid] \n");
	printf("\t--cost-function				set the type of cost function (squaredError, crossEntropy, and cosineDist) [squaredError] \n");
	printf("\n");

	exit(-1);
}

int main(int argc, char *argv[])
{
	unsigned int pass,sample_idx,end_sample_idx,i;
	size_t D,M,F=-1;
	FILE *fp;
	bool verbose = true;
	bool is_normalize = false;
	float l1_lambda = 0;
	float l2_lambda = 0;
	actFuncType actFunc = sigmoid;
	errFuncType errFunc = squaredError;
	unsigned int max_pass = 10;
	size_t mb_sz = 1000;
	//const element momentum = 0.1;
	element eta_base = 0.1;
	char *trd_path,*vad_path;
	trd_path=NULL;	vad_path=NULL;

	if(argc < 2)	{
		printf("%s: no options specified\n", argv[0]);
		printf("%s: Use --help for more information.\n", argv[0]);
		exit(-1);
	}

	for(i=1;i<argc;i++) {
		if(!strcmp(argv[i], "--trd-path")) {
			trd_path=argv[i+1];	i++;	
		} else if(!strcmp(argv[i], "--vad-path")) {
			vad_path=argv[i+1];	i++;	
		} else if(!strcmp(argv[i], "--hiddenlayer-size")) {
			F=atoi(argv[i+1]);		i++;
		} else if(!strcmp(argv[i], "--learn-rate")) {
			eta_base=atof(argv[i+1]);		i++;
		} else if(!strcmp(argv[i], "--minibatch-size")) {
			mb_sz=atoi(argv[i+1]);	i++;
		} else if(!strcmp(argv[i], "--max-iter")) {
			max_pass=atoi(argv[i+1]);	i++;
		} else if(!strcmp(argv[i], "--normailze")) {
			is_normalize = true;
		} else if(!strcmp(argv[i], "--l1-lambda")) {
			l1_lambda=atof(argv[i+1]);	i++;
		} else if(!strcmp(argv[i], "--l2-lambda")) {
			l2_lambda=atof(argv[i+1]);	i++;
		} else if(!strcmp(argv[i], "--activation-function")) {
			if(strcmp(argv[i+1], "sigmoid")==0) {
				actFunc = sigmoid;
			} else if(strcmp(argv[i+1], "tanh")==0) {
				actFunc = _tanh;
			} else if(strcmp(argv[i+1], "relu")==0) {
				actFunc = rectified;
			} else {
				fprintf(stderr,"--activation-function: %s: unsupported activation function\n", argv[i+1]);
				exit(-1);
			}
			i++;
		} else if(!strcmp(argv[i], "--cost-function")) {
			if(strcmp(argv[i+1], "squaredError")==0) {
				errFunc = squaredError;
			} else if(strcmp(argv[i+1], "crossEntropy")==0) {
				errFunc = crossEntropy;
			} else if(strcmp(argv[i+1], "cosineDist")==0) {
				errFunc = cosineDist;
			} else {
				fprintf(stderr,"--cost-function: %s: unsupported cost function\n", argv[i+1]);
				exit(-1);
			}
			i++;
		} else if(!strcmp(argv[i], "--help")) {
			print_help(argv[0]);
		} else {
			printf("Invalid argument: %s\n", argv[i]);
			print_help(argv[0]);
		}
	}

	if(!trd_path)	NN_error("training data and label files should be specified\n\tExited..");

	srand(time(NULL));

	regularization_option reg_option;
	reg_option.l1_lambda = l1_lambda;
	reg_option.l2_lambda = l2_lambda;
	reg_option.normalize = is_normalize;

	NN_Matrix *W, *hb, *vb, *dW, *dhb, *dvb;
	NN_Matrix *eta_W,*eta_hb,*eta_vb;
	NN_Matrix *adaGrad_W,*adaGrad_hb,*adaGrad_vb;
	//element *grad_W,*grad_hb,*grad_vb;
	NN_Matrix *minibatch;
	element cost;
	cs *traindata, *permuted_traindata;
	csi *ri,*ci;

	// load training examples from a file
	fp = NN_fopen(trd_path,"r");
	traindata = NN_load_spdata(fp);
	fprintf(stdout, "Training data is loaded..\n");

	D = traindata->m;	/* dim of data */
	M = traindata->n;	/* the number of training examples */

	ri = cs_randperm(D,0);
	ci = cs_randperm(M,time(NULL));
	permuted_traindata = cs_permute(traindata,cs_pinv(ri,D),ci,1);
	cs_spfree(traindata);
	traindata = permuted_traindata;		permuted_traindata=NULL;

	AE_arch netconfig;
	netconfig.D = D;
	netconfig.M = M;
	netconfig.F = F;

	fp = fopen("netconfig.bin", "wb");
	fwrite(&netconfig, sizeof(netconfig), 1, fp);
	fclose(fp);	

	if(verbose)	fprintf(stdout,"D: %d\nM: %d\nF: %d\n", (int)D,(int)M,(int)F);

	// initialize learning parameters
	W = NN_rand_matrix(F,D);
	element r = sqrt(6)/sqrt(F+D+1);
	W = NN_addsi(1.0,NN_muli(W,2*r),-r);
	hb = NN_zeros_matrix(F,1);
	vb = NN_zeros_matrix(D,1);

	dW = NN_zeros_matrix(F,D);
	dhb = NN_zeros_matrix(F,1);
	dvb = NN_zeros_matrix(D,1);	
	adaGrad_W = NN_zeros_matrix(F,D);
	adaGrad_hb = NN_zeros_matrix(F,1);
	adaGrad_vb = NN_zeros_matrix(D,1);

	AE_interim* tmp_vars = (AE_interim*) malloc(sizeof(AE_interim));
	tmp_vars->A1 	= NN_zeros_matrix(F,mb_sz);
	tmp_vars->H 	= NN_zeros_matrix(F,mb_sz);
	tmp_vars->dH 	= NN_zeros_matrix(F,mb_sz); 
	tmp_vars->A2 	= NN_zeros_matrix(D,mb_sz);
	tmp_vars->O 	= NN_zeros_matrix(D,mb_sz);
	tmp_vars->dO 	= NN_zeros_matrix(D,mb_sz);
	if(errFunc==cosineDist)	{
		tmp_vars->J 	= NN_zeros_matrix(1,mb_sz);
	} else {
		tmp_vars->J 	= NN_zeros_matrix(D,mb_sz);
	}
	tmp_vars->dJ 	= NN_zeros_matrix(D,mb_sz);
/*
	grad_W = NN_zeros_matrix(F*D);
	grad_hb = NN_zeros_matrix(F);
	grad_vb = NN_zeros_matrix(D);
*/

	double* train_cost_per_minibatch = malloc((int)(M/mb_sz)*sizeof(double));
	minibatch = NN_init_matrix(D,mb_sz);

	// run
	for(pass=0; pass < max_pass; pass++) {
		if(verbose)	fprintf(stdout, "Pass: %d/%d\n", pass+1, max_pass);	fflush(stdout);

		memset(train_cost_per_minibatch,0,(int)(M/mb_sz)*sizeof(double));

		for(sample_idx=0; (sample_idx+mb_sz) <= M; sample_idx+=mb_sz) {

			// extract mini-batch of data
			if(sample_idx+mb_sz-1>=M)	end_sample_idx = M-1;
			else						end_sample_idx = sample_idx+mb_sz-1;

			NN_isubmatrix_sp(traindata,0,D-1,sample_idx,end_sample_idx,minibatch);

			// compute cost and gradients
			AE(W,hb,vb,minibatch,minibatch,reg_option,actFunc,errFunc,&cost,dW,dhb,dvb,tmp_vars);
			train_cost_per_minibatch[(int)(sample_idx/mb_sz)] = cost;

			if(verbose)	{
				NN_current_time();
				fprintf(stdout, "\t%u/%lu\tcost: %.4f\n", sample_idx,M,cost);	fflush(stdout);
			}

			// update adaGrad
			//printf("update adaGrad\n");
			NN_Matrix* sq_dW = NN_pow(dW,2);
			NN_Matrix* sq_dhb = NN_pow(dhb,2);
			NN_Matrix* sq_dvb = NN_pow(dvb,2);
			adaGrad_W = addi(1.0,sq_dW,1.0,adaGrad_W);		NN_free(sq_dW);
			adaGrad_hb = addi(1.0,sq_dhb,1.0,adaGrad_hb);		NN_free(sq_dhb);
			adaGrad_vb = addi(1.0,sq_dvb,1.0,adaGrad_vb);		NN_free(sq_dvb);

			// update eta & parameters
			//printf("update eta\n");
			eta_W = NN_muli(NN_powi(NN_sqrt(adaGrad_W),-1),eta_base);	eta_W = NN_infinity_to_zero(eta_W);
			W = addi(-1,elemwise_mmi(dW,eta_W),1,W);		NN_free(eta_W);

			eta_hb = NN_muli(NN_powi(NN_sqrt(adaGrad_hb),-1),eta_base);	eta_hb=NN_infinity_to_zero(eta_hb);
			hb = addi(-1,elemwise_mmi(dhb,eta_hb),1,hb);	NN_free(eta_hb);

			eta_vb = NN_muli(NN_powi(NN_sqrt(adaGrad_vb),-1),eta_base);	eta_vb=NN_infinity_to_zero(eta_vb);
			vb = addi(-1,elemwise_mmi(dvb,eta_vb),1,vb);	NN_free(eta_vb);	
			//printf("Done a parameter update\n");
/*
			grad_W=iadd_vv(momentum,dW,1-momentum,grad_W,F*D);
			grad_hb=iadd_vv(momentum,dhb,1-momentum,grad_hb,F);
			grad_vb=iadd_vv(momentum,dvb,1-momentum,grad_vb,D);
			W = iadd_vv(-eta_base,grad_W,1,W,F*D);		NN_free(dW);
			hb = iadd_vv(-eta_base,grad_hb,1,hb,F);		NN_free(dhb);
			vb = iadd_vv(-eta_base,grad_vb,1,vb,D);		NN_free(dvb);
*/
		}

		// compute averaged cost
		double avg_cost = 0.0;
		for(i=0;i<floor(M/mb_sz);i++) {
			avg_cost+= train_cost_per_minibatch[i];
		}

		if(verbose)	fprintf(stdout, "%.4f\n", avg_cost/floor(M/mb_sz));

		// store intermediate parameters to continue training afterwards
/*
		AE_learned_params learned_params;
		learned_params.W = W;
		learned_params.hb = hb;
		learned_params.vb = vb;
		learned_params.adaGrad_W = adaGrad_W;
		learned_params.adaGrad_hb = adaGrad_hb;
		learned_params.adaGrad_vb = adaGrad_vb;
		
		char filename[80];
		sprintf(filename, "intermediate_leared_parameters_at_%d_pass.bin", pass);
		fp = fopen(filename, "wb");
		AE_fwrite_results(&learned_params, D, M, F, fp);
		fclose(fp);
*/

		// shuffling data
		if(verbose)	fprintf(stdout, "shuffling data\n");
		ri = cs_randperm(D,0);
		ci = cs_randperm(M,time(NULL));
		permuted_traindata = cs_permute(traindata,cs_pinv(ri,D),ci,1);
		cs_spfree(traindata);
		traindata = permuted_traindata;		permuted_traindata=NULL;
	}

	// store learning results
/*

	AE_learned_params final_params;
	final_params.W = W;
	final_params.hb = hb;
	final_params.vb = vb;
	final_params.adaGrad_W = adaGrad_W;
	final_params.adaGrad_hb = adaGrad_hb;
	final_params.adaGrad_vb = adaGrad_vb;
		
	fp = fopen("final_learned_parameters.bin", "wb");
	AE_fwrite_results(&final_params, D, M, F, fp);
	fclose(fp);
*/
	// release resources
	free(train_cost_per_minibatch);			
	cs_spfree(traindata);

	NN_free(W);
	NN_free(hb);
	NN_free(vb);

	NN_free(adaGrad_W);
	NN_free(adaGrad_hb);
	NN_free(adaGrad_vb);
	return 0;
}
