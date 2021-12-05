#include "header.h"

feature_t aggregation (graph_t graph_c, feature_t in_feature_c) {
	int i, k, j;
	feature_t out_feature_c;

	printf("AGGREGATION: A[%d][%d] * X[%d][%d] = X'[%d][%d]\n", in_feature_c.node_num, in_feature_c.node_num, in_feature_c.node_num, in_feature_c.feature_num, in_feature_c.node_num, in_feature_c.feature_num);
	
	// Same number of features for IN and OUT
	out_feature_c.feature_num = in_feature_c.feature_num;
	// Same number of Nodes for IN and OUT
	out_feature_c.node_num = in_feature_c.node_num;
	// Allocate Memory
	out_feature_c.features = (float**) malloc (in_feature_c.feature_num * sizeof(float*));
	
	//Possibly add a step to process all in shared memory instead of global memory since there is multiple access in the triple for loop
	
	// Double for loop for initialization
	// Initialization of the features: Allocation and setting the value as 0s
	for (i = 0; i < in_feature_c.feature_num; ++i) {
		out_feature_c.features[i] = (float*) malloc (in_feature_c.node_num * sizeof(float));
		for (j = 0; j < in_feature_c.node_num; ++j) {
			out_feature_c.features[i][j] = 0;
		}
	}
	
	// Triple for loop from node 0 to N
	for (i = 0; i < in_feature_c.node_num; ++i) {
		printf("\r%.2f%% Completed!", (float)i * 100.00 / (float)in_feature_c.node_num);
	    fflush(stdout);
		
		// Double for loop setting the values of the out_feature nodes
		// Number of Features 
		for (k = 0; k < in_feature_c.feature_num; ++k) {
			// sum of all the values in the in_feature edge connections to out_features
			for (j = graph_c.indexes[i]; j < graph_c.indexes[i + 1]; ++j) {
				// Change to a adjacency matrix and perform row by operations
				out_feature_c.features[k][i] += in_feature_c.features[k][graph_c.neighbours[j]];
			}
			// Divide by the differences in the indexes for nodes
			out_feature_c.features[k][i] /= ((float)graph_c.indexes[i + 1] - graph_c.indexes[i]);
		}
	}
	printf("\r\t\t\t\t\t\r");

	return out_feature_c;
}
// CUDA CODE FOR THIS SECTION
// The features and parpameters are dismantled so they can be read in for CUDA
__global__ void combination_v0( float* in_features, int in_feature_num, int in_node_num, //feature_t in_feature
								float* out_features, //feature_t out_feature
								float* biases, float* weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
								bool relu){				 
	// Keep the same checks as before
	if (in_feature_num != in_feature_num_p) {
    	printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_num, in_feature_num_p);
    	// exit(-1);
	}
	// set values of the out_feature_c
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is out feature num
	// y is node dimension
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;

	// Single read in of biases, no need for shared mem
	if( index_y < in_node_num && index_x < out_feature_num_p ){
		out_features[index_x * in_node_num + index_y] =  biases[index_x];

		float val = 0.0f;
		for(int k = 0; k < in_feature_num_p; ++k){
			// atomic add for future versions
			val += in_features[k * in_node_num + index_y] * weights[k * out_feature_num_p + index_x];
		}
		out_features[index_x * in_node_num + index_y] += val;
		__syncthreads();
		if(relu){
			out_features[index_x * in_node_num + index_y] = MAX(0.00000, out_features[index_x * in_node_num + index_y]);
		}
	}
	__syncthreads();
}

__global__ void combination_v1( float* in_features, int in_feature_num, int in_node_num, //feature_t in_feature
			     float* out_features, //feature_t out_feature
			     float* biases, float* weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
			     bool relu){
	
	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;
	
	// in-feature will be read in # row times in the overall combination
	__shared__ float in [TILED_SIZE][TILED_SIZE];
	// parameter will be called # column number of times
	__shared__ float weight [TILED_SIZE][TILED_SIZE];
	
	// x is out feature num
	// y is node dimension
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;
	
	// initialize with biases
	if( index_y < in_node_num && index_x < out_feature_num_p ){
		out_features[index_x * in_node_num + index_y] =  biases[index_x];
	}
	// Tiled Matrix Multiplication
	float val = 0.0f;
	for(int m = 0; m < (in_feature_num_p / float(TILED_SIZE)); ++m){
		// Read in from global memory to shared memory
		if(m * TILED_SIZE + tx < in_feature_num_p && index_y < in_node_num)
			in[tx][ty] = in_features[((m * TILED_SIZE + tx) * in_node_num + index_y)];
		else
			in[tx][ty] = 0.0f;
		
		if( m * TILED_SIZE + ty < in_feature_num_p && index_x < out_feature_num_p)
			weight[ty][tx] = weights[((m * TILED_SIZE + ty) * out_feature_num_p + index_x)];
		else
			weight[ty][tx] = 0.0f;
		__syncthreads();
		// ith column of in with jth column of weight is the (j,i) of the out_features
		for(int k = 0; k < TILED_SIZE; ++k){
			val += in[k][ty] * weight[k][tx];
		}
		__syncthreads();	
		
	}
	if(index_x == 16 && index_y == 0){
		printf("Something is wrong here: bias and val is %f %f \n", out_features[index_x * in_node_num + index_y], val);
	}
	out_features[index_x * in_node_num + index_y] += val;
	__syncthreads();
	if(relu && index_y < in_node_num && index_x < out_feature_num_p){
		out_features[index_x * in_node_num + index_y] = MAX(0.00000, out_features[index_x * in_node_num + index_y]);
	}

}


// combination_v2 will have the biases in const memory
__global__ void combination_v2( float* in_features, int in_feature_num, int in_node_num, //feature_t in_feature
			     float* out_features, //feature_t out_feature
			     float* biases, float* weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
			     bool relu){
	
	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;
	
	// in-feature will be read in # row times in the overall combination
	__shared__ float in [TILED_SIZE][TILED_SIZE];
	// parameter will be called # column number of times
	__shared__ float weight [TILED_SIZE][TILED_SIZE];
	// biases will be called in_node_num times for each
	__cost__ float bias[out_feature_num_p];
	// x is out feature num
	// y is node dimension
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;
	
	if( index_x < out_feature_num_p){
		bias[index_x] = biases[index_x]
	}
	// initialize with biases
	if( index_y < in_node_num && index_x < out_feature_num_p ){
		out_features[index_x * in_node_num + index_y] =  bias[index_x];
	}
	// Tiled Matrix Multiplication
	float val = 0.0f;
	for(int m = 0; m < (in_feature_num_p / float(TILED_SIZE)); ++m){
		// Read in from global memory to shared memory
		if(m * TILED_SIZE + tx < in_feature_num_p && index_y < in_node_num)
			in[tx][ty] = in_features[((m * TILED_SIZE + tx) * in_node_num + index_y)];
		else
			in[tx][ty] = 0.0f;
		
		if( m * TILED_SIZE + ty < in_feature_num_p && index_x < out_feature_num_p)
			weight[ty][tx] = weights[((m * TILED_SIZE + ty) * out_feature_num_p + index_x)];
		else
			weight[ty][tx] = 0.0f;
		__syncthreads();
		// ith column of in with jth column of weight is the (j,i) of the out_features
		for(int k = 0; k < TILED_SIZE; ++k){
			val += in[k][ty] * weight[k][tx];
		}
		__syncthreads();	
		
	}
	if(index_x == 16 && index_y == 0){
		printf("Something is wrong here: bias and val is %f %f \n", out_features[index_x * in_node_num + index_y], val);
	}
	out_features[index_x * in_node_num + index_y] += val;
	__syncthreads();
	if(relu && index_y < in_node_num && index_x < out_feature_num_p){
		out_features[index_x * in_node_num + index_y] = MAX(0.00000, out_features[index_x * in_node_num + index_y]);
	}

}


feature_t combination (feature_t in_feature_c, parameter_t parameter_c, bool relu) {
	int i, j, k;
	feature_t out_feature_c;

	if (in_feature_c.feature_num != parameter_c.in_feature_num) {
    	printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_c.feature_num, parameter_c.in_feature_num);
    	exit(-1);
	}

	printf("COMBINATION: X'[%d][%d] * W[%d][%d] = X[%d][%d]\n", in_feature_c.node_num, in_feature_c.feature_num, parameter_c.in_feature_num, parameter_c.out_feature_num, in_feature_c.node_num, parameter_c.out_feature_num);
	
	// Same number of nodes, features and allocation of memory
	out_feature_c.node_num = in_feature_c.node_num;
	out_feature_c.feature_num = parameter_c.out_feature_num;
	out_feature_c.features = (float**) malloc (parameter_c.out_feature_num * sizeof(float*));
	
	// Allocate Memory
	for (i = 0; i < parameter_c.out_feature_num; ++i) {
		out_feature_c.features[i] = (float*) malloc (in_feature_c.node_num * sizeof(float));
	}

	// Same optimization process could be applied as aggregation
	for (i = 0; i < in_feature_c.node_num; ++i) {
		printf("\r%.2f%% Completed!", (float)i * 100.00 / (float)in_feature_c.node_num);
	    fflush(stdout);
		for (j = 0; j < parameter_c.out_feature_num; ++j) {
			out_feature_c.features[j][i] = parameter_c.biasses[j];
			// Unlike aggregation, no division afterward needed
			for (k = 0; k < parameter_c.in_feature_num; ++k) {
				out_feature_c.features[j][i] += in_feature_c.features[k][i] * parameter_c.weights[k][j];
			}
			// Not a divergence here since this if statement will be applied to every thread if we use ReLU
			if(relu)
				out_feature_c.features[j][i] = MAX(0.00000, out_feature_c.features[j][i]);
		}
	}
	printf("\r\t\t\t\t\t\r");
	
	return out_feature_c;
}

void analyzer (feature_t feature_c, label_t label_c) {
	int i, j;
	int correct_num = 0;

	// Same double for loop as aggregation and combination
	for (i = 0; i < feature_c.node_num; ++i) {
		float max_feature = feature_c.features[0][i];
		int max_idx = 0;
		for (j = 1; j < feature_c.feature_num; ++j) {
			// Divergence that cannot be removed -> findimg max elelemt
			// Optimzation possible with different scan methods
			if(feature_c.features[j][i] > max_feature) {
				max_feature = feature_c.features[j][i];
				max_idx = j;
			}
		}
		// __syncthreads(); here
		// This divergence condition is necessary and cannot be removed
		if (max_idx == label_c[i]) {
			correct_num++;
		}
	}
	
	printf("Accuracy: %.2f%%\n", (float)correct_num * 100.00 / (float)feature_c.node_num);
}
