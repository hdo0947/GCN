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
__global__ void combination_v0( float* &in_features, int &in_feature_num, int &in_node_num, //feature_t in_feature
			     float* &out_features, int &out_feature_num, int &out_node_num, //feature_t out_feature
			     float* &biases, float* &weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
			     bool relu){
	int i,j,k;
	// Keep the same checks as before
	if (in_feature_num != in_feature_num_p) {
    		printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_num, in_feature_num_p);
    		exit(-1);
	}
	// set values of the out_feature_c
	out_feature_num = out_feature_num_p;
	out_node_num = in_node_num;
	//out_feature_c.features = (float**) malloc (parameter_c.out_feature_num * sizeof(float*));
	
	/*
	--------------------Not for v0---------------
	__shared__ out_features [numRow][numCol];
	// in-feature will be read in # row times in the overall combination
	__shared__ in_features [k][numCol];
	// parameter will be called # column number of times
	__shared__ features [k][numRow];
	// K will work like the TILESIZE in matrix multiplication?
	--------------------Not for v0---------------
	*/
	// TILED_SIZE == blocksize == 16
	int col = blockIdx.x * TILED_SIZE + threadIdx.x;
    	int row = blockIdx.y * TILED_SIZE + threadIdx.y;
	
	// Single read in of biases, no need for shared mem
	out_features[row * out_node_num + col] =  biases[row];
			 
	if( row < out_feature_num && col < in_node_num){
		// Consider matrix kernel multiplication methods, since we can read in whole rows at a time
		float val = 0.0f;
		for(int k = 0; k < in_feature_num_p; ++l){
			// atomic add for future versions
			val += in_features[k * out_node_num + col] * weights[k * out_node_num + row];
		}
		out_features[row * out_node_num + col] = val;
		__syncthreads();
		if(relu)
			out_features[row * out_node_num + col] = MAX(0.00000, out_features[row * out_node_num + col]);
		
	}

}

// combination_v1 will start reading in the variables from global to shared
__global__ void combination_v1( float* &in_features, int in_feature_num, int in_node_num, //feature_t in_feature
			     float* &out_features, int &out_feature_num, int &out_node_num, //feature_t out_feature
			     float* &biases, float* &weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
			     bool relu){
	int i,j,k;
	// Keep the same checks as before
	if (in_feature_num != in_feature_num_p) {
    		printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_num, in_feature_num_p);
    		exit(-1);
	}
	// set values of the out_feature_c
	out_feature_num = out_feature_num_p;
	out_node_num = in_node_num;
	//out_feature_c.features = (float**) malloc (parameter_c.out_feature_num * sizeof(float*));
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int k = in_feature_num;
	// in-feature will be read in # row times in the overall combination
	__shared__ in [TILED_SIZE][TILED_SIZE];
	// parameter will be called # column number of times
	__shared__ weight [TILED_SIZE][TILED_SIZE];
	// K will work like the TILESIZE in matrix multiplication?

	// TILESIZE == blocksize == 16
	int col = blockIdx.x * TILED_SIZE + threadIdx.x;
    	int row = blockIdx.y * TILED_SIZE + threadIdx.y;
	
	// initialize with biases
	if( row < out_feature_num && col < in_node_num){
		out_features[row * out_node_num + col] =  biases[row];
	}
	// Tiled Matrix Multiplication
	for(int m = 0; m < (in_feature_num / TILED_SIZE); ++m){
		// Read in from global memory to shared memory
		if(m * TILE_WIDTH + tx < in_node_num && row < in_feature_num_p)
		    in[ty][tx] = in_features[Row * in_feature_num_p + m * TILE_WIDTH + tx];
		else
		    in[ty][tx] = 0.0f;
		
		if( m * TILE_WIDTH + ty < out_feature_num_p && col < in_feature_num_p)
		    weight[ty][tx] = weights[((m * TILE_WIDTH + ty) * in_feature_num_p + Col)];
		else
		    weight[ty][tx] = 0.0f;
		__syncthreads();
		// ith column of in with jth column of weight is the (j,i) of the out_features
		for(int n = 0; n < TILE_WIDTH; ++n){
			// Not coalesed read; want in[n][tx] * weight[n][ty] so we read row by row
			val += in[tx][n] * weight[ty][n];
		}
		__syncthreads();	
		
	}
    	if(row < out_feature_num_p && col < in_node_num)
		if(relu){
			out_features[row * in_node_num + col] = MAX(0.00000, val);
		}
		else{
			out_features[row * in_node_num + col] = val;
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
