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


__global__ void combination(feature_t in_feature_c, feature_t out_feature_c, parameter_t parameter_c, bool relu){
	int i,j,k;
	// Keep the same checks as before
	if (in_feature_c.feature_num != parameter_c.in_feature_num) {
    	printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_c.feature_num, parameter_c.in_feature_num);
    	exit(-1);
	}
	// set values of the out_feature_c
	out_feature_c.node_num = in_feature_c.node_num;
	out_feature_c.feature_num = parameter_c.out_feature_num;
	//out_feature_c.features = (float**) malloc (parameter_c.out_feature_num * sizeof(float*));
	
	// Declare shared Variable to reduce global reads and writes
	// One thing to do is we need the value in the numCol to be max of in_feature_c.node_num
	__shared__ features [parameter_c.out_feature_num][max(in_feature_c.node_num)];
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	features[row][col] =  parameter_c.biasses[j];
	// We only read in in_feature once, no need for shared memory
	// parameter_c.weights could be stored in shared memory as well... 
	//one of the various test could be shared vs. global for the in_features and parameter
	out_feature_c.features[row][col] += in_feature_c.features[k][i] * parameter_c.weights[k][j];
	
	out_feature_c.features = features
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
			// Unlike aggregation, no division afterword needed
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
