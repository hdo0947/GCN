#include "header.h"

#define TILED_SIZE 16

__global__ void aggregation_cuda_v0(float* inputfeature, float* outputfeature, int* indexes, int* neighbours, int feature_num, int node_num, int edge_num) {
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is feature dimension
	// y is node dimension 
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;
	if (index_x < feature_num && index_y < node_num){
		for (int j = indexes[index_y]; j < indexes[index_y + 1]; ++j) {
			outputfeature[index_x * node_num + index_y] += inputfeature[index_x * node_num + neighbours[j]];
		}
		outputfeature[index_x * node_num+ index_y] /= (float)(indexes[index_y + 1] - indexes[index_y]);
	} 
}

__global__ void combination_v0( float* in_features, int in_feature_num, int in_node_num, //feature_t in_feature
			     float* out_features, int out_feature_num, int out_node_num, //feature_t out_feature
			     float* biases, float* weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
			     bool relu){
						 
	// Keep the same checks as before
	if (in_feature_num != in_feature_num_p) {
    	printf("ERROR: Incompatible number of features in feature (%d) and parameter (%d) objects!\n", in_feature_num, in_feature_num_p);
    	// exit(-1);
	}
	// set values of the out_feature_c
	out_feature_num = out_feature_num_p;
	out_node_num = in_node_num;

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is out feature num
	// y is node dimension 
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;

	// Single read in of biases, no need for shared mem
	
			 
	if( index_y < out_feature_num && index_x < in_node_num){
		// printf("hi\n");
		out_features[index_y * out_feature_num + index_x] =  biases[index_y];
		// Consider matrix kernel multiplication methods, since we can read in whole index_ys at a time
		float val = 0.0f;
		for(int k = 0; k < in_feature_num_p; ++k){
			// atomic add for future versions
			val += in_features[k * out_feature_num + index_x] * weights[k * out_feature_num + index_y];
		}
		out_features[index_y * out_feature_num + index_x] += val;
		__syncthreads();
		if(relu)
			out_features[index_y * out_feature_num + index_x] = MAX(0.00000, out_features[index_y * out_feature_num + index_x]);
			// out_features[index_y * out_node_num + index_x] = fmaxf(0.00000, out_features[index_y * out_node_num + index_x]);
	}
	__syncthreads();

}

// __global__ void combination_v0( float* in_features, int in_feature_num, int in_node_num, //feature_t in_feature
// 			     float* out_features, int out_feature_num, int out_node_num, //feature_t out_feature
// 			     float* biases, float* weights, int in_feature_num_p, int out_feature_num_p, //parameter_t
// 			     bool relu){
// 	int bx = blockIdx.x; int by = blockIdx.y;
// 	int tx = threadIdx.x; int ty = threadIdx.y; 
// 	// x is feature dimension
// 	// y is node dimension 
// 	int index_x = bx * TILED_SIZE + tx;	
// 	int index_y = by * TILED_SIZE + ty;
// 	if (index_x == 0 && index_y  == 16 ){
// 		printf("1111111111111111111111the device weight is \n");
// 	}
// }


__global__ void analyzer_cuda_v0(float* inputfeature, int* label, int feature_num, int node_num, int* correctness){
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is feature dimension
	// y is node dimension 
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;	
	if (index_y < node_num){
		correctness[index_y] = 1;
		
	}
	__syncthreads();
	if (index_x < feature_num && index_y < node_num){
		if (index_x == 0 && index_y == 1){
			// printf("The input featuer is %f, the label feature is %f\n", 
			// 		inputfeature[index_x * node_num + index_y], 
			// 		label[index_y] * node_num + index_y);
		}
		if (inputfeature[index_x * node_num + index_y] > inputfeature[label[index_y] * node_num + index_y]){
			correctness[index_y] = 0;
		}
	}
	__syncthreads();
	// if (index_x == 0 && correctness[index_y] == 1){
	// 	printf("The output correctness is %d for node index %d\n", correctness[index_y], index_y);
	// }
}

bool verified_feature(float* feature_device, float** feature_host_true, int feature_num, int node_num){
	float* feature_host = (float *) malloc (feature_num * node_num * sizeof(float));
	cudaMemcpy(feature_host, feature_device, feature_num * node_num * sizeof(float), cudaMemcpyDeviceToHost);
	for (int f = 0; f < feature_num; ++ f){
		for (int n = 0 ; n < node_num ; ++ n){
			if ( abs(feature_host_true[f][n] - feature_host[f * node_num + n]) < 1e-4 ){
				continue;
			} else{
				printf("The first wrong answer is %f for the %d feature and %d node, the true value is %f \n", 
					  feature_host[f * node_num + n], f, n, feature_host_true[f][n]);
				return false;
			}
		}
	}
	return true;
}

void convert2DarrayTo1Darray(float** input, float* output, int x_length, int y_length){
	for (int x = 0; x < x_length; ++ x){
		for (int y = 0 ; y < y_length ; ++ y){
			output[x * y_length + y] = input[x][y];
			// printf("%f\n", output[x * y_length + y]);
		}	
	}
}

int main(int argc, char const *argv[]) {
	if ((argc != 2) || ((strcmp(argv[1], "cora") != 0) && (strcmp(argv[1], "citeseer") != 0) && (strcmp(argv[1], "reddit") != 0))) {
		printf("ERROR: usage \"%s [cora|citeseer|reddit]\"\n", argv[0]);
		return -1;
	}
	GCN_t GCN_c = GCN_parser((char*)argv[1]);
	float* inputfeatures_device;
	float* outputfeatures_agg1_device;
	int* indexes_deivce;
	int* neighbours_device;
	cudaMalloc((float**)&inputfeatures_device, sizeof(float) * GCN_c.feature_c.feature_num * GCN_c.feature_c.node_num);
	cudaMalloc((float**)&outputfeatures_agg1_device, sizeof(float) * GCN_c.feature_c.feature_num * GCN_c.feature_c.node_num);
	cudaMalloc((int**)&indexes_deivce, sizeof(int) * (GCN_c.spec_c.nodes + 1));
	cudaMalloc((int**)&neighbours_device, sizeof(int) * (GCN_c.spec_c.edges));
	float* features = (float*) malloc(GCN_c.feature_c.node_num * GCN_c.feature_c.feature_num * sizeof(float));

	convert2DarrayTo1Darray(GCN_c.feature_c.features, features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num);

	cudaMemcpy(inputfeatures_device, features, sizeof(float) * GCN_c.feature_c.feature_num * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	cudaMemcpy(indexes_deivce, GCN_c.graph_c.indexes, sizeof(int) * (GCN_c.spec_c.nodes + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(neighbours_device, GCN_c.graph_c.neighbours, sizeof(int) * (GCN_c.spec_c.edges), cudaMemcpyHostToDevice);


	/////////////////////////// Test first aggregation //////////////////////////////////////

	// CUDA version
	dim3 gridDim(int(ceil(GCN_c.feature_c.feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	dim3 blockDim(TILED_SIZE, TILED_SIZE);
	
	auto started = std::chrono::high_resolution_clock::now();
	aggregation_cuda_v0<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_agg1_device, 
												indexes_deivce, neighbours_device, 
												GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges);
	cudaDeviceSynchronize();
	auto done = std::chrono::high_resolution_clock::now();
	// printf("\n");
	printf("Time cost for GPU version v0 of fisrt aggregation is %d nanoseconds. \n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());

	// CPU version
	feature_t feature_c;
	started = std::chrono::high_resolution_clock::now();
	feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
	done = std::chrono::high_resolution_clock::now();
	printf("Time cost for CPU version of fisrt aggregation is %d nanoseconds. \n\n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());

	std::cout << "The GPU version v0 of aggregation result is " << 
			  verified_feature(outputfeatures_agg1_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;
	
	/////////////////////////// Test first combination //////////////////////////////////////

	// CUDA version
	// TODO
	
	float* outputfeatures_comb1_device;
	float* biases_comb1_device;
	float* weights_comb1_device;
	cudaMalloc((float**)&outputfeatures_comb1_device, sizeof(float) * GCN_c.l1_parameter_c.out_feature_num * GCN_c.feature_c.node_num);
	cudaMalloc((float**)&biases_comb1_device, sizeof(float) * GCN_c.l1_parameter_c.out_feature_num);
	cudaMalloc((float**)&weights_comb1_device, sizeof(float) * GCN_c.l1_parameter_c.out_feature_num * GCN_c.l1_parameter_c.in_feature_num);
	float* weights_comb1 = (float*) malloc(GCN_c.l1_parameter_c.out_feature_num * GCN_c.l1_parameter_c.in_feature_num * sizeof(float));

	convert2DarrayTo1Darray(GCN_c.l1_parameter_c.weights, weights_comb1, GCN_c.l1_parameter_c.in_feature_num, GCN_c.l1_parameter_c.out_feature_num);

	cudaMemcpy(biases_comb1_device, GCN_c.l1_parameter_c.biasses, sizeof(float) * GCN_c.l1_parameter_c.out_feature_num, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_comb1_device, weights_comb1, sizeof(float) * GCN_c.l1_parameter_c.out_feature_num * GCN_c.l1_parameter_c.in_feature_num, cudaMemcpyHostToDevice);

	// for (int i = 0; i < GCN_c.l1_parameter_c.in_feature_num; i ++){
	// 	for (int j = 0; j < GCN_c.l1_parameter_c.out_feature_num; j ++){
	// 		printf("for the index i = %d, j = %d, the value is: %f \n", i, j, GCN_c.l1_parameter_c.weights[i][j]);
	// 	}
	// }
	printf("for the index i = %d, j = %d, the value is: %f \n", 9, 9, GCN_c.l1_parameter_c.weights[9][9]);

	gridDim = dim3(int(ceil(GCN_c.l1_parameter_c.out_feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	blockDim = dim3(TILED_SIZE, TILED_SIZE);

	printf("out_feature_num is %d \n", GCN_c.l1_parameter_c.out_feature_num);
	printf("Dim grid is x:%d, y: %d \n", gridDim.x, gridDim.y);

	started = std::chrono::high_resolution_clock::now();
	// combination_v0<<<gridDim, blockDim>>>(outputfeatures_agg1_device, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, //feature_t in_feature
	// 									outputfeatures_comb1_device, GCN_c.l1_parameter_c.out_feature_num, GCN_c.feature_c.node_num, //feature_t out_feature
	// 									biases_comb1_device, weights_comb1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.l1_parameter_c.out_feature_num, //parameter_t
	// 									true);
	combination_v0<<<gridDim, blockDim>>>(outputfeatures_agg1_device, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, //feature_t in_feature
										outputfeatures_comb1_device, GCN_c.l1_parameter_c.out_feature_num, GCN_c.feature_c.node_num, //feature_t out_feature
										biases_comb1_device, weights_comb1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.l1_parameter_c.out_feature_num, //parameter_t
										true);
	cudaDeviceSynchronize();
	done = std::chrono::high_resolution_clock::now();	
	printf("Time cost for CPU version of fisrt combination is %d nanoseconds. \n\n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());


	// CPU version
	started = std::chrono::high_resolution_clock::now();
	feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);
	done = std::chrono::high_resolution_clock::now();
	printf("Time cost for CPU version of fisrt combination is %d nanoseconds. \n\n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());

	std::cout << "The GPU version v0 of combination result is " << 
			  verified_feature(outputfeatures_comb1_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;	


	/////////////////////////// Test Analyzer //////////////////////////////////////
	int* correctness;
	int* label_device;
	cudaMalloc((int**)&correctness, sizeof(int) * GCN_c.feature_c.node_num);
	cudaMalloc((int**)&label_device, sizeof(int) * GCN_c.feature_c.node_num);
	cudaMemcpy(label_device, GCN_c.label_c, sizeof(int) * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	// CUDA version
	started = std::chrono::high_resolution_clock::now();
	analyzer_cuda_v0<<<gridDim, blockDim>>>(inputfeatures_device, 
											label_device, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, 
											correctness);
	cudaDeviceSynchronize();
	done = std::chrono::high_resolution_clock::now();
	printf("Time cost for GPU version v0 of analyzer is %d nanoseconds. \n\n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());


	// CPU version
	started = std::chrono::high_resolution_clock::now();
	analyzer(feature_c, GCN_c.label_c);
	done = std::chrono::high_resolution_clock::now();
	printf("Time cost for CPU version of analyzer is %d nanoseconds. \n\n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());
	return 0;
}