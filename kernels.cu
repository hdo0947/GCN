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
	float* outputfeatures_device;
	int* indexes_deivce;
	int* neighbours_device;
	cudaMalloc((float**)&inputfeatures_device, sizeof(float) * GCN_c.feature_c.feature_num * GCN_c.feature_c.node_num);
	cudaMalloc((float**)&outputfeatures_device, sizeof(float) * GCN_c.feature_c.feature_num * GCN_c.feature_c.node_num);
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
	aggregation_cuda_v0<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_device, 
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
	printf("Time cost for CPU version of fisrt aggregation is %d nanoseconds. \n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());

	std::cout << "The GPU version v0 of aggregation result is " << 
			  verified_feature(outputfeatures_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;
	
	/////////////////////////// Test first combination //////////////////////////////////////

	// CUDA version
	// TODO

	// CPU version
	started = std::chrono::high_resolution_clock::now();
	feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);
	done = std::chrono::high_resolution_clock::now();
	printf("Time cost for CPU version of fisrt combination is %d nanoseconds. \n", std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count());


	return 0;
}