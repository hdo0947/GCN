#include "header.h"

#define TILED_SIZE 16

__global__ void aggregation_cuda_v0(float* inputfeature, float* outputfeature, int* indexes, int* neighbours, int feature_num, int node_num, int edge_num) {
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	int index_y = by * TILED_SIZE + ty;
	int index_x = bx * TILED_SIZE + tx;	
	// printf("input %d: \n", inputfeature[index_x * node_num + index_y]);
	printf("index %d: \n", indexes[index_y]);
	printf("output %f: \n", outputfeature[index_x * node_num + index_y]);
	if (index_y < node_num && index_x < feature_num){
		for (int j = indexes[index_y]; j < indexes[index_y + 1]; ++j) {
			// Change to a adjacency matrix and perform row by operations
			outputfeature[index_x * node_num + index_y] += inputfeature[index_x * node_num + neighbours[j]];
		}
		outputfeature[index_x * node_num + index_y] /= ((float)indexes[index_y + 1] - indexes[index_y]);
		printf("input %d: \n", inputfeature[index_x * node_num + index_y]);
		printf("index %d: \n", indexes[index_y]);
		printf("output %f: \n", outputfeature[index_x * node_num + index_y]);
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

	cudaMemcpy(inputfeatures_device, GCN_c.feature_c.features, sizeof(float) * GCN_c.feature_c.feature_num * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	cudaMemcpy(indexes_deivce, GCN_c.graph_c.indexes, sizeof(float) * (GCN_c.spec_c.nodes + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(neighbours_device, GCN_c.graph_c.neighbours, sizeof(float) * (GCN_c.spec_c.edges), cudaMemcpyHostToDevice);
	// int *indexes_test = GCN_c.graph_c.indexes;
	// for (int i = 0 ; i < GCN_c.feature_c.node_num; ++i){
	// 	printf("The %dth node: %d \n", i, indexes_test[i + 1]);
	// }

	// printf("nodes %d, edges %d, features %d, hidden %d, labels %d : \n", 
	// 	GCN_c.spec_c.nodes, GCN_c.spec_c.edges, GCN_c.spec_c.features, GCN_c.spec_c.hidden, GCN_c.spec_c.labels);
	
	// printf("%d\n", GCN_c.feature_c.node_num);

	dim3 gridDim(int(ceil(GCN_c.feature_c.feature_num/float(TILED_SIZE))), 
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE))));
  	dim3 blockDim(TILED_SIZE, TILED_SIZE);
	auto started = std::chrono::high_resolution_clock::now();
	aggregation_cuda_v0<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_device, 
												indexes_deivce, neighbours_device, 
												GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges);
	auto done = std::chrono::high_resolution_clock::now();
	// printf("\n");
	printf("Time cost for GPU version v0 of aggregation is %d microseconds. \n", std::chrono::duration_cast<std::chrono::microseconds>(done-started).count());

	feature_t feature_c;
	started = std::chrono::high_resolution_clock::now();
	feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
	done = std::chrono::high_resolution_clock::now();
	printf("Time cost for CPU version of aggregation is %d microseconds. \n", std::chrono::duration_cast<std::chrono::microseconds>(done-started).count());
	return 0;
}