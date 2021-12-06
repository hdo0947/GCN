#include "header.h"

#define TILED_SIZE 16
#define TILED_SIZE_agg_cuda_v1 1024
#define MAX_LENGTH_NEIGHBOUR_agg_cuda_v1 1024
#define TILED_SIZE_agg_cuda_v3 1024
#define MAX_RAW_agg_cuda_v3 40
#define MAX_LENGTH_BIAS_comb_cuda_v2 128
#define TEST_NUM 1
// A Parallel SpMV/CSR version Kernel
__global__ void aggregation_cuda_v0(float* inputfeature, float* outputfeature, 
									int* indexes, int* neighbours, 
									int feature_num, int node_num, int edge_num) {
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is feature dimension
	// y is node dimension 
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;

	if (index_x < feature_num && index_y < node_num){
		outputfeature[index_x * node_num+ index_y] = 0;
		for (int j = indexes[index_y]; j < indexes[index_y + 1]; ++j) {
			outputfeature[index_x * node_num + index_y] += 
					inputfeature[index_x * node_num + neighbours[j]];
		}
		outputfeature[index_x * node_num+ index_y] /= 
					(float)(indexes[index_y + 1] - indexes[index_y]);
	} 
}

// Load neighbour to shared memory
__global__ void aggregation_cuda_v1(float* inputfeature, float* outputfeature, int* indexes, int* neighbours, int feature_num, int node_num, int edge_num) {
	// x is feature dimension
	// y is node dimension 
	int block_feature_dim = int(ceil(feature_num /float(TILED_SIZE_agg_cuda_v1)));
	int bx = blockIdx.x; 
	int tx = threadIdx.x; 
	int index_x = (bx % block_feature_dim) * TILED_SIZE_agg_cuda_v1 + tx;	
	int index_y = bx / block_feature_dim;
	__shared__ int neighbours_shared [MAX_LENGTH_NEIGHBOUR_agg_cuda_v1];

	int neighbour_index_start = indexes[index_y];
	int neighbour_index_end = indexes[index_y + 1];
	int total_neighbours = neighbour_index_end - neighbour_index_start;

	// About 0.1% of time
	for (int i = 0; i < total_neighbours; i+= feature_num){
		int load_neighbour_index = i + tx;
		if( load_neighbour_index < total_neighbours){
			neighbours_shared[load_neighbour_index] = neighbours[load_neighbour_index + neighbour_index_start];
		}
	}
	// Main cost of time
	__syncthreads();
	if (index_x < feature_num && index_y < node_num){
		float val= 0.0f;
		for (int j = 0; j < total_neighbours; ++j) {
			val += inputfeature[index_x * node_num + neighbours_shared[j]];
		}	
		outputfeature[index_x * node_num + index_y] = val/(float)total_neighbours;
		// outputfeature[index_x * node_num + index_y] = ;
	}
}

__global__ void aggregation_cuda_v2(float* inputfeature, float* outputfeature, float* ELL_value, int* ELL_row, int feature_num, int node_num, int ELL_row_num) {
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is feature dimension
	// y is node dimension 
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;
	if (index_x < feature_num && index_y < node_num){
		float sum = 0.0f;
		for (int i = 0; i < ELL_row_num; ++i) {
			sum += inputfeature[index_x * node_num + ELL_row[i * node_num + index_y]] * ELL_value[i * node_num + index_y];
		}
		outputfeature[index_x * node_num+ index_y] = sum;
	} 
}

__global__ void aggregation_cuda_v3_ELL(float* inputfeature, float* outputfeature, float* ELL_value, int* ELL_row, int feature_num, int node_num, int ELL_row_num) {
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	// x is feature dimension
	// y is node dimension 
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;
	if (index_x < feature_num && index_y < node_num){
		float sum = 0.0f;
		for (int i = 0; i < ELL_row_num; ++i) {
			sum += inputfeature[index_x * node_num + ELL_row[i * node_num + index_y]] * ELL_value[i * node_num + index_y];
		}
		outputfeature[index_x * node_num+ index_y] = sum;
	} 
}

__global__ void aggregation_cuda_v3_COO(float* inputfeature, float* outputfeature, 
										int feature_num, int node_num,
										float* Hybrid_COO_value, int* Hybrid_COO_row, int* Hybrid_COO_col,
										int Hybrid_COO_length) {
	int bx = blockIdx.x; 
	int tx = threadIdx.x; 
	// x is feature dimension
	int index_x = bx * TILED_SIZE_agg_cuda_v3 + tx;	
	__shared__ int Hybrid_COO_col_shared [TILED_SIZE_agg_cuda_v3];
	__shared__ int Hybrid_COO_row_shared [TILED_SIZE_agg_cuda_v3];
	__shared__ float Hybrid_COO_value_shared [TILED_SIZE_agg_cuda_v3];

	if (tx < Hybrid_COO_length){
		Hybrid_COO_col_shared[tx] = Hybrid_COO_col[tx];
		Hybrid_COO_row_shared[tx] = Hybrid_COO_row[tx];
		Hybrid_COO_value_shared[tx] = Hybrid_COO_value[tx];
	}
	__syncthreads();

	if (index_x < feature_num){
		for (int i = 0; i < Hybrid_COO_length; ++i) {
			outputfeature[index_x * node_num+ Hybrid_COO_col_shared[i]] += inputfeature[index_x * node_num+ Hybrid_COO_row_shared[i]]  * Hybrid_COO_value_shared[i];
		}
	} 
}




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
	//
	__shared__ float bias[MAX_LENGTH_BIAS_comb_cuda_v2];
	// x is out feature num
	// y is node dimension
	int index_x = bx * TILED_SIZE + tx;	
	int index_y = by * TILED_SIZE + ty;

	__syncthreads();
	// initialize with biases
	if(index_x < out_feature_num_p){
		bias[index_x] = biases[index_x];
	}

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

bool verified_feature(float* feature_device, float** feature_host_true, int feature_num, int node_num, bool vis_true = false){
	float* feature_host = (float *) malloc (feature_num * node_num * sizeof(float));
	cudaMemcpy(feature_host, feature_device, feature_num * node_num * sizeof(float), cudaMemcpyDeviceToHost);
	for (int f = 0; f < feature_num; ++ f){
		for (int n = 0 ; n < node_num ; ++ n){
			if ( abs(feature_host_true[f][n] - feature_host[f * node_num + n]) < 1e-4 ){
				if(vis_true){
					printf("The %f for the %d feature and %d node, the true value is %f \n", 
						feature_host[f * node_num + n], f, n, feature_host_true[f][n]);
				}
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

void convertSparseToELLFormat(int* indexes, int* neighbours, 
							  int feature_num, int node_num, int edge_num, int ELL_format_row,
							  float* ELL_value, int* ELL_row){

	for(int i = 0; i < node_num ; ++i){
		int adj_node_num = indexes[i + 1] - indexes[i];
		for(int j = 0; j < adj_node_num; ++j){
			ELL_value[node_num * j + i] = float(1)/adj_node_num;
			ELL_row[node_num * j + i] = neighbours[indexes[i] + j];
		}
	}
}

void convertSparseToHybridFormat(int* indexes, int* neighbours, 
								 int feature_num, int node_num, int edge_num, 
							     float* Hybrid_ELL_value, int* Hybrid_ELL_row,
								 int Hybrid_ELL_row_num,
								 float* Hybrid_COO_value, int* Hybrid_COO_row, int* Hybrid_COO_col,
								 int Hybrid_COO_length){
	int index_COO = 0;
	for(int i = 0; i < node_num ; ++i){
		int adj_node_num = indexes[i + 1] - indexes[i];
		for(int j = 0; j < Hybrid_ELL_row_num; ++j){
			if (j < adj_node_num){
				Hybrid_ELL_value[node_num * j + i] = float(1)/adj_node_num;
				Hybrid_ELL_row[node_num * j + i] = neighbours[indexes[i] + j];
			}
		} 
		if (adj_node_num > Hybrid_ELL_row_num){
			for(int j = Hybrid_ELL_row_num; j < adj_node_num; ++j){
				Hybrid_COO_value[index_COO] = float(1)/adj_node_num;
				Hybrid_COO_row[index_COO] = neighbours[indexes[i] + j];
				Hybrid_COO_col[index_COO] = i;
				index_COO += 1;
			}
		}
	}

}

class time_calculator{
	public:
		int time[TEST_NUM];
	float CalculateMean(){
        float sum = 0;
        for(int i = 0; i < TEST_NUM; i++)
            sum += (float)time[i];
        return (sum / TEST_NUM);
    }
	float CalculateVar(){
		float mean = CalculateMean();
        float temp = 0;
        for(int i = 0; i < TEST_NUM; i++){
             temp += (float)(time[i] - mean) * (time[i] - mean) ;
        }
        return sqrt(temp/TEST_NUM);
    }
};

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
	// CPU version
	feature_t feature_c;
	time_calculator time_CPU_agg;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();
		feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
		auto done = std::chrono::high_resolution_clock::now();
		time_CPU_agg.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}
	
	// CUDA version
	dim3 gridDim(int(ceil(GCN_c.feature_c.feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	dim3 blockDim(TILED_SIZE, TILED_SIZE);
	
	time_calculator time_GPU_agg_v0;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();
		aggregation_cuda_v0<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_agg1_device, 
													indexes_deivce, neighbours_device, 
													GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		
		time_GPU_agg_v0.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}


	std::cout << "The GPU version v0 of aggregation result is " << 
			  verified_feature(outputfeatures_agg1_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;
	printf("Time cost for GPU v0 of fisrt aggregation is %f nanoseconds with var %f, which is %f tims faster than the CPU version. \n\n", time_GPU_agg_v0.CalculateMean(), time_GPU_agg_v0.CalculateVar(), time_CPU_agg.CalculateMean()/time_GPU_agg_v0.CalculateMean());
	
	gridDim = dim3(int(ceil(GCN_c.feature_c.feature_num /float(TILED_SIZE_agg_cuda_v1))) * GCN_c.feature_c.node_num );
  	blockDim = dim3(TILED_SIZE_agg_cuda_v1);
	// printf("The gridDim is : %d, the blockDim is : %d \n\n\n\n.", gridDim.x, blockDim.x);

	time_calculator time_GPU_agg_v1;
	for(int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();  
		aggregation_cuda_v1<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_agg1_device, 
													indexes_deivce, neighbours_device, 
													GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		time_GPU_agg_v1.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}

	std::cout << "The GPU version v1 of aggregation result is " << 
			  verified_feature(outputfeatures_agg1_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;
	printf("Time cost for GPU v1 of fisrt aggregation is %f nanoseconds, the varance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_agg_v1.CalculateMean(), time_GPU_agg_v1.CalculateVar(), time_CPU_agg.CalculateMean()/time_GPU_agg_v1.CalculateMean());
	
	
	int ELL_row_num = 0;
	for(int i = 0; i < GCN_c.feature_c.node_num ; ++i){
		if(ELL_row_num < GCN_c.graph_c.indexes[i + 1] - GCN_c.graph_c.indexes[i]){
			ELL_row_num = GCN_c.graph_c.indexes[i + 1] - GCN_c.graph_c.indexes[i];
		}
	}
	float* ELL_value; 
	int* ELL_row ;
	ELL_value = (float*) malloc(GCN_c.feature_c.node_num * ELL_row_num * sizeof(float));
	ELL_row = (int*) malloc(GCN_c.feature_c.node_num * ELL_row_num * sizeof(int));
	convertSparseToELLFormat(GCN_c.graph_c.indexes, GCN_c.graph_c.neighbours, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges, ELL_row_num,
							 ELL_value, ELL_row);
	float* ELL_value_device; 
	int* ELL_row_device;
	cudaMalloc((float**)&ELL_value_device, sizeof(float) * ELL_row_num * GCN_c.feature_c.node_num);
	cudaMalloc((int**)&ELL_row_device, sizeof(int) * ELL_row_num * GCN_c.feature_c.node_num);
	cudaMemcpy(ELL_value_device, ELL_value, sizeof(float) * ELL_row_num * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	cudaMemcpy(ELL_row_device, ELL_row, sizeof(int) * ELL_row_num * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	gridDim = dim3(int(ceil(GCN_c.feature_c.feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	blockDim = dim3(TILED_SIZE, TILED_SIZE);
	time_calculator time_GPU_agg_v2;
	for (int i = 0; i < TEST_NUM; ++ i){
		auto started = std::chrono::high_resolution_clock::now();
		aggregation_cuda_v2<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_agg1_device, 
												ELL_value_device, ELL_row_device, 
												GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, ELL_row_num);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		
		time_GPU_agg_v2.time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}


	std::cout << "The GPU version v2 of aggregation result is " << 
			  verified_feature(outputfeatures_agg1_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;
	printf("Time cost for GPU v2 of fisrt aggregation is %f nanoseconds, the variance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_agg_v2.CalculateMean(), time_GPU_agg_v2.CalculateVar(), time_CPU_agg.CalculateMean()/time_GPU_agg_v2.CalculateMean());
	


	int Hybrid_ELL_row_num = MAX_RAW_agg_cuda_v3;
	int Hybrid_COO_length = 0;
	for(int i = 0; i < GCN_c.feature_c.node_num ; ++i){
		if(Hybrid_ELL_row_num < GCN_c.graph_c.indexes[i + 1] - GCN_c.graph_c.indexes[i]){
			Hybrid_COO_length += GCN_c.graph_c.indexes[i + 1] - GCN_c.graph_c.indexes[i] - Hybrid_ELL_row_num;
		}
	}

	printf("For the GPU version v3 of aggregation (Hybrid version),  the ELL format has the row num %d, the COO format has the length %d.\n", 
			Hybrid_ELL_row_num, Hybrid_COO_length);
	float* Hybrid_ELL_value; // Hybrid_ELL_row_num X node_num
	int* Hybrid_ELL_row ;
	Hybrid_ELL_value = (float*) malloc(GCN_c.feature_c.node_num * Hybrid_ELL_row_num * sizeof(float));
	Hybrid_ELL_row = (int*) malloc(GCN_c.feature_c.node_num * Hybrid_ELL_row_num * sizeof(int));
	
	float* Hybrid_COO_value; // Hybrid_COO_length
	int* Hybrid_COO_row ;	
	int* Hybrid_COO_col ;	
	Hybrid_COO_value = (float*) malloc(Hybrid_COO_length * sizeof(float));
	Hybrid_COO_row = (int*) malloc(Hybrid_COO_length * sizeof(int));	
	Hybrid_COO_col = (int*) malloc(Hybrid_COO_length * sizeof(int));
	convertSparseToHybridFormat(GCN_c.graph_c.indexes, GCN_c.graph_c.neighbours, 
								GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges, 
								Hybrid_ELL_value, Hybrid_ELL_row,
								Hybrid_ELL_row_num,
								Hybrid_COO_value, Hybrid_COO_row, Hybrid_COO_col,
								Hybrid_COO_length);

	float* Hybrid_ELL_value_device; 
	float* Hybrid_COO_value_device;
	int* Hybrid_ELL_row_device;
	int* Hybrid_COO_row_device;
	int* Hybrid_COO_col_device;
	cudaMalloc((float**)&Hybrid_ELL_value_device, sizeof(float) * Hybrid_ELL_row_num * GCN_c.feature_c.node_num);
	cudaMalloc((int**)&Hybrid_ELL_row_device, sizeof(int) * Hybrid_ELL_row_num * GCN_c.feature_c.node_num);

	cudaMalloc((float**)&Hybrid_COO_value_device, sizeof(float) * Hybrid_COO_length);
	cudaMalloc((int**)&Hybrid_COO_row_device, sizeof(int) * Hybrid_COO_length);
	cudaMalloc((int**)&Hybrid_COO_col_device, sizeof(int) * Hybrid_COO_length);
	cudaMemcpy(Hybrid_ELL_value_device, Hybrid_ELL_value, sizeof(float) * Hybrid_ELL_row_num * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	cudaMemcpy(Hybrid_ELL_row_device, Hybrid_ELL_row, sizeof(int) * Hybrid_ELL_row_num * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	cudaMemcpy(Hybrid_COO_value_device, Hybrid_COO_value, sizeof(float) * Hybrid_COO_length, cudaMemcpyHostToDevice);
	cudaMemcpy(Hybrid_COO_row_device, Hybrid_COO_row, sizeof(int) * Hybrid_COO_length, cudaMemcpyHostToDevice);
	cudaMemcpy(Hybrid_COO_col_device, Hybrid_COO_col, sizeof(int) * Hybrid_COO_length, cudaMemcpyHostToDevice);
	gridDim = dim3(int(ceil(GCN_c.feature_c.feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	blockDim = dim3(TILED_SIZE, TILED_SIZE);
	
	dim3 gridDim_COO(int(ceil(GCN_c.feature_c.feature_num/float(TILED_SIZE_agg_cuda_v3))));
  	dim3 blockDim_COO(TILED_SIZE_agg_cuda_v3);
	time_calculator time_GPU_agg_v3;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();
		aggregation_cuda_v3_ELL<<<gridDim, blockDim>>>(inputfeatures_device, outputfeatures_agg1_device, 
												Hybrid_ELL_value_device, Hybrid_ELL_row_device, 
												GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, Hybrid_ELL_row_num);
		
		cudaDeviceSynchronize();
		aggregation_cuda_v3_COO<<<gridDim_COO, blockDim_COO>>>(inputfeatures_device, outputfeatures_agg1_device, 
																GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num, 
																Hybrid_COO_value_device, Hybrid_COO_row_device, Hybrid_COO_col_device,
																Hybrid_COO_length);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		time_GPU_agg_v3.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}

	
	std::cout << "The GPU version v3 of aggregation result is " << 
			  verified_feature(outputfeatures_agg1_device, feature_c.features, GCN_c.feature_c.feature_num, GCN_c.feature_c.node_num) 
			  << std::endl;
	printf("Time cost for GPU v3 of fisrt aggregation is %f nanoseconds, the variance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_agg_v3.CalculateMean(), time_GPU_agg_v3.CalculateVar(), time_CPU_agg.CalculateMean()/time_GPU_agg_v3.CalculateMean());
	
	
	
	/////////////////////////// Test first combination //////////////////////////////////////

	// CPU version
	time_calculator time_CPU_comb;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();
		combination(feature_c, GCN_c.l1_parameter_c, true);
		auto done = std::chrono::high_resolution_clock::now();
		time_CPU_comb.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}
	feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);

	// CUDA version	
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

	gridDim = dim3(int(ceil(GCN_c.l1_parameter_c.out_feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	blockDim = dim3(TILED_SIZE, TILED_SIZE);
	time_calculator time_GPU_comb_v0;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();

		combination_v0<<<gridDim, blockDim>>>(outputfeatures_agg1_device, 
											GCN_c.l1_parameter_c.in_feature_num, GCN_c.feature_c.node_num, //feature_t in_feature
											outputfeatures_comb1_device, //feature_t out_feature
											biases_comb1_device, weights_comb1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.l1_parameter_c.out_feature_num, //parameter_t
											true);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		time_GPU_comb_v0.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}


	std::cout << "The GPU version v0 of combination result is " << 
			verified_feature(outputfeatures_comb1_device, feature_c.features, GCN_c.l1_parameter_c.out_feature_num, GCN_c.feature_c.node_num) 
			<< std::endl;	
	printf("Time cost for GPU v0 of fisrt combination is %f nanoseconds, the variance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_comb_v0.CalculateMean(), time_GPU_comb_v0.CalculateVar(), float(time_CPU_comb.CalculateMean())/time_GPU_comb_v0.CalculateMean());

	time_calculator time_GPU_comb_v1;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();

		combination_v1<<<gridDim, blockDim>>>(outputfeatures_agg1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.feature_c.node_num, //feature_t in_feature
											outputfeatures_comb1_device, //feature_t out_feature
											biases_comb1_device, weights_comb1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.l1_parameter_c.out_feature_num, //parameter_t
											true);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		time_GPU_comb_v1.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();	
	}	

	
	std::cout << "The GPU version v1 of combination result is " << 
			verified_feature(outputfeatures_comb1_device, feature_c.features, GCN_c.l1_parameter_c.out_feature_num, GCN_c.feature_c.node_num) 
			<< std::endl;	
	printf("Time cost for GPU v1 of fisrt combination is %f nanoseconds, the variance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_comb_v1.CalculateMean(), time_GPU_comb_v1.CalculateVar(), float(time_CPU_comb.CalculateMean())/time_GPU_comb_v1.CalculateMean());

	time_calculator time_GPU_comb_v2;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();

		combination_v2<<<gridDim, blockDim>>>(outputfeatures_agg1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.feature_c.node_num, //feature_t in_feature
											outputfeatures_comb1_device, //feature_t out_feature
											biases_comb1_device, weights_comb1_device, GCN_c.l1_parameter_c.in_feature_num, GCN_c.l1_parameter_c.out_feature_num, //parameter_t
											true);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		time_GPU_comb_v2.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();	
	}	

	
	std::cout << "The GPU version v2 of combination result is " << 
			verified_feature(outputfeatures_comb1_device, feature_c.features, GCN_c.l1_parameter_c.out_feature_num, GCN_c.feature_c.node_num) 
			<< std::endl;	
	printf("Time cost for GPU v2 of fisrt combination is %f nanoseconds, the variance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_comb_v2.CalculateMean(), time_GPU_comb_v2.CalculateVar(), float(time_CPU_comb.CalculateMean())/time_GPU_comb_v2.CalculateMean());



	////// second aggregation part ////

	feature_c = aggregation(GCN_c.graph_c, feature_c);
	float* outputfeatures2_agg1_device;
	cudaMalloc((float**)&outputfeatures2_agg1_device, sizeof(float) * GCN_c.l1_parameter_c.out_feature_num * GCN_c.feature_c.node_num);
	gridDim = dim3(int(ceil(GCN_c.l1_parameter_c.out_feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	blockDim = dim3(TILED_SIZE, TILED_SIZE);	

	aggregation_cuda_v0<<<gridDim, blockDim>>>(outputfeatures_comb1_device, outputfeatures2_agg1_device, 
												indexes_deivce, neighbours_device, 
												GCN_c.l1_parameter_c.out_feature_num, GCN_c.feature_c.node_num, GCN_c.spec_c.edges);
	std::cout << "The GPU version v0 of second aggregation result is " << 
			verified_feature(outputfeatures2_agg1_device, feature_c.features, GCN_c.l2_parameter_c.out_feature_num, GCN_c.feature_c.node_num) 
			<< std::endl;		
	
	////// second combination part ////

	float* outputfeatures2_comb_device;
	float* biases2_comb_device;
	float* weights2_comb_device;
	cudaMalloc((float**)&outputfeatures2_comb_device, sizeof(float) * GCN_c.l2_parameter_c.out_feature_num * GCN_c.feature_c.node_num);
	cudaMalloc((float**)&biases2_comb_device, sizeof(float) * GCN_c.l2_parameter_c.out_feature_num);
	cudaMalloc((float**)&weights2_comb_device, sizeof(float) * GCN_c.l2_parameter_c.out_feature_num * GCN_c.l2_parameter_c.in_feature_num);
	float* weights_comb2 = (float*) malloc(GCN_c.l2_parameter_c.out_feature_num * GCN_c.l2_parameter_c.in_feature_num * sizeof(float));

	convert2DarrayTo1Darray(GCN_c.l2_parameter_c.weights, weights_comb2, GCN_c.l2_parameter_c.in_feature_num, GCN_c.l2_parameter_c.out_feature_num);
	cudaMemcpy(biases2_comb_device, GCN_c.l2_parameter_c.biasses, sizeof(float) * GCN_c.l2_parameter_c.out_feature_num, cudaMemcpyHostToDevice);
	cudaMemcpy(weights2_comb_device, weights_comb2, sizeof(float) * GCN_c.l2_parameter_c.out_feature_num * GCN_c.l2_parameter_c.in_feature_num, cudaMemcpyHostToDevice);
	gridDim = dim3(int(ceil(GCN_c.l2_parameter_c.out_feature_num/float(TILED_SIZE))),
				 int(ceil(GCN_c.feature_c.node_num/float(TILED_SIZE)))
				 );
  	blockDim = dim3(TILED_SIZE, TILED_SIZE);	

	feature_c = combination(feature_c, GCN_c.l2_parameter_c, false);

	combination_v2<<<gridDim, blockDim>>>(outputfeatures2_agg1_device, 
										GCN_c.l2_parameter_c.in_feature_num, GCN_c.feature_c.node_num, //feature_t in_feature
										outputfeatures2_comb_device, //feature_t out_feature
										biases2_comb_device, weights2_comb_device, GCN_c.l2_parameter_c.in_feature_num, GCN_c.l2_parameter_c.out_feature_num, //parameter_t
										false);
	std::cout << "The GPU version v0 of second combination result is " << 
			verified_feature(outputfeatures2_comb_device, feature_c.features, GCN_c.l2_parameter_c.out_feature_num, GCN_c.feature_c.node_num, false) 
			<< std::endl;		
	
	/////////////////////////// Test Analyzer //////////////////////////////////////
	int* correctness;
	int* label_device;
	cudaMalloc((int**)&correctness, sizeof(int) * GCN_c.feature_c.node_num);
	cudaMalloc((int**)&label_device, sizeof(int) * GCN_c.feature_c.node_num);
	cudaMemcpy(label_device, GCN_c.label_c, sizeof(int) * GCN_c.feature_c.node_num, cudaMemcpyHostToDevice);
	// CUDA version
	time_calculator time_GPU_ana_v0;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();
		analyzer_cuda_v0<<<gridDim, blockDim>>>(outputfeatures2_comb_device, 
												label_device, GCN_c.l2_parameter_c.out_feature_num, GCN_c.feature_c.node_num, 
												correctness);
		cudaDeviceSynchronize();
		auto done = std::chrono::high_resolution_clock::now();
		time_GPU_ana_v0.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}


	// CPU version
	time_calculator time_CPU_ana;
	for (int num = 0; num < TEST_NUM; num ++){
		auto started = std::chrono::high_resolution_clock::now();
		analyzer(feature_c, GCN_c.label_c);
		auto done = std::chrono::high_resolution_clock::now();
		time_CPU_ana.time[num] = std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count();
	}

	printf("Time cost for GPU v0 of fisrt analyzer is %f nanoseconds, the variance is %f nanoseconds, which is %f tims faster than the CPU version. \n\n", time_GPU_ana_v0.CalculateMean(), time_GPU_ana_v0.CalculateVar(), time_CPU_ana.CalculateMean()/float(time_GPU_ana_v0.CalculateMean()));
	return 0;
}