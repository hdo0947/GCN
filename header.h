#ifndef __HEADER_H__
#define __HEADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <iostream>
#include <math.h>
#include <sys/types.h>
#include <ctime>
#include <chrono>
#include <math.h> 

#define DEBUG false
#define DATASET_DIR "./GCN_data/"
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) < (Y)) ? (Y) : (X))

typedef struct spec_t {
	int nodes;
	int edges;
	int features;
	int hidden;
	int labels;
} spec_t;

typedef struct feature_t {
	float** features;
	int feature_num;
	int node_num;
} feature_t;

typedef int* label_t;

typedef struct graph_t {
	int* indexes;
	int* neighbours;
} graph_t;

typedef struct parameter_t {
	float* biasses;
	float** weights;
	int in_feature_num;
	int out_feature_num;
} parameter_t;

typedef struct GCN_t {
	spec_t spec_c;
	feature_t feature_c;
	label_t label_c;
	graph_t graph_c;
	parameter_t l1_parameter_c;
	parameter_t l2_parameter_c;
} GCN_t;

#include "utilities.h"
#include "kernels.h"

#endif
