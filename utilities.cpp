#include "header.h"

spec_t spec_parser(char* spec_f) {
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    spec_t spec_c;

    fp = fopen(spec_f, "r");
    if (fp == NULL){
        printf("ERROR: No such file (%s)\n", spec_f);
	    exit(-1);
    }

    if(getline(&line, &len, fp) != -1) {
    	line[strcspn(line, "\n")] = 0;
    	spec_c.nodes = atoi(line);
    } else {
    	printf("ERROR: No number of nodes in spec!\n");
    	exit(-1);
    }

    if(getline(&line, &len, fp) != -1) {
    	line[strcspn(line, "\n")] = 0;
    	spec_c.edges = atoi(line);
    } else {
    	printf("ERROR: No number of edges in spec!\n");
    	exit(-1);
    }

    if(getline(&line, &len, fp) != -1) {
    	line[strcspn(line, "\n")] = 0;
    	spec_c.features = atoi(line);
    } else {
    	printf("ERROR: No number of features in spec!\n");
    	exit(-1);
    }

    if(getline(&line, &len, fp) != -1) {
    	line[strcspn(line, "\n")] = 0;
    	spec_c.hidden = atoi(line);
    } else {
    	printf("ERROR: No number of hidden layers in spec!\n");
    	exit(-1);
    }

    if(getline(&line, &len, fp) != -1) {
    	line[strcspn(line, "\n")] = 0;
    	spec_c.labels = atoi(line);
    } else {
    	printf("ERROR: No number of labels in spec!\n");
    	exit(-1);
    }

    fclose(fp);
    if (line)
        free(line);
    
    return spec_c;
}

feature_t feature_parser(spec_t spec_c, char* feature_f) {
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
	feature_t feature_c;
	int node_idx = 0;
	int feature_idx = 0;
	int i;

	feature_c.feature_num = spec_c.features;
	feature_c.node_num = spec_c.nodes;
	feature_c.features = (float**) malloc ((spec_c.features) * sizeof(float*));
	for (i = 0; i < spec_c.features; ++i) {
		feature_c.features[i] = (float*) malloc ((spec_c.nodes) * sizeof(float)); 
	}

	fp = fopen(feature_f, "r");
    if (fp == NULL) {
        printf("ERROR: No such file (%s)\n", feature_f);
        exit(-1);
    }

    while ((read = getline(&line, &len, fp)) != -1) {
		char * feature = strtok(line, " \t\n");
		feature_idx = 0;
		while(feature != NULL) {
			feature_c.features[feature_idx][node_idx] = atof(feature);
			feature = strtok(NULL, " \t\n");
			feature_idx++;
		}
		if (feature_idx != spec_c.features) {
	    	printf("ERROR: Incompatible number of features in spec (%d) and feature (%d) files!\n", spec_c.features, feature_idx);
	    	exit(-1);
	    }
    	node_idx++;
    }

    if (node_idx != spec_c.nodes) {
    	printf("ERROR: Incompatible number of nodes in spec (%d) and feature (%d) files!\n", spec_c.nodes, node_idx);
    	exit(-1);
    }

    fclose(fp);
    if (line)
        free(line);
    
    return feature_c;
}

label_t label_parser(spec_t spec_c, char* label_f) {
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
	label_t label_c;
	int node_idx = 0;

	label_c = (int*) malloc ((spec_c.nodes) * sizeof(int));

	fp = fopen(label_f, "r");
    if (fp == NULL) {
        printf("ERROR: No such file (%s)\n", label_f);
        exit(-1);
    }

    while ((read = getline(&line, &len, fp)) != -1) {
		char * label = strtok(line, " \t\n");
		while(label != NULL) {
			label_c[node_idx] = atoi(label);
			label = strtok(NULL, " \t\n");
		}
    	node_idx++;
    }

    if (node_idx != spec_c.nodes) {
    	printf("ERROR: Incompatible number of nodes in spec (%d) and label (%d) files!\n", spec_c.nodes, node_idx);
    	exit(-1);
    }

    fclose(fp);
    if (line)
        free(line);
    
    return label_c;
}

graph_t graph_parser(spec_t spec_c, char* graph_f) {
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
	graph_t graph_c;
	int node_idx = 0;
	int edge_idx = 0;

	graph_c.indexes = (int*) malloc ((spec_c.nodes + 1) * sizeof(int));
	graph_c.neighbours = (int*) malloc (spec_c.edges * sizeof(int));

	fp = fopen(graph_f, "r");
    if (fp == NULL) {
        printf("ERROR: No such file (%s)\n", graph_f);
        exit(-1);
    }

    while ((read = getline(&line, &len, fp)) != -1) {
		char * neighbour = strtok(line, " \t\n");
		graph_c.indexes[node_idx] = edge_idx;
		while(neighbour != NULL) {
			graph_c.neighbours[edge_idx] = atoi(neighbour);
			neighbour = strtok(NULL, " \t\n");
			edge_idx++;
		}
    	node_idx++;
    }

    if (edge_idx != spec_c.edges) {
    	printf("ERROR: Incompatible number of edges in spec (%d) and graph (%d) files!\n", spec_c.edges, edge_idx);
    	exit(-1);
    }

    graph_c.indexes[node_idx] = edge_idx;

    fclose(fp);
    if (line)
        free(line);
    
    return graph_c;
}

parameter_t parameter_parser(int input_nodes, int output_nodes, char* weight_f, char* bias_f) {
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
	parameter_t parameter_c;
	int node_idx = 0;
	int weight_idx = 0;
	int i;

	parameter_c.biasses = (float*) malloc (output_nodes * sizeof(float));
	parameter_c.weights = (float**) malloc (input_nodes * sizeof(float*));
	for (i = 0; i < input_nodes; ++i) {
		parameter_c.weights[i] = (float*) malloc (output_nodes * sizeof(float));
	}
	parameter_c.in_feature_num = input_nodes;
	parameter_c.out_feature_num = output_nodes;

	fp = fopen(weight_f, "r");
    if (fp == NULL) {
        printf("ERROR: No such file (%s)\n", weight_f);
        exit(-1);
    }

    while ((read = getline(&line, &len, fp)) != -1) {
		char * weight = strtok(line, " \t\n");
		while(weight != NULL) {
			parameter_c.weights[weight_idx][node_idx] = atof(weight);
			weight = strtok(NULL, " \t\n");
			node_idx++;
		}
		if (node_idx != output_nodes) {
	    	printf("ERROR: Incompatible number of outputs in spec (%d) and weight (%d) files!\n", output_nodes, node_idx);
	    	exit(-1);
	    }
		node_idx = 0;
    	weight_idx++;
    }
    
	if (weight_idx != input_nodes) {
		printf("ERROR: Incompatible number of inputs in spec (%d) and weight (%d) files!\n", input_nodes, weight_idx);
    	exit(-1);
	}

    fclose(fp);

    fp = fopen(bias_f, "r");
    if (fp == NULL) {
        printf("ERROR: No such file (%s)\n", bias_f);
        exit(-1);
    }

    node_idx = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
    	line[strcspn(line, "\n")] = 0;
		parameter_c.biasses[node_idx] = atof(line);
    	node_idx++;
    }

    if (node_idx != output_nodes) {
    	printf("ERROR: Incompatible number of outputs in spec (%d) and bias (%d) files!\n", output_nodes, node_idx);
    	exit(-1);
    }

    fclose(fp);
    if (line)
        free(line);
    
    return parameter_c;
}

void print_spec (spec_t spec_c) {
	printf("# of nodes: %d, edges: %d, features: %d, hidden layers: %d, labels: %d\n", spec_c.nodes, spec_c.edges, spec_c.features, spec_c.hidden, spec_c.labels);
}

void print_features (feature_t feature_c) {
	int i, j;
	printf("features[%d][%d]: \n", feature_c.feature_num, feature_c.node_num);
	for (i = 0; i < MIN(4, feature_c.feature_num); ++i) {
		printf("\t");
		for (j = 0; j < MIN(4, feature_c.node_num); ++j) {
			printf("%.2f\t", feature_c.features[i][j]);
		}
		if (feature_c.node_num > 4)
			printf("...\t%.2f\n", feature_c.features[i][feature_c.node_num-1]);
		else
			printf("\n");
	}
	if (feature_c.feature_num > 4){
		printf("\t.\n\t.\n\t.\n\t");
		for (j = 0; j < MIN(4, feature_c.node_num); ++j) {
			printf("%.2f\t", feature_c.features[feature_c.feature_num-1][j]);
		}
		if (feature_c.node_num > 4)
			printf("...\t%.2f\n", feature_c.features[feature_c.feature_num-1][feature_c.node_num-1]);
		else
			printf("\n");
	}
}

void print_labels (spec_t spec_c, label_t label_c) {
	int i;
	printf("labels[%d]:\t", spec_c.nodes);
	for (i = 0; i < MIN(4, spec_c.nodes); ++i)
		printf("%d\t", label_c[i]);
	if (spec_c.nodes > 4)
		printf("...\t%d\n", label_c[spec_c.nodes-1]);
	else
		printf("\n");
}

void print_graph (spec_t spec_c, graph_t graph_c) {
	int i, j;
	printf("neighbours[%d]: \n", spec_c.nodes);
	for (i = 0; i < MIN(4, spec_c.nodes); ++i) {
		printf("[%d]:\t\t", i);
		for (j = graph_c.indexes[i]; j < MIN(graph_c.indexes[i] + 4, graph_c.indexes[i + 1]); ++j) {
			printf("%d\t", graph_c.neighbours[j]);
		}
		if (graph_c.indexes[i + 1] > graph_c.indexes[i] + 4)
			printf("...\t%d\n", graph_c.neighbours[graph_c.indexes[i + 1] - 1]);
		else
			printf("\n");
	}
	if (spec_c.nodes > 4){
		printf(".\n.\n.\n[%d]:\t\t", spec_c.nodes - 1);
		for (j = graph_c.indexes[spec_c.nodes - 1]; j < MIN(graph_c.indexes[spec_c.nodes - 1] + 4, graph_c.indexes[spec_c.nodes]); ++j) {
			printf("%d\t", graph_c.neighbours[j]);
		}
		if (graph_c.indexes[spec_c.nodes] > graph_c.indexes[spec_c.nodes - 1] + 4)
			printf("...\t%d\n", graph_c.neighbours[graph_c.indexes[spec_c.nodes] - 1]);
		else
			printf("\n");
	}
}

void print_parameter (parameter_t parameter_c) {
	int i, j;
	printf("biasses[%d]: \n\t", parameter_c.out_feature_num);
	for (i = 0; i < MIN(4, parameter_c.out_feature_num); ++i) {
		printf("%.2f\t", parameter_c.biasses[i]);
	}
	if (parameter_c.out_feature_num > 4)
		printf("...\t%.2f\n", parameter_c.biasses[parameter_c.out_feature_num-1]);
	else
		printf("\n");

	printf("weights[%d][%d]: \n", parameter_c.in_feature_num, parameter_c.out_feature_num);
	for (i = 0; i < MIN(4, parameter_c.in_feature_num); ++i) {
		printf("\t");
		for (j = 0; j < MIN(4, parameter_c.out_feature_num); ++j) {
			printf("%.2f\t", parameter_c.weights[i][j]);
		}
		if (parameter_c.out_feature_num > 4)
			printf("...\t%.2f\n", parameter_c.weights[i][parameter_c.out_feature_num-1]);
		else
			printf("\n");
	}
	if (parameter_c.in_feature_num > 4){
		printf("\t.\n\t.\n\t.\n\t");
		for (j = 0; j < MIN(4, parameter_c.out_feature_num); ++j) {
			printf("%.2f\t", parameter_c.weights[parameter_c.in_feature_num-1][j]);
		}
		if (parameter_c.out_feature_num > 4)
			printf("...\t%.2f\n", parameter_c.weights[parameter_c.in_feature_num-1][parameter_c.out_feature_num-1]);
		else
			printf("\n");
	}
}

char* str_concat(char* first, char* second, char* third) {
	char* str = (char*) malloc (256 * sizeof(char));
	strcpy(str, first);
	strcat(str, second);
	strcat(str, third);
	return str;
}

GCN_t GCN_parser (char* dataset) {
	GCN_t GCN_c;
	GCN_c.spec_c = spec_parser(str_concat(DATASET_DIR, dataset, "_ds/spec.txt"));
#if DEBUG
	print_spec(GCN_c.spec_c);
#endif
	GCN_c.feature_c = feature_parser(GCN_c.spec_c, str_concat(DATASET_DIR, dataset, "_ds/features.txt"));
#if DEBUG
	print_features(GCN_c.feature_c);
#endif
	GCN_c.label_c = label_parser(GCN_c.spec_c, str_concat(DATASET_DIR, dataset, "_ds/labels.txt"));
#if DEBUG
	print_labels(GCN_c.spec_c, GCN_c.label_c);
#endif
	GCN_c.graph_c = graph_parser(GCN_c.spec_c, str_concat(DATASET_DIR, dataset, "_ds/edges.txt"));
#if DEBUG
	print_graph(GCN_c.spec_c, GCN_c.graph_c);
#endif
	GCN_c.l1_parameter_c = parameter_parser(GCN_c.spec_c.features, GCN_c.spec_c.hidden, str_concat(DATASET_DIR, dataset, "_ds/l1_weight.txt"), str_concat(DATASET_DIR, dataset, "_ds/l1_bias.txt"));
#if DEBUG
	print_parameter(GCN_c.l1_parameter_c);
#endif
	GCN_c.l2_parameter_c = parameter_parser(GCN_c.spec_c.hidden, GCN_c.spec_c.labels, str_concat(DATASET_DIR, dataset, "_ds/l2_weight.txt"), str_concat(DATASET_DIR, dataset, "_ds/l2_bias.txt"));
#if DEBUG
	print_parameter(GCN_c.l2_parameter_c);
#endif
	return GCN_c;
}