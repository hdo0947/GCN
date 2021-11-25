#include "header.h"

int main(int argc, char const *argv[]) {
	if ((argc != 2) || ((strcmp(argv[1], "cora") != 0) && (strcmp(argv[1], "citeseer") != 0) && (strcmp(argv[1], "reddit") != 0))) {
		printf("ERROR: usage \"%s [cora|citeseer|reddit]\"\n", argv[0]);
		return -1;
	}
	GCN_t GCN_c = GCN_parser((char*)argv[1]);
	feature_t feature_c;
	feature_c = aggregation(GCN_c.graph_c, GCN_c.feature_c);
	feature_c = combination(feature_c, GCN_c.l1_parameter_c, true);
	feature_c = aggregation(GCN_c.graph_c, feature_c);
	feature_c = combination(feature_c, GCN_c.l2_parameter_c, false);
	analyzer(feature_c, GCN_c.label_c);
	return 0;
}