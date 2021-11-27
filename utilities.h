#ifndef __UTILITIES_H__
#define __UTILITIES_H__

spec_t spec_parser(char* spec_f);

feature_t feature_parser(spec_t spec_c, char* feature_f);

label_t label_parser(spec_t spec_c, char* label_f);

graph_t graph_parser(spec_t spec_c, char* graph_f);

parameter_t parameter_parser(int input_nodes, int output_nodes, char* weight_f, char* bias_f);

void print_spec (spec_t spec_c);

void print_features (feature_t feature_c);

void print_labels (spec_t spec_c, label_t label_c);

void print_graph (spec_t spec_c, graph_t graph_c);

void print_parameter (parameter_t parameter_c);

char* str_concat(char* first, char* second, char* third);

GCN_t GCN_parser (char* dataset);

#endif
