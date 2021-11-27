#ifndef __GCN_H__
#define __GCN_H__

feature_t aggregation (graph_t graph_c, feature_t in_feature_c);

feature_t combination (feature_t in_feature_c, parameter_t parameter_c, bool relu);

void analyzer (feature_t feature_c, label_t label_c);

#endif