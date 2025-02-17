#!/usr/bin/bash

for i in {3..20}
do
    python fedxgboost_multiclass_script.py -reshape 0 -inputs "binary" -num_clients $i
    python fedxgboost_multiclass_script.py -reshape 1 -inputs "binary" -num_clients $i
    python fedxgboost_multiclass_script.py -reshape 0 -inputs "soft" -num_clients $i
    python fedxgboost_multiclass_script.py -reshape 1 -inputs "soft" -num_clients $i
done

#matlab -nodisplay -nodesktop -r "run feature_extractor.m"