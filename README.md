## Notebook (example) 
Start from notebook (install requirements.txt with python 3.8 (3.8.19 tested)
note_fedxgboost.ipynb

## Federated data: create a federated dataset and distribute to clients 
```python
python -m classes.Datasets.data_generator -data $data -samples $samples -data $data -niid $n_iid -alpha 0.1
$data choices=['stroke'], default='stroke', help = 'Type of data',
$samples default=100, help="sets a fixed number samples per device"
$n_iid choices=['iid', 'sample'], default='iid', help="Heterogeneity type"
NOTE: stroke dataset can be shared upon request
```
## FedXGBoost with learnable learning rates 
> xgboost_binary_stroke_example: fedXGBoostllr with stroke data

## FedAvg benchmark
> fedavg_binary_stroke_example: benchmark FedAvg under same environment (stroke data)

## Note
add folder xgb_models before running
