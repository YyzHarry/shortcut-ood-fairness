# The Limits of Fair Medical Imaging AI In The Wild

### Downloading Data

To download all the datasets used in this study, please follow instructions in [DataSources.md](./DataSources.md).

As the original image files are often high resolution, we cache the images as downsampled copies to speed things up for some datasets. You should run 
```
python -m scripts.cache_cxr --data_path <data_path> --dataset <dataset>
``` 
where datasets can be `mimic`, `vindr`, `siim`, `isic`, or `odir`. This process is required for `vindr` and `siim`, and is optional for the remaining datasets.


### Training a Single Model

use `train.py`

### Training and Evaluating a Grid of Models

1. use `sweep_grid.py` to run `train.py` with experiments in `experiments.py` to train models
2. use `compute_optimal_thres.ipynb` to generate optimal thresholds based on F1 score maximization. We provide the thresholds we used (the output of this notebook) in `opt_thres.pkl`.
3. use `sweep_grid.py` to run `eval.py` with `experiments.py` to evaluate trained models at the best thresholds. Be sure to use the `--no_output_dir` argument when calling `sweep_grid.py`

