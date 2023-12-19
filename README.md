# The Limits of Fair Medical Imaging AI In The Wild

[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/YyzHarry/shortcut-ood-fairness/blob/main/LICENSE)
![](https://img.shields.io/github/stars/YyzHarry/shortcut-ood-fairness)
![](https://img.shields.io/github/forks/YyzHarry/shortcut-ood-fairness)
![](https://visitor-badge.laobi.icu/badge?page_id=YyzHarry.shortcut-ood-fairness&right_color=%23FFA500)

[[Paper](https://arxiv.org/abs/2312.10083)]

**Summary**: As artificial intelligence (AI) rapidly approaches human-level performance in medical imaging, it is crucial that it does not exacerbate or propagate healthcare disparities. Prior research has established AI’s capacity to infer demographic data from chest X-rays, leading to a key concern: do models using demographic shortcuts have unfair predictions across subpopulations? In this study, we conduct a thorough investigation into the extent to which medical AI utilizes demographic encodings, focusing on potential fairness discrepancies within both in-distribution training sets and external test sets. Our analysis covers three key medical imaging disciplines: radiology, dermatology, and ophthalmology, and incorporates data from six global chest X-ray datasets. We confirm that medical imaging AI leverages demographic shortcuts in disease classification. While correcting shortcuts algorithmically effectively addresses fairness gaps to create "locally optimal" models within the original data distribution, this optimality is not true in new test settings. Surprisingly, we find that models with less encoding of demographic attributes are often most "globally optimal", exhibiting better fairness during model evaluation in new test environments. Our work establishes best practices for medical imaging models which maintain their performance and fairness in deployments beyond their initial training contexts, underscoring critical considerations for AI clinical deployments across populations and sites.

### Downloading Data

To download all the datasets used in this study, please follow instructions in [DataSources.md](./DataSources.md).

As the original image files are often high resolution, we cache the images as downsampled copies to speed training up for certain datasets. To do so, run
```bash
python -m scripts.cache_cxr --data_path <data_path> --dataset <dataset>
``` 
where datasets can be `mimic`, `vindr`, `siim`, `isic`, or `odir`. This process is required for `vindr` and `siim`, and is optional for the remaining datasets.

### Training and Evaluating a Single Model

```bash
python -m train \
       --algorithm <algo> \
       --dataset <dset> \
       --task <task> \
       --attr <attr> \
       --data_dir <data_path> \
       --output_dir <output_path>
```

### Training and Evaluating a Grid of Models

1. Use `sweep_grid.py` to run `train.py` with experiments in `experiments.py` to train models
    ```bash
    python -m sweep_grid launch \
           --experiment <exp_train> \
           ...
    ```

2. Use [`compute_optimal_thres.ipynb`](./notebooks/compute_optimal_thres.ipynb) to generate optimal thresholds based on F1-score maximization. We provide the thresholds used in the study (the output of this notebook) in [`opt_thres.pkl`](./notebooks/opt_thres.pkl).
3. Use `sweep_grid.py` to run `eval.py` with `experiments.py` to evaluate trained models at the best thresholds. Be sure to use the `--no_output_dir` argument when calling `sweep_grid.py`
    ```bash
    python -m sweep_grid launch \
           --experiment <exp_eval> \
           --no_output_dir \
           ...
    ```

### Acknowledgements
This code is partly based on the open-source implementations from [SubpopBench](https://github.com/YyzHarry/SubpopBench).

### Citation
If you find this code or idea useful, please cite our work:

```bibtex
@article{yang2023limits,
  title={The Limits of Fair Medical Imaging AI In The Wild},
  author={Yuzhe Yang and Haoran Zhang and Judy W Gichoya and Dina Katabi and Marzyeh Ghassemi},
  journal={arXiv preprint arXiv:2312.10083},
  year={2023}
}
```

### Contact
If you have any questions, feel free to contact us through email (yuzhe@mit.edu & haoranz@mit.edu) or GitHub issues. Enjoy!
