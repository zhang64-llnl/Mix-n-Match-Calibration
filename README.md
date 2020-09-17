# Mix-n-Match-Calibration

This repository contains code that accompanies the paper [Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning](https://arxiv.org/abs/2003.07329). Please see the paper for more details.

LLNL CP Number: CP02333 

## Citation

If you find this library useful please consider citing our paper:

    @inproceedings{zhang2020mix,
      author={Zhang, Jize and Kailkhura, Bhavya and Han, T},
      booktitle={International Conference on Machine Learning (ICML)},
      title = {Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning},
      year = {2020},
    }


## To use in a project

The file `demo_calibration.py` is a template to conduct calibration and evaluate their performance with various methods.

The file `util_calibration.py` contains the functions describing the proposed mix-n-match calibration methods.

The file `util_evaluation.py` contains the functions describing the proposed mix-n-match evaluation methods.
