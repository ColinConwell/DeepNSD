# DeepNSD
 
 This repository contains code for the ongoing DeepNSD project, an attempt to characterize the representational structure of human visual cortex with the massive [NSD](http://naturalscenesdataset.org/) fMRI dataset and a bountiful cornucopia of deep neural network models.
 
 Our Google Colab tutorial ([bit.ly/Deep-NSD-Tutorial](https://bit.ly/Deep-NSD-Tutorial)) provides a step by step demonstration of the main functions in this pipeline, fitting the representations of a [CLIP](https://github.com/openai/CLIP) model to a subset of the fMRI data using the [DeepDive](https://github.com/ColinConwell/DeepDive) package.
 
 You can use this codebase to quickly load (in a unified API) a number of models and their associated transforms. (Please note that -- pending further development -- you will have to install the underlying model packages manually, as they often require machine-specific settings during installation.) 
 
 Models we've preprocessed include:
 
 - the [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models) library
- the [Torchvision](https://pytorch.org/vision/stable/models.html) model zoo
- the [Taskonomy](http://taskonomy.stanford.edu/) project
- the [VISSL](https://vissl.ai/) (SSL) model zoo
- ISL's [MiDas](https://github.com/isl-org/MiDaS) models zoo
- FaceBook's [DINO](https://github.com/facebookresearch/dino) models...
 
 A [manuscript](https://www.biorxiv.org/content/10.1101/2022.03.28.485868v1.abstract) that details results obtained using this pipeline may be found at the reference below.
 
 ```bibtex
@article{conwell2022pressures,
  title={What can 1.8 billion regressions tell us about the pressures shaping high-level visual representation in brains and machines?},
  author={Conwell, Colin and Prince, Jacob S and Kay, Kendrick N and Alvarez, George A and Konkle, Talia},
  journal={BioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
