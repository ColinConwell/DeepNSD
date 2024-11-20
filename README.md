# DeepNSD: Controlled DNN Modeling of Human Visual Brain Data
 
 This repository contains code for the DeepNSD project, an attempt to characterize the representational structure of human visual cortex with the massive [NSD](http://naturalscenesdataset.org/) fMRI dataset and a bountiful cornucopia of deep neural network models. It also contains all code and access to data for reproducing the associated manuscript: ["What can 1.8 billion regressions tell us about the pressures shaping high-level visual representation in brains and machines?"](https://www.biorxiv.org/content/10.1101/2022.03.28.485868v1.abstract), currently in-press at Nature Communications.
 
 Our Google Colab tutorial ([bit.ly/Deep-NSD-Tutorial](https://bit.ly/Deep-NSD-Tutorial)) provides a step by step demonstration of the main functions in this pipeline, fitting the representations of a [CLIP](https://github.com/openai/CLIP) model to a single subject subset of the fMRI data using the [DeepDive](https://github.com/ColinConwell/DeepDive) package (soon to be re-released as [DeepJuice](https://deepjuice.io/)).
 
 You can use this codebase to quickly load (in a unified API) a number of models and their associated transforms. (Please note that -- pending further development -- you will have to install the underlying model packages manually, as they often require machine-specific settings during installation.) 
 
 Models we've preprocessed include:
 
 - the [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models) library
- the [Torchvision](https://pytorch.org/vision/stable/models.html) model zoo
- the [Taskonomy](http://taskonomy.stanford.edu/) project
- the [VISSL](https://vissl.ai/) (SSL) model zoo
- ISL's [MiDas](https://github.com/isl-org/MiDaS) models zoo
- FaceBook's [DINO](https://github.com/facebookresearch/dino) models...

(Note, the [neural_data](./neural_data) is included as a legacy folder, to preserve compatibility with older versions of the Colaboratory tutorial.)
 
To cite this repository, or the associated [publication](https://www.nature.com/articles/s41467-024-53147-y), please use the following BibTex:
 
 ```bibtex
@article{conwell2024large,
 title={A large-scale examination of inductive biases shaping high-level visual representation in brains and machines},
 author={Conwell, Colin and Prince, Jacob S and Kay, Kendrick N and Alvarez, George A and Konkle, Talia},
 journal={Nature Communications},
 volume={15},
 number={1},
 pages={9383},
 year={2024},
 publisher={Nature Publishing Group UK London} 
}
```

## 2024 Update: *DeepDive* to *DeepJuice*

+ **Squeezing your deep nets for science!**

Recently, our team has been working on a new, highly-accelerated version of this codebase called **Deepjuice** -- effectively, a bottom-up reimplementation of all DeepDive functionalities that allows for end-to-end benchmarking (feature extraction, SRP, PCA, CKA, RSA, and regression) without ever removing data from the GPU. 

**DeepJuice** is currently in private beta, but if you're interested in trying out, please feel free to contact me (Colin Conwell) by email: conwell[at]g[dot]harvard[dot]edu
