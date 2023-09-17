# CALIME
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://pypi.org/project/biasondemand) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`CALIME` is a GitHub repository containing the **CALIME** algorithm. It refers to the paper titled *Causality-Aware Local Interpretable Model-Agnostic Explanations*.

Check out the pdf here: [[pdf](https://arxiv.org/pdf/2212.05256.pdf)]

## Abstract

A significant drawback of eXplainable Artificial Intelligence (XAI) approaches is the assumption of feature independence. This paper focuses on integrating causal knowledge in XAI methods to increase trust and help users assess explanations’ quality. We propose a novel exten- sion to a widely used local and model-agnostic explainer that explicitly encodes causal relationships in the data generated around the input in- stance to explain. Extensive experiments show that our method achieves superior performance comparing the initial one for both the fidelity in mimicking the black-box and the stability of the explanations.

## Setup

The packages requires a python version >=3.8, as well as some libraries listed in requirements file. For some additional functionalities, more libraries are needed for these extra functions and options to become available. 

```
git clone https://github.com/marti5ini/CALIME.git
cd CALIME
```

Dependencies are listed in requirements.txt, a virtual environment is advised:

```
python3 -m venv ./venv # optional but recommended
pip install -r requirements.txt
```
## Citation

If you use `CALIME` in your research, please cite our paper:

```
@@article{cinquini2022calime,
  title={CALIME: Causality-Aware Local Interpretable Model-Agnostic Explanations},
  author={Cinquini, Martina and Guidotti, Riccardo},
  journal={arXiv preprint arXiv:2212.05256},
  year={2022}
}
```
