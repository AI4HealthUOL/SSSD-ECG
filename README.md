# Diffusion-based Conditional ECG Generation with Structured State Space Models

This is the official repository for the paper [Diffusion-based Conditional ECG Generation with Structured State Space Models](https://arxiv.org/abs/2301.08227) <ins>accepted by Computers in Biology and Medicine</ins>. We propose diverse algorithms (primarly SSSD-ECG) for the generation of 12-lead ECG signals conditioned on disease labels.


<a href="https://figshare.com/s/43df16e4a50e4dd0a0c5" alt="Dataset: https://figshare.com/s/43df16e4a50e4dd0a0c5">
  <img src="https://img.shields.io/badge/Dataset-10.6084%2Fm9.figshare.21922947-red" /></a>
<a href="https://figshare.com/s/81834b24a4711c2a5c55" alt="Model: https://figshare.com/s/81834b24a4711c2a5c55">
  <img src="https://img.shields.io/badge/Model-10.6084%2Fm9.figshare.21922875-red" /></a>
<a href="https://zenodo.org/account/settings/github/repository/AI4HealthUOL/SSSD-ECG" alt="Code: https://zenodo.org/account/settings/github/repository/AI4HealthUOL/SSSD-ECG"> <img src="https://img.shields.io/badge/Code-10.5281%2Fzenodo.7551714-blue" /></a>
  
 

![alt text](https://github.com/AI4HealthUOL/SSSD-ECG/blob/main/clinical%20evaluation/diagnosis%20on%20normal%20samples/plots/reports/SSSD.png?style=centerme)


### Please cite our publication if you found our research to be helpful.

```bibtex
@article{ALCARAZ2023107115,
title = {Diffusion-based conditional ECG generation with structured state space models},
journal = {Computers in Biology and Medicine},
volume = {163},
pages = {107115},
year = {2023},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2023.107115},
url = {https://www.sciencedirect.com/science/article/pii/S0010482523005802},
author = {Juan Miguel Lopez Alcaraz and Nils Strodthoff},
keywords = {Cardiology, Electrocardiography, Signal processing, Synthetic data, Diffusion models, Time series},
abstract = {Generating synthetic data is a promising solution for addressing privacy concerns that arise when distributing sensitive health data. In recent years, diffusion models have become the new standard for generating various types of data, while structured state space models have emerged as a powerful approach for capturing long-term dependencies in time series. Our proposed solution, SSSD-ECG, combines these two technologies to generate synthetic 12-lead electrocardiograms (ECGs) based on over 70 ECG statements. As reliable baselines are lacking, we also propose conditional variants of two state-of-the-art unconditional generative models. We conducted a thorough evaluation of the quality of the generated samples by assessing pre-trained classifiers on the generated data and by measuring the performance of a classifier trained only on synthetic data. SSSD-ECG outperformed its GAN-based competitors. Our approach was further validated through experiments that included conditional class interpolation and a clinical Turing test, which demonstrated the high quality of SSSD-ECG samples across a wide range of conditions.}
}

```
