# DESM (Decoupling and super-resolution microscopy)

Self-supervised single-channel multiplexed super-resolution microscopy

This repository provides the PyTorch implementation for our DESM framework, a self-supervised deep learning method for joint channel decoupling and super-resolution reconstruction from single-channel confocal microscopy data.

> Note: some scripts and file names in this repository still keep the historical `DSCM` naming. They correspond to the current DESM codebase.

---

## Overview

DESM is designed for multiplexed super-resolution microscopy from single-channel laser-scanning microscopy data.

The framework supports two modes:

- **DESM-IO**: intensity-only mode for standard confocal data
- **DESM-IL**: intensity-lifetime mode for FLIM-assisted reconstruction

The codebase includes:

- training on simulation / prepared datasets
- evaluation on synthetic and semi-synthetic data
- FLIM-assisted inference and evaluation
- visualization scripts for reconstructed results

---

## Paper

**Self-supervised single-channel multiplexed super-resolution microscopy**

Qinglin Chen<sup>1,†</sup>, Luwei Wang<sup>1,†</sup>, Min Yi<sup>1</sup>, Jia Li<sup>1</sup>, Dan Shao<sup>2</sup>, Xiaoyu Weng<sup>1</sup>, Liwei Liu<sup>1</sup>, Dayong Jin<sup>2,3,*</sup>, Junle Qu<sup>1,*</sup>

<sup>1</sup> State Key Laboratory of Radio Frequency Heterogeneous Integration (Shenzhen University) & Key Laboratory of Optoelectronic Devices and Systems, College of Physics and Optoelectronic Engineering, Shenzhen University, Shenzhen 518060, China

<sup>2</sup> Institute for Biomedical Materials and Devices (IBMD), Faculty of Science, University of Technology Sydney, NSW 2007, Australia

<sup>3</sup> Zhejiang Provincial Engineering Research Center for Organelles Diagnostics and Therapy, Eastern Institute of Technology, Ningbo 315200, China

† Equal contribution  
* Corresponding authors: dayong.jin@eitech.edu.cn, jlqu@szu.edu.cn

---

## Environment

The current codebase is tested with the following software environment:

- Python 3.8
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA-compatible GPU recommended

Install dependencies with:

```bash
pip install -r requirements.txt
