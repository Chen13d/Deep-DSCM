Deep-DSCM

Multi-colour super-resolution imaging in single-channel confocal microscopy by deep learning

This repository provides the demo code for the paper:
Deep-DSCM: multi-colour super-resolution imaging in single-channel confocal microscopy by deep learning

📄 Citation

Authors: Qinglin Chen<sup>1,†</sup>, Luwei Wang<sup>1,†</sup>, Jia Li<sup>1</sup>, Dan Shao<sup>2</sup>, Xiaoyu Weng<sup>1</sup>, Liwei Liu<sup>1</sup>, Dayong Jin<sup>2,3,</sup>, Junle Qu<sup>1,</sup>

<sup>1</sup> State Key Laboratory of Radio Frequency Heterogeneous Integration (Shenzhen University) & Key Laboratory of Optoelectronic Devices and Systems, College of Physics and Optoelectronic Engineering, Shenzhen University, Shenzhen 518060, China

<sup>2</sup> Institute for Biomedical Materials and Devices (IBMD), Faculty of Science, University of Technology Sydney, NSW 2007, Australia

<sup>3</sup> Zhejiang Provincial Engineering Research Center for Organelles Diagnostics and Therapy, Eastern Institute of Technology, Ningbo 315200, China

† Equal contribution: Qinglin Chen, Luwei Wang
* Corresponding authors: dayong.jin@eitech.edu.cn, jlqu@szu.edu.cn

⚙️ Environment

Python 3.8.20

CUDA 12.0

PyTorch 2.0.1

📂 File Structure

DSCM-demo.ipynb → Main entry point for:

Validation on synthetic data

Testing on real data

🚀 Quick Start

Clone this repository:

git clone https://github.com/your-repo/Deep-DSCM.git
cd Deep-DSCM


Install dependencies:

pip install -r requirements.txt


Run the demo notebook:

jupyter notebook DSCM-demo.ipynb

📌 Notes

This demo includes both synthetic dataset validation and real microscopy data testing.

For training with your own dataset, please adapt the notebook accordingly.

📜 License

This project is for academic research only. Please contact the authors for commercial usage.

