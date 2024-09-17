# CrossEM
**The dataset and code are for research purpose only**

Deep learning side-channel attacks can recover encryption keys on a target by analyzing power consumption or electromagnetic (EM) signals. However, they are less portable when there are domain shifts between training and test data. While existing studies have shown that pre-processing and unsupervised domain adaptation can enhance the portability of deep learning side-channel attacks given domain shifts over EM traces, the findings are limited to easy targets (e.g. 8-bit microcontrollers).

In this project, we investigate the portability of deep learning side-channel attacks over EM traces acquired from more challenging targets, including 32-bit microcontrollers and EM traces with random delay. We study domain shifts introduced by the combination of hardware variations, distinct keys, and inconsistent probe locations between two targets. In addition, we perform comparative analyses of multiple existing (and new) pre-processing and unsupervised domain adaptation methods. We conduct a series of comprehensive experiments and derive three main observations. (1) Pre-processing and unsupervised domain adaptation methods can enhance the portability of deep learning side-channel attacks over more challenging targets. (2) The effectiveness of each method, however, varies depending on the target and probe locations in use. In other words, observations of a method on easy targets do not necessarily generalize to challenging targets. (3) None of the methods can constantly outperform others. Moreover, we highlight two types of pitfalls that could lead to over-optimistic attack results in cross-device evaluations. We also contribute a large-scale public dataset (with 3 million EM traces from 9 probe locations over multiple targets) for benchmarking and reproducibility of side-channel attacks tackling domain shifts over EM traces.  

**This project establishes a large-scale EM dataset for side-channel attacks from 8-bit and 32-bit targets using TinyAES, with samples taken from 9 different locations for each chip.**

## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

Mabon Ninan, Evan Nimmo, Shane Reilly, Channing Smith, Wenhai Sun, Boyang Wang, and John M. Emmert, "A Second Look at the Portability of Deep Learning Side-Channel Attacks over EM Traces," The 27th International Symposium on Research in Attacks, Intrusions and Defenses (**RAID 2024**), Padua, Italy, Sept. 30 -- Oct. 2, 2024


## Requirements
This project is written in Python 3.8, Tensorflow 2.3.1 and Pytorch 2.0. Our experiments were performed on the following machines:

* GPU machine (Intel i9 CPU, 128GB memory,and a NVIDIA  NVIDIA 4080 GPU).
* GPU Server (Optional for accelarted Performance:AMD EPYC 7742 64-Core processor with 512 gB memory and 4 NVIDIA A100-SXM GPU)

## Datasets 

Our datasets used in this study can be accessed through the link below (last modified: **Sept. 2024**):

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EmwWSN0WkY1HuYST2xfZ5Q4Baw0hJ15QHtPuC2tIcPCQNQ?e=k7HPUq

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can (typically within 1~2 days). Thanks!

## Code:
The codebased include 3 folders: 
* BaseLine
  * Convolutional Neural Network (CNN)
  * Most Significant Byte (MSB)
  * Multilayer Perceptron (MLP)
* Domain Adaptation for Side-Channel
  * Adversarial Domain Adaptation (ADA) 
  * Maximum-Mean-Discrepancy Domain Adaptation (MMD)
  * On-the-Fly Fine Tuning (FT)
  * Zero-Mean and Unit-Variance Normalization (MVN)
* Pre-Processing Methods for Side-Channel
  * Discrete Fourier Transform (DFT)
  * Linear Discriminant Analysis (LDA)
  * Principle Component Analysis (PCA)

Note: Each with their own ReadMe Files 


# Contacts
Mabon Ninan ninanmm@mail.uc.edu
Boyang Wang boyang.wang@uc.edu
