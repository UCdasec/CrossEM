# CrossEM A Second Look at the Portability of Deep Learning Side-Channel Attacks over EM Traces

This on-going project examines the impact of EM probe locations on the results of deep-learning side-channel attacks. 

**The dataset and code are for research purpose only**

## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

Mabon Ninan, Evan Nimmo, Shane Reilly, Channing Smith, Wenhai Sun, Boyang Wang, and John M. Emmert. 2024. "A Second Look at the Portability of Deep Learning Side-Channel Attacks over EM Traces".The 27th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2024), Padua (Italy)


## Requirements
This project is written in Python 3.8, Tensorflow 2.3.1 and Pytorch 2.0 . 
Our experiments is running with:

* GPU machine (Intel i9 CPU, 128GB memory,and a NVIDIA  NVIDIA 4080 GPU).
* GPU Server (Optional for accelarted Performance:AMD EPYC 7742 64-Core processor with 512 gB memory and 4 NVIDIA A100-SXM GPU)

## Datasets (**Access is limited to our lab members only at this point**) 

Our datasets used in this study can be accessed through the link below (last modified March 2023):

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EmwWSN0WkY1HuYST2xfZ5Q4Baw0hJ15QHtPuC2tIcPCQNQ?e=k7HPUq

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can. Thanks!

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

Boyang Wang wang2ba@ucmail.uc.edu
