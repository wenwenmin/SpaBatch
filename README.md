# SpaBatch
![image][(https://github.com/wenwenmin/SpaBatch/blob/main/F1.pdf)]

## Overview
With the rapid accumulation of spatial transcriptomics (ST) data across diverse tissues, individuals, and technological platforms, there is an urgent need for a robust and reliable multi-slice integration framework to enable three-dimensional (3D) spatial domain identification. However, existing methods largely focus on two-dimensional (2D) spatial domain identification within individual slices and fail to adequately account for inter-slice spatial correlations and batch effect correction, thereby limiting the accuracy of cross-slice 3D spatial domain identification.
In this study, we present SpaBatch, a novel framework for integrating and analyzing multi-slice ST data, which effectively corrects batch effects and enables cross-slice 3D spatial domain identification.
To demonstrate the power of SpaBatch, we apply SpaBatch to seven real ST datasets, including human cortical slices from different individuals, mouse brain slices generated using two different techniques, mouse embryo slices, human embryonic heart slices, and HER2+ breast cancer tissues.
Comprehensive validation demonstrates that SpaBatch consistently outperforms state-of-the-art methods in 3D spatial domain identification while effectively correcting batch effects. Moreover, SpaBatch efficiently captures conserved tissue architectures and cancer-associated substructures across slices, and leverages limited annotations to predict cancer regions in unannotated sections, highlighting its potential for tissue-structure interpretation and developmental biology studies.

## Datasets
All data used in this work are available at: https://zenodo.org/uploads/15233992.

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 4090.
- `pip install -r requiremnt.txt`
  
## Running demo
We provide demos for all datasets as references. The Tutorials section includes the running results of SpaBatch on all datasets used in this study, along with the code for downstream analysis. The results of all baseline methods can be found in the Baselines folder.

## Contact details
If you have any questions, please contact niujinyun@aliyun.com and minwenwen@ynu.edu.cn.

## Article link
https://www.biorxiv.org/content/10.1101/2025.03.24.645150v1
