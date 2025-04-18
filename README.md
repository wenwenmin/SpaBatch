# SpaBatch
![image](https://github.com/wenwenmin/SpaBatch/blob/main/Figure1.png)

## Overview
With the rapid generation of spatial transcriptomics (ST) data across diverse tissues, individuals, and platforms, there is an urgent need for a robust integration algorithm for multi-slice joint analysis. However, existing spatial domain identification methods often struggle with batch effects and have limited generalization capabilities across slices. To address these challenges, we propose SpaBatch, a novel framework for the integration and analysis of multi-slice ST data, enabling effective data integration, cross-slice spatial domain identification, and downstream biological analysis. We apply SpaBatch to a variety of ST datasets, including human cortical slices from different individuals, mouse brain slices generated by two different technologies, mouse embryonic slices, human embryonic heart slices, and HER2+ breast cancer tissues. SpaBatch efficiently captures shared tissue structures across slices, cancer-related substructures, and leverages limited annotations to predict cancer regions across slices. Extensive experiments on multiple real ST datasets demonstrate that SpaBatch outperforms existing methods in both spatial domain identification and batch effect correction, showing great potential in tissue structure interpretation and developmental dynamics studies.

## Datasets
All data used in this work are available at: https://zenodo.org/uploads/15233992.

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 4090)
- `pip install -r requiremnt.txt`
  
## Running demo
We provide an example to test SpaBatch on the DLPFC dataset. You can test the DLPFC dataset by running the Run_DLPFC_151507.ipynb file.

## Contact details
If you have any questions, please contact niujinyun@aliyun.com and minwenwen@ynu.edu.cn.

## Article link
https://www.biorxiv.org/content/10.1101/2025.03.24.645150v1
