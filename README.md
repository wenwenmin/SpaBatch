# SpaBatch
![image](https://github.com/wenwenmin/SpaBatch/blob/main/F1.jpg)

## Overview
With the rapid accumulation of spatial transcriptomics (ST) data across diverse tissues, individuals, and technological platforms, there is an urgent need for a robust and reliable multi-slice integration framework to enable three-dimensional (3D) spatial domain identification. However, existing methods largely focus on two-dimensional (2D) spatial domain identification within individual slices and fail to adequately account for inter-slice spatial correlations and batch effect correction, thereby limiting the accuracy of cross-slice 3D spatial domain identification.
In this study, we present SpaBatch, a novel framework for integrating and analyzing multi-slice ST data, which effectively corrects batch effects and enables cross-slice 3D spatial domain identification.
To demonstrate the power of SpaBatch, we apply SpaBatch to seven real ST datasets, including human cortical slices from different individuals, mouse brain slices generated using two different techniques, mouse embryo slices, human embryonic heart slices, and HER2+ breast cancer tissues.
Comprehensive validation demonstrates that SpaBatch consistently outperforms state-of-the-art methods in 3D spatial domain identification while effectively correcting batch effects. Moreover, SpaBatch efficiently captures conserved tissue architectures and cancer-associated substructures across slices, and leverages limited annotations to predict cancer regions in unannotated sections, highlighting its potential for tissue-structure interpretation and developmental biology studies.

## Datasets
All data used in this work are available at: https://zenodo.org/uploads/15233992.

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 4090).
- `pip install -r requiremnt.txt`
  
## Running demo
We provide demos for all datasets as references. The [Tutorials folders](https://github.com/wenwenmin/SpaBatch/tree/main/Tutorials) includes the running results of SpaBatch on all datasets used in this study, along with the code for downstream analysis. The results of all baseline methods can be found in the [Baselines folders](https://github.com/wenwenmin/SpaBatch/tree/main/Baselines).

## Contact details
If you have any questions, please contact niujinyun@aliyun.com and minwenwen@ynu.edu.cn.

## Citing
<p>The corresponding BiBTeX citation are given below:</p>
<div class="highlight-none"><div class="highlight"><pre>
@article{niu2025SpaBatch,
author = {Niu, Jinyun and Fang, Donghai and Chen, Jinyu and Xiong, Yi and Liu, Juan and Min, Wenwen},
title = {SpaBatch: Deep Learning-Based Cross-Slice Integration and 3D Spatial Domain Identification in Spatial Transcriptomics},
journal = {Advanced Science},
volume = {12},
number = {44},
pages = {e09090},
doi = {https://doi.org/10.1002/advs.202509090},
year = {2025}
}
</pre></div>

## Article link

Free Download: https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202509090
