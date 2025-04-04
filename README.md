---

<div align="center">    
 
# Session-Guided Attention in Continuous Learning with Few Samples

[Zicheng Pan](https://zichengpan.github.io/), Xiaohan Yu, and Yongsheng Gao

![Paper](https://img.shields.io/badge/paper-TIP%202025-blue)

</div>

## Abstract
Few-shot class-incremental learning (FSCIL) aims to learn from a sequence of incremental data sessions with a limited number of samples in each class. The main issues it encounters are the risk of forgetting previously learned data when introducing new data classes, as well as not being able to adapt the old model to new data due to limited training samples. Existing state-of-the-art solutions normally utilize pre-trained models with fixed backbone parameters to avoid forgetting old knowledge. While this strategy preserves previously learned features, the fixed nature of the backbone limits the model's ability to learn optimal representations for unseen classes, which compromises performance on new class increments. In this paper, we propose a novel SEssion-Guided Attention framework (SEGA) to tackle this challenge. SEGA exploits the class relationships within each incremental session by assessing how test samples relate to class prototypes. This allows accurate incremental session identification for test data, leading to more precise classifications. In addition, an attention module is introduced for each incremental session to further utilize the feature from the fixed backbone. As the session of the testing image is determined, we can fine-tune the feature with the corresponding attention module to better cluster the sample within the selected session. Our approach adopts the fixed backbone strategy to avoid forgetting the old knowledge while achieving novel data adaptation. Experimental results on three FSCIL datasets consistently demonstrate the superior adaptability of the proposed SEGA framework in FSCIL tasks.

