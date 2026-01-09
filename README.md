# Quantifying and Alleviating Co-Adaptation in <br/> Sparse-View 3D Gaussian Splatting

[![Project Website](https://img.shields.io/badge/Project_Website-4CAF50?logo=googlechrome&logoColor=white)](https://chenkangjie1123.github.io/Co-Adaptation-3DGS/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2508.12720)
[![Videos Compare](https://img.shields.io/badge/Videos_Compare-red?style=flat&logo=youtube&logoColor=white)](https://chenkangjie1123.github.io/Co-Adaptation-3DGS/#videos_compare)


[Kangjie Chen<sup>1</sup>](https://github.com/chenkangjie1123), [Yingji Zhong<sup>2</sup>](https://github.com/zhongyingji), [Zhihao Li<sup>3</sup>](https://scholar.google.com/citations?hl=en&user=4cuefJ0AAAAJ), [Jiaqi Lin<sup>1</sup>](), [Youyu Chen<sup>4</sup>](https://github.com/YouyuChen0207), [Minghan Qin<sup>1</sup>](https://minghanqin.github.io/), [Haoqian Wang<sup>1</sup>](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm) <br />
<sup>1</sup> Tsinghua University, <sup>2</sup> HKUST, <sup>3</sup> Huawei Noah’s Ark Lab, <sup>4</sup> Harbin Institute of Technology <br />



## TL;DR
This paper introduces the concept of **co-adaptation** in 3D Gaussian Splatting (3DGS), analyzes its role in rendering artifacts, and proposes two strategies:  
- 🎲 **Dropout Regularization** – Randomly drops subsets of Gaussians to prevent over-co-adaptation.  
- 🌫️ **Opacity Noise Injection** – Adds noise to opacity values, suppressing spurious fitting and enhancing robustness.

Besides, we further explore noise injection on other Gaussian attributes and advanced dropout variants. More details can be found in the Appendix of our paper.

*The code is based on [Binocular3DGS](https://github.com/hanl2010/Binocular3DGS). Thanks for their great work!*

## Abstract
3D Gaussian Splatting (3DGS) has demonstrated impressive performance in novel view synthesis under dense-view settings. However, in sparse-view scenarios, despite the realistic renderings in training views, 3DGS occasionally manifests appearance artifacts in novel views. This paper investigates the appearance artifacts in sparse-view 3DGS and uncovers a core limitation of current approaches: the optimized Gaussians are overly-entangled with one another to aggressively fit the training views, which leads to a neglect of the real appearance distribution of the underlying scene and results in appearance artifacts in novel views. The analysis is based on a proposed metric, termed Co-Adaptation Score (CA), which quantifies the entanglement among Gaussians, i.e., co-adaptation, by computing the pixel-wise variance across multiple renderings of the same viewpoint, with different random subsets of Gaussians. The analysis reveals that the degree of co-adaptation is naturally alleviated as the number of training views increases. Based on the analysis, we propose two lightweight strategies to explicitly mitigate the co-adaptation in sparse-view 3DGS: (1) random gaussian dropout; (2) multiplicative noise injection to the opacity. Both strategies are designed to be plug-and-play, and their effectiveness is validated across various methods and benchmarks. We hope that our insights into the co-adaptation effect will inspire the community to achieve a more comprehensive understanding of sparse-view 3DGS.


## Why Color Artifacts in Sparse-View 3DGS?
or Why Dropout and Noise Injection Work in Sparse-View 3DGS?
<p align="center">
  <img width="90%" alt="Visualization" src="https://github.com/user-attachments/assets/a5653fb8-15bf-44bc-88eb-fd207193708d" />
</p>

**Visualization of 3DGS behaviors under different levels of co-adaptation.** Thin gray arrows indicate training views, bold arrows indicate a novel view. Green arrow denotes correct color prediction, while red indicates color errors. (a) Simulates a 3DGS model trained with dense views, where Gaussian ellipsoids contribute evenly to pixel color across views, resulting in accurate rendering from the novel view. (b)(c)(d) Simulate various cases of 3DGS trained under sparse-view settings. (b) and (c) show that co-adaptation in the training views — where Gaussians contribute unequally to pixel colors — results in thin and thick artifacts under novel views. (d) shows a highly co-adapted case where multiple Gaussians with distinct colors collectively overfit a single grayscale pixel in the training view, resulting in severe wrong color artifacts under the novel view. 
- Thin gray arrows → training views  
- Bold arrows → novel view  
- Green arrow → correct color prediction ✅
- Red arrow → color artifacts ❌

## Setup
#### Installation
Clone [Co-Adaptation-of-3DGS](https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS.git) and Setup Anaconda Environment
```
git clone --recursive https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS.git
conda create -n coadaptation3dgs python=3.10
conda activate coadaptation3dgs
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
#### Dataset
- Download the processed datasets: [LLFF](https://drive.google.com/file/d/1XlnLk5SSzZ9bNdne5Wx5niA2ypo_aXAO/view?usp=drive_link) and [DTU](https://drive.google.com/file/d/13tEn6BxA3bKbTVc6xpWAbPHLALBSHOMK/view?usp=sharing)
- Download the NeRF Synthetic dataset from [here](https://drive.google.com/file/d/1RwXCLEDxm8ssWvp-qhCBWRcechk3lsQJ/view?usp=sharing)
#### Checkpoints
[Binocular3DGS](https://github.com/hanl2010/Binocular3DGS) use the pre-trained [PDCNet+](https://github.com/PruneTruong/DenseMatching) to generate dense initialization point clouds. The pre-trained PDCNet+ model can be downloaded [here](https://drive.google.com/file/d/151X9ovbOG35tbPjioV5CYk_5GKQ8FErw/view?usp=sharing). Put the pre-trained model in `submodules/dense_matcher/pre_trained_models`

## Training and Evaluation
#### LLFF dataset
```
python script/run_llff.py
```
#### DTU dataset
```
python script/run_dtu.py
```
#### NeRF Synthetic dataset (Blender)
When training on the Blender dataset, the evaluation metrics vary significantly between using a white background and a black background. In the paper, we adopt the white background setting while using a black background here.
```
python script/run_blender.py
```


## Integration into Your Project
Our strategy is designed for **sparse-view settings** and is fully compatible with the **3D Gaussian Splatting (3DGS)** framework. It can be seamlessly integrated into your own 3DGS-based project with only minimal changes.  

To integrate 🎲 **Dropout Regularization**, simply add the following lines to your **`./gaussian_renderer/__init__.py`** file in the 3DGS framework:  

```python
# init random dropout mask
if dropout_factor > 0.0 and train:
    dropout_mask = torch.rand(pc.get_opacity.shape[0], device=pc.get_opacity.device).cuda()
    dropout_mask = dropout_mask < (1 - dropout_factor)

# randomly dropout 3DGS points during training
if dropout_factor > 0.0 and train:
    means3D   = means3D[dropout_mask]
    means2D   = means2D[dropout_mask]
    shs       = shs[dropout_mask]
    opacity   = opacity[dropout_mask]
    scales    = scales[dropout_mask]
    rotations = rotations[dropout_mask]
elif not train:
    # scale opacity for test stage rendering
    opacity *= 1 - dropout_factor
```

To integrate 🌫️ **Opacity Noise Injection**, simply add the following lines:  

```python
# add noise to opacity during training
if train and sigma_noise > 0.0:
    # sigma_noise = 0.8  # example value
    epsilon_opacity = torch.randn_like(opacity, device=opacity.device) * sigma_noise
    epsilon_opacity = torch.clamp(epsilon_opacity, min=-sigma_noise, max=sigma_noise)
    opacity = torch.clamp(opacity * (1.0 + epsilon_opacity), min=0.0, max=1.0)
```

> 💡 Note: The optimal parameter setting of **Opacity Noise Injection** may vary across different baselines, configurations, scenes, and dataset types. Therefore, we generally recommend using **Dropout Regularization**, which provides more robust and stable parameter choices in practice.


## Quantitative Comparison on LLFF and DTU Datasets

We evaluated existing 3DGS-based sparse-view reconstruction methods with and without our proposed **co-adaptation suppression strategies** — *dropout regularization* and *opacity noise injection*. We report **PSNR**, **SSIM**, **LPIPS**, and **Co-Adaptation (CA)** scores on both **training** and **novel views** to assess reconstruction quality and co-adaptation reduction. Experiments are conducted on **LLFF** and **DTU** datasets, using **3 training views** following prior works ([Binocular3DGS](https://github.com/hanl2010/Binocular3DGS), [Cor-GS](https://github.com/jiaw-z/CoR-GS), [FSGS](https://github.com/VITA-Group/FSGS), [RegNeRF](https://github.com/google-research/google-research/tree/master/regnerf), [FreeNeRF](https://github.com/Jiawei-Yang/FreeNeRF)). Input images are downsampled by **8×** for LLFF and **4×** for DTU relative to their original resolutions.
<!-- <img width="1498" height="553" alt="image" src="https://github.com/user-attachments/assets/89cca527-30a6-483e-a6ec-dbfb773f0465" /> -->

| Method | Setting | PSNR↑ | SSIM↑ | LPIPS↓ | TrainCA↓ | TestCA↓ | PSNR↑ | SSIM↑ | LPIPS↓ | TrainCA↓ | TestCA↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | <center>**LLFF**</center> | | | | | <center>**DTU**</center> | | | | |
| **3DGS** | baseline | 19.36 | 0.651 | 0.232 | 0.007543 | 0.008206 | 17.30 | 0.824 | 0.152 | 0.002096 | 0.002869 |
| | w/ dropout | **20.20** | **0.691** | **0.211** | 0.001752 | 0.002340 | **17.75** | **0.850** | **0.135** | **0.000757** | **0.002263** |
| | w/ opacity noise | 19.91 | 0.676 | 0.223 | **0.001531** | **0.002300** | 17.27 | 0.839 | 0.140 | 0.001203 | 0.002390 |
| **DNGaussian** | baseline | 18.93 | 0.599 | 0.295 | 0.007234 | 0.007645 | 18.91 | 0.790 | 0.176 | 0.005113 | 0.005744 |
| | w/ dropout | **19.43** | **0.623** | 0.302 | **0.003242** | **0.003821** | **19.86** | **0.828** | **0.149** | **0.001201** | **0.001917** |
| | w/ opacity noise | 19.15 | 0.608 | **0.294** | 0.004507 | 0.005071 | 19.52 | 0.813 | 0.153 | 0.001901 | 0.002641 |
| **FSGS** | baseline | 20.43 | 0.682 | 0.248 | 0.004580 | 0.004758 | 17.34 | 0.818 | 0.169 | 0.002078 | 0.003313 |
| | w/ dropout | **20.82** | **0.716** | **0.200** | 0.001930 | 0.002205 | **17.86** | **0.838** | **0.153** | **0.001057** | 0.002376 |
| | w/ opacity noise | 20.59 | 0.706 | 0.210 | **0.001666** | **0.002020** | 17.81 | 0.834 | 0.156 | 0.001096 | **0.002245** |
| **CoR-GS** | baseline | 20.17 | 0.703 | **0.202** | 0.005025 | 0.005159 | 19.21 | 0.853 | 0.119 | 0.001090 | 0.001199 |
| | w/ dropout | **20.64** | **0.712** | 0.217 | **0.001442** | **0.001617** | **19.94** | **0.868** | 0.118 | **0.000378** | **0.000611** |
| | w/ opacity noise | 20.28 | 0.705 | **0.202** | 0.003731 | 0.003867 | 19.54 | 0.862 | **0.115** | 0.000558 | 0.000814 |
| **Binocular3DGS** | baseline | 21.44 | 0.751 | 0.168 | 0.001845 | 0.001951 | 20.71 | 0.862 | 0.111 | 0.001399 | 0.001579 |
| ⭐ | w/ dropout | **22.12** | 0.777 | **0.154** | 0.000875 | 0.000978 | 21.03 | **0.875** | 0.108 | 0.000752 | 0.001146 |
| ⭐ | w/ opacity noise | **22.12** | 0.780 | 0.155 | **0.000660** | **0.000762** | 20.92 | 0.866 | **0.107** | 0.001346 | 0.001548 |
| ⭐ | w/ both | 22.11 | **0.781** | 0.157 | 0.000673 | **0.000762** | **21.05** | **0.875** | 0.109 | **0.000736** | **0.001143** |

## Citation
If you find our work helpful, please ⭐ our repository and cite:
```bibtex
@article{chen2025quantifying,
  title={Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting},
  author={Chen, Kangjie and Zhong, Yingji and Li, Zhihao and Lin, Jiaqi and Chen, Youyu and Qin, Minghan and Wang, Haoqian},
  journal={arXiv preprint arXiv:2508.12720},
  year={2025}
}
```


### Concurrent Works  

There are two other concurrent works that also use dropout to boost sparse-view 3DGS:  

- [DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting ](https://arxiv.org/abs/2504.00773) 
- [DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](https://arxiv.org/abs/2504.09491)  

They attribute the effectiveness of dropout to empirical factors — such as reducing overfitting through fewer active splats (*DropoutGS*), or enhancing gradient flow to distant Gaussians (*DropGaussian*).  

We respect these insights and are pleased that several works highlight the benefits of dropout in sparse-view 3DGS. Our work complements these findings by offering a deeper analysis of **co-adaptation**, with the goal of stimulating broader discussion on more generalizable 3D representations.  
