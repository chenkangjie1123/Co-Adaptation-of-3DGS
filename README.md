<p align="center">
  <h1 align="center">🌈 Quantifying and Alleviating Co-Adaptation in <br/> Sparse-View 3D Gaussian Splatting </h1>
</p>


<p align="center">
  <a href="https://github.com/chenkangjie1123">Kangjie Chen</a><sup>1</sup>  &nbsp;&nbsp
  <a href="https://github.com/zhongyingji">Yingji Zhong</a><sup>2</sup>  &nbsp;&nbsp
  <a href="https://scholar.google.com/citations?hl=en&user=4cuefJ0AAAAJ">Zhihao Li</a><sup>3</sup>  &nbsp;&nbsp
  <a>Jiaqi Lin</a><sup>1</sup>  <br>
  <a href="https://github.com/YouyuChen0207">Youyu Chen</a><sup>4</sup>  &nbsp;&nbsp
  <a href="https://minghanqin.github.io/">Minghan Qin</a><sup>1</sup>  &nbsp;&nbsp
  <a href="https://www.sigs.tsinghua.edu.cn/whq_en/main.htm">Haoqian Wang</a><sup>1</sup> 📪  
  <br>📪 corresponding author<br>
  <sup>1</sup> Tsinghua University &nbsp;&nbsp;
  <sup>2</sup> HKUST &nbsp;&nbsp;
  <sup>3</sup> Huawei Noah’s Ark Lab &nbsp;&nbsp;<br>
  <sup>4</sup> Harbin Institute of Technology
</p>

<div align="center" style="text-align: center;">

[![Project Page](https://img.shields.io/badge/🌐-Project_Page-blueviolet)](https://chenkangjie1123.github.io/Co-Adaptation-3DGS/#)
[![arXiv Paper](https://img.shields.io/badge/📜-arXiv:2508-12720)](https://arxiv.org/abs/2508.12720)
[![Videos Compare](https://img.shields.io/badge/📺-Videos%20Compare-00a1d6)](https://chenkangjie1123.github.io/Co-Adaptation-3DGS/#videos_compare)

</div>


## 📌 TL;DR
This paper introduces the concept of **co-adaptation** in 3D Gaussian Splatting (3DGS), analyzes its role in rendering artifacts, and proposes two strategies:  
- 🎲 **Dropout Regularization** – Randomly drops subsets of Gaussians to prevent over-co-adaptation.  
- 🌫️ **Opacity Noise Injection** – Adds noise to opacity values, suppressing spurious fitting and enhancing robustness.  

*The code is based on [Binocular3DGS](https://github.com/hanl2010/Binocular3DGS). Thanks for their great work!*


## 🌟 Abstract
3D Gaussian Splatting (3DGS) has demonstrated impressive performance in novel view synthesis under dense-view settings. However, in sparse-view scenarios, despite the realistic renderings in training views, 3DGS occasionally manifests appearance artifacts in novel views. This paper investigates the appearance artifacts in sparse-view 3DGS and uncovers a core limitation of current approaches: the optimized Gaussians are overly-entangled with one another to aggressively fit the training views, which leads to a neglect of the real appearance distribution of the underlying scene and results in appearance artifacts in novel views. The analysis is based on a proposed metric, termed Co-Adaptation Score (CA), which quantifies the entanglement among Gaussians, i.e., co-adaptation, by computing the pixel-wise variance across multiple renderings of the same viewpoint, with different random subsets of Gaussians. The analysis reveals that the degree of co-adaptation is naturally alleviated as the number of training views increases. Based on the analysis, we propose two lightweight strategies to explicitly mitigate the co-adaptation in sparse-view 3DGS: (1) random gaussian dropout; (2) multiplicative noise injection to the opacity. Both strategies are designed to be plug-and-play, and their effectiveness is validated across various methods and benchmarks. We hope that our insights into the co-adaptation effect will inspire the community to achieve a more comprehensive understanding of sparse-view 3DGS.


## 🌟 Why Color Artifacts in Sparse-View 3DGS?
<p align="center">
  <img width="90%" alt="Visualization" src="https://github.com/user-attachments/assets/a5653fb8-15bf-44bc-88eb-fd207193708d" />
</p>

*Visualization of 3DGS behaviors under different levels of co-adaptation.*  
- Thin gray arrows → training views  
- ✅ ❌ Bold arrows → novel view  
- ✅ Green arrow → correct color prediction  
- ❌ Red arrow → color errors  

## 🛠️ Setup
#### Installation
Clone [Co-Adaptation-of-3DGS](https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS.git)
```
git clone --recursive https://github.com/chenkangjie1123/Co-Adaptation-of-3DGS.git
```
Setup Anaconda Environment
```
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
[Binocular3DGS](https://github.com/hanl2010/Binocular3DGS) use the pre-trained [PDCNet+](https://github.com/PruneTruong/DenseMatching) to generate dense initialization point clouds. The pre-trained PDCNet+ model can be downloaded [here](https://drive.google.com/file/d/151X9ovbOG35tbPjioV5CYk_5GKQ8FErw/view?usp=sharing).

Put the pre-trained model in `submodules/dense_matcher/pre_trained_models`

## 📊 Training and Evaluation
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


## 📖 Citation
If you find our work helpful, please ⭐ our repository and cite:
```bibtex
@article{chen2025quantifying,
  title={Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting},
  author={Chen, Kangjie and Zhong, Yingji and Li, Zhihao and Lin, Jiaqi and Chen, Youyu and Qin, Minghan and Wang, Haoqian},
  journal={arXiv preprint arXiv:2508.12720},
  year={2025}
}
