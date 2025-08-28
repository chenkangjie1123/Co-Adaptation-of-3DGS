<p align="center">
  <h1 align="center">🌈 Quantifying and Alleviating Co-Adaptation in <br/> Sparse-View 3D Gaussian Splatting </h1>
</p>

<p align="center">
  <a href="https://github.com/chenkangjie1123">Kangjie Chen</a>,  
  <a href="https://github.com/zhongyingji">Yingji Zhong</a>,  
  <a href="https://scholar.google.com/citations?hl=en&user=4cuefJ0AAAAJ">Zhihao Li</a>,  
  <a>Jiaqi Lin</a>,  <br>
  <a href="https://github.com/YouyuChen0207">Youyu Chen</a>,  
  <a href="https://minghanqin.github.io/">Minghan Qin</a>,  
  <a href="https://www.sigs.tsinghua.edu.cn/whq_en/main.htm">Haoqian Wang</a> 📪  
  <br>📪 corresponding author<br>
</p>

<p align="center">
  <a href="https://chenkangjie1123.github.io/Co-Adaptation-3DGS/#">Webpage</a> •
  <a href="https://arxiv.org/abs/2508.12720">arXiv</a> •
  <a href="https://arxiv.org/pdf/2508.12720">Paper</a>
</p>


## 📝 TL;DR
This paper introduces the concept of **co-adaptation** in 3D Gaussian Splatting (3DGS), analyzes its role in rendering artifacts, and proposes two strategies:  
- 🎲 **Dropout Regularization** – Randomly drops subsets of Gaussians to prevent over-co-adaptation.  
- 🌫️ **Opacity Noise Injection** – Adds noise to opacity values, suppressing spurious fitting and enhancing robustness.  

👉 *The code is based on [Binocular3DGS](https://github.com/hanl2010/Binocular3DGS). Thanks for their great work!* 🙌  


## 🌟 Why Color Artifacts in Sparse-View 3DGS?
<p align="center">
  <img width="90%" alt="Visualization" src="https://github.com/user-attachments/assets/a5653fb8-15bf-44bc-88eb-fd207193708d" />
</p>

*Visualization of 3DGS behaviors under different levels of co-adaptation.*  
- Thin gray arrows → training views  
- ✅ ❌ Bold arrows → novel view  
- ✅ Green arrow → correct color prediction  
- ❌ Red arrow → color errors  


## 📚 Citation
```bibtex
@article{chen2025quantifying,
  title={Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting},
  author={Chen, Kangjie and Zhong, Yingji and Li, Zhihao and Lin, Jiaqi and Chen, Youyu and Qin, Minghan and Wang, Haoqian},
  journal={arXiv preprint arXiv:2508.12720},
  year={2025}
}
