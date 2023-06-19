# High-Fidelity Guided Image Synthesis with Latent Diffusion Models (CVPR 2023)

> Controllable image synthesis with user scribbles has gained huge public interest with the recent advent of text-conditioned latent diffusion models. The user scribbles control the color composition while the text prompt provides control over the overall image semantics. However, we find that prior works suffer from an intrinsic domain shift problem wherein the generated outputs often lack details and resemble simplistic representations of the target domain. In this paper, we propose a novel guided image synthesis framework, which addresses this problem by modeling the output image as the solution of a constrained optimization problem. We show that while computing an exact solution to the optimization is infeasible, an approximation of the same can be achieved while just requiring a single pass of the reverse diffusion process. Additionally, we show that by simply defining a cross-attention based correspondence between the input text tokens and the user stroke-painting, the user is also able to control the semantics of different painted regions without requiring any conditional training or finetuning. Human user study results show that the proposed approach outperforms the previous state-of-the-art by over 85.32% on the overall user satisfaction scores.

<!-- [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Singh_High-Fidelity_Guided_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf)][[Project Page](https://1jsingh.github.io/gradop)][[Demo](http://exposition.cecs.anu.edu.au:6009/)][[Citation](#citation)] -->


<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Singh_High-Fidelity_Guided_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
<a href="https://1jsingh.github.io/gradop"><img src="https://img.shields.io/badge/Project-Page-succees?style=for-the-badge&logo=GitHub" height=22.5></a>
<a href="http://exposition.cecs.anu.edu.au:6009/"><img src="https://img.shields.io/badge/Online-Demo-blue?style=for-the-badge&logo=Streamlit" height=22.5></a>
<a href="#citation"><img src="https://img.shields.io/badge/Paper-Citation-green?style=for-the-badge&logo=Google%20Scholar" height=22.5></a>
<!-- <a href="https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2F1jsingh%2Fpaint2pix&text=Unleash%20your%20inner%20artist%20...%20synthesize%20amazing%20artwork%2C%20and%20realistic%20image%20content%20or%20simply%20perform%20a%20range%20of%20diverse%20real%20image%20edits%20using%20just%20coarse%20user%20scribbles.&hashtags=Paint2Pix%2CECCV2022"><img src="https://img.shields.io/badge/Share--white?style=for-the-badge&logo=Twitter" height=22.5></a> -->

<p align="center">
<img src="https://1jsingh.github.io/docs/gradop/overview-v3.png" width="800px"/>  
<br>
We propose a novel stroke based guided image synthesis framework which (Left) resolves the intrinsic domain shift problem in prior works (b), wherein the final images lack details and often resemble simplistic representations of the target domain (e) (generated using only text-conditioning). Iteratively reperforming the guided synthesis with the generated outputs (c) seems to improve realism but it is expensive and the generated outputs tend to lose faithfulness with the reference (a) with each iteration. (Right) Additionally, the user is also able to specify the semantics of different painted regions without requiring any additional training or finetuning.
</p>

## Description   
Official implementation of our CVPR 2023 paper with streamlit demo. By modelling the guided image synthesis output as the solution of a constrained optimization problem, we imporve output realism <em>w.r.t</em> the target domain (e.g. realistic photos) when performing guided image synthesis from coarse user scribbles.

<!-- ## Updates

* **(10/06/23)** Our [project demo](http://exposition.cecs.anu.edu.au:6009/) is online. Try performing realistic image synthesis right from your browser!  -->

<!-- https://user-images.githubusercontent.com/25987491/185323657-a71c239c-892c-4202-b753-a84c0bf19a30.mp4 -->


## Table of Contents
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
- [Pretrained Models](#pretrained-models)
  * [Paint2pix models](#paint2pix-models)
- [Using the Demo](#using-the-demo)
- [Example Results](#example-results)
  * [Progressive Image Synthesis](#progressive-image-synthesis)
  * [Real Image Editing](#real-image-editing)
  * [Artistic Content Generation](#artistic-content-generation)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>



## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3
- Tested on Ubuntu 20.04, Nvidia RTX 3090 and CUDA 11.5

### Installation
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/paint2pix_env.yaml`.

## Pretrained Models
Please download the following pretrained models essential for running the provided demo.

### Paint2pix models
| Path | Description
| :--- | :----------
|[Canvas Encoder - ReStyle](https://drive.google.com/file/d/1ufKEtDXEG6o96KjLh-i6EL7Ir9TlwPcs/view?usp=sharing)  | Paint2pix Canvas Encoder trained with a ReStyle architecture.
|[Identity Encoder - ReStyle](https://drive.google.com/file/d/1KT3YmSHgMJM3b7Ox9zciyo3FSELtJsyS/view?usp=sharing)  | Paint2pix Identity Encoder trained with a ReStyle architecture.
|[StyleGAN - Watercolor Painting](https://drive.google.com/file/d/1WW_a589lv7R9-PNvKlVkVxITIZnW7xlv/view?usp=sharing)  | StyleGAN decoder network trained to generate watercolor paintings. Used for artistic content generation with paint2pix.
|[IR-SE50 Model](https://drive.google.com/file/d/1U4q_o20uGMozSetOkMGddUcAWf_ons2-/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID loss and id-encoder training.


Please download and save the above models to the directory `pretrained_models`. 


## Using the Demo

We provide a streamlit-drawble canvas based demo for trying out different features of the Paint2pix model. To start the demo use,

```
CUDA_VISIBLE_DEVICES=2 streamlit run demo.py --server.port 6009
```

The demo can then be accessed on the local machine or ssh client via [localhost](http://localhost:6009).

The demo has been divided into 3 convenient sections:
    
1. **Real Image Editing**: Allows the user to edit real images using coarse user scribbles
2. **Progressive Image Synthesis**: Start from an empty canvas and design your desired image output using just coarse scribbles.
3. **Artistic Content Generation**: Unleash your inner artist! create highly artistic portraits using just coarse scribbles.


## Example Results

### Progressive Image Synthesis

<p align="center">
<img src="docs/prog-synthesis.png" width="800px"/>  
<br>
Paint2pix for progressive image synthesis
</p>

### Real Image Editing

<p align="center">
<img src="docs/custom-color-edits.png" width="800px"/>  
<br>
Paint2pix for achieving diverse custom real-image edits
</p>

### Artistic Content Generation


<p align="center">
<img src="docs/watercolor-synthesis.png" width="800px"/>  
<br>
Paint2pix for generating highly artistic content using coarse scribbles
</p>

## Acknowledgments
This code borrows heavily from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), 
[encoder4editing](https://github.com/omertov/encoder4editing) and [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder). 

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{singh2023high,
  title={High-Fidelity Guided Image Synthesis With Latent Diffusion Models},
  author={Singh, Jaskirat and Gould, Stephen and Zheng, Liang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5997--6006},
  year={2023}
}
```

