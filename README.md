# High-Fidelity Guided Image Synthesis with Latent Diffusion Models (CVPR 2023)

> Controllable image synthesis with user scribbles has gained huge public interest with the recent advent of text-conditioned latent diffusion models. The user scribbles control the color composition while the text prompt provides control over the overall image semantics. However, we find that prior works suffer from an intrinsic domain shift problem wherein the generated outputs often lack details and resemble simplistic representations of the target domain. In this paper, we propose a novel guided image synthesis framework, which addresses this problem by modeling the output image as the solution of a constrained optimization problem. We show that while computing an exact solution to the optimization is infeasible, an approximation of the same can be achieved while just requiring a single pass of the reverse diffusion process. Additionally, we show that by simply defining a cross-attention based correspondence between the input text tokens and the user stroke-painting, the user is also able to control the semantics of different painted regions without requiring any conditional training or finetuning. Human user study results show that the proposed approach outperforms the previous state-of-the-art by over 85.32% on the overall user satisfaction scores.

<!-- [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Singh_High-Fidelity_Guided_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf)][[Project Page](https://1jsingh.github.io/gradop)][[Demo](http://exposition.cecs.anu.edu.au:6009/)][[Citation](#citation)] -->


<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Singh_High-Fidelity_Guided_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
<a href="https://1jsingh.github.io/gradop"><img src="https://img.shields.io/badge/Project-Page-succees?style=for-the-badge&logo=GitHub" height=22.5></a>
<a href="http://exposition.cecs.anu.edu.au:6009/"><img src="https://img.shields.io/badge/Online-Demo-blue?style=for-the-badge&logo=Streamlit" height=22.5></a>
<a href="#citation"><img src="https://img.shields.io/badge/Paper-Citation-green?style=for-the-badge&logo=Google%20Scholar" height=22.5></a>
<!-- <a href="https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2F1jsingh%2Fpaint2pix&text=Unleash%20your%20inner%20artist%20...%20synthesize%20amazing%20artwork%2C%20and%20realistic%20image%20content%20or%20simply%20perform%20a%20range%20of%20diverse%20real%20image%20edits%20using%20just%20coarse%20user%20scribbles.&hashtags=Paint2Pix%2CECCV2022"><img src="https://img.shields.io/badge/Share--white?style=for-the-badge&logo=Twitter" height=22.5></a> -->

<p align="center">
<img src="./docs/overview-final-v1.jpg" width="800px"/>  
<br>
We propose a novel stroke based guided image synthesis framework which (Left) resolves the intrinsic domain shift problem in prior works (b), wherein the final images lack details and often resemble simplistic representations of the target domain (e) (generated using only text-conditioning). Iteratively reperforming the guided synthesis with the generated outputs (c) seems to improve realism but it is expensive and the generated outputs tend to lose faithfulness with the reference (a) with each iteration. (Right) Additionally, the user is also able to specify the semantics of different painted regions without requiring any additional training or finetuning.
</p>

## Description   
Official implementation of our CVPR 2023 paper with streamlit demo. By modelling the guided image synthesis output as the solution of a constrained optimization problem, we improve output realism <em>w.r.t</em> the target domain (e.g. realistic photos) when performing guided image synthesis from coarse user scribbles.

## Updates

* **(20/06/23)** Our project code and [demo](http://exposition.cecs.anu.edu.au:6009/) are online. Try performing realistic stroke-to-image synthesis right from your browser! 

https://github.com/1jsingh/GradOP-Guided-Image-Synthesis/assets/25987491/1549da28-e45b-4fab-a198-71508b237e4d



## Table of Contents
  * [Description](#description)
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Setup](#setup)
    + [Hugging Face Diffusers Library (Stable Diffusion)](#hugging-face-diffusers-library--stable-diffusion-)
  * [Usage](#usage)
  * [Example Results](#example-results)
    + [Stroke Guided Image Synthesis](#stroke-guided-image-synthesis)
    + [Visualizing the Optimization Process.](#visualizing-the-optimization-process)
    + [More Results](#more-results)
  * [Acknowledgments](#acknowledgments)
  * [Citation](#citation)



## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3
- Tested on Ubuntu 20.04, Nvidia RTX 3090 and CUDA 11.5 (though will likely run on other setups without modification)

### Setup
- Dependencies:  
Our code builds on the requirement of the official Stable Diffusion repository. To set up the environment, please run:
```
conda env create -f environment/environment.yaml
conda activate gradop-guided-synthesis
```

### Hugging Face Diffusers Library (Stable Diffusion)
Our code uses the Hugging Face [diffusers](https://github.com/huggingface/diffusers) library for downloading the Stable Diffusion v1.4 text-to-image model.

## Usage

### General

The GradOP+ model is provided in a simple [diffusers](https://github.com/huggingface/diffusers) pipeline for easy use:

* First load the pipeline with Stable Diffusion Weights
``` python
from pipeline_gradop_stroke2img import GradOPStroke2ImgPipeline
pipeline = GradOPStroke2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",torch_dtype=torch.float32)
```

* Simply load user-scribbles image and perform inference using GradOP+
``` python
# define the guidance inputs: 1) text prompt and 2) guidance image containing coarse scribbles
seed = 0
prompt = "a photo of a fox beside a tree"
stroke_img = Image.open('./input-images/fox.png').convert('RGB').resize((512,512))

# perform img2img guided synthesis using gradop+
generator = torch.Generator(device=device).manual_seed(seed)
out = pipeline.gradop_plus_stroke2img(prompt, stroke_img, strength=0.8, num_iterative_steps=3, grad_steps_per_iter=12, generator=generator)
```

Notes:

* Increase the number of `grad_steps_per_iter` (8-16) for improving faithfulness with the input reference.
* Increasing the `num_iterative_steps` between 3-5 can also help with the same.
* We can also compare the performance with standard (SDEdit-based) diffusers img2img predictions for same seed,
``` python
generator = torch.Generator(device=device).manual_seed(seed)
out = pipeline.sdedit_img2img(prompt=prompt, image=stroke_img, generator=generator)
```

* Similarly, visualization of the target data subspace (images conditioned only on the text prompt) can be done as follows,
``` python
prompt = "a photo of a fox beside a tree"
text_conditioned_outputs = pipeline.text2img_prediction(prompt, num_images_per_prompt=4).images
``` 

### Script

To generate an image, you can also simply run `run.py` script. For example,
``` bash
python run.py --img_path ./input-images/fox.png --prompt "a photo of a fox beside a tree" --seed 0 --num_iterative_steps=3 --grad_steps_per_iter=12
```

Notes:

* Increase the number of `--grad_steps_per_iter` (8-16) for improving faithfulness with the input reference.
* Increasing the `--num_iterative_steps` between 2-5 can also help with the same.
* For generating baseline images with standard (SDEdit-based) diffusers img2img prediction use `--method=sdedit` option as,

``` bash
python run.py --method sdedit --img_path ./input-images/fox.png --prompt "a photo of a fox beside a tree" --strength=0.8 --seed 0 
```

### Notebooks

We also provide a demo Jupyter notebook for detailed analysis and comparison with prior SDEdit based guided synthesis. Please see `notebooks/demo-gradop.ipynb`
for step-by-step analysis including:

* Comparison with SDEdit under changing hyperparameter
* Additional results across diverse data modalities (e.g. realistic photos, anime scenes etc.)
* Visualization of GradOP+ outputs under changing number of gradient descent steps.


## Example Results

### Stroke Guided Image Synthesis

<p align="center">
<img src="https://1jsingh.github.io/docs/gradop/sdedit-var-p4-v3.png" width="800px"/>  
<img src="https://1jsingh.github.io/docs/gradop/sdedit-var-p7.png" width="800px"/>  
<br>
As compared to prior works, our method provides a more practical approach for improving output realism (with respect to the target domain) while still maintaining the faithfulness with the reference painting.
</p>

### Visualizing the Optimization Process.
A key component of the proposed GradOP/GradOP+ solution, is to model the guided image synthesis problem as a constrained optimization problem and solve the same approximately using simple gradient descent. Here, we visualize the variation in output performance as the number of gradient descent steps are increased.

<p align="center">
<img src="./docs/ngrad-var-final-v1.jpg" width="800px"/>  
<br>
As the number of gradient steps for GradOP+ optimization increase (left to right) the output converges to more and more faithful (yet realistic) representation of the input reference painting / scribbles.
</p>

### More Results

<p align="center">
<img src="./docs/sample-results-ours-p3.png" width="800px"/>
<img src="./docs/sample-results-ours-p2.png" width="800px"/>
<img src="./docs/sample-results-ours-v3.png" width="800px"/>
<br>
Our approach allows the user to easily generate realistic image outputs across a range of data modalities.
</p>

## Acknowledgments
This code is builds on the code from the img2img stable diffusion pipeline from the [diffusers](https://github.com/huggingface/diffusers) library. 

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

