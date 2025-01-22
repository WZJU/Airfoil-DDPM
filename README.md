# README

Thank you for your interest in this work.

This repository holds the code for our study, which is submitting.

We will upload our pre-trained model as soon as possible.



## How to train

```
import airfoil_DDPM
df_model=Unet(1, 20,context_size_1=3, context_size_2=3, down_sampling_dim=2, dropout = 0.)
```

where context_size_1 and context_size_2 are the size of context, you can change it on your need.

## How to use

cd [generate.py](https://github.com/WZJU/Airfoil-DDPM/blob/main/generate.py)

upload a .pth file you trained at line 94

Input leading edge radius R, maximum thickness t_max, and the corresponding position X_c of maximum thickness at line 110.

Input any set of [Cl,Cd,Cm] at line 111.

Line 115 to line 120 is normalization, you can change it on your need.

run [generate.py](https://github.com/WZJU/Airfoil-DDPM/blob/main/generate.py), the code will print CST paras. Copy it

cd [plot.ipynb](https://github.com/WZJU/Airfoil-DDPM/blob/main/plot.ipynb)

paste CST paras in block 2, line 4

paste constraint in block 2, line 2 to 3

## Reference

https://github.com/CompVis/stable-diffusion/tree/main

https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file

https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py
