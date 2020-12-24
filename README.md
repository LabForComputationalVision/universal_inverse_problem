# Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser 
Paper: https://arxiv.org/abs/2007.13640 \
Zahra Kadkhodaie, Eero P. Simoncelli,<br>

### Image priors, manifolds, and noisy observations
Visual images lie on a low-dimensional manifold, spanned by various natural deformations. Images on this manifold are approximately equally probable - at least locally. Probability of <img src="https://render.githubusercontent.com/render/math?math=x"> being a natural image, <img src="https://render.githubusercontent.com/render/math?math=p(x)">, is zero everywhere except for <img src="https://render.githubusercontent.com/render/math?math=x"> drawn from the manifold. 
![](figs/fig1.png)

An observed image, <img src="https://render.githubusercontent.com/render/math?math=y">, contaminated with Gaussian noise, <img src="https://render.githubusercontent.com/render/math?math=z\sim \mathcal N(0,\sigma^2)"> is drawn from an observation density, <img src="https://render.githubusercontent.com/render/math?math=p(y)">, which is a Gaussian-blurred version of the image prior. Moreover, the family of observation densities over different noise variances, <img src="https://render.githubusercontent.com/render/math?math=p_{\sigma}(y)">, forms a Gaussian scale-space representation of the prior analogous to the temporal evolution of a diffusion process

![](figs/fig2.png)

### Least squares denoising
For a noisy observation, ![y=x+z](https://latex.codecogs.com/svg.latex?;y=x+z), the least squares estimate of the true signal is the conditional mean of the posterior:\
![\hat{x}(y)=min{\hat{x}}\int||\hat{x}-x||^2p(x|y)dx](https://latex.codecogs.com/svg.latex?;\hat{x}(y)=min_{\hat{x}}\int||\hat{x}-x||^2p(x|y)dx) 
![=\int xp(x|y)dx](https://latex.codecogs.com/svg.latex?;=\int xp(x|y)dx) 

![](figs/fig3.png)

### Exposing the implicit prior through Empirical Bayes estimation
For Gaussian noise contamination, the least squares estimate can be written (exactly) as:\
![\hat{x}(y)=\intxp(x|y)dx](https://latex.codecogs.com/svg.latex?;\hat{x}(y)=\intxp(x|y)dx)\ 
![\hat{x}(y)=y+\sigma^2\nabla_y\log p(y)](https://latex.codecogs.com/svg.latex?;\hat{x}(y)=y+\sigma^2\nabla_y\log p(y)) 

This is Miyasawa’s Empirical Bayes formulation (1961), which expresses the denoising operation in terms of the gradient of the prior predictive density, ![p(y)](https://latex.codecogs.com/svg.latex?;p(y)). 
Below, we show a two-dimensional simulation/visualization.End of red line segments shows the least-squares optimal denoising solution ![\hat{x}](https://latex.codecogs.com/svg.latex?;\hat{x}) for each noisy signal, ![y](https://latex.codecogs.com/svg.latex?;y).

![](figs/fig4.png)

### Drawing high-probability samples from the implicit prior


Algorithm in a nutshell:
• Use denoiser-defined gradient to go uphill in probability \ 
• Do this iteratively \
• On each step, effect noise decreases, and effective prior becomes less blurred. Gradient step size automatically adapts to each noise level. \
• This coarse to fine optimization procedure converges to a point on the manifold! \

Two-dimensional visualization: trajectory of our iterative coarse-to-fine inverse algorithm
![](figs/fig8.png)

Click [here]() to watch a video of an animation of the two-dimensional simulatoin.

Two sequences of images, yt, from the iterative sampling procedure, with different initializations, y0, and added noise, β:
![](figs/synthesis_progression.png)
![](figs/synthesis_color_4.png)
![](figs/synthesis_color_2.png)
![](figs/synthesis_mnist_1.png)
![](figs/synthesis_mnist_2.png)

### Solving linear inverse problems using the implicit prior
![](figs/fig9.png)

Given a set of linear measurements of an image, xc = M T x, where M is a low-rank measurement matrix, we use an enhanced version of our algorithm to recover the original image

#### Inpainting
![](figs/inpaint_samples_flower.png)
![](figs/inpaint_samples_vase.png)
![](figs/inpaint_samples_zebra.png)
#### Missing random pixels
10% of pixels retained 
![](figs/random_pix_samples_zebra.png)

#### Super resolution
2x super resolution 
![](figs/super_res_flower.png)
4x super resolution 
![](figs/super_res_pepper.png)

#### Spectral super resolution 
10% of fourier coefficients retained (low frequencies)
![](figs/deblur_butterfly.png)

#### random basis - compressive sensing
Dimensionality reduced to 10%.
![](figs/rand_basis_soldier.png)


# In this repository
### Pre-trained denoisers
The directory [denoisers](denoisers) contains denoisers trained for removing Gaussian noise from natural images with the objective of minimizing mean square error. The prior embedded in a denoiser depends on the architecture of the model as well as the data used during training. The [denoisers](denoisers)  directory contains a separate folder for each denoiser with a specific architecture. The code for each architecture can be found in [code/network.py](code/network.py). Under each architecure directory, there are multiple folders for the denoiser trained on a particular dataset, and a specific noise range. 

### Code
The code directory contains code for the [algorithm](code/algorithm_inv_prob.py), the pre-trained [denoisers architecture](code/network.py), and [helper functions](code/Utils_inverse_prob.py). 

### test_images
Multiple commonly used [color](test_images/color) and [grayscale](test_images/grayscale) image datasets are uploaded in the test_images directory.

### Demo:
The [Demo.ipynb](Demo.ipynb) contains code for loading a per-trained denoiser and using it to generate samples from the prior of natural images implicit in the denoiser. It also contains code for solving various linear inverse problems (e.g. inpainting, deblurring, super-resolution, randomly-dropped pixels, compressive sensing). Notice that there is no training involved for solving these problems, as long as there is a universal blind denoiser at hand. The algorithm simply uses the prior embedded in the denoiser to generate the missing parts of the partially measured image. 

[Demo.ipynb](Demo.ipynb) can be executed on CPU or GPU. 

### Requirements 
Here is the list of libraries you need to install to execute the code: 

python  3.7.6 

numpy 1.19.4 \
skimage 0.17.2 \
matplotlib 1.19.4 \
PyTorch 1.7.0 \
argparse 1.1 \
os \
time\ 
sys \
gzip 
