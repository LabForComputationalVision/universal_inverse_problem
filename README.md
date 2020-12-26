# Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser 
Paper: https://arxiv.org/abs/2007.13640 \
Zahra Kadkhodaie, Eero P. Simoncelli,<br>

## In this repository
### Pre-trained denoisers
The directory [denoisers](denoisers) contains denoisers trained for removing Gaussian noise from natural images with the objective of minimizing mean square error. The prior embedded in a denoiser depends on the architecture of the model as well as the data used during training. The [denoisers](denoisers)  directory contains a separate folder for each denoiser with a specific architecture. The code for each architecture can be found in [code/network.py](code/network.py). Under each architecure directory, there are multiple folders for the denoiser trained on a particular dataset, and a specific noise range. 

### Code
The code directory contains code for the [algorithm](code/algorithm_inv_prob.py), the pre-trained [denoisers architecture](code/network.py), and [helper functions](code/Utils_inverse_prob.py). 

### Test_images
Multiple commonly used [color](test_images/color) and [grayscale](test_images/grayscale) image datasets are uploaded in the test_images directory.

### Demo
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


## Summary 
### Image priors, manifolds, and noisy observations
Visual images lie on a low-dimensional manifold, spanned by various natural deformations. Images on this manifold are approximately equally probable - at least locally. Probability of ![](https://latex.codecogs.com/svg.latex?x) being a natural image, ![](https://latex.codecogs.com/svg.latex?p(x)), is zero everywhere except for ![](https://latex.codecogs.com/svg.latex?x) drawn from the manifold. 

<img src="figs/fig1.png" width="680" height="250">

An observed image, ![](https://latex.codecogs.com/svg.latex?y), contaminated with Gaussian noise, <img src="https://render.githubusercontent.com/render/math?math=z\sim \mathcal N(0,\sigma^2)"> is drawn from an observation density, ![](https://latex.codecogs.com/svg.latex?p(y)), which is a Gaussian-blurred version of the image prior. Moreover, the family of observation densities over different noise variances, ![](https://latex.codecogs.com/svg.latex?p_{\sigma}(y)), forms a Gaussian scale-space representation of the prior analogous to the temporal evolution of a diffusion process

![](figs/fig2.png)

### Least squares denoising
For a noisy observation, ![y=x+z](https://latex.codecogs.com/svg.latex?;y=x+z), the least squares estimate of the true signal is the conditional mean of the posterior:

<img src="figs/fig3.png" width="630" height="265">

![](https://latex.codecogs.com/svg.latex?\hat{x}(y)=min_{\hat{x}}\int||\hat{x}-x||^2p(x|y)dx=\int(xp(x|y)dx)) 

### Exposing the implicit prior through Empirical Bayes estimation
For Gaussian noise contamination, the least squares estimate can be written (exactly) as:

![](https://latex.codecogs.com/svg.latex?\hat{x}(y)=\int(xp(x|y)dx)=y+\sigma^2\nabla_y\log(p(y)))


This is Miyasawa’s Empirical Bayes formulation (1961), which expresses the denoising operation in terms of the gradient of the prior predictive density, ![p(y)](https://latex.codecogs.com/svg.latex?p(y)). 
Below, we show a two-dimensional simulation/visualization. End of red line segments shows the least-squares optimal denoising solution ![\hat{x}](https://latex.codecogs.com/svg.latex?\hat{x}) for each noisy signal, ![y](https://latex.codecogs.com/svg.latex?;y):

![](figs/fig4.png)

### Drawing high-probability samples from the implicit prior

Algorithm in a nutshell:
* Use denoiser-defined gradient to go uphill in probability 
*  Do this iteratively 
* On each step, effective noise decreases, and effective prior becomes less blurred. Gradient step size automatically adapts to each noise level. 
* This coarse to fine optimization procedure converges to a point on the manifold! 

Below is a two-dimensional visualization of trajectory of our iterative coarse-to-fine inverse algorithm:

<img src="figs/fig8.png" width="250" height="250">

Click [here]() to watch a video of the two-dimensional simulatoin.

Sequences of images, ![](https://latex.codecogs.com/svg.latex?y_t), from the iterative sampling procedure, with different initializations, ![](https://latex.codecogs.com/svg.latex?y_0), and added noise, ![](https://latex.codecogs.com/svg.latex?\beta) are shown below. This is equivalent to the above simluation, but in the image space. Here we use a denoiser (BF-CNN) denoiser trained on (1) grayscale natural images (2) color natural images and (3) MNIST dataset. Starting from noise, the algorithm follow a trajectory to eventually sample from the manifold embedded in denoiser in use. 

![](figs/synthesis_progression.png)
![](figs/synthesis_progression2.png)
![](figs/synthesis_color_4.png)
![](figs/synthesis_color_2.png)
![](figs/synthesis_mnist_1.png)
![](figs/synthesis_mnist_2.png)

### Solving linear inverse problems using the implicit prior
Given a set of linear measurements of an image, ![](https://latex.codecogs.com/svg.latex?x_c) = ![](https://latex.codecogs.com/svg.latex?M^Tx), where M is a low-rank measurement matrix, we use an enhanced version of our algorithm to recover the original image. This is equivalent to restricting the algorithm to converge to the intersection of the manifold and the hyperplane spanned by the column space of M. To demonstrate this, we show partially linearly measured images and their reconstrcution for 5 different types of measurement matrices, M. 

<img src="figs/fig8.png" width="250" height="250">

#### 1. Inpainting
![](figs/inpaint_gray.png)
![](figs/inpaint_samples_flower.png)
![](figs/inpaint_mnist.png)

#### 2. Missing random pixels
10% of pixels retained 

![](figs/random_pix_gray.png)
![](figs/random_pix_samples_zebra.png)
![](figs/random_pix_mnist.png)

#### 3. Super resolution
4x super resolution (~6% dimensions retained )
![](figs/super_res_gray.png)
![](figs/super_res_pepper.png)
![](figs/super_res_mnist.png)

#### 4. Spectral super resolution 
10% of fourier coefficients retained (low frequencies)
![](figs/deblur_gray2.png)
![](figs/deblur_butterfly.png)
![](figs/deblur_mnist.png)

#### 5. Random basis - compressive sensing
Dimensionality reduced to 10%.
![](figs/rand_basis_gray.png)
![](figs/rand_basis_soldier.png)
![](figs/rand_basis_mnist.png)


