# Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser

Prior probability models are a central component of many image processing problems, but density estimation is notoriously difficult for high-dimensional signals such as photographic images. Deep neural networks have provided state-of-the-art solutions for problems such as denoising, which implicitly rely on a prior probability model of natural images. Here, we develop a robust and general methodology for making use of this implicit prior. We rely on a little-known statistical result due to Miyasawa (1961), who showed that the least-squares solution for removing additive Gaussian noise can be written directly in terms of the gradient of the log of the noisy signal density. We use this fact to develop a stochastic coarse-to-fine gradient ascent procedure for drawing high-probability samples from the implicit prior embedded within a CNN trained to perform blind (i.e., unknown noise level) least-squares denoising. A generalization of this algorithm to constrained sampling provides a method for using the implicit prior to solve any linear inverse problem, with no additional training. We demonstrate this general form of transfer learning in multiple applications, using the same algorithm to produce high-quality solutions for deblurring, super-resolution, inpainting, and compressive sensing.

![alt text](test.png?raw=true)


## Pre-trained denoisers
The directory [denoisers](denoisers) contains denoisers trained for removing Gaussian noise from natural images with the objective of minimizing mean square error. The prior embedded in a denoiser depends on the architecture of the model as well as the data used during training. This directory contains a separate folder for each denoiser with a specific architecture. The code for each architecture can be found in [code/network.py](code/network.py). Under each architecure directory, there are multiple folders for the denoiser trained on a particular dataset, and a specific noise range. 

## Code
The code directory contains code for the [algorithm](code/algorithm_inv_prob.py), the pre-trained denoisers architecture, and helper functions. 

## test_images
Multiple commonly used 

## Requirements 
Here is the list of libraries you need to install to execute the code: 


## Citation

