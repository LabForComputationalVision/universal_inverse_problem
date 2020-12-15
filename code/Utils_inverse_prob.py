import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.fft
################################################# Helper Functions #################################################


def single_image_loader(data_set_dire_path, image_number):
    all_names = os.listdir(data_set_dire_path)
    file_name = all_names[image_number]
    im = plt.imread(data_set_dire_path + file_name)
    if len(im.shape) == 3:
        x = torch.tensor(im).permute(2,0,1)
    elif len(im.shape) == 2:
        x = torch.tensor(im.reshape(1, im.shape[0], im.shape[1]))
    return x

def load_mnist_image( folder_path):
    
    # read and prep train images    
    f = gzip.open(folder_path,'r')
    image_size = 28
    f.read(16)
    buf = f.read(image_size * image_size )
    data = np.frombuffer(buf, dtype=np.uint8).astype(float)/255
    image = torch.tensor(data.reshape( 1,image_size, image_size).astype('float32'))

def rescale_image(im):
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')


def plot_synthesis(intermed_Ys, sample):
    f, axs = plt.subplots(1,len(intermed_Ys), figsize = ( 4*len(intermed_Ys),4))
    axs = axs.ravel()

    for ax in range(len(intermed_Ys)):
        x = intermed_Ys[ax].permute(1,2,0).detach().numpy()
        if x.shape[2] == 1:
            fig = axs[ax].imshow(rescale_image(x[:,:,0]), 'gray')
        else:
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')

    sample = sample.permute(1,2,0).detach().numpy()
    if sample.shape[2] == 1:
        fig = axs[-1].imshow(rescale_image(sample[:,:,0]),'gray' )
    else:
        fig = axs[-1].imshow(rescale_image(sample))

    axs[-1].axis('off')
    print('value range', np.round(np.min(sample ),2), np.round(np.max(sample),2) )

def plot_sample(x, corrupted, sample):
    f, axs = plt.subplots(1,3, figsize = (15,5))
    axs = axs.ravel()
    x = x.permute(1,2,0)
    corrupted = corrupted.permute(1,2,0)
    sample = sample.detach().permute(1,2,0)
    
    if x.size()!=corrupted.size():    
        h_diff = x.size()[0] - corrupted.size()[0]
        w_diff = x.size()[1] - corrupted.size()[1]
        x = x[0:x.size()[0]-h_diff,0:x.size()[1]-w_diff,: ]
        print('WARNING: psnr and ssim calculated using a cropped original image, because the original image is not divisible by the downsampling scale factor.')
        
    if x.shape[2] == 1: # if gray scale image
        fig = axs[0].imshow( x.squeeze(-1), 'gray', vmin=0, vmax = 1)
        axs[0].set_title('original')
        
        fig = axs[1].imshow(corrupted.squeeze(-1), 'gray',vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.squeeze(-1).numpy(), corrupted.squeeze(-1).numpy()  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy() ))
        axs[1].set_title('corrupted image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );  
        
        fig = axs[2].imshow(sample.squeeze(-1),'gray' ,vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.squeeze(-1).numpy(), sample.squeeze(-1).numpy()  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy() ))
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );

            
    else: # if color image
        fig = axs[0].imshow( x, vmin=0, vmax = 1)
        axs[0].set_title('original')        
        
        fig = axs[1].imshow( corrupted, vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.numpy(), corrupted.numpy(), multichannel=True  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy() ))
        axs[1].set_title('corrupted image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );  
        
        fig = axs[2].imshow(sample,vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True) ,3)
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy() ))   
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );
            
            
    for i in range(3): 
        axs[i].axis('off')

def plot_all_samples(sample, intermed_Ys):
    n_rows = int(np.ceil(len(intermed_Ys)/4))

    f, axs = plt.subplots(n_rows,4, figsize = ( 4*4, n_rows*4))
    axs = axs.ravel()

    for ax in range(len(intermed_Ys)):
        x = intermed_Ys[ax].detach().permute(1,2,0)
        if x.shape[2] == 1:
            fig = axs[ax].imshow(x[:,:,0], 'gray')
        else:
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')

    sample = sample.detach().permute(1,2,0)
    if sample.shape[2] == 1:
        fig = axs[-1].imshow(sample[:,:,0],'gray' )
    else:
        fig = axs[-1].imshow(rescale_image(sample))
    axs[-1].axis('off')
    plt.colorbar(fig, ax=axs[-1], fraction=.05)


    for ax in range(len(intermed_Ys),n_rows*4 ):
        axs[ax].axis('off')

###################################### Inverse problems Tasks ##################################
#############################################################################################
class synthesis:
    def __init__(self):
        super(synthesis, self).__init__()

    def M_T(self, x):
        return torch.zeros_like(x)

    def M(self, x):
        return torch.zeros_like(x)

#############################################################################################
class inpainting:
    '''
    makes a blanked area in the center
    @x_size : image size, tuple of (n_ch, im_d1,im_d2)
    @w: width of the blanked area
    @h: height of the blanked area
    '''
    def __init__(self, x_size,h, w):
        super(inpainting, self).__init__()

        n_ch , im_d1, im_d2 = x_size
        self.mask = torch.ones(x_size)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        c1, c2 = int(im_d1/2), int(im_d2/2)
        h , w= int(h/2), int(w/2)
        self.mask[0:n_ch, c1-h : c1+h , c2-w:c2+w] = 0

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask
#############################################################################################
class rand_pixels:
    '''
    @x_size : tuple of (n_ch, im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(rand_pixels, self).__init__()

        self.mask = torch.tensor(np.random.binomial(n=1, p=p, size = x_size ))
        if torch.cuda.is_available():
            self.mask = se.l.mask.cuda()

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask
#############################################################################################

class super_resolution:
    '''
    block averaging for super resolution.
    creates a low rank matrix (thin and tall) for down sampling
    @s: downsampling factor, int
    @x_size: tuple of three int  (n_ch, im_d1, im_d2)
    '''

    def __init__(self, x_size, s):
        super(super_resolution, self).__init__()

        if x_size[1]%2 !=0 or x_size[2]%2 != 0 :
            raise Exception("image dimensions need to be even")

        self.down_sampling_kernel = torch.ones(x_size[0],1,s,s)
        self.down_sampling_kernel = self.down_sampling_kernel/np.linalg.norm(self.down_sampling_kernel[0,0])
        if torch.cuda.is_available():
            self.down_sampling_kernel = self.down_sampling_kernel.cuda()
        self.x_size = x_size
        self.s = s

    def M_T(self, x):
        down_im = torch.nn.functional.conv2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])
        return down_im[0]

    def M(self, x):
        rec_im = torch.nn.functional.conv_transpose2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])

        return rec_im[0]

#############################################################################################

class random_basis:
    '''
    @x_size : tuple of (im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(random_basis, self).__init__()
        n_ch , im_d1, im_d2 = x_size
        self.x_size = x_size
        self.U, _ = torch.qr(torch.randn(int(np.prod(x_size)),int(np.prod(x_size)*p) ))
        if torch.cuda.is_available():
            self.U = self.U.cuda()

    def M_T(self, x):
        # gets 2d or 3d image and returns flatten partial measurement(1d)
        return torch.matmul(self.U.T,x.flatten())

    def M(self, x):
        # gets flatten partial measurement (1d), and returns 2d or 3d reconstruction
        return torch.matmul(self.U,x).reshape(self.x_size[0], self.x_size[1], self.x_size[2])

#############################################################################################
#### important: when using fftn from torch the reconstruction is more lossy than when fft2 from numpy
#### the difference between reconstruction and clean image in pytorch is of order of e-8, but in numpy is e-16

#### fix: only for images of odd dims. extend!
class spectral_super_resolution:
    '''
    creates a mask for dropping high frequency coefficients
    @im_d: dimension of the input image is (im_d, im_d)
    @p: portion of coefficients to keep
    '''
    def __init__(self, x_size, p):
        super(spectral_super_resolution, self).__init__()
        if (x_size[0] % 2 or x_size[1]%2) == 0 :
            raise Exception('need odd dim')

        self.x_size = x_size
        f0 = int(x_size[0]*p/2)
        f1 = int(x_size[1]*p/2)
        mask = torch.ones((x_size[0],x_size[1]))
        mask[f0+1:x_size[0]-f0,:]=0
        mask[:, f1+1:x_size[1]-f1]=0
        self.mask = mask


    def M_T(self, x):
        return self.mask*torch.fft.fftn(x, norm= 'ortho', s = self.x_size)

    def M(self, x):
        return torch.real(torch.fft.ifftn(x, norm= 'ortho', s = self.x_size))





