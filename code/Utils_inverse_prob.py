import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.fft
import gzip
import argparse
from network import BF_CNN

################################################# Helper Functions #################################################
def load_denoiser(architecture,grayscale, training_data, training_noise): 
    if architecture=='BF_CNN': 
        model = load_BF_CNN(grayscale, training_data, training_noise)

        
    return model

def load_BF_CNN(grayscale, training_data, training_noise): 
    '''
    @ grayscale: if True, number of input and output channels are set to 1. Otherwise 3
    @ training_data: models provided in here have been trained on {BSD400, mnist, BSD300}
    @ training_noise: standard deviation of noise during training the denoiser
    '''
    parser = argparse.ArgumentParser(description='BF_CNN_color')
    parser.add_argument('--dir_name', default= '../noise_range_')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 20)
    if grayscale is True: 
        parser.add_argument('--num_channels', default= 1)
    else:
        parser.add_argument('--num_channels', default= 3)
    
    args = parser.parse_args('')

    model = BF_CNN(args)
    if torch.cuda.is_available():
        model = model.cuda()
    model_path = os.path.join('denoisers/BF_CNN',training_data,training_noise,'model.pt')
    if torch.cuda.is_available():
        learned_params =torch.load(model_path)

    else:
        learned_params =torch.load(model_path, map_location='cpu' )
    model.load_state_dict(learned_params)
    return model

#################################################
def single_image_loader(data_set_dire_path, image_number):
    
    if 'mnist' in data_set_dire_path.split('/'): 
        f = gzip.open(data_set_dire_path + '/t10k-images-idx3-ubyte.gz','r')
        f.read(16)
        buf = f.read(28 * 28 *10000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(float)/255
        x = torch.tensor(data.reshape( 10000,28, 28).astype('float32'))[image_number:image_number+1]
        
    else: 
        all_names = os.listdir(data_set_dire_path)
        file_name = all_names[image_number]
        im = plt.imread(data_set_dire_path + file_name)
        if len(im.shape) == 3:
            x = torch.tensor(im).permute(2,0,1)
        elif len(im.shape) == 2:
            x = torch.tensor(im.reshape(1, im.shape[0], im.shape[1]))
            
    return x

class test_image: 
    def __init__(self, grayscale,path, image_num):
        super(test_image, self).__init__()
        
        self.grayscale = grayscale
        self.path = path
        self.image_num = image_num
        
        self.im = single_image_loader(self.path,self.image_num)
        if self.im.dtype == torch.uint8: 
            self.im = self.im/255
        if self.im.size()[0] == 3 and grayscale==True: 
            raise Exception('model is trained for grayscale images. Load a grayscale image')
        elif self.im.size()[0] == 1 and grayscale==False: 
            raise Exception('model is trained for color images. Load a color image')
        if torch.cuda.is_available():
            self.im = self.im.cuda()
        
    def show(self):
        if self.grayscale is True: 
            if torch.cuda.is_available():
                plt.imshow(self.im.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
            else: 
                plt.imshow(self.im.squeeze(0), 'gray', vmin=0, vmax = 1)                
        else: 
            if torch.cuda.is_available():
                plt.imshow(self.im.permute(1,2,0).cpu(), vmin=0, vmax = 1)
            else: 
                plt.imshow(self.im.permute(1,2,0), vmin=0, vmax = 1)

        plt.title('test image')
        plt.colorbar()
#         plt.axis('off');

    def crop(self, x0,y0,h,w):
        self.cropped_im = self.im[:, x0:x0+h, y0:y0+w]             
        if self.grayscale is True: 
            if torch.cuda.is_available():
                plt.imshow(self.cropped_im.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
            else: 
                plt.imshow(self.cropped_im.squeeze(0), 'gray', vmin=0, vmax = 1)
                
        else: 
            if torch.cuda.is_available():            
                plt.imshow(self.cropped_im.permute(1,2,0).cpu(), vmin=0, vmax = 1)
            else: 
                plt.imshow(self.cropped_im.permute(1,2,0), vmin=0, vmax = 1)

        plt.title('cropped test image')
        plt.colorbar()
#         plt.axis('off');        
        return self.cropped_im


#################################################
def rescale_image(im):
    if type(im) == torch.Tensor: 
        im = im.numpy()
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')


def plot_synthesis(intermed_Ys, sample):
    f, axs = plt.subplots(1,len(intermed_Ys), figsize = ( 4*len(intermed_Ys),4))
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()
            
        x = intermed_Ys[ax].permute(1,2,0).detach().numpy() 
        if x.shape[2] == 1: # if grayscale
            fig = axs[ax].imshow(x.squeeze(-1), 'gray')
        else: # if color
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')

    #### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()
        
    sample = sample.permute(1,2,0).detach().numpy()
    if sample.shape[2] == 1: # if grayscale
        fig = axs[-1].imshow(sample.squeeze(-1),'gray' )
    else: # if color
        fig = axs[-1].imshow(rescale_image(sample))

    axs[-1].axis('off')
    print('value range', np.round(np.min(sample ),2), np.round(np.max(sample),2) )


def plot_sample(x, corrupted, sample):
    if torch.cuda.is_available():
        x = x.cpu()
        corrupted = corrupted.cpu()
        sample = sample.cpu()
        
    x = x.permute(1,2,0)
    corrupted = corrupted.permute(1,2,0)
    sample = sample.detach().permute(1,2,0)
        
    if x.size()!=corrupted.size():    
        h_diff = x.size()[0] - corrupted.size()[0]
        w_diff = x.size()[1] - corrupted.size()[1]
        x = x[0:x.size()[0]-h_diff,0:x.size()[1]-w_diff,: ]
        print('WARNING: psnr and ssim calculated using a cropped original image, because the original image is not divisible by the downsampling scale factor.')
        
    f, axs = plt.subplots(1,3, figsize = (15,5))
    axs = axs.ravel()        
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
        
        fig = axs[1].imshow( torch.clip(corrupted,0,1), vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.numpy(), corrupted.numpy(), multichannel=True  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy() ))
        axs[1].set_title('corrupted image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );  
        
        fig = axs[2].imshow(torch.clip(sample, 0,1),vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True) ,3)
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy() ))   
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );
            
            
    for i in range(3): 
        axs[i].axis('off')
    
    


def plot_all_samples(sample, intermed_Ys):
    n_rows = int(np.ceil(len(intermed_Ys)/4))

    f, axs = plt.subplots(n_rows,4, figsize = ( 4*4, n_rows*4))
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()
            
        x = intermed_Ys[ax].detach().permute(1,2,0).numpy()
        if x.shape[2] == 1:
            fig = axs[ax].imshow(x.squeeze(-1), 'gray')
        else:
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')
    
    #### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()
        
    sample = sample.detach().permute(1,2,0).numpy()
    if sample.shape[2] == 1:
        fig = axs[-1].imshow(sample.squeeze(-1),'gray' )
    else:
        fig = axs[-1].imshow(rescale_image(sample))
    axs[-1].axis('off')
    plt.colorbar(fig, ax=axs[-1], fraction=.05)


    for ax in range(len(intermed_Ys),n_rows*4 ):
        axs[ax].axis('off')


def plot_corrupted_im(x_c): 
    try:

        if torch.cuda.is_available():
            plt.imshow(x_c.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
        else: 
            plt.imshow(x_c.squeeze(0), 'gray', vmin=0, vmax = 1)
    except TypeError: 
        if torch.cuda.is_available():
            plt.imshow(x_c.permute(1,2,0).cpu(), vmin=0, vmax = 1)
        else: 
            plt.imshow(x_c.permute(1,2,0) , vmin=0, vmax = 1)

    plt.colorbar()    
    
    
###################################### Inverse problems Tasks ##################################
#############################################################################################
class synthesis:
    def __init__(self):
        super(synthesis, self).__init__()

    def M_T(self, x):
        return torch.zeros_like(x)

    def M(self, x):
        return torch.zeros_like(x)

class inpainting:
    '''
    makes a blanked area in the center
    @x_size : image size, tuple of (n_ch, im_d1,im_d2)
    @x0,y0: center of the blanked area
    @w: width of the blanked area
    @h: height of the blanked area
    '''
    def __init__(self, x_size,x0,y0,h, w):
        super(inpainting, self).__init__()

        n_ch , im_d1, im_d2 = x_size
        self.mask = torch.ones(x_size)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        c1, c2 = int(x0), int(y0)
        h , w= int(h/2), int(w/2)
        self.mask[0:n_ch, c1-h : c1+h , c2-w:c2+w] = 0

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask

    
class rand_pixels:
    '''
    @x_size : tuple of (n_ch, im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(rand_pixels, self).__init__()

        self.mask = torch.tensor(np.random.binomial(n=1, p=p, size = x_size ))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask

    
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


#### important: when using fftn from torch the reconstruction is more lossy than when fft2 from numpy
#### the difference between reconstruction and clean image in pytorch is of order of e-8, but in numpy is e-16

class spectral_super_resolution:
    '''
    creates a mask for dropping high frequency coefficients
    @im_d: dimension of the input image is (im_d, im_d)
    @p: portion of coefficients to keep
    '''
    def __init__(self, x_size, p):
        super(spectral_super_resolution, self).__init__()

        self.x_size = x_size
        f1 = int(x_size[1]*p/2)
        f2 = int(x_size[2]*p/2)
        mask = torch.ones((x_size[1],x_size[2]))
        mask[f1+1:x_size[1]-f1,:]=0
        mask[:, f2+1:x_size[2]-f2]=0
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def M_T(self, x):
        return self.mask*torch.fft.fftn(x, norm= 'ortho', s = (self.x_size[1], self.x_size[2]) )

    def M(self, x):
        return torch.real(torch.fft.ifftn(x, norm= 'ortho', s = (self.x_size[1], self.x_size[2]) ))










