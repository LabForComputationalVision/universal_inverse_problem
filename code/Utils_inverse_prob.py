import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import os
import time
from PIL import Image
from math import  inf
from skimage import color
import scipy.fftpack as spfft
import pywt

#########################################load Functions #########################################
def mean_squared_error(ref, A):
    return (( ref - A )**2).mean(axis=None)


def PSNR(A, ref):
    if mean_squared_error(ref, A) == 0:
        return inf
    else:
        return 10* np.log10(1 /mean_squared_error(ref, A))


def single_image_loader(data_set_dire_path, image_number):
    all_names = os.listdir(data_set_dire_path)
    file_name = all_names[image_number]
    im = plt.imread(data_set_dire_path + file_name).astype(float)
    return im

# rewrite the load function into more general one
# def load_Berkley_dataset( train_folder_path, test_folder_path,set12_path):
#     image_dict = {};
#     # read and prep train images
#     train_images = []
#     train_names = os.listdir(train_folder_path)
#     test_names = os.listdir(test_folder_path)

#     for file_name in train_names:
#         train_images.append(io.imread(train_folder_path + file_name).astype(float)/255 );

#     # read and prep test images
#     test_images = []
#     for file_name in test_names:
#         image = io.imread(test_folder_path + file_name).astype(float)
#         if image.shape[0] > image.shape[1]:
#             image = image.T
#         test_images.append(image/255 );

#     #read and prep set12
#     images_set12 = []
#     set12_names = os.listdir(set12_path)

#     for file_name in set12_names:
#         images_set12.append(io.imread(set12_path + file_name).astype(float)/255 );

#     image_dict['train'] = np.array(train_images)
#     image_dict['test'] = np.array(test_images)
#     image_dict['set12'] = np.array(images_set12)

#     return image_dict

def load_Berkley_dataset( train_folder_path, test_folder_path,set12_path):
    image_dict = {};
    # read and prep train images
    train_images = []
    train_names = os.listdir(train_folder_path)
    test_names = os.listdir(test_folder_path)

    for file_name in train_names:
        train_images.append(plt.imread(train_folder_path + file_name).astype(float) );

    # read and prep test images
    test_images = []
    for file_name in test_names:
        image = plt.imread(test_folder_path + file_name).astype(float)
        if image.shape[0] > image.shape[1]:
            image = image.T
        test_images.append(image );

    #read and prep set12
    images_set12 = []
    set12_names = os.listdir(set12_path)

    for file_name in set12_names:
        images_set12.append(plt.imread(set12_path + file_name).astype(float) );

    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)
    image_dict['set12'] = np.array(images_set12)

    return image_dict

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_set14( path):
    test_names = os.listdir(path)

    # read and prep test images
    test_images = []
    for file_name in test_names:
        image = plt.imread(path + file_name).astype(float)
        image = image
        image = color.rgb2gray(image)
        # image = rgb2gray(image)
        test_images.append(image );

    return np.array(test_images)


#############################################################################################

# fix for non-square images
def mask_centrosymm(im_d = 41 ,p=.5):
    '''
    im_d: image size is (im_d, im_d)
    p: dropping factor
    returns a mask of size (im_d,im_d) for dropping fourier coefficents ina centro-symmetric way
    '''
    if im_d % 2 == 0 :
        raise Exception('need odd dim')

    b = np.random.randint(0,im_d, size = (int((p/2)*((im_d**2)-1)),2))
    b = np.concatenate([b, im_d-1-b])
    b = np.unique(b, axis=0)
    uniq_shape = b.shape[0]

    # remove the center
    b = b.tolist()
    if [int(im_d/2),int(im_d/2)] in b:
        b.remove([int(im_d/2),int(im_d/2)])
    b = np.array(b)


    while uniq_shape < int(p*((im_d**2)-1)):
        idx_needed = int((int(p*((im_d**2)-1)) - uniq_shape)/2 )
        new_idx = np.random.randint(0,im_d, size = (idx_needed,2))
        b = np.concatenate([b, new_idx, im_d-1-new_idx ])
        b = np.unique(b, axis=0)
        uniq_shape = b.shape[0]
        # remove the center
        b = b.tolist()
        if [int(im_d/2),int(im_d/2)] in b:
            b.remove([int(im_d/2),int(im_d/2)])
        b = np.array(b)


    mask = np.zeros([im_d,im_d])
    for i, j in zip (b[:,0], b[:,1]):
        mask[i,j] = 1

    # set the center to one to keep the DC term
    mask[int(im_d/2),int(im_d/2)] = 1
    return mask

###################################### Inverse problems Tasks ##################################
#############################################################################################
class synthesis:
    def __init__(self):
        super(synthesis, self).__init__()

    def M_T(self, x):
        return np.zeros_like(x)

    def M(self, x):
        return np.zeros_like(x)
#############################################################################################
class inpainting:
    '''
    makes a 20x20 blanked area in the center
    @x_size : tuple of (im_d1,im_d2)
    '''
    def __init__(self, x_size):
        super(inpainting, self).__init__()
        im_d1, im_d2 = x_size
        self.mask = np.ones(x_size)
        self.mask[int(im_d1/2)-10:int(im_d1/2)+10,int(im_d1/2)-10:int(im_d1/2)+10] = 0

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask

#############################################################################################
class rand_pixels:
    '''
    @x_size : tuple of (im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(rand_pixels, self).__init__()

        self.mask = np.random.binomial(n=1, p=p, size = x_size )

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
    @x_size: tuple of two int
    '''

    def __init__(self, x_size, s):
        super(super_resolution, self).__init__()

        Mat_T = []
        im_d1, im_d2 = x_size

        if im_d1%2 !=0 or im_d2%2 != 0 :
            raise Exception("image dimensions need to be even")

        for j in range(0,im_d1*im_d2, im_d1 *s):
            for i in range(0,im_d1,s):
                row = np.zeros(im_d1*im_d2)
                for k in range(s):
                    row[j+i+k*im_d1 : j+i+k*im_d1+s] = 1
                Mat_T.append(row/np.linalg.norm(row))

        self.down_sampling = np.array(Mat_T).T
        self.x_size = x_size

    def M_T(self, x):
        return np.dot(self.down_sampling.T , x.flatten() )

    def M(self, x):
        return np.dot(self.down_sampling,x.flatten()).reshape(self.x_size[0], self.x_size[1])




#############################################################################################
class spectral_super_resolution:
    '''
    creates a mask for dropping either low frequency fourier coeffs or high frequenc-y dropping
    @low_pass: True if want to blur, i.e. keep the lower frequencies.
    @im_d: dimension of the input image is (im_d, im_d)
    @filt_size: size of the filter
    '''
    def __init__(self, low_pass, im_d, filt_size):
        super(spectral_super_resolution, self).__init__()
        if im_d % 2 == 0 :
            raise Exception('need odd dim')

        if low_pass is True:
            mask = np.ones((im_d,im_d))
            mask[int(filt_size/2)+1:im_d-int(filt_size/2),:]=0
            mask[:, int(filt_size/2)+1:im_d-int(filt_size/2)]=0

        else:
            mask = np.ones((im_d,im_d))
            mask[0:int(filt_size/2)+1, 0:int(filt_size/2)+1]=0
            mask[im_d-int(filt_size/2)::, im_d-int(filt_size/2)::]=0
            mask[0:int(filt_size/2)+1, im_d-int(filt_size/2)::]=0
            mask[im_d-int(filt_size/2)::, 0:int(filt_size/2)+1]=0


        self.mask = mask


    def M_T(self, x):
        return self.mask*np.fft.fft2(x)

    def M(self, x):
        return np.real(np.fft.ifft2(x))



#############################################################################################

class random_basis:
    '''
    @x_size : tuple of (im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(random_basis, self).__init__()
        im_d1, im_d2 = x_size
        self.x_size = x_size
        self.U, _ = np.linalg.qr(np.random.randn(int(im_d1*im_d2),int(im_d1*im_d2*p) ))

    def M_T(self, x):
        return np.dot(self.U.T,x.flatten())

    def M(self, x):
        return np.dot(self.U,x).reshape(self.x_size[0], self.x_size[1])





########################### funtion needed to perform compressive sensing

def dct2_matrix(dim):
    '''dim: matrix dimension. an integer, because matrix is square
    returns: each sinosuid sits in a row
       '''

    A = []
    temp2 = np.zeros((dim,dim))

    for i in range(temp2.shape[0]):
        for j in range(temp2.shape[1]):
            temp2[i,j] = 1
            A.append(spfft.idctn(temp2, norm='ortho').flatten())
            temp2[i,j] = 0
    return np.array(A)



def wavelet_matrix(dim, wave_name):
    '''dim: matrix dimension. an integer, because matrix is square
       wave_name: string. wavelet name
       returns: each wavelet sits in a row
    '''

    A = []
    temp2 = pywt.wavedec2(np.zeros((dim,dim)), wave_name)

    for i in range(temp2[0].shape[0]):
        for j in range(temp2[0].shape[1]):
            temp2[0][i,j] = 1
            A.append(pywt.waverec2(temp2, wave_name).flatten())
            temp2[0][i,j] = 0


    for level in range(1,len(temp2)):
        for orient in range(len(temp2[level])):
            for i in range(temp2[level][orient].shape[0]):
                for j in range(temp2[level][orient].shape[1]):
                    temp2[level][orient][i,j] = 1
                    A.append(pywt.waverec2(temp2, wave_name).flatten())
                    temp2[level][orient][i,j] = 0
    return np.array(A)


