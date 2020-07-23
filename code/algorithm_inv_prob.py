import numpy as np
import os
import time
import torch



def univ_inv_sol(model, x_c , M_T , M  ,sig_0, sig_L, h0 , beta , freq, save_interm):
    '''
    @x_c:  M^T.x)
    @M: a dxd binary mask in the transformed space
    @sig_0: init sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    '''

    (im_d1,im_d2) = M(x_c).shape
    intermed_Ys=[]


    if np.mean(M(x_c)) == 0 :
        y =  np.random.randn(im_d1,im_d2)*sig_0 + .5
    else:
        y =  np.random.randn(im_d1,im_d2)*sig_0 + np.mean(M(x_c))

    if save_interm is True:
        intermed_Ys.append(y)

    f_y = model(torch.FloatTensor(y).unsqueeze(0).unsqueeze(0))[0,0].data.numpy()
    sigma = np.linalg.norm(f_y)/np.sqrt(im_d1*im_d2)


    t=1
    start_time_total = time.time()
    while sigma > sig_L:


        h = h0*t/(1+ (h0*(t-1)) )

        f_y = model(torch.FloatTensor(y).unsqueeze(0).unsqueeze(0))[0,0].data.numpy()

        d = f_y - M(M_T(f_y)) + ( M(M_T(y))  - M(x_c) )


        sigma = np.linalg.norm(d)/np.sqrt(im_d1*im_d2)

        gamma = sigma*np.sqrt( (1 - (beta*h))**2 - (1-h)**2)

        y = y -  h*d + gamma*np.random.randn(im_d1,im_d2)

        if t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma )

            if save_interm is True:
                intermed_Ys.append(y)


        t +=1


    print("-------- total number of iterations, " , t )
    print("-------- average time per iteration (s), " , np.round((time.time() - start_time_total)/(t-1)  ,4) )

    denoised_y = y - model(torch.FloatTensor(y).unsqueeze(0).unsqueeze(0))[0,0].data.numpy()

    return denoised_y, intermed_Ys

