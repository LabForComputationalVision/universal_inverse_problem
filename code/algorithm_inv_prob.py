import numpy as np
import torch
import time
import os


### all in pytorch. Takes a tensor of size (n_ch, im_d1, im_d2)
### and returns a tensor of size (n_ch, im_d1, im_d2)
def univ_inv_sol(model, x_c , M_T , M  ,sig_0, sig_L, h0 , beta , freq, save_interm):
    '''
    @x_c:  M^T.x)
    @M: low rank measurement matrix - in function form
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @freq: if save_interm is true, outputs will be stored with this frequency
    '''

    n_ch, im_d1,im_d2 = M(x_c).size()
    N = n_ch* im_d1*im_d2
    intermed_Ys=[]

    # initialize y
    e =  torch.ones_like(M(x_c), requires_grad= False )
    y = torch.normal((e - M(M_T(e)))*.5 + M(x_c), sig_0)
    y = y.unsqueeze(0)
    y.requires_grad = False

    if save_interm is True:
        intermed_Ys.append(y.squeeze(0))


    if torch.cuda.is_available():
        y = y.cuda()

    f_y = model(y)


    sigma = torch.norm(f_y)/np.sqrt(N)


    t=1
    start_time_total = time.time()
    while sigma > sig_L:


        h = h0*t/(1+ (h0*(t-1)) )
        with torch.no_grad():
            f_y = model(y)

        d = f_y - M(M_T(f_y[0])) + ( M(M_T(y[0]))  - M(x_c) )


        sigma = torch.norm(d)/np.sqrt(N)

        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))

        noise = torch.randn(n_ch, im_d1,im_d2)

        if torch.cuda.is_available():
            noise = noise.cuda()

        y = y -  h*d + gamma*noise

        if t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma.item() )

            if save_interm is True:
                intermed_Ys.append(y.squeeze(0))


        t +=1


    print("-------- total number of iterations, " , t )
    print("-------- average time per iteration (s), " , np.round((time.time() - start_time_total)/(t-1)  ,4) )

    denoised_y = y - model(y)



    return denoised_y.squeeze(0), intermed_Ys


