import scipy
from scipy.linalg import dft
import torch
import math
from ch_est_net.preloading import *

def DFT_matrixes(freq,time, return_IDFT = False):
    shape = max(freq,time)
    coeff = torch.sqrt(torch.tensor(shape, dtype = torch.float))
    DFT = torch.tensor(dft(shape), dtype = torch.cfloat)/coeff
    
    DFT_re = torch.real(DFT)
    DFT_im = torch.imag(DFT)
    DFT_conj = torch.conj(DFT)
    IDFT_re = torch.real(DFT_conj)
    IDFT_im = torch.imag(DFT_conj)
    if return_IDFT:
        return DFT_re[:time,:freq], DFT_im[:time,:freq], IDFT_re[:freq,:time], IDFT_im[:freq,:time]
    return DFT_re[:time,:freq], DFT_im[:time,:freq]

def get_DFT(size, is_complex = True):
    
    
    DFT = scipy.linalg.dft(size)/math.sqrt(size)
    IDFT = DFT.conj()

    DFT = torch.tensor(DFT, dtype= torch.cfloat)
    IDFT = torch.tensor(IDFT, dtype= torch.cfloat)

    if is_complex:
        return DFT, IDFT
    else:
        DFT_re = torch.real(DFT)
        DFT_im = torch.imag(DFT)
        IDFT_re = torch.real(IDFT)
        IDFT_im = torch.imag(IDFT)
    
    return DFT_re, DFT_im, IDFT_re, IDFT_im


def MM(N_re, N_im, M_re, M_im):
    ''' 
    Matrix multiplication for Complex matrices 
    '''
    re = torch.sub(torch.mm(N_re, M_re) , torch.mm(N_im, M_im))
    im = torch.add(torch.mm(N_re, M_im) , torch.mm(N_im, M_re))
    return re, im 

def shape_to_complex(X):
    assert len(X.shape) == 3
    return X[:,:,0] + 1j*X[:,:,1]

def complex_to_shape(X):
    assert len(X.shape) == 2
    return torch.stack((X.real, X.imag), dim = 2)


def get_detector_error(method, is_complex, dtype, onePilotFolder,dataL, ml, 
            lossVersion='detector', # 'detector' or 'relError'
            inds = range(1,141), 
            SNR_L = range(-10,-1), 
            seed = 4, 
            max_iter = 3, 
            ml_version = 12, 
            SNRscaleFactor = 1.,
            scen = None, ## <- Shouldn't be none 
            scale = True):
    assert lossVersion in ['detector', 'relError']

    N_used = scen.RB_num*scen.RB_size
    loss = []
    comb = scen.comb

    z = torch.zeros(64, 512, 2, requires_grad = False)
    h_hat = torch.zeros(64, 512, 2, requires_grad = False) 

    losses = []

    if lossVersion == 'detector':
        N_pilot_sym = scen.N_pilot*scen.N_TTI
        N_data_sym = (14-N_pilot_sym)*scen.N_TTI;
        for SNR in SNR_L:
            loss_current = 0
            for ind in inds:
                h_pilot, h_data = data_load(scen, dtype = dtype, onePilotFolder = onePilotFolder,dataL = dataL, ind = ind+1 ,use_preloaded = False)
                h_pilot_noisy, _ = add_noise(h_pilot, SNR, scen = scen, dtype = dtype, seed = seed)
                h_data_noisy, data_noise_power = add_noise_data(h_data, SNR, dtype= dtype, seed = seed) 

                u = h_pilot_noisy.mean(dim=2)

                # if mehtod works with complex values - transfrom dims to complex numbers                
                if is_complex:
                    u = torch.tensor(u[:,:,0] + 1j*u[:,:,1], dtype = torch.complex64)
                
                # CHANNEL ESTIMATION
                # working method 
                h_pilot_rec = method(u)

                # if methods works with complex numbers, transform the to dims 
                if is_complex:
                    h_pilot_rec = complex_to_shape(h_pilot_rec)
                

                # Show time domain recovered and initial signals
                # if ind % 70==-1:
                #     h_pilot_rec_numpy = upsampling(scen, h_pilot_rec, inverse=False).detach().numpy()
                #     plt.plot(h_pilot_rec_numpy[0,:,0]**2+h_pilot_rec_numpy[0,:,1]**2)
                #     h_f = h_pilot
                #     if len(h_f.shape) == 4:
                #         h_f = h_f.mean(dim=2)
                #     h_f = upsampling(scen, h_f, inverse=False).detach().numpy()
                #     plt.plot(h_f[0,:,0]**2+h_f[0,:,1]**2)
                #     plt.show() 
                #            
                assert h_pilot_rec.shape[1] == N_used
                H_re = h_pilot_rec[:, :, 0]
                H_im = -h_pilot_rec[:, :, 1]
                
                # Detector error calculation
                err_data = 0
                for k in range (N_data_sym):    
                    det_data = torch.zeros((N_used, 2))   
                    assert h_data_noisy.shape[1] == N_used 
                    Y = h_data_noisy[:, :, k, :]                    
                    det_data[:,0] = (torch.sum(Y[:,:,0]*H_re-Y[:,:,1]*H_im, dim=0)/
                                    (data_noise_power+torch.sum(H_re**2+H_im**2, dim=0)))
                    det_data[:,1] = (torch.sum(Y[:,:,1]*H_re+Y[:,:,0]*H_im, dim=0)/
                                    (data_noise_power+torch.sum(H_re**2+H_im**2, dim=0)))
                      
                    err = det_data - torch.Tensor([1.,0.])  
                    err_data = err_data+torch.sum(err**2)
    
                loss_current += err_data/(N_data_sym*N_used)

                cur = {'loss': err_data.detach().numpy() ,'SNR': SNR, 'file': ind}
                losses.append(cur)

            loss.append(loss_current)        
            
    return loss, losses