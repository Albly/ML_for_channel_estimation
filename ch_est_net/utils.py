import scipy
from scipy.linalg import dft
import torch
import math

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
