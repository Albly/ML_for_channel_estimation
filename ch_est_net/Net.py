import torch
from torch.nn import Module
from ch_est_net.activation import sigmoid
import ch_est_net.Net
from torch import nn
from ch_est_net.utils import DFT_matrixes, MM


class Net_Layer(torch.nn.Module):
    
    def __init__(self, cfg, freq_samples, time_samples, Trainable = True, isPass = False):
        super(Net_Layer,self).__init__()

        self.S1 = torch.nn.Parameter(torch.tensor([cfg.S1]), requires_grad = False)
        self.S2 = torch.nn.Parameter(torch.tensor([cfg.S2]), requires_grad = False)
        
        self.DFT_re, self.DFT_im, self.IDFT_re, self.IDFT_im = DFT_matrixes(freq_samples, time_samples, True)

        self.isPass = isPass
        self.cfg = cfg


    def setTrainable(self, value: bool):
        if value == True: 
            self.S1.requires_grad = True
            self.S2.requires_grad = True
        
        else:
            self.S1.requires_grad = False
            self.S1.grad = None
            self.S2.requires_grad = False
            self.S2.grad = None
    

    def forward(self, u_re, u_im, z_re, z_im, H_hat_re, H_hat_im):
        if self.isPass == False:

            Z_re, Z_im = MM(z_re, z_im, self.IDFT_re , self.IDFT_im)

            R_re = H_hat_re + Z_re
            R_im = H_hat_im + Z_im

            R_mean = torch.sqrt(torch.mean((R_re**2 + R_im**2), dim = 0))

            H_hat_mean = sigmoid(R_mean, self.S1, self.S2) 

            H_hat_re = R_re * H_hat_mean
            H_hat_im = R_im * H_hat_mean

            h_hat_re, h_hat_im = MM(H_hat_re, H_hat_im, self.DFT_re, self.DFT_im)

            z_re = u_re - h_hat_re
            z_im = u_im - h_hat_im

        return z_re, z_im, H_hat_re, H_hat_im


class Net(torch.nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.network = nn.ModuleList([Net_Layer(cfg,48,512) for i in range(cfg.layers)])

        print("Created Net with ", cfg.layers, "layers")

        self.DFT_re, self.DFT_im = DFT_matrixes(48,512)


    def forward(self, u):
        H_hat_re = torch.zeros([64,512])
        H_hat_im = torch.zeros([64,512])

        u, maximum = self.normalize(u)

        u_re = u[:,:,0]
        u_im = u[:,:,1] 

        z_re = u_re
        z_im = u_im

        for layer in self.network:
            z_re, z_im, H_hat_re, H_hat_im = layer(u_re, u_im, z_re, z_im, H_hat_re, H_hat_im)
        
        
        h_hat_re, h_hat_im = MM(H_hat_re, H_hat_im, self.DFT_re, self.DFT_im) 

        out = torch.stack((h_hat_re, h_hat_im), dim = 2)

        out = self.denormalize(out, maximum)
        return out



    def normalize(self, u):
        maximum = abs(u.max())
        u = 5*u/maximum
        return u, maximum



    def denormalize(self, u, maximum):
        u = u/5*maximum
        return u 

 
    def setState(self, trainable_code, pass_code):

        c = [bool(int(d)) for d in str(trainable_code)]
        p = [bool(int(d)) for d in str(pass_code)]

        for i, module in enumerate(self.network):
            module.setTrainable(c[i])
            module.isPass = p[i]

