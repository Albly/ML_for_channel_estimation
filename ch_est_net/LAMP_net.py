import torch
from torch import nn
from torch._C import get_default_dtype
import ch_est_net.activation
from ch_est_net.utils import DFT_matrixes, MM, get_DFT


class BEAM_LAMP_layer(torch.nn.Module):
    def __init__(self, cfg, freq_samples, time_samples, act_class):
        super(BEAM_LAMP_layer, self).__init__()
        
        DFT_re, DFT_im, IDFT_re, IDFT_im = get_DFT(size = 512, is_complex = False)
        
        self.B_re = nn.Parameter(torch.tensor(DFT_re[:,:48], requires_grad = True))
        self.B_im = nn.Parameter(torch.tensor(DFT_im[:,:48], requires_grad = True))

        self.A_re = nn.Parameter(torch.tensor(IDFT_re[:,:48], requires_grad = False))
        self.A_im = nn.Parameter(torch.tensor(IDFT_im[:,:48], requires_grad = False))

        self.activation = getattr(ch_est_net.activation, 'exponential')
        self.theta = nn.Parameter(torch.tensor([1.0, 1.0, -1.0] , requires_grad= True))
        

    def forward(self, u_re, u_im, z_re, z_im, H_hat_re, H_hat_im):
        Np = torch.tensor([48.0])

        Z_re, Z_im = MM(z_re, z_im, self.B_re.T, self.B_im.T)

        R_re = H_hat_re + Z_re
        R_im = H_hat_im + Z_im

        sigma = torch.norm(torch.stack((z_re, z_im)), 2) / torch.sqrt(Np)

        H_hat_re = self.activation.function(R_re, sigma, self.theta)
        H_hat_im = self.activation.function(R_im, sigma, self.theta)

        #check derivative
        b_re = self.activation.derivative(R_re,sigma, self.theta).sum() / Np
        b_im = self.activation.derivative(R_im,sigma, self.theta).sum() / Np


        h_hat_re, h_hat_im = MM(
            H_hat_re, H_hat_im, self.A_re, self.A_im
        )

        z_re = u_re - h_hat_re + b_re*z_re
        z_im = u_im - h_hat_im + b_im*z_im


        return z_re, z_im, H_hat_re, H_hat_im







class LAMP_layer(torch.nn.Module):
    def __init__(self, cfg, freq_samples, time_samples, act_class):
        super(LAMP_layer, self).__init__()
    

        DFT_re, DFT_im = DFT_matrixes(freq_samples, time_samples)
        
        self.A_re = nn.Parameter(torch.tensor(DFT_re, requires_grad = True))
        self.A_im = nn.Parameter(torch.tensor(DFT_im, requires_grad = True))
        self.theta = nn.Parameter(torch.tensor([1.0, 1.0, -1.0] , requires_grad= True))

        self.activation = getattr(ch_est_net.activation, act_class)


    def forward(self, u_re, u_im, z_re, z_im, H_hat_re, H_hat_im):
        Np = torch.tensor([48.0])

        Z_re, Z_im = MM(z_re, z_im, self.A_re.T, self.A_im.T)

        R_re = H_hat_re + Z_re
        R_im = H_hat_im + Z_im

        sigma = torch.norm(torch.stack((z_re,z_im)), 2 ) / torch.sqrt(Np)

        H_hat_re = self.activation.function(R_re, sigma, self.theta)
        H_hat_im = self.activation.function(R_im, sigma, self.theta)

        b_re = self.activation.derivative(R_re,sigma, self.theta).sum() / Np
        b_im = self.activation.derivative(R_im,sigma, self.theta).sum() / Np

        h_hat_re, h_hat_im = MM(
            H_hat_re, H_hat_im, self.A_re, self.A_im
        )

        z_re = u_re - h_hat_re + b_re*z_re
        z_im = u_im - h_hat_im + b_im*z_im


        return z_re, z_im, H_hat_re, H_hat_im

class LAMP(nn.Module):
    def __init__(self,cfg):
        super(LAMP,self).__init__()

        self.net = nn.ModuleList(
            [LAMP_layer(cfg, 48, 512, cfg.activation) for i in range(cfg.layers)]
        )
        print("Created LAMP with ", cfg.layers, "layers")
        self.DFT_re, self.DFT_im = DFT_matrixes(48, 512)
        self.recived = cfg.recived
        self.norm = cfg.normalize

    def forward(self, u):
        H_hat_re = torch.zeros([self.recived, 512])
        H_hat_im = torch.zeros([self.recived, 512])

        if self.norm:
            u, maximum = self.normalize(u)

        u_re = u[:, :, 0]
        u_im = u[:, :, 1]

        z_re = u_re
        z_im = u_im

        for layer in self.net:
            z_re, z_im, H_hat_re, H_hat_im = layer(
                u_re, u_im, z_re, z_im, H_hat_re, H_hat_im)

        h_hat_re, h_hat_im = MM(H_hat_re, H_hat_im, self.DFT_re, self.DFT_im)

        out = torch.stack((h_hat_re, h_hat_im), dim=2)
        
        if self.norm:
            out = self.denormalize(out, maximum)
        return out


    def normalize(self, u):
        maximum = abs(u.max())
        u = 5*u/maximum
        return u, maximum


    def denormalize(self, u, maximum):
        u = u/5*maximum
        return u