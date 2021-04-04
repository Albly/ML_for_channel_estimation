
import torch
from torch import nn

# def MSE_loss(h_rec, h_ideal):
#     loss = torch.nn.MSELoss()
#     return loss(h_rec[:,:48, :], h_ideal)


def MSE_detector_loss(h_rec, h_data_noisy, data_noise_power, scen0):
    
    H_re = h_rec[:,:48,0].clone()
    H_im = -h_rec[:,:48,1].clone()

    err_data = 0

    N_pilot_sym = scen0.N_pilot*scen0.N_TTI
    N_data_sym = (14 - N_pilot_sym)*scen0.N_TTI
    N_used = scen0.RB_num * scen0.RB_size

    for k in range (N_data_sym):
        det_data = torch.zeros((N_used, 2))
        assert h_data_noisy.shape[1] == N_used
        Y = h_data_noisy[:,:,k,:]

        det_data[:,0] = (torch.sum(Y[:,:,0]*H_re-Y[:,:,1]*H_im, dim=0)/
                                    (data_noise_power+torch.sum(H_re**2+H_im**2, dim=0)))
        
        det_data[:,1] = (torch.sum(Y[:,:,1]*H_re+Y[:,:,0]*H_im, dim=0)/
                                    (data_noise_power+torch.sum(H_re**2+H_im**2, dim=0)))

        err = det_data - torch.Tensor([1.,0.])
        err_data = err_data + torch.sum(err**2)

    loss_current = err_data/(N_data_sym*N_used)

    return loss_current
