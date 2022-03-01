from turtle import forward
import torch
import numpy as np
from abc import ABC, abstractmethod


#================================================================
class Activation(ABC):
    @property
    def num_learn_vars(self):
        raise NotImplementedError


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def get_default_values():
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


    def derivative(self, x, *args, **kwargs):
        return self.autograd_torch(self.forward)(x,*args,**kwargs)
        
    
    def autograd_torch(self, f):
        def result(x, *args, **kwargs):
            with torch.enable_grad():
                x_ = x.detach().requires_grad_(True) # make leaf variables out of the inputs

                list_args = []
                dict_kwargs = {}

                for arg in args:
                    if torch.is_tensor(arg):
                        arg = arg.detach().requires_grad_(True)
                    list_args.append(arg)
            
                for key, value in kwargs.items():
                    if torch.is_tensor(value):
                        value = value.detach().requires_grad_(True)    
                    dict_kwargs[key] = value
                
                f(x_,*list_args, **dict_kwargs).sum().backward()
                return x_.grad
        return result

#================================================================

class Sigmoid_avg(Activation):
    num_learn_vars = 2

    def get_default_values(self):
        S1 = 5.0
        S2 = 10.0
        return S1,S2

    def forward(self, r, sigma, S1, S2):
        assert len(r.shape) == 2, 'Should be 2D matrix'
        N, Rx = r.shape

        power_vec = torch.sqrt(torch.mean(abs(r)**2, dim = 1)) 
        #R_mean = torch.sqrt(torch.mean((R_re**2 + R_im**2), dim=0))
        one = torch.tensor([1.0])
        mask_vec = one/(one+torch.exp(-S1*power_vec+S2))
        mask_vec = torch.reshape(mask_vec, (N, 1))
        mask_mat = mask_vec.expand((N, Rx))

        #if r.is_complex():
        #    mask_mat = mask_mat + 1j*mask_mat
    
        return r*mask_mat
    


class Sigmoid(Activation):
    num_learn_vars = 2

    def get_default_values(self):
        S1 = 5.0
        S2 = 10.0
        return S1, S2

    def forward(self, r, sigma, S1, S2):
        assert len(r.shape) == 2, 'Should be 2D matrix'
        one = torch.tensor([1.0])
        mask_mat = one/(one+torch.exp(-S1*r+S2))
        return r*mask_mat
    


class Scaled_soft_threshold(Activation):
    num_learn_vars = 2

    def get_default_values(self):
        alpha = 0.1
        beta = 1.0

        return alpha, beta

    def forward(self, r, sigma, alpha, beta):
        
        zero = torch.zeros(r.shape)
        sgn = torch.sgn(r)
        res = beta*sgn*torch.maximum(torch.abs(r) - alpha*sigma, zero)
        return res
    
    #def derivative(self, output, beta):
        # @output - output of forward 
    #    return beta*torch.linalg.vector_norm(output, ord = 0)



class Pwlin(Activation):
    num_learn_vars = 5

    def get_default_values(self):
        theta1 = 2.0
        theta2 = 4.0
        theta3 = 0.1
        theta4 = 1.5
        theta5 = 0.95
        return theta1, theta2, theta3, theta4, theta5

    def forward(self,r, sigma, theta_0,theta_1,theta_2,theta_3,theta_4):
        scale_out = sigma**0.5
        scale_in = 1/scale_out

        rs = torch.sgn(r*scale_in)
        ra = torch.abs(r*scale_in)

        rgn0 = (ra<theta_0).type(torch.float32)
        rgn1 = (ra<theta_1).type(torch.float32) - rgn0
        rgn2 = (ra>=theta_1).type(torch.float32)

        xhat = scale_out * rs * (
            rgn0*theta_2*ra+
            rgn1*(theta_3*(ra-theta_0) + theta_2*theta_0)+
            rgn2*(theta_4*(ra-theta_1)+ theta_2*theta_0 + theta_3*(theta_1-theta_0))
        )

    #    self.dxdr = theta_2*rgn0 + theta_3*rgn1 + theta_4*rgn2

        return xhat 
    
    #def derivative(self):
    #    return self.dxdr




class complex_soft_threshold():
    def function(r,lambd, type = 'soft'):
        if type != 'soft': 
            raise Exception('No {} type for threshold function'.format(type))          #TODO: add hard-threshold

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       #TODO: add possibility of GPU

        zero = torch.zeros(r.shape, device = "cpu")         # zero vector with same shape as input
        sgn = torch.sgn(r)                  # sgn of vector. Equivalent to sign if v - real

        return sgn*torch.maximum(torch.abs(r) - lambd , zero) 

class exponential():
    def function(r,sigma, theta):
        return theta[1]*r + theta[2]*r*torch.exp(-(r**2)/(2*theta[0]**2*sigma**2))
    
    def derivative(r, sigma,theta):

        theta_sigma = theta[0]**2 * sigma**2 
        exponent = torch.exp(-(r**2)/(2*theta_sigma))
        mult = theta_sigma - r**2

        return theta[2]*exponent*mult/theta_sigma + theta[1] 



def sigmoid(r, S1, S2, recived):
    one = torch.tensor([1])
    res = one/(1+torch.exp(-S1*r+S2))
    res = torch.reshape(res, (1, 512))
    res = res.expand((recived, 512))
    return res


def arctan(r, S1, S2):
    pi = torch.tensor(np.pi)
    one_half = torch.tensor([0.5])
    res = torch.atan(S1*(r+S2))/pi + one_half
    res = torch.reshape(res, (1, 512))
    res = res.expand((64, 512))

    return res


def tanh(r, S1, S2):
    one_half = torch.tensor([0.5])
    res = one_half * torch.tanh(S1*(r+S2)) + one_half
    res = torch.reshape(res, (1, 512))
    res = res.expand((64, 512))

    return res


def sqrt(r, S1, S2):
    one_half = torch.tensor([0.5])
    one = torch.tensor([1])
    res = one_half * (S1*(r+S2)) / torch.sqrt(one + (S1*(r+S2))**2) + one_half
    res = torch.reshape(res, (1, 512))
    res = res.expand((64, 512))

    return res


def rat_sigmoid(r, S1, S2):
    one_half = torch.tensor([0.5])
    one = torch.tensor([1])
    res = one_half * S1*(r+S2) / (abs(S1*(r+S2))+1) + one_half
    res = torch.reshape(res, (1, 512))
    res = res.expand((64, 512))

    return res
