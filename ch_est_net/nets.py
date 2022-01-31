from turtle import forward, shape
from sympy import false, true
import ch_est_net.activation
from torch import linalg
from torch import nn
import torch

def threshold(v, lambd, type = 'soft'):
    '''
    Thresholding activation function generalized to complex values. 
    if @type is 'soft' applies SoftThresholding function
    
    About functions
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.1124&rep=rep1&type=pdf
    '''

    if type != 'soft': 
        raise Exception('No {} type for threshold function'.format(type))

    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    zero = torch.zeros(v.shape, device = device)         # zero vector with same shape as input
    sgn = torch.sgn(v)                  # sgn of vector. Equivalent to sign if v - real

    return sgn*torch.maximum(torch.abs(v) - lambd , zero) 



class Lista(torch.nn.Module):
    def __init__(self, A, layers, beta = 1.0) -> None:
        super(Lista, self).__init__()
        INIT_LAMBDA = 0.01

        M,N = A.shape[0], A.shape[1]
        In = torch.eye(N)
        self.A = A
        B = beta * torch.conj(A).T

        self.B = torch.nn.Parameter(B, requires_grad = True)
        self.S = torch.nn.Parameter(In-B@A, requires_grad = True)

        self.lambdas = torch.nn.Parameter(torch.ones(layers+1)*INIT_LAMBDA, requires_grad = True)      
        self.layers = layers
        
    def forward(self, y):
        By = self.B @ y
        x_hat = threshold(By, lambd = self.lambdas[0], type= 'soft')

        for layer in range(self.layers):
            r = By + self.S @ x_hat
            x_hat = threshold(r, lambd=self.lambdas[layer+1])
        
        return (self.A@x_hat).T




class LAMP_layer(torch.nn.Module):
    def __init__(self, A, B, alpha, beta, activation_class) -> None:
        super(LAMP_layer,self).__init__()

        # NON TRAINABLE
        self.A = A
        self.M, self.N = A.shape
        self.activation = getattr(ch_est_net.activation, activation_class)
        
        # TRAINABLE PARAMETERS
        self.beta = nn.Parameter(beta, requires_grad = True)
        self.alpha = nn.Parameter(alpha, requires_grad = True)
        self.B = nn.Parameter(A.conj().T, requires_grad = True)

        self._isPass = False
        
    def setTrainable(self, state: bool):
        if state == True:
            self.beta.requires_grad = True
            self.alpha.requires_grad = True
            self.B.requires_grad = True
        
        else:
            self.beta.requires_grad = False
            self.beta.grad = None
            self.alpha.requires_grad = False
            self.alpha.grad = None
            self.B.requires_grad = False 
            self.B.grad = None

    def forward(self, x_hat, v, y):
        if self._isPass == False:
            r = x_hat + self.B @ v
            
            sigma = 1/torch.sqrt(self.M) * torch.linalg.vector_norm(v, ord = 2)
            lambd = self.alpha/torch.sqrt(self.M) * sigma

            x_hat = self.beta * self.activation.function(r,lambd)
            v = y - self.A @ x_hat  + 1/self.M * self.activation.derivative(r, lambd)

        return x_hat, v

class Tied_LAMP(torch.nn.Module):
    def __init__(self, A, layers: int, activation_class = 'complex_soft_threshold'):
        super(Tied_LAMP, self).__init__()

        self.activation = getattr(ch_est_net.activation, activation_class)
        self.layers = layers
        self.M, self.N = torch.tensor(A.shape[0]), torch.tensor(A.shape[1])
        self.A = A

        assert self.M <= self.N, 'M assumed to be <= N'

        self.alphas = nn.Parameter(torch.ones(layers, dtype = torch.float64), requires_grad = True)
        self.betas = nn.Parameter(torch.ones(layers, dtype = torch.float64), requires_grad = True)
        self.B = nn.Parameter(A.conj().T, requires_grad = True)


    def forward(self, y):
        x_hat = torch.zeros(self.N, y.shape[1], dtype = self.A.dtype)
        v = torch.zeros(y.shape, dtype = self.A.dtype)
        
        assert list(x_hat.shape) == [512,64], 'Size is {0}'.format(list(x_hat.shape))
        assert list(v.shape) == [48,64], 'Size is {0}'.format(list(v.shape))
        
        for t in range(self.layers):
            r = x_hat + self.B @ v
            sigma = 1.0/torch.sqrt(self.M)*torch.linalg.vector_norm(v, ord = 2)
            lambd = self.alphas[t] * sigma
            x_hat = self.betas[t] * self.activation.function(r,lambd)
            b = self.betas[t]/self.M * torch.linalg.vector_norm(x_hat, ord = 0)
            v = y - self.A @ x_hat + b*v
        
        return x_hat, v




        

class LAMP(torch.nn.Module):
    def __init__(self, A, layers, alpha, beta, delta, activation_class, B_type = 'fft') -> None:
        super().__init__()
        
        self.network = torch.nn.ModuleList(
            [LAMP_layer(A, alpha, beta, activation_class, B_type)]
        )

        print("Created LAMP wiht ", layers, 'layers')
        self.is_normalize = True
        self.delta = delta 
        self.A = A

        self.M, self.N = A.shape
        assert self.M <= self.N, 'M should be <= than N'
    
    def forward(self, y):
        x_hat = torch.zeros([self.M, self.N])

        if self.is_normalize:
            y, maximum = self.normalize(y, self.delta)
        
        z = y 

        for layer in self.network:
            x_hat, v = layer(x_hat, v, y)
        
        if self.is_normalize:
            x_hat = self.denormalize(x_hat, maximum)

        return x_hat @ self.A


    def normalize(self, y, delta = 1.0 ):
        maximum = torch.linalg.vector_norm(y, ord = 'inf')
        y = delta*y/maximum
        return y , maximum

    def denormalize(self, y, maximum, delta = 1.0):
        y = y/delta*maximum
        return y



