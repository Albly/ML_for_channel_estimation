import torch

def sigmoid(r, S1, S2):
    one = torch.tensor([1])
    res = one/(1+torch.exp(-S1*r+S2))
    res = torch.reshape(res, (1,512))
    res = res.expand((64,512))
    return res

def arctan(r, S1, S2):
    pi = torch.tensor(np.pi)
    one_half = torch.tensor([0.5])
    res = torch.atan(S1*(r+S2))/pi + one_half
    res = torch.reshape(res, (1,512))
    res = res.expand((64,512))
    
    return res

def tanh(r,S1,S2):
    one_half = torch.tensor([0.5])
    res = one_half * torch.tanh(S1*(r+S2)) + one_half
    res = torch.reshape(res, (1,512))
    res = res.expand((64,512))
    
    return res

def sqrt(r, S1, S2):
    one_half = torch.tensor([0.5])
    one = torch.tensor([1])
    res = one_half * (S1*(r+S2)) / torch.sqrt(one+ (S1*(r+S2))**2 ) + one_half
    res = torch.reshape(res, (1,512))
    res = res.expand((64,512))
    
    return res

def rat_sigmoid(r,S1,S2):
    one_half = torch.tensor([0.5])
    one = torch.tensor([1])
    res = one_half* S1*(r+S2) / (abs(S1*(r+S2))+1) + one_half
    res = torch.reshape(res, (1,512))
    res = res.expand((64,512))
    
    return res
