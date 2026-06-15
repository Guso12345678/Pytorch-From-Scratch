##SOLO VAMOS A IMPLEMENTAR EL FORWARD DE LA GRU.py
import torch 
from typing import Any 
import math

class GRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx:Any,inputs:torch.Tensor,h0:torch.Tensor,W_z:torch.Tensor,W_r:torch.Tenso,W_h:torch.Tensor,b_z:torch.Tensor,b_r:torch.Tensor,b_h:torch.Tensor):
        tensor_ht_guardar = torch.zeros(size=(inputs.size(0),inputs.size(1),h0.size(2)),dtype=torch.float64) 
        for i in range(inputs.size(1)): 
            #Primero hacemos la concatenacion como hemos hecho antes: 
            concatenacion = torch.cat([inputs[:,i,:],h0],dim=1)
            #Hacemos el calculo de las puertas: 
            #1. Update Gate: 
            zt = (concatenacion @ W_z) + b_z
            sigmoid_zt = (1)/(1 + torch.exp(-zt))
            #2. Reset Gate: 
            rt = (concatenacion @ W_r) + b_r
            sigmoid_rt = (1)/(1 + torch.exp(-rt))

            #Sacamos el estado candidato y para ello primero necesitamos la nueva concatenacion:
            concatenacion2 = torch.cat([inputs[:,i,:],sigmoid_rt*h0],dim=1) 
            estado_cand = (concatenacion2 @ W_h) + b_h 
            tanh_estado_cand = (torch.exp(estado_cand) - torch.exp(-estado_cand))/(torch.exp(estado_cand) + torch.exp(-estado_cand)) 

            #Actualizamos el estado oculto: 
            new_state = ((1 - sigmoid_zt)*h0) + (sigmoid_zt*tanh_estado_cand)
            tensor_ht_guardar[:,i,:] = new_state
            h0 = new_state
        return tensor_ht_guardar, tensor_ht_guardar.transpose(0,1)[-1,:,:].unsqueeze(0)
    
class GRU(torch.nn.Module): 
    def __init__(self,input_dim:int, hidden_size:int): 
        self.hidden_size = hidden_size
        self.W_z = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.W_r = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.W_h = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.b_z = torch.nn.Parameter(torch.empty(size=(hidden_size,)))
        self.b_r = torch.nn.Parameter(torch.empty(size=(hidden_size,)))
        self.b_h = torch.nn.Parameter(torch.empty(size=(hidden_size,))) 
        torch.nn.init.xavier_uniform_(self.W_z)
        torch.nn.init.xavier_uniform_(self.W_r)
        torch.nn.init.xavier_uniform_(self.W_h)
        torch.nn.init.xavier_uniform_(self.b_z)
        torch.nn.init.xavier_uniform_(self.b_r)
        torch.nn.init.xavier_uniform_(self.b_h)
        self.fn = GRUFunction.apply

    def forward(self,inputs:torch.Tensor,h0:torch.Tensor):
        return self.fn(inputs,h0,self.W_z,self.W_r,self.W_h,self.b_z,self.b_r,self.b_h)
