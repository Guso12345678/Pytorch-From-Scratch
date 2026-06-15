import torch 
from typing import Any 
import math 

#Primera Versión con concatenacion de la entrada y el estado_oculto en cada time step, la segunda version es lo mismo con mas pesos y sin la concatenacion. 
class LSTMFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:Any, inputs:torch.Tensor, h0:torch.Tensor, c0:torch.Tensor, W_f:torch.Tensor, W_o:torch.Tensor, W_i:torch.Tensor, W_c:torch.Tensor, b_f:torch.Tensor, b_i:torch.Tensor, b_o:torch.Tensor, b_c:torch.Tensor): 
        tensor_ht_guardar = torch.zeros(size=(inputs.size(0),inputs.size(1),h0.size(2)),dtype=torch.float64)
        tensor_ct_guardar = torch.zeros(size=(inputs.size(0),inputs.size(1),c0.size(2)),dtype=torch.float64)

        for i in range(inputs.size(1)):
            #Concatenacion de las entradas y el estado: 
            concatenacion = torch.cat([inputs[:,i,:],h0],dim=1)
            #Calculo de la puertas: 
            # 1. Olvido 
            forget_gate = (concatenacion @ W_f) + b_f 
            sigmoid_forget_gate = 1/(1+ torch.exp(-forget_gate))

            #2. Entrada 
            input_gate = (concatenacion @ W_i) + b_i 
            sigmoid_input_gate = (1)/(1 + torch.exp(-input_gate))

            #3. Salida 
            output_gate = (concatenacion @ W_o) + b_o
            sigmoid_output_gate = (1)/(1 + torch.exp(-output_gate))

            #4. Celula_provisional 
            cell_gate = (concatenacion @ W_c) + b_c 
            tanh_cell_gate = (torch.exp(cell_gate) - torch.exp(-cell_gate))/(torch.exp(cell_gate) + torch.exp(-cell_gate))

            #Calculo de la nueva celula de estado y actualizacion de la misma: 
            new_cell = (sigmoid_forget_gate*c0) + (sigmoid_input_gate*tanh_cell_gate)
            tensor_ct_guardar[:,i,:] = new_cell 
            c0 = new_cell

            #Calculo del nuevo estado de memoria y actualizacion del mismo:
            new_ht = (sigmoid_output_gate) * ((torch.exp(c0) - torch.exp(-c0))/(torch.exp(c0) + torch.exp(-c0)))
            tensor_ht_guardar[:,i,:] = new_ht
            h0 = new_ht 
        
        return tensor_ht_guardar, tensor_ht_guardar.transpose(0,1)[-1,:,:].unsqueeze(0) 
    

class LSTM(torch.nn.Module): 
    def __init__(self,input_dim:int, hidden_size:int): 
        self.hidden_size = hidden_size
        self.W_f = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.W_o = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.W_i = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.W_c = torch.nn.Parameter(torch.empty(size=(hidden_size+input_dim,hidden_size)))
        self.b_f = torch.nn.Parameter(torch.empty(size=(hidden_size,)))
        self.b_i = torch.nn.Parameter(torch.empty(size=(hidden_size,)))
        self.b_o = torch.nn.Parameter(torch.empty(size=(hidden_size,))) 
        self.b_c = torch.nn.Parameter(torch.empty(size=(hidden_size,))) 
        torch.nn.init.xavier_uniform_(self.W_f)
        torch.nn.init.xavier_uniform_(self.W_o)
        torch.nn.init.xavier_uniform_(self.W_i)
        torch.nn.init.xavier_uniform_(self.W_c)
        torch.nn.init.xavier_uniform_(self.b_f)
        torch.nn.init.xavier_uniform_(self.b_i)
        torch.nn.init.xavier_uniform_(self.b_o)
        torch.nn.init.xavier_uniform_(self.b_c)
        self.fn = LSTMFunction.apply
    
    def forward(self,inputs:torch.Tensor,h0:torch.Tensor,c0:torch.Tensor):
        return self.fn(inputs,h0,c0,self.W_f,self.W_o,self.W_i,self.W_c,self.b_f,self.b_i,self.b_o,self.b_c) 

