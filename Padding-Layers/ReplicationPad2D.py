import torch 

class ReplicationPad2DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size): 
        """
            The input shape is: [N,C,Hin,Win]
            Te output shape is: [N,C,Hout,Wout]
        """
        N, C, H, W = inputs.shape
        pleft, pright, ptop, pbottom = padding_size
        H_out = H + ptop + pbottom
        W_out = W + pleft + pright

        outputs = torch.zeros((N,C,H_out,W_out))

         # Crear tensor vacío
        outputs = torch.empty((N, C, H_out, W_out), device=inputs.device, dtype=inputs.dtype)

        # DERECHA: copiar última columna a las columnas de la derecha
        outputs[:, :, ptop:ptop+H, pleft+W:] = inputs[:, :, :, -1:].expand(N, C, H, pright)

        # IZQUIERDA: copiar primera columna a las columnas de la izquierda
        outputs[:, :, ptop:ptop+H, :pleft] = inputs[:, :, :, 0:1].expand(N, C, H, pleft)

        # ARRIBA: copiar primera fila del bloque central a las filas superiores
        outputs[:, :, :ptop, :] = outputs[:, :, ptop:ptop+1, :].expand(N, C, ptop, W_out)

        # ABAJO: copiar última fila del bloque central a las filas inferiores
        outputs[:, :, ptop+H:, :] = outputs[:, :, ptop+H-1:ptop+H, :].expand(N, C, pbottom, W_out)

        # CENTRO: copiar el input a su posición central
        outputs[:, :, ptop:ptop+H, pleft:pleft+W] = inputs

        return outputs 
    
###EL BACKWARD ES LO DE SIEMPRE SOLO VA A FLUIR GRADIENTE POR LOS ELEMENTOS DEL TENSOR ORIGINAL ES DECIR POR EL CENTRO. 
