import torch 

class UnFold1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,kernel_size,dilatation,padding,stride): 
        """
            Nuestro tensor de entrada es: (N,C,d1) donde N es el batch, C es el canal y d1 es la dimension espacial suponemos 1D
        """
        N,C,d1 = inputs.shape 
        shape_output = C*kernel_size #Si no fuese una Conv1D tendriamos que multiplicar por tantos kernel_sizes como spatial_dimensiones tenga
        L_out = int(((d1 + 2*padding -dilatation*(kernel_size -1) -1)/(stride)) + 1)#Formula que se aplica para el unfold
        outputs = torch.empty((N,shape_output,L_out))
        # for i in range(shape_output): #Esto representara las filas 
        #     for j in range(L_out): #Esto representa como las columnas 
        #         fila_start = i*stride
        #         columna_start = j*stride
        #         fila_end = fila_start + kernel_size
        #         columna_end = fila_end + kernel_size
        #         window = inputs[:,fila_start:fila_end,columna_start:columna_end]
        #         outputs[:,i,j] = window

        """
            Version con mas bucles: 
        """
        for n in range(N):
            for c in range(C):
                for k in range(kernel_size):
                    offset = k * dilatation
                    for l in range(L_out):
                        start = l * stride + offset
                        outputs[n, c * kernel_size + k, l] = inputs[n, c, start]
        """
            Version con dos bucles: 
        """
        ctx.N = N 
        ctx.C = C 
        ctx.d1 = d1 
        ctx.kernel_size = kernel_size 
        ctx.dilatation = dilatation
        ctx.stride = stride
        out = torch.empty((N,kernel_size*C,L_out))
        for l in range(L_out): #Iteramos como sobre las columnas. 
            for k in range(kernel_size): #Iteramos sobre los kernel_size que sera unicamente como estara puesto. 
                start = l*stride + k*dilatation #Con esto sacamos el indice del elemento de inputs que queremos. 
                out[:,k::kernel_size,l] = inputs[:,:,start] #Aqui montamos los elementos de dos filas en dos. 
        return out  #Nuestro tensor de salida va a tener un shape de: [N,C*kernel_size,L_out]
    
    @staticmethod
    def backward(ctx, grad_outputs):
        """
            El grad_outputs va tener el shape de: [N,C*kernel_size,L_out]. 
            Lo que vamos a tener que hacer es hacer como un fold del tensor de grad_outputs, que consistiria en como redistribuir el gradiente de cada uno. 
        """
        N = ctx.N 
        C = ctx.C
        L_in = ctx.d1 
        kernel_size = ctx.kernel_size
        dilatation = ctx.dilatation
        stride = ctx.stride 
        _,shape_output,Lout = grad_outputs.shape
        grad_inputs = torch.empty((N,C,L_in))
        for l in range(Lout): 
            for k in range(kernel_size): 
                start = l*stride + k*dilatation
                grad_inputs[:,:,start] += grad_outputs[:,k::kernel_size,l]
        return grad_inputs,None,None,None,None  


