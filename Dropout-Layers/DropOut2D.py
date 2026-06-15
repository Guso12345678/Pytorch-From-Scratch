import torch  
from utils_for_dropout2D import get_dropout_random_indexes

class Dropout2D(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """

        # TODO
        super().__init__()
        self.p = p 
        self.inplace = inplace 
    
    def forward(self,inputs:torch.Tensor): 
        """
            The input shape is: [N, C, H, W]
            The output shape is: [N, C, H, W]
        """

        if not self.training: 
            return inputs 
        shape = inputs.shape 
        mascara_ceros = get_dropout_random_indexes(shape,self.p).float()
        if self.inplace == True: 
            inputs.mul_(1 - mascara_ceros)
            inputs.mul_(1/(1 - self.p))
            return inputs 
        outputs = inputs * (1 - mascara_ceros)
        outputs = outputs * (1/(1 - self.p))
        return outputs 