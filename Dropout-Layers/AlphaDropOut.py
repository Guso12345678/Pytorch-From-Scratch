import torch  
from utils_for_alpha_dropout import get_dropout_random_indexes

class AlphaDropout(torch.nn.Module):
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
        ##Constantes de la SELU: 
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.alpha_prime = -self.alpha * self.scale
    
    def forward(self, inputs: torch.Tensor): 
        if not self.training or self.p == 0.:
            return inputs
        
        mask = get_dropout_random_indexes(inputs.shape,self.p)
        dropped = inputs * mask + self.alpha_prime * (1 - mask)

        a = ((1 - self.p) * (1 + self.p * self.alpha_prime**2)) ** -0.5
        b = -a * self.p * self.alpha_prime

        return a * dropped + b