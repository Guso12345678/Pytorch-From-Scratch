import torch  
from utils_for_dropout import get_dropout_random_indexes

class Dropout(torch.nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forwward pass.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs. Dimensions: [*], same as inputs tensor.
        """
        shape = inputs.shape 
        mascara_ceros = get_dropout_random_indexes(shape,self.p)
        if self.inplace == True: 
            inputs *= (1 - mascara_ceros)
            inputs *= (1/(1 - self.p))
            return inputs 
        outputs = inputs * (1 - mascara_ceros)
        outputs = outputs * (1/(1 - self.p))
        return outputs 
