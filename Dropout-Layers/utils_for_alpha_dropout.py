import torch 
def get_dropout_random_indexes(shape: torch.Size, p: float,alphaprime) -> torch.Tensor:
    """
    This function get the indexes to put elements at zero for the
    dropout layer. It ensures the elements are selected following the
    same implementation than the pytorch layer.

    Args:
        shape: shape of the inputs to put it at zero. Dimensions: [*].
        p: probability of the dropout.

    Returns:
        indexes to put elements at zero in dropout layer.
            Dimensions: shape.
    """

    # get inputs indexes
    inputs: torch.Tensor = torch.ones(shape)

    # get indexes
    indexes: torch.Tensor = torch.nn.functional.dropout(inputs,p)

    indexes = (indexes == 0).int()

    return indexes