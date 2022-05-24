import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.

        # 1) initialine the nn.Module class
        super().__init__()

        # 2) create a list of all the dimensions
        all_dims = [in_dim, *dims]

        # 3) create a list of all layers (add a non-linearity at the end of each layer)
        layers = []
        for dim_in, dim_out, nonlin in zip(all_dims[:-1], all_dims[1:], nonlins):
            # 3.1) add a linear layer
            layers += [nn.Linear(dim_in, dim_out, bias=True)]

            # 3.2) add a non-linearity to the end of the layer
            if isinstance(nonlin, str) and (nonlin in ACTIVATIONS.keys()):
                layers += [ACTIVATIONS[nonlin](**ACTIVATION_DEFAULT_KWARGS[nonlin])]
            elif isinstance(nonlin, nn.Module):
                layers += [nonlin]
            else:
                raise TypeError("Expected `nonlin` to be an `str` or an instance of `nn.Module. "
                                "Got {} instead.".format(type(nonlin)))

        # 4) create a sequence of the input layers
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.

        # 1) flatten the input to (n_samples, num_of_features)
        x_flattened = torch.reshape(x, (x.shape[0], -1))

        # 2) pass the input through the MLP and return the prediction
        y_pred = self.fc_layers(x_flattened)
        return y_pred
