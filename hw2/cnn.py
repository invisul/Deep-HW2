import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.
    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        # Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.

        p_idx = 0
        for ii in range(len(self.channels)):
            if ii == 0:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=self.channels[ii], **self.conv_params)]
            else:
                layers += [nn.Conv2d(in_channels=self.channels[ii-1], out_channels=self.channels[ii],
                                     **self.conv_params)]
            layers += [ACTIVATIONS[self.activation_type](**self.activation_params)]
            p_idx += 1
            if p_idx == self.pool_every:
                layers += [POOLINGS[self.pooling_type](**self.pooling_params)]
                p_idx = 0

        seq = nn.Sequential(*layers)

        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # 0) Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # 1) generate a random input with the same size as defined in __init__
            dummy_input = torch.randn([1, *self.in_size])

            # 2) pass the generated input through the feature extractor
            extractor_output = self.feature_extractor(dummy_input)

            # 3) The number of features the extractor calculates is a product of all the dimensions of the output size
            return int(torch.prod(torch.tensor(extractor_output.shape[1:])))
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.

        # 1) get the depth of the MLP (not including the output layer
        d = len(self.hidden_dims)

        # 2) specify the width of each layer, including the output layer
        dims = [*self.hidden_dims, self.out_classes]

        # 3) initialize the activation for each linear layer. The activation for the output layer should be Identity
        activations = []
        for ii in range(d):
            activations += [ACTIVATIONS[self.activation_type](**self.activation_params)]
        activations += [ACTIVATIONS['none']()]

        # 4) initialize the MLP model
        mlp: MLP = MLP(in_dim=self._n_features(), dims=dims, nonlins=activations)

        return mlp

    def forward(self, x: Tensor):
        # Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None

        # 1) extract the features from the image
        features = self.feature_extractor(x)

        # 2) flatten the features, so we would have (num_of_pixels_in_last_layer x depth x K x K)
        features = features.view(features.size(0), -1)

        # 3) use the classifier to predict the score for each class
        out = self.mlp(features)

        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.

        # 1) build main path
        main_path = []
        for ii in range(len(channels)):
            # 1.0) calculate padding
            padding = int((kernel_sizes[ii] - 1) / 2)

            # 1.1) add convolutional layer to main path
            if ii == 0:
                main_path += [nn.Conv2d(in_channels=in_channels,
                                        out_channels=channels[ii],
                                        kernel_size=kernel_sizes[ii], bias=True, padding=padding)]
            else:
                main_path += [nn.Conv2d(in_channels=channels[ii-1],
                                        out_channels=channels[ii],
                                        kernel_size=kernel_sizes[ii], bias=True, padding=padding)]

            if ii != (len(channels) - 1):
                # 1.2) add dropout layer to the main path
                if dropout > 0:
                    main_path += [nn.Dropout2d(dropout)]

                # 1.3) add batchnorm layer to the main path
                if batchnorm:
                    main_path += [nn.BatchNorm2d(channels[ii])]

                # 1.4) add the activation layer to the main path
                main_path += [ACTIVATIONS[activation_type](**activation_params)]

        # 1.5) add all main path layers as a sequential model
        self.main_path = nn.Sequential(*main_path)

        # 2) build shortcut path
        shortcut_path = []

        # 2.1) if the number of output channels of the main path is equal to the number of input channels
        if in_channels == channels[-1]:
            shortcut_path += [nn.Identity()]
        else:
            # 2.2) add a non-biased convolutional layer to equalize the number of channels
            shortcut_path += [nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False, padding=0)]

        # 2.3) add shortcut path layer as a sequential model
        self.shortcut_path = nn.Sequential(*shortcut_path)

    def forward(self, x: Tensor):
        # Implement the forward pass. Save the main and residual path to `out`.
        # out: Tensor = None
        out = self.main_path(x) + self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        # assert len(inner_channels) == len(inner_kernel_sizes)

        assert len(inner_channels) == len(inner_kernel_sizes)
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        super().__init__(in_channels=in_out_channels,
                         channels=[inner_channels[0], *inner_channels, in_out_channels],
                         kernel_sizes=[1, *inner_kernel_sizes, 1],
                         **kwargs)


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.

        N = len(self.channels)
        P = self.pool_every

        # num_of_blocks = N//P
        activations = ACTIVATIONS[self.activation_type]
        pool = POOLINGS[self.pooling_type]

        # [-> (CONV -> ACT)*P -> POOL]*(N/P) means it happens N//p times!!!
        blockParams = dict(batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                                activation_params=self.activation_params)
        bottleneck = self.bottleneck
        block_channels = []
        for i,channel  in enumerate(self.channels):
            block_channels += [channel]
            if ((i+1)%P) == 0:
                out_channel = block_channels[-1]
                if bottleneck and in_channels == out_channel:
                    layers += [ResidualBottleneckBlock(
                        in_out_channels=in_channels,
                        inner_channels=block_channels[1:-1],
                        inner_kernel_sizes=[3] * len(block_channels[1:-1]),
                        **blockParams
                    )]
                else:
                    layers += [ResidualBlock(
                        in_channels=in_channels,
                        channels=block_channels,
                        kernel_sizes= [3] * P,
                        **blockParams,
                    )]
                # update the channel for the net residualblock we make
                in_channels = channel
                block_channels =[] # empty it cuz we used it in block
                # add a pooling layer (after the conv act * P)
                # print(self.pooling_params.keys())
                layers += [pool(**self.pooling_params)]


        # if we got here it means thre is some block_channels left!!
        if N % P != 0:
            out_channel = block_channels[-1]
            if bottleneck and in_channels == out_channel:
                layers += [ResidualBottleneckBlock(
                    in_out_channels=in_channels,
                    inner_channels=block_channels[1:-1],
                    inner_kernel_sizes=[3] * len(block_channels[1:-1]),
                    **blockParams
                )]
            else:
                layers += [ResidualBlock(
                    in_channels=in_channels,
                    channels=block_channels,
                    kernel_sizes=[3] * len(block_channels),
                    **blockParams,
                )]

        seq = nn.Sequential(*layers)
        return seq


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, c_red: dict, c_out: dict, act_fn, dropout):
        """
        Inputs:
            in_channels - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        if 0 < dropout < 1:
            args = nn.Dropout2d(dropout)
        else:
            args = nn.Identity()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, c_out["1x1"], kernel_size=1),
            args,
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, c_red["3x3"], kernel_size=1),
            args,
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, c_red["5x5"], kernel_size=1),
            args,
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels, c_out["max"], kernel_size=1),
            args,
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class MyInceptionBlock(InceptionBlock):
    def __init__(self, in_channels, out_channels, c_red: dict, c_out: dict, act_fn, dropout):
        super().__init__(in_channels, c_red, c_out, act_fn,dropout)
        inception_channels_out = sum([val for _, val in c_out.items()])
        conv_out = torch.nn.Conv2d(in_channels=inception_channels_out,
                                   kernel_size=1,
                                   out_channels=int(sum(out_channels)))
        self.conv_out = nn.Sequential(conv_out, act_fn())




class YourCNN2(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.

        N = len(self.channels)
        P = self.pool_every

        act_fun = torch.nn.LeakyReLU
        dropout = 0.4

        # num_of_blocks = N//P
        pool = POOLINGS[self.pooling_type]

        # [-> (CONV -> ACT)*P -> POOL]*(N/P) means it happens N//p times!!!
        block_channels = []
        for i,channel  in enumerate(self.channels):
            block_channels += [channel]
            if ((i+1)%P) == 0:
                # in_channels, out_channels, c_red: dict, c_out: dict, act_fn,dropout
                layers += [MyInceptionBlock(
                    in_channels=in_channels,
                    out_channels= block_channels,
                    c_red = {"3x3": 32, "5x5": 16},
                    c_out= {"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                    act_fn=act_fun,
                    dropout=dropout
                )]
                # update the channel for the net residualblock we make
                in_channels = channel
                block_channels =[] # empty it cuz we used it in block
                # add a pooling layer (after the conv act * P)
                layers += [pool(**self.pooling_params)]


        # if we got here it means thre is some block_channels left!!
        if N % P != 0:
                layers += [MyInceptionBlock(
                    in_channels=in_channels,
                    out_channels= sum(block_channels),
                    c_red = {"3x3": 32, "5x5": 16},
                    c_out= {"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                    act_fn=act_fun,
                    dropout=dropout
                )]
        if dropout > 0:
            layers += [nn.Dropout2d(dropout)]

        seq = nn.Sequential(*layers)
        return seq


class YourCNN(ResNet):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims, **kwargs):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims, True, 0, True,
                         pooling_params=dict(kernel_size=2), **kwargs)
