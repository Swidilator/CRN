import torch
import torch.nn as nn

# from apex.normalization import FusedLayerNorm

from typing import Tuple, List, Any


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class RefinementModule(nn.Module):
    r"""
    One 3 layer module making up a segment of a CRN. Mask input tensor & prior layers get resized.

    Args:
        prior_layer_channel_count(int): number of input channels from previous layer
        semantic_input_channel_count(int): number of input channels from semantic annotation
        output_channel_count(int): number of output channels
        input_height_width(tuple(int)): input image height and width
        is_final_module(bool): is this the final module in the network
    """

    def __init__(
        self,
        prior_layer_channel_count: int,  # Number of channels from previous RM or noise
        semantic_input_channel_count: int,  # Number of channels from one-hot encoded semantic input
        output_channel_count: int,  # Number of channels to be outputted by this RM
        input_height_width: tuple,  # Input image height and width
        use_feature_encoder: bool,
        is_final_module: bool = False,  # Is this RM the final in the network
        final_channel_count: int = 3,  # If this RM is final, how many channels must it output
    ):
        super(RefinementModule, self).__init__()

        self.input_height_width: tuple = input_height_width
        self.use_feature_encoder = use_feature_encoder
        # Total number of input channels
        self.total_input_channel_count: int = (
            prior_layer_channel_count
            + semantic_input_channel_count
            + (3 if self.use_feature_encoder else 0)
        )
        self.output_channel_count: int = output_channel_count
        self.is_final_module: bool = is_final_module
        self.final_channel_count = final_channel_count

        # self.dropout = nn.Dropout2d(p=0.1)

        # Module architecture
        self.conv_1 = nn.Conv2d(
            self.total_input_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        RefinementModule.init_conv_weights(self.conv_1)

        self.layer_norm_1 = nn.LayerNorm(
            RefinementModule.change_output_channel_size(
                input_height_width, self.output_channel_count
            ),
            # torch.Size(input_height_width),
        )

        self.conv_2 = nn.Conv2d(
            self.output_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        RefinementModule.init_conv_weights(self.conv_2)

        self.layer_norm_2 = nn.LayerNorm(
            RefinementModule.change_output_channel_size(
                input_height_width, self.output_channel_count
            ),
            # torch.Size(input_height_width),
        )

        if self.is_final_module:
            self.final_conv = nn.Conv2d(
                self.output_channel_count,
                self.final_channel_count,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            RefinementModule.init_conv_weights(self.final_conv)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)

    @staticmethod
    def init_conv_weights(conv: nn.Module) -> None:
        nn.init.normal_(conv.weight, mean=0.0, std=0.02)
        # nn.init.xavier_uniform_(conv.weight, gain=1)
        nn.init.constant_(conv.bias, 0)

    @staticmethod
    def change_output_channel_size(
        input_height_width: tuple, output_channel_number: int
    ):
        size_list = list(input_height_width)
        size_list.insert(0, output_channel_number)
        # print(size_list)
        return torch.Size(size_list)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        # Separate inputs
        mask: torch.Tensor = inputs[0]
        feature_selection: torch.Tensor = inputs[1]
        prior_layers: torch.Tensor = inputs[2]

        # Downsample mask for current RM
        mask = torch.nn.functional.interpolate(
            input=mask, size=self.input_height_width, mode="nearest"
        )

        # If there are prior layers, upsample them and concatenate them onto the mask input
        # otherwise, don't
        x: torch.Tensor = mask
        if prior_layers is not None:
            prior_layers = torch.nn.functional.interpolate(
                input=prior_layers, size=self.input_height_width, mode="bilinear"
            )
            x = torch.cat((x, prior_layers), dim=1)
        if self.use_feature_encoder:
            feature_selection = torch.nn.functional.interpolate(
                input=feature_selection, size=self.input_height_width, mode="nearest"
            )
            x = torch.cat((x, feature_selection), dim=1)

        x = self.conv_1(x)
        x = self.layer_norm_1(x)
        x = self.leakyReLU(x)

        x = self.conv_2(x)
        x = self.layer_norm_2(x)
        x = self.leakyReLU(x)

        if self.is_final_module:
            x = self.final_conv(x)

        return x
