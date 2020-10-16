import torch
import torch.nn as nn


class RefinementModule(nn.Module):
    def __init__(
        self,
        semantic_input_channel_count: int,
        feature_encoder_input_channel_count: int,
        edge_map_input_channel_count: int,
        base_conv_channel_count: int,
        prior_conv_channel_count: int,
        final_conv_output_channel_count: int,
        input_height_width: tuple,
        norm_type: str,
    ):
        super(RefinementModule, self).__init__()

        self.input_height_width: tuple = input_height_width

        # Total number of input channels
        self.total_input_channel_count: int = (
            semantic_input_channel_count
            + feature_encoder_input_channel_count
            + edge_map_input_channel_count
            + prior_conv_channel_count
        )

        self.base_conv_channel_count: int = base_conv_channel_count

        self.final_conv_output_channel_count: int = final_conv_output_channel_count
        self.use_feature_encoder: int = feature_encoder_input_channel_count > 0
        self.is_final_module: bool = self.final_conv_output_channel_count > 0
        # self.dropout = nn.Dropout2d(p=0.1)

        # Module architecture
        self.conv_1 = nn.Conv2d(
            self.total_input_channel_count,
            self.base_conv_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        RefinementModule.init_conv_weights(self.conv_1)

        if norm_type == "full":
            self.norm_1 = nn.LayerNorm(
                RefinementModule.change_output_channel_size(
                    input_height_width, self.base_conv_channel_count
                ),
            )
        elif norm_type == "half":
            self.norm_1 = nn.LayerNorm(torch.Size(input_height_width))
        elif norm_type == "none":
            self.norm_1 = lambda x: x
        elif norm_type == "group":
            self.norm_1 = nn.GroupNorm(1, self.base_conv_channel_count)
        elif norm_type == "instance":
            self.norm_1 = nn.InstanceNorm2d(
                self.base_conv_channel_count,
                eps=1e-05,
                momentum=0.1,
                affine=False,
                track_running_stats=False,
            )
        elif norm_type == "weight":
            self.norm_1 = lambda x: x
            self.conv_1 = nn.utils.weight_norm(self.conv_1)

        self.leakyReLU_1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_2 = nn.Conv2d(
            self.base_conv_channel_count,
            self.base_conv_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        RefinementModule.init_conv_weights(self.conv_2)

        if norm_type == "full":
            self.norm_2 = nn.LayerNorm(
                RefinementModule.change_output_channel_size(
                    input_height_width, self.base_conv_channel_count
                ),
            )
        elif norm_type == "half":
            self.norm_2 = nn.LayerNorm(torch.Size(input_height_width))
        elif norm_type == "none":
            self.norm_2 = lambda x: x
        elif norm_type == "group":
            self.norm_2 = nn.GroupNorm(1, self.base_conv_channel_count)
        elif norm_type == "instance":
            self.norm_2 = nn.InstanceNorm2d(
                self.base_conv_channel_count,
                eps=1e-05,
                momentum=0.1,
                affine=False,
                track_running_stats=False,
            )
        elif norm_type == "weight":
            self.norm_2 = lambda x: x
            self.conv_2 = nn.utils.weight_norm(self.conv_2)

        self.leakyReLU_2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.is_final_module:
            self.final_conv = nn.Conv2d(
                self.base_conv_channel_count,
                self.final_conv_output_channel_count,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            RefinementModule.init_conv_weights(self.final_conv)

    @staticmethod
    def init_conv_weights(conv: nn.Module) -> None:
        # nn.init.normal_(conv.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(conv.weight, gain=1)
        nn.init.constant_(conv.bias, 0)

    @staticmethod
    def change_output_channel_size(
        input_height_width: tuple, output_channel_number: int
    ):
        size_list = list(input_height_width)
        size_list.insert(0, output_channel_number)
        # print(size_list)
        return torch.Size(size_list)

    def forward(
        self,
        mask: torch.Tensor,
        prior_layers: torch.Tensor,
        feature_selection: torch.Tensor,
        edge_map: torch.Tensor
    ):

        # Downsample mask for current RM
        mask = torch.nn.functional.interpolate(
            input=mask, size=self.input_height_width, mode="nearest"
        )

        # If there are prior layers, upsample them and concatenate them onto the mask input
        # otherwise, don't
        x: torch.Tensor = mask
        if prior_layers is not None:
            prior_layers = torch.nn.functional.interpolate(
                input=prior_layers,
                size=self.input_height_width,
                mode="bilinear",
                align_corners=True,
            )
            x = torch.cat((x, prior_layers), dim=1)
        if self.use_feature_encoder:
            feature_selection = torch.nn.functional.interpolate(
                input=feature_selection, size=self.input_height_width, mode="nearest"
            )
            edge_map = torch.nn.functional.interpolate(
                input=edge_map, size=self.input_height_width, mode="nearest"
            )
            x = torch.cat((x, feature_selection, edge_map), dim=1)

        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.leakyReLU_1(x)

        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.leakyReLU_2(x)

        if self.is_final_module:
            x = self.final_conv(x)

        return x
