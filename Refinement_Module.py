import torch
import torch.nn as nn
from typing import List

from support_scripts.components import RMBlock


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
        prev_frame_count: int,
    ):
        super().__init__()

        self.input_height_width: tuple = input_height_width
        self.prev_frame_count: int = prev_frame_count

        # Total number of input channels
        self.total_semantic_input_channel_count: int = (
            semantic_input_channel_count
            + feature_encoder_input_channel_count
            + edge_map_input_channel_count
            + prior_conv_channel_count
            + (prev_frame_count * semantic_input_channel_count)
        )

        # Total number of input channels
        self.total_image_input_channel_count: int = (prev_frame_count * 3)

        self.base_conv_channel_count: int = base_conv_channel_count

        self.final_conv_output_channel_count: int = final_conv_output_channel_count
        self.use_feature_encoder: int = feature_encoder_input_channel_count > 0
        self.is_final_module: bool = self.final_conv_output_channel_count > 0

        self.rm_block_1_semantic = RMBlock(
            self.base_conv_channel_count,
            self.total_semantic_input_channel_count,
            self.input_height_width,
            kernel_size=3,
            norm_type=norm_type,
            num_conv_groups=1,
        )

        if prev_frame_count > 0:
            self.rm_block_1_image = RMBlock(
                self.base_conv_channel_count,
                self.total_image_input_channel_count,
                self.input_height_width,
                kernel_size=3,
                norm_type=norm_type,
                num_conv_groups=1,
            )

        self.rm_block_2 = RMBlock(
            self.base_conv_channel_count,
            self.base_conv_channel_count,
            self.input_height_width,
            kernel_size=3,
            norm_type=norm_type,
            num_conv_groups=1,
        )

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

    def forward(
        self,
        mask: torch.Tensor,
        prior_layers: torch.Tensor,
        feature_selection: torch.Tensor,
        edge_map: torch.Tensor,
        prev_frames: torch.Tensor = None,
        prev_masks: torch.Tensor = None,
    ):

        # Interpolate inputs to current RM module resolution
        inputs: list = [
            (mask, "nearest"),
            (prior_layers, "bilinear"),
            (feature_selection, "nearest"),
            (edge_map, "nearest"),
            (prev_frames, "bilinear"),
            (prev_masks, "nearest"),
        ]

        (
            mask,
            prior_layers,
            feature_selection,
            edge_map,
            prev_frames,
            prev_masks,
        ) = interpolate_inputs(self.input_height_width, inputs)

        # Concatenate semantic inputs together for use in semantic entry conv
        x_semantic_input: list = [
            x
            for x in (mask, prior_layers, feature_selection, edge_map, prev_masks)
            if x is not None
        ]
        x_semantic: torch.Tensor = torch.cat(x_semantic_input, dim=1)
        x = self.rm_block_1_semantic(x_semantic, relu_loc="after")

        # If previous frames are present, then pass them into the image entry conv and add them to the semantic output
        if self.prev_frame_count > 0:
            x_image: torch.Tensor = self.rm_block_1_image(prev_frames, relu_loc="after")
            x = x + x_image

        # Continue as normal
        x = self.rm_block_2(x, relu_loc="after")

        if self.is_final_module:
            x = self.final_conv(x)

        return x


def interpolate_inputs(
    input_height_width: tuple, inputs: List[tuple]
) -> List[torch.Tensor]:
    # Function for interpolating any number of inputs
    output_list: list = []

    for single_input in inputs:
        if single_input[0] is not None:
            if single_input[1] == "nearest":
                output_list.append(
                    torch.nn.functional.interpolate(
                        input=single_input[0], size=input_height_width, mode="nearest"
                    )
                )
            elif single_input[1] == "bilinear":
                output_list.append(
                    torch.nn.functional.interpolate(
                        input=single_input[0],
                        size=input_height_width,
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            else:
                raise ValueError("Invalid interpolation type,")
        else:
            output_list.append(None)

    return output_list
