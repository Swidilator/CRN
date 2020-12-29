import torch
import torch.nn as nn
from typing import List

from support_scripts.components import RMBlock, ResNetBlock, Block


class RefinementModule(nn.Module):
    def __init__(
        self,
        semantic_input_channel_count: int,
        feature_encoder_input_channel_count: int,
        edge_map_input_channel_count: int,
        base_conv_channel_count: int,
        prior_conv_channel_count: int,
        final_conv_output_channel_count: int,
        is_final_module: bool,
        input_height_width: tuple,
        norm_type: str,
        num_prior_frames: int,
        num_resnet_processing_rms: int,
        resnet_mode: bool,
        resnet_no_add: bool,
        use_semantic_input: bool,
        use_image_input: bool,
        is_flow_output: bool,
        is_twin_model: bool,
    ):
        super().__init__()

        self.input_height_width: tuple = input_height_width
        self.prior_conv_channel_count: int = prior_conv_channel_count
        self.num_prior_frames: int = num_prior_frames
        self.resnet_mode: bool = resnet_mode
        self.resnet_no_add: bool = resnet_no_add
        self.num_resnet_processing_rms: int = num_resnet_processing_rms
        self.use_semantic_input: bool = use_semantic_input
        self.use_image_input: bool = use_image_input
        self.is_twin_model: bool = is_twin_model
        self.is_flow_output: bool = is_flow_output
        self.is_final_module: bool = is_final_module
        self.base_conv_channel_count: int = base_conv_channel_count

        # Total number of input channels
        self.total_image_input_channel_count: int = num_prior_frames * 3
        self.final_conv_output_channel_count: int = final_conv_output_channel_count
        self.use_feature_encoder: int = feature_encoder_input_channel_count > 0

        self.is_first_module: bool = self.prior_conv_channel_count == 0

        self.complex_input_mode: bool = self.is_twin_model or self.resnet_mode

        # Total number of input channels
        self.total_semantic_input_channel_count: int = (
            (semantic_input_channel_count * (num_prior_frames + 1))
            + feature_encoder_input_channel_count
            + edge_map_input_channel_count
            + (prior_conv_channel_count if not self.complex_input_mode else 0)
            + (
                self.total_image_input_channel_count
                if not self.complex_input_mode
                else 0
            )
        )

        if self.use_semantic_input:
            if self.total_semantic_input_channel_count == 0:
                raise ValueError(
                    "total_semantic_input_channel_count is 0 but semantic input is required."
                )
            self.rm_block_1_semantic = RMBlock(
                self.base_conv_channel_count,
                self.total_semantic_input_channel_count,
                self.input_height_width,
                kernel_size=3,
                norm_type=norm_type,
                num_conv_groups=1,
            )

        if self.use_image_input and self.complex_input_mode:
            if self.num_prior_frames == 0:
                raise ValueError("num_prior_frames is 0 but image input requested.")
            self.rm_block_1_image = RMBlock(
                self.base_conv_channel_count,
                self.total_image_input_channel_count,
                self.input_height_width,
                kernel_size=3,
                norm_type=norm_type,
                num_conv_groups=1,
            )

        if not self.is_first_module and self.complex_input_mode:
            self.rm_block_1_prior_layer = RMBlock(
                self.base_conv_channel_count,
                self.prior_conv_channel_count,
                self.input_height_width,
                kernel_size=3,
                norm_type=norm_type,
                num_conv_groups=1,
            )

        if not resnet_mode:
            self.rm_block_2 = RMBlock(
                self.base_conv_channel_count,
                self.base_conv_channel_count,
                self.input_height_width,
                kernel_size=3,
                norm_type=norm_type,
                num_conv_groups=1,
            )
            if self.complex_input_mode:
                self.rm_block_3 = RMBlock(
                    self.base_conv_channel_count,
                    self.base_conv_channel_count,
                    self.input_height_width,
                    kernel_size=3,
                    norm_type=norm_type,
                    num_conv_groups=1,
                )

        else:
            self.resnet_block_1 = ResNetBlock(
                self.base_conv_channel_count,
                self.input_height_width,
                self.resnet_no_add,
            )
            self.resnet_block_2 = ResNetBlock(
                self.base_conv_channel_count,
                self.input_height_width,
                self.resnet_no_add,
            )

        if self.is_final_module:
            # if self.resnet_mode:
            self.rm_final_processing_list: nn.ModuleList = nn.ModuleList()
            if self.num_resnet_processing_rms > 0:
                for i in range(self.num_resnet_processing_rms):
                    self.rm_final_processing_list.append(
                        RMBlock(
                            self.base_conv_channel_count,
                            self.base_conv_channel_count,
                            self.input_height_width,
                            kernel_size=3,
                            norm_type=norm_type,
                            num_conv_groups=1,
                        )
                    )

            if self.final_conv_output_channel_count > 0:
                self.final_conv = nn.Conv2d(
                    self.base_conv_channel_count,
                    self.final_conv_output_channel_count,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                Block.init_conv_weights(
                    self.final_conv, init_type="xavier", zero_bias=True
                )

            if self.is_flow_output:
                # Conv for generating optical flow
                self.flow_conv_out: nn.Sequential = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(
                        self.base_conv_channel_count,
                        2,
                        kernel_size=7,
                        padding=0,
                    ),
                )

                # Conv for generating merge mask
                self.mask_conv_out: nn.Sequential = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(
                        self.base_conv_channel_count,
                        1,
                        kernel_size=7,
                        padding=0,
                    ),
                    nn.Sigmoid(),
                )

        # Compatibility with old saves, resnet does not have compat
        if not self.resnet_mode and self.use_semantic_input:
            self.conv_1 = self.rm_block_1_semantic.conv_1
            self.norm_1 = self.rm_block_1_semantic.norm_1
            self.conv_2 = self.rm_block_2.conv_1
            self.norm_2 = self.rm_block_2.norm_1

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

        # Process semantic input
        if self.use_semantic_input:
            if not self.complex_input_mode:
                # Concatenate semantic inputs together for use in semantic entry conv
                x_semantic_input: list = [
                    x
                    for x in (
                        mask,
                        prior_layers,
                        feature_selection,
                        edge_map,
                        prev_masks,
                        prev_frames,
                    )
                    if x is not None
                ]
            else:
                x_semantic_input: list = [
                    x
                    for x in (mask, feature_selection, edge_map, prev_masks)
                    if x is not None
                ]

            x_semantic: torch.Tensor = torch.cat(x_semantic_input, dim=1)
            x = self.rm_block_1_semantic(x_semantic, relu_loc="after")
        else:
            mask: torch.Tensor
            x = None

        if not self.is_first_module and self.complex_input_mode:
            x_prior_layers: torch.Tensor = self.rm_block_1_prior_layer(prior_layers)
            x = x_prior_layers + x if x is not None else x_prior_layers

        # If previous frames are present, then pass them into the image entry conv and add them to the semantic output
        if self.use_image_input and self.complex_input_mode:
            x_image: torch.Tensor = self.rm_block_1_image(prev_frames, relu_loc="after")
            x = x_image + x if x is not None else x_image

        if not self.resnet_mode:
            # Continue as normal
            x = self.rm_block_2(x, relu_loc="after")
            if self.complex_input_mode:
                x = self.rm_block_3(x, relu_loc="after")
        else:
            x = self.resnet_block_1(x)
            x = self.resnet_block_2(x)

            # Processing intended for final rm

        out_img = None
        out_flow = None
        out_mask = None

        if self.is_final_module:
            # if self.resnet_mode:
            for item in self.rm_final_processing_list:
                if self.resnet_mode:
                    x = item(x, relu_loc="before")
                else:
                    x = item(x, relu_loc="after")
            if self.final_conv_output_channel_count > 0:
                out_img = self.final_conv(x)
            if self.is_flow_output:
                out_flow = self.flow_conv_out(x)
                out_mask = self.mask_conv_out(x)

        output_dict: dict = {
            "x": x,
            "out_img": out_img,
            "out_flow": out_flow,
            "out_mask": out_mask,
        }

        return output_dict


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
