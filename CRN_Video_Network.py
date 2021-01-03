from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
from math import log2

from CRN.Refinement_Module import RefinementModule
from support_scripts.components import FlowNetWrapper


class CRNVideo(torch.nn.Module):
    def __init__(
        self,
        use_tanh: bool,
        input_tensor_size: Tuple[int, int],
        final_image_size: Tuple[int, int],
        num_classes: int,
        num_inner_channels: int,
        use_feature_encoder: bool,
        layer_norm_type: str,
        use_resnet_rms: bool,
        num_resnet_processing_rms: int,
        num_prior_frames: int,
        use_optical_flow: bool,
        use_edge_map: bool,
        use_twin_network: bool,
        num_output_images: int,
    ):
        super(CRNVideo, self).__init__()

        self.use_tanh: bool = use_tanh
        self.input_tensor_size: Tuple[int, int] = input_tensor_size
        self.final_image_size: Tuple[int, int] = final_image_size
        self.num_classes: int = num_classes
        self.num_inner_channels: int = num_inner_channels
        self.use_feature_encoder: bool = use_feature_encoder
        self.layer_norm_type: str = layer_norm_type
        self.use_resnet_rms: bool = use_resnet_rms
        self.num_resnet_processing_rms: int = num_resnet_processing_rms
        self.num_prior_frames: int = num_prior_frames
        self.use_optical_flow: bool = use_optical_flow
        self.use_edge_map: bool = use_edge_map
        self.use_twin_network: bool = use_twin_network
        self.num_output_images: int = num_output_images

        self.num_output_image_channels: int = 3

        # Checking settings are valid
        if self.num_output_images > 1:
            assert (
                self.num_prior_frames == 0 and self.use_optical_flow is False
            ), "num_prior_frames > 0 required if use_optical_flow == True"

        if self.use_optical_flow:
            assert (
                self.num_prior_frames > 0
            ), "num_prior_frames > 0 required if use_optical_flow == True"

        # To manage memory usage, number of conv filters in each RM decreases over time
        base_rms_conv_channel_settings: list = [
            1024,
            1024,
            1024,
            1024,
            1024,
            512,
            512,
            128,
            32,
        ]
        # Modify settings to match input value for max number of conv filters
        self.rms_conv_channel_settings: list = [
            min(self.num_inner_channels, x) for x in base_rms_conv_channel_settings
        ]

        # Calculate number of RMs based on output image size
        self.num_rms: int = int(log2(final_image_size[0])) - 1

        # Create and pupulate lists with RMs
        self.rms_list: nn.ModuleList = nn.ModuleList(
            [
                RefinementModule(
                    semantic_input_channel_count=self.num_classes,
                    feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                    edge_map_input_channel_count=(self.use_edge_map * 1),
                    base_conv_channel_count=self.rms_conv_channel_settings[0],
                    prior_conv_channel_count=0,
                    final_conv_output_channel_count=0,
                    is_final_module=False,
                    input_height_width=self.input_tensor_size,
                    norm_type=self.layer_norm_type,
                    num_prior_frames=self.num_prior_frames,
                    num_resnet_processing_rms=0,
                    resnet_mode=self.use_resnet_rms,
                    resnet_no_add=False,
                    use_semantic_input=True,
                    use_image_input=(
                        self.num_prior_frames > 0 and not self.use_twin_network
                    ),
                    is_flow_output=False,
                    is_twin_model=self.use_twin_network,
                )
            ]
        )

        self.rms_list.extend(
            [
                RefinementModule(
                    semantic_input_channel_count=self.num_classes,
                    feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                    edge_map_input_channel_count=(self.use_edge_map * 1),
                    base_conv_channel_count=self.rms_conv_channel_settings[i],
                    prior_conv_channel_count=self.rms_conv_channel_settings[i - 1],
                    final_conv_output_channel_count=0,
                    is_final_module=False,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    norm_type=self.layer_norm_type,
                    num_prior_frames=self.num_prior_frames,
                    num_resnet_processing_rms=0,
                    resnet_mode=self.use_resnet_rms,
                    resnet_no_add=False,
                    use_semantic_input=True,
                    use_image_input=(
                        self.num_prior_frames > 0 and not self.use_twin_network
                    ),
                    is_twin_model=self.use_twin_network,
                    is_flow_output=False,
                )
                for i in range(1, self.num_rms - 1)
            ]
        )

        self.rms_list.append(
            RefinementModule(
                semantic_input_channel_count=self.num_classes,
                feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                edge_map_input_channel_count=(self.use_edge_map * 1),
                base_conv_channel_count=self.rms_conv_channel_settings[
                    self.num_rms - 1
                ],
                prior_conv_channel_count=self.rms_conv_channel_settings[
                    self.num_rms - 2
                ],
                final_conv_output_channel_count=self.num_output_image_channels
                * self.num_output_images,
                is_final_module=True,
                input_height_width=final_image_size,
                norm_type=self.layer_norm_type,
                num_prior_frames=self.num_prior_frames,
                num_resnet_processing_rms=self.num_resnet_processing_rms,
                resnet_mode=self.use_resnet_rms,
                resnet_no_add=False,
                use_semantic_input=True,
                use_image_input=(
                    self.num_prior_frames > 0 and not self.use_twin_network
                ),
                is_flow_output=self.use_optical_flow and not self.use_twin_network,
                is_twin_model=self.use_twin_network,
            )
        )

        # Flow network
        self.rms_list_twin: nn.ModuleList = nn.ModuleList(
            [
                RefinementModule(
                    semantic_input_channel_count=0,
                    feature_encoder_input_channel_count=0,
                    edge_map_input_channel_count=0,
                    base_conv_channel_count=self.rms_conv_channel_settings[0],
                    prior_conv_channel_count=0,
                    final_conv_output_channel_count=0,
                    is_final_module=False,
                    input_height_width=self.input_tensor_size,
                    norm_type=self.layer_norm_type,
                    num_prior_frames=self.num_prior_frames,
                    num_resnet_processing_rms=0,
                    resnet_mode=self.use_resnet_rms,
                    resnet_no_add=False,
                    use_semantic_input=False,
                    use_image_input=self.num_prior_frames > 0,
                    is_twin_model=True,
                    is_flow_output=False,
                )
                if self.use_twin_network and self.num_prior_frames > 0
                else nn.Identity()
            ]
        )

        self.rms_list_twin.extend(
            [
                RefinementModule(
                    semantic_input_channel_count=0,
                    feature_encoder_input_channel_count=0,
                    edge_map_input_channel_count=0,
                    base_conv_channel_count=self.rms_conv_channel_settings[i],
                    prior_conv_channel_count=self.rms_conv_channel_settings[i - 1],
                    final_conv_output_channel_count=0,
                    is_final_module=False,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    norm_type=self.layer_norm_type,
                    num_prior_frames=self.num_prior_frames,
                    num_resnet_processing_rms=0,
                    resnet_mode=self.use_resnet_rms,
                    resnet_no_add=False,
                    use_semantic_input=False,
                    use_image_input=self.num_prior_frames > 0,
                    is_flow_output=False,
                    is_twin_model=True,
                )
                if self.use_twin_network
                else nn.Identity()
                for i in range(1, self.num_rms - 1)
            ]
        )

        self.rms_list_twin.append(
            RefinementModule(
                semantic_input_channel_count=0,
                feature_encoder_input_channel_count=0,
                edge_map_input_channel_count=0,
                base_conv_channel_count=self.rms_conv_channel_settings[
                    self.num_rms - 1
                ],
                prior_conv_channel_count=self.rms_conv_channel_settings[
                    self.num_rms - 2
                ],
                final_conv_output_channel_count=0,
                is_final_module=True,
                input_height_width=final_image_size,
                norm_type=self.layer_norm_type,
                num_prior_frames=self.num_prior_frames,
                num_resnet_processing_rms=self.num_resnet_processing_rms,
                resnet_mode=self.use_resnet_rms,
                resnet_no_add=False,
                use_semantic_input=False,
                use_image_input=self.num_prior_frames > 0,
                is_flow_output=True,
                is_twin_model=True,
            )
            if self.use_twin_network and self.use_optical_flow
            else nn.Identity()
        )

        if self.use_tanh:
            self.tan_h = nn.Tanh()

        # Grid for warping
        if self.use_optical_flow:
            self.grid: torch.Tensor = FlowNetWrapper.get_grid(
                1, self.final_image_size, torch.device("cuda:0")
            )

    def forward(
        self,
        msk: torch.Tensor,
        feature_encoding: torch.Tensor,
        edge_map: torch.Tensor,
        prev_images: torch.Tensor = None,
        prev_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # List of output, both flow and gen nets sum and are stored here
        output_list: list = []

        output_1 = self.rms_list[0](
            msk, None, feature_encoding, edge_map, prev_images, prev_masks
        )["x"]
        if self.use_twin_network and self.num_prior_frames > 0:
            output_1 = (
                output_1
                + self.rms_list_twin[0](
                    msk, None, feature_encoding, edge_map, prev_images, prev_masks
                )["x"]
            )
        output_list.append(output_1)

        for i in range(1, len(self.rms_list) - 1):
            output_i = self.rms_list[i](
                msk,
                output_list[-1],
                feature_encoding,
                edge_map,
                prev_images,
                prev_masks,
            )["x"]

            if self.use_twin_network:  # Preparation for Siamese network test
                output_i = output_i + self.rms_list_twin[i](
                    msk,
                    output_list[-1],
                    feature_encoding,
                    edge_map,
                    prev_images,
                    prev_masks,
                )["x"]

            output_list.append(output_i)

        # Generated image, use final gen RM
        output_final_rms_list: dict = self.rms_list[-1](
            msk,
            output_list[-1],
            feature_encoding,
            edge_map,
            prev_images,
            prev_masks,
        )
        output_gen = (output_final_rms_list["out_img"] + 1.0) / 2.0

        output_flow = None
        output_mask = None
        output_warped = None

        # Optical flow and merge mask, use final flow RM
        if self.use_optical_flow:
            if not self.use_twin_network:
                output_flow: Optional[torch.Tensor] = output_final_rms_list["out_flow"]
                output_mask: Optional[torch.Tensor] = output_final_rms_list["out_mask"]
            else:
                output_flow_and_mask = self.rms_list_twin[-1](
                    msk,
                    output_list[-1],
                    feature_encoding,
                    edge_map,
                    prev_images,
                    prev_masks,
                )
                output_flow: Optional[torch.Tensor] = output_flow_and_mask["out_flow"]
                output_mask: Optional[torch.Tensor] = output_flow_and_mask["out_mask"]

            # Warp prior frame with flow
            output_warped: Optional[torch.Tensor] = FlowNetWrapper.resample(
                prev_images[:, 0:3], output_flow, self.grid
            )
            output: torch.Tensor = (output_mask * output_gen) + (
                (torch.ones_like(output_mask) - output_mask) * output_warped
            )
        else:
            output = output_gen
            output_gen = None

        # For multiple output images, split final output and put back together as separate images.
        if self.num_output_images > 1:
            a, b, c = torch.chunk(output.unsqueeze(2), 3, 1)
            output = torch.cat((a, b, c), 2)
        else:
            output = output.unsqueeze(1)

        return (
            output,
            output_gen,
            output_warped,
            output_flow,
            output_mask,
        )
