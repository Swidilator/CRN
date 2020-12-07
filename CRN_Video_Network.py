from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
from math import log2

from CRN.Refinement_Module import RefinementModule
from GAN.Blocks import UCIRBlock
from support_scripts.components import ResNetBlock, FlowNetWrapper


class CRNVideo(torch.nn.Module):
    def __init__(
        self,
        use_tanh: bool,
        input_tensor_size: Tuple[int, int],
        final_image_size: Tuple[int, int],
        num_output_images: int,
        num_classes: int,
        num_inner_channels: int,
        use_feature_encoder: bool,
        layer_norm_type: str,
    ):
        super(CRNVideo, self).__init__()

        self.use_tanh: bool = use_tanh
        self.input_tensor_size: Tuple[int, int] = input_tensor_size
        self.final_image_size: Tuple[int, int] = final_image_size
        self.num_output_images: int = num_output_images
        self.num_classes: int = num_classes
        self.num_inner_channels: int = num_inner_channels
        self.use_feature_encoder = use_feature_encoder
        self.layer_norm_type = layer_norm_type

        self.__NUM_OUTPUT_IMAGE_CHANNELS__: int = 3

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
        self.rms_conv_channel_settings: list = [
            min(self.num_inner_channels, x) for x in base_rms_conv_channel_settings
        ]

        self.num_rms: int = int(log2(final_image_size[0])) - 1

        self.rms_list: nn.ModuleList = nn.ModuleList(
            [
                RefinementModule(
                    semantic_input_channel_count=self.num_classes,
                    feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                    edge_map_input_channel_count=(self.use_feature_encoder * 1),
                    base_conv_channel_count=self.rms_conv_channel_settings[0],
                    prior_conv_channel_count=0,
                    final_conv_output_channel_count=0,
                    input_height_width=self.input_tensor_size,
                    norm_type=self.layer_norm_type,
                    prev_frame_count=2,
                    resnet_mode=True,
                    no_semantic_input=False,
                    no_image_input=True,
                )
            ]
        )

        self.rms_list.extend(
            [
                RefinementModule(
                    semantic_input_channel_count=self.num_classes,
                    feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                    edge_map_input_channel_count=(self.use_feature_encoder * 1),
                    base_conv_channel_count=self.rms_conv_channel_settings[i],
                    prior_conv_channel_count=self.rms_conv_channel_settings[i - 1],
                    final_conv_output_channel_count=0,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    norm_type=self.layer_norm_type,
                    prev_frame_count=2,
                    resnet_mode=True,
                    no_semantic_input=False,
                    no_image_input=True,
                )
                for i in range(1, self.num_rms - 1)
            ]
        )

        self.rms_list.append(
            RefinementModule(
                semantic_input_channel_count=self.num_classes,
                feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                edge_map_input_channel_count=(self.use_feature_encoder * 1),
                base_conv_channel_count=self.rms_conv_channel_settings[
                    self.num_rms - 1
                ],
                prior_conv_channel_count=self.rms_conv_channel_settings[
                    self.num_rms - 2
                ],
                final_conv_output_channel_count=self.__NUM_OUTPUT_IMAGE_CHANNELS__
                * self.num_output_images,
                input_height_width=final_image_size,
                norm_type=self.layer_norm_type,
                prev_frame_count=2,
                resnet_mode=True,
                no_semantic_input=False,
                no_image_input=True,
            )
        )

        # Flow network
        self.rms_list_flow: nn.ModuleList = nn.ModuleList(
            [
                RefinementModule(
                    semantic_input_channel_count=0,
                    feature_encoder_input_channel_count=0,
                    edge_map_input_channel_count=0,
                    base_conv_channel_count=self.rms_conv_channel_settings[0],
                    prior_conv_channel_count=0,
                    final_conv_output_channel_count=0,
                    input_height_width=self.input_tensor_size,
                    norm_type=self.layer_norm_type,
                    prev_frame_count=2,
                    resnet_mode=True,
                    no_semantic_input=True,
                    no_image_input=False,
                )
            ]
        )

        self.rms_list_flow.extend(
            [
                RefinementModule(
                    semantic_input_channel_count=0,
                    feature_encoder_input_channel_count=0,
                    edge_map_input_channel_count=0,
                    base_conv_channel_count=self.rms_conv_channel_settings[i],
                    prior_conv_channel_count=self.rms_conv_channel_settings[i - 1],
                    final_conv_output_channel_count=0,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    norm_type=self.layer_norm_type,
                    prev_frame_count=2,
                    resnet_mode=True,
                    no_semantic_input=True,
                    no_image_input=False,
                )
                for i in range(1, self.num_rms - 1)
            ]
        )

        self.rms_list_flow.append(
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
                input_height_width=final_image_size,
                norm_type=self.layer_norm_type,
                prev_frame_count=2,
                resnet_mode=True,
                no_semantic_input=True,
                no_image_input=False,
            )
        )

        # Conv for generating optical flow
        self.flow_conv_out: nn.Sequential = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                self.rms_conv_channel_settings[self.num_rms - 1],
                2,
                kernel_size=7,
                padding=0,
            ),
        )

        # Conv for generating merge mask
        self.mask_conv_out: nn.Sequential = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                self.rms_conv_channel_settings[self.num_rms - 1],
                1,
                kernel_size=7,
                padding=0,
            ),
            nn.Sigmoid(),
        )

        if self.use_tanh:
            self.tan_h = nn.Tanh()

        # Grid for warping
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
        output_list: list = [
            self.rms_list[0](
                msk, None, feature_encoding, edge_map, prev_images, prev_masks
            )
            + self.rms_list_flow[0](
                msk, None, feature_encoding, edge_map, prev_images, prev_masks
            )
        ]
        for i in range(1, len(self.rms_list) - 1):
            output_list.append(
                self.rms_list[i](
                    msk,
                    output_list[-1],
                    feature_encoding,
                    edge_map,
                    prev_images,
                    prev_masks,
                )
                + self.rms_list_flow[i](
                    msk,
                    output_list[-1],
                    feature_encoding,
                    edge_map,
                    prev_images,
                    prev_masks,
                )
            )

        # Generated image, use final gen rm
        output_gen: torch.Tensor = self.rms_list[-1](
            msk, output_list[-1], feature_encoding, edge_map, prev_images, prev_masks,
        )
        output_gen = (output_gen + 1.0) / 2.0

        # Optical flow and merge mask, use final flow rm
        output_flow_and_mask = self.rms_list_flow[-1](
            msk, output_list[-1], feature_encoding, edge_map, prev_images, prev_masks,
        )
        output_flow: torch.Tensor = self.flow_conv_out(output_flow_and_mask)
        output_mask: torch.Tensor = self.mask_conv_out(output_flow_and_mask)

        # Warp prior frame with flow
        output_warped: torch.Tensor = FlowNetWrapper.resample(
            prev_images[:, 0:3], output_flow, self.grid
        )

        # Merge generated image with warped image using merge mask
        output: torch.Tensor = (output_mask * output_gen) + (
            (torch.ones_like(output_mask) - output_mask) * output_warped
        )

        return (
            output,
            output_gen,
            output_warped,
            output_flow,
            output_mask,
        )
