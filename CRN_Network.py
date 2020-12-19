from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
from math import log2

from CRN.Refinement_Module import RefinementModule


class CRN(torch.nn.Module):
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
        use_resnet_rms: bool,
        num_resnet_processing_rms: int,
    ):
        super(CRN, self).__init__()

        self.use_tanh: bool = use_tanh
        self.input_tensor_size: Tuple[int, int] = input_tensor_size
        self.final_image_size: Tuple[int, int] = final_image_size
        self.num_output_images: int = num_output_images
        self.num_classes: int = num_classes
        self.num_inner_channels: int = num_inner_channels
        self.use_feature_encoder: bool = use_feature_encoder
        self.layer_norm_type: str = layer_norm_type
        self.use_resnet_rms: bool = use_resnet_rms
        self.num_resnet_processing_rms: int = num_resnet_processing_rms

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
                    is_final_module=False,
                    input_height_width=self.input_tensor_size,
                    norm_type=self.layer_norm_type,
                    prev_frame_count=0,
                    resnet_mode=self.use_resnet_rms,
                    resnet_no_add=False,
                    num_resnet_processing_rms=0,
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
                    is_final_module=False,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    norm_type=self.layer_norm_type,
                    prev_frame_count=0,
                    resnet_mode=self.use_resnet_rms,
                    resnet_no_add=False,
                    num_resnet_processing_rms=0,
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
                is_final_module=True,
                input_height_width=final_image_size,
                norm_type=self.layer_norm_type,
                prev_frame_count=0,
                resnet_mode=self.use_resnet_rms,
                resnet_no_add=True,
                num_resnet_processing_rms=self.num_resnet_processing_rms,
            )
        )

        if self.use_tanh:
            self.tan_h = nn.Tanh()

    def forward(
        self, msk: torch.Tensor, feature_encoding: torch.Tensor, edge_map: torch.Tensor
    ) -> torch.Tensor:

        output: torch.Tensor = self.rms_list[0](msk, None, feature_encoding, edge_map)
        for i in range(1, len(self.rms_list)):
            output = self.rms_list[i](msk, output, feature_encoding, edge_map)

        a, b, c = torch.chunk(output.unsqueeze(2), 3, 1)
        output = torch.cat((a, b, c), 2)

        # TanH for squeezing outputs to [-1, 1]
        if self.use_tanh:
            output = self.tan_h(output).clone()

        # Squeeze output to [0,1]
        output = (output + 1.0) / 2.0
        return output
