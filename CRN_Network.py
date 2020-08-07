from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
from math import log2

from CRN.Refinement_Module import RefinementModule
from support_scripts.components import FeatureEncoder


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
    ):
        super(CRN, self).__init__()

        self.use_tanh: bool = use_tanh
        self.input_tensor_size: Tuple[int, int] = input_tensor_size
        self.final_image_size: Tuple[int, int] = final_image_size
        self.num_output_images: int = num_output_images
        self.num_classes: int = num_classes
        self.num_inner_channels: int = num_inner_channels
        self.use_feature_encoder = use_feature_encoder
        self.layer_norm_type = layer_norm_type

        self.__NUM_NOISE_CHANNELS__: int = 0
        self.__NUM_OUTPUT_IMAGE_CHANNELS__: int = 3

        if self.use_feature_encoder:
            # Todo Find better way of setting these parameters
            self.feature_encoder: FeatureEncoder = FeatureEncoder(3, 3, 4)

        self.num_rms: int = int(log2(final_image_size[0])) - 1

        self.rms_list: nn.ModuleList = nn.ModuleList(
            [
                RefinementModule(
                    semantic_input_channel_count=self.num_classes,
                    noise_input_channel_count=self.__NUM_NOISE_CHANNELS__,
                    feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                    base_inner_channel_count=self.num_inner_channels,
                    final_conv_output_channel_count=0,
                    first_rm=True,
                    input_height_width=self.input_tensor_size,
                    norm_type=self.layer_norm_type,
                )
            ]
        )

        self.rms_list.extend(
            [
                RefinementModule(
                    semantic_input_channel_count=self.num_classes,
                    noise_input_channel_count=0,
                    feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                    base_inner_channel_count=self.num_inner_channels,
                    final_conv_output_channel_count=0,
                    first_rm=False,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    norm_type=self.layer_norm_type,
                )
                for i in range(1, self.num_rms - 1)
            ]
        )

        self.rms_list.append(
            RefinementModule(
                semantic_input_channel_count=self.num_classes,
                noise_input_channel_count=0,
                feature_encoder_input_channel_count=(self.use_feature_encoder * 3),
                base_inner_channel_count=self.num_inner_channels,
                final_conv_output_channel_count=self.__NUM_OUTPUT_IMAGE_CHANNELS__
                * num_output_images,
                first_rm=False,
                input_height_width=final_image_size,
                norm_type=self.layer_norm_type,
            )
        )
        if self.use_tanh:
            self.tan_h = nn.Tanh()

    def forward(self, msk, real_img, instance_original, noise):
        if self.use_feature_encoder:
            feature_selection: Optional[torch.Tensor] = self.feature_encoder(
                real_img, instance_original
            )
        else:
            feature_selection: Optional[torch.Tensor] = None

        return self.generate_output(msk, feature_selection, noise)

    def generate_output(
        self,
        msk: torch.Tensor,
        feature_selection: torch.Tensor,
        noise: Union[torch.Tensor, None],
    ) -> torch.Tensor:

        output: torch.Tensor = self.rms_list[0](msk, noise, feature_selection)
        for i in range(1, len(self.rms_list)):
            output = self.rms_list[i](msk, output, feature_selection)

        a, b, c = torch.chunk(output.permute(1, 0, 2, 3).unsqueeze(0), 3, 1)
        output = torch.cat((a, b, c), 2)

        # TanH for squeezing outputs to [-1, 1]
        if self.use_tanh:
            output = self.tan_h(output).clone()

        # Squeeze output to [0,1]
        output = (output + 1.0) / 2.0
        return output
