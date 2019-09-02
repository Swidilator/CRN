import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
from math import log2
import time
import random

from Helper_Stuff import *
from Data_Management import CRNDataset
from CRN.Perceptual_Loss import PerceptualLossNetwork
from Training_Framework import MastersModel

import wandb


class CRNFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        input_tensor_size: image_size,
        max_input_height_width: image_size,
        num_output_images: int,
        num_inner_channels: int,
        num_classes: int,
        batch_size: int,
        num_loader_workers: int,
        history_len: int,
    ):
        super(CRNFramework, self).__init__(device=device)
        self.batch_size = batch_size
        self.input_tensor_size = input_tensor_size

        self.model_name: str = "CRN_Latest.pt"

        self.__set_data_loader__(
            data_path,
            max_input_height_width,
            num_classes,
            batch_size,
            num_loader_workers,
        )
        self.__set_model__(
            input_tensor_size,
            max_input_height_width,
            num_output_images,
            num_classes,
            num_inner_channels,
            history_len,
        )

    @property
    def wandb_trainable_model(self):
        return self.crn

    def __set_data_loader__(
        self,
        data_path,
        max_input_height_width,
        num_classes,
        batch_size,
        num_loader_workers,
    ):
        self.__data_set_train__ = CRNDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="train",
            num_classes=num_classes,
            should_flip=True,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_test__ = CRNDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="test",
            num_classes=num_classes,
            should_flip=False,
        )

        self.data_loader_test: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_test__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        # self.__data_set_val__ = CRNDataset(
        #     max_input_height_width=max_input_height_width,
        #     root=data_path,
        #     split="val",
        #     num_classes=num_classes,
        #     should_flip=False
        # )
        #
        # self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        #     self.__data_set_val__,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_loader_workers,
        # )

    def __set_model__(
        self,
        input_tensor_size,
        max_input_height_width,
        num_output_images,
        num_classes,
        num_inner_channels,
        history_len,
    ) -> None:
        self.crn: CRN = CRN(
            input_tensor_size=input_tensor_size,
            final_image_size=max_input_height_width,
            num_output_images=num_output_images,
            num_classes=num_classes,
            num_inner_channels=num_inner_channels,
        )
        IMAGE_CHANNELS = 3
        self.crn = self.crn.to(self.device)

        self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
            (IMAGE_CHANNELS, max_input_height_width[0], max_input_height_width[1]),
            history_len,
        )
        self.loss_net = self.loss_net.to(self.device)

        # self.optimizer = torch.optim.SGD(self.crn.parameters(), lr=0.01, momentum=0.9)

        self.optimizer = torch.optim.Adam(
            self.crn.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

    def save_model(self, model_dir: str, snapshot: bool = False) -> None:
        localtime: time.localtime() = time.localtime(time.time())
        model_snapshot: str = "CRN" + "_"
        model_snapshot = model_snapshot + str(localtime.tm_year) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_mon) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_mday) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_hour) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_min) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_sec) + ".pt"

        if snapshot:
            torch.save(
                {
                    "model_state_dict": self.crn.state_dict(),
                    "loss_layer_scales": self.loss_net.loss_layer_scales,
                    "loss_history": self.loss_net.loss_layer_history,
                },
                model_dir + model_snapshot,
            )
        torch.save(
            {
                "model_state_dict": self.crn.state_dict(),
                "loss_layer_scales": self.loss_net.loss_layer_scales,
                "loss_history": self.loss_net.loss_layer_history,
            },
            model_dir + self.model_name,
        )

    def load_model(self, model_dir: str, model_snapshot: str = None) -> None:
        if model_snapshot is not None:
            checkpoint = torch.load(
                model_dir + model_snapshot, map_location=self.device
            )
            self.crn.load_state_dict(checkpoint["model_state_dict"])
            self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
            self.loss_net.loss_layer_history = checkpoint["loss_history"]
        else:
            checkpoint = torch.load(
                model_dir + self.model_name, map_location=self.device
            )
            self.crn.load_state_dict(checkpoint["model_state_dict"])
            self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
            self.loss_net.loss_layer_history = checkpoint["loss_history"]

    def train(self, update_lambdas: bool) -> epoch_output:
        self.crn.train()
        torch.cuda.empty_cache()

        loss_ave: float = 0.0
        loss_total: float = 0.0

        if update_lambdas:
            self.loss_net.update_lambdas()

        for batch_idx, (img, msk) in enumerate(self.data_loader_train):
            self.optimizer.zero_grad()
            img: torch.Tensor = img.to(self.device)
            msk: torch.Tensor = msk.to(self.device)
            this_batch_size: int = msk.shape[0]
            # noise: torch.Tensor = torch.randn(
            #     this_batch_size,
            #     1,
            #     self.input_tensor_size[0],
            #     self.input_tensor_size[1],
            #     device=self.device,
            # )
            # noise = noise.to(self.device)

            # out: torch.Tensor = self.crn(inputs=(msk, noise, self.batch_size))
            out: torch.Tensor = self.crn(inputs=(msk, None, self.batch_size))

            img = CRNFramework.__normalise__(img)
            out = CRNFramework.__normalise__(out)

            loss: torch.Tensor = self.loss_net((out, img))
            loss.backward()
            loss_ave += loss.item()
            loss_total += loss.item()
            if batch_idx * self.batch_size % 120 == 112:
                print(
                    "Batch: {batch}\nLoss: {loss_val}".format(
                        batch=batch_idx, loss_val=loss_ave * this_batch_size
                    )
                )
                # WandB logging, if WandB disabled this should skip the logging without error
                no_except(wandb.log, {"Batch Loss": loss_ave * this_batch_size})
                loss_ave = 0.0
            self.optimizer.step()
            # del loss, msk, noise, img
            del loss, msk, img
        del loss_ave
        return loss_total, None

    def eval(self) -> epoch_output:
        self.crn.eval()
        loss_total: torch.Tensor = torch.Tensor([0.0]).to(self.device)
        loss_total.requires_grad = False
        for batch_idx, (img, msk) in enumerate(self.data_loader_val):
            img: torch.Tensor = img.to(self.device)
            msk: torch.Tensor = msk.to(self.device)
            # noise: torch.Tensor = torch.randn(
            #     msk.shape[0],
            #     1,
            #     self.input_tensor_size[0],
            #     self.input_tensor_size[1],
            #     device=self.device,
            # )
            # noise = noise.to(self.device)

            out: torch.Tensor = self.crn(inputs=(msk, None, self.batch_size))

            img = CRNFramework.__normalise__(img)
            out = CRNFramework.__normalise__(out)

            loss: torch.Tensor = self.loss_net((out, img))
            loss_total = loss_total + loss
            del loss, msk, img
        return loss_total.item(), None

    def sample(self, k: int) -> sample_output:
        self.crn.eval()
        sample_list: list = random.sample(range(len(self.__data_set_test__)), k)
        outputs: sample_output = []
        # noise: torch.Tensor = torch.randn(
        #     1,
        #     1,
        #     self.input_tensor_size[0],
        #     self.input_tensor_size[1],
        #     device=self.device,
        # )
        transform: transforms.ToPILImage = transforms.ToPILImage()
        for i, val in enumerate(sample_list):
            img, msk = self.__data_set_test__[val]
            msk = msk.to(self.device).unsqueeze(0)
            img_out: torch.Tensor = self.crn(inputs=(msk, None, self.batch_size))
            img_out = img_out.cpu()[0]
            outputs.append(transform(img_out))
            del img, msk
        return outputs

    @staticmethod
    def __single_channel_normalise__(
        channel: torch.Tensor, params: tuple
    ) -> torch.Tensor:
        # channel = [H ,W]   params = (mean, std)
        return (channel - params[0]) / params[1]

    @staticmethod
    def __single_image_normalise__(image: torch.Tensor, mean, std) -> torch.Tensor:
        for i in range(3):
            image[i] = CRNFramework.__single_channel_normalise__(
                image[i], (mean[i], std[i])
            )
        return image

    @staticmethod
    def __normalise__(input: torch.Tensor) -> torch.Tensor:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if len(input.shape) == 4:
            for i in range(input.shape[0]):
                input[i] = CRNFramework.__single_image_normalise__(input[i], mean, std)
        else:
            input = CRNFramework.__single_image_normalise__(input, mean, std)
        return input


class RefinementModule(modules.Module):
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
        prior_layer_channel_count: int,
        semantic_input_channel_count: int,
        output_channel_count: int,
        input_height_width: tuple,
        is_final_module: bool = False,
        final_channel_count: int = 3,
    ):
        super(RefinementModule, self).__init__()

        self.input_height_width: tuple = input_height_width
        self.total_input_channel_count: int = (
            prior_layer_channel_count + semantic_input_channel_count
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
        nn.init.xavier_uniform_(self.conv_1.weight, gain=1)
        nn.init.constant_(self.conv_1.bias, 0)

        self.layer_norm_1 = nn.LayerNorm(
            RefinementModule.change_output_channel_size(
                input_height_width, self.output_channel_count
            )
        )

        self.conv_2 = nn.Conv2d(
            self.output_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.conv_2.weight, gain=1)
        nn.init.constant_(self.conv_2.bias, 0)

        # self.layer_norm_2 = nn.LayerNorm(
        #     RefinementModule.change_output_channel_size(
        #         input_height_width, self.output_channel_count
        #     )
        # )
        #
        # self.conv_3 = nn.Conv2d(
        #     self.output_channel_count,
        #     self.output_channel_count,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )
        # nn.init.xavier_uniform(self.conv_3.weight, gain=1)
        # nn.init.constant(self.conv_3.bias, 0)

        if not self.is_final_module:
            self.layer_norm_2 = nn.LayerNorm(
                RefinementModule.change_output_channel_size(
                    input_height_width, self.output_channel_count
                )
            )
        else:
            self.final_conv = nn.Conv2d(
                self.output_channel_count,
                self.final_channel_count,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.xavier_uniform_(self.final_conv.weight, gain=1)
            nn.init.constant_(self.final_conv.bias, 0)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)

    @staticmethod
    def change_output_channel_size(
        input_height_width: tuple, output_channel_number: int
    ):
        size_list = list(input_height_width)
        size_list.insert(0, output_channel_number)
        # print(size_list)
        return torch.Size(size_list)

    def forward(self, inputs: tuple):
        mask: torch.Tensor = inputs[0]
        prior_layers: torch.Tensor = inputs[1]
        mask = torch.nn.functional.interpolate(
            input=mask, size=self.input_height_width, mode="nearest"
        )
        if prior_layers is not None:
            prior_layers = torch.nn.functional.interpolate(
                input=prior_layers, size=self.input_height_width, mode="bilinear"
            )

            x = torch.cat((mask, prior_layers), dim=1)
        else:
            x = mask

        x = self.conv_1(x)
        # print(x.size())
        x = self.layer_norm_1(x)
        x = self.leakyReLU(x)

        x = self.conv_2(x)
        # print(x.size())
        # x = self.layer_norm_2(x)
        # x = self.leakyReLU(x)

        # x = self.conv_3(x)
        # print(x.size())
        if not self.is_final_module:
            x = self.layer_norm_2(x)
            x = self.leakyReLU(x)
            # x = self.dropout(x)
        else:
            x = self.final_conv(x)
        return x


class CRN(torch.nn.Module):
    def __init__(
        self,
        input_tensor_size: image_size,
        final_image_size: image_size,
        num_output_images: int,
        num_classes: int,
        num_inner_channels: int,
    ):
        super(CRN, self).__init__()

        self.input_tensor_size: image_size = input_tensor_size
        self.final_image_size: image_size = final_image_size
        self.num_output_images: int = num_output_images
        self.num_classes: int = num_classes
        self.num_inner_channels: int = num_inner_channels

        self.__NUM_NOISE_CHANNELS__: int = 0
        self.__NUM_OUTPUT_IMAGE_CHANNELS__: int = 3

        self.num_rms: int = int(log2(final_image_size[0])) - 1

        self.rms_list: list = []

        self.rms_list.append(
            RefinementModule(
                prior_layer_channel_count=self.__NUM_NOISE_CHANNELS__,
                semantic_input_channel_count=num_classes,
                output_channel_count=self.num_inner_channels,
                input_height_width=input_tensor_size,
                is_final_module=False,
            )
        )

        for i in range(1, self.num_rms - 1):
            self.rms_list.append(
                RefinementModule(
                    prior_layer_channel_count=self.num_inner_channels,
                    semantic_input_channel_count=num_classes,
                    output_channel_count=self.num_inner_channels,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    is_final_module=False,
                )
            )

        self.rms_list.append(
            RefinementModule(
                prior_layer_channel_count=self.num_inner_channels,
                semantic_input_channel_count=num_classes,
                output_channel_count=self.num_inner_channels,
                input_height_width=final_image_size,
                is_final_module=True,
                final_channel_count=self.__NUM_OUTPUT_IMAGE_CHANNELS__
                * num_output_images,
            )
        )

        self.rms = nn.Sequential(*self.rms_list)
        self.tan_h = nn.Tanh()

    def __del__(self):
        del self.rms

    def forward(self, inputs: list):
        mask: torch.Tensor = inputs[0]
        noise: torch.Tensor = inputs[1]
        batch_size: int = inputs[2]
        x: torch.Tensor = self.rms[0]([mask, noise])
        for i in range(1, self.num_rms):
            x = self.rms[i]([mask, x])
        # TanH for squeezing outputs to [-1, 1]
        x = self.tan_h(x).clone()
        return x
