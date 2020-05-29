import torch
import torch.nn as nn
import torch.nn.modules as modules
import torchvision

import wandb
from typing import Tuple, List, Any

# from torchvision.transforms import Resize
# from copy import copy


# class PerceptualDifference(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, img, trth):
#         result = (img - trth).abs().sum()
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None


class CircularList:
    def __init__(self, input: int):
        self.len = input
        self.data: list = [1.0 for x in range(input)]
        self.pointer: int = 0

    def update(self, input: float) -> None:
        self.data[self.pointer] = input
        if self.pointer + 1 == self.len:
            self.pointer = 0
        else:
            self.pointer += 1

    def sum(self) -> float:
        return sum(self.data)

    def mean(self) -> float:
        return sum(self.data) / self.len


def get_layer_values(
    self: torch.nn.modules.conv.Conv2d, input: tuple, output: torch.Tensor
) -> None:
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested in
    self.stored_output = output


class PerceptualLossNetwork(modules.Module):
    def __init__(
        self,
        history_len: int,
        base_model: str,
        device: torch.device,
        use_loss_output_image: bool,
    ):
        super(PerceptualLossNetwork, self).__init__()

        self.device: torch.device = device
        self.base_model = base_model
        self.use_loss_output_image = use_loss_output_image

        self.output_feature_layers: list = []

        if self.base_model == "VGG":
            vgg = torchvision.models.vgg19(pretrained=True, progress=True)
            self.feature_network = vgg.features
            del vgg
        elif self.base_model == "MobileNet":
            mobile_net = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
            self.feature_network = mobile_net.features
            del mobile_net

        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()
        torch.cuda.empty_cache()

        self.norm = torch.nn.modules.normalization

        # loss_layer_numbers: tuple = (2, 7, 12, 21, 30)
        if self.base_model == "VGG":
            loss_layer_numbers: tuple = (2, 7, 12, 21, 30)
            for i in loss_layer_numbers:
                self.output_feature_layers.append(self.feature_network[i])
        elif self.base_model == "MobileNet":
            loss_layer_numbers: tuple = (2, 4, 6, 11, 15)
            for i in loss_layer_numbers:
                self.output_feature_layers.append(self.feature_network[i].conv[2])

        self.loss_layer_history: list = []
        # Values taken from official source code, no idea how they got them
        self.loss_layer_scales = [2.6, 4.8, 3.7, 5.6, 0.15, 1.0]

        # History
        for i in range(len(self.loss_layer_scales)):
            self.loss_layer_history.append(CircularList(history_len))

        # Loss layer coefficient base calculations
        for layer in self.output_feature_layers:
            layer.register_forward_hook(get_layer_values)

    def update_lambdas(self) -> None:
        avg_list: list = [
            self.loss_layer_history[i].mean()
            for i in range(len(self.loss_layer_history))
        ]
        avg_total: float = sum(avg_list) / len(avg_list)

        for i, val in enumerate(avg_list):
            scale_factor: float = val / avg_total
            self.loss_layer_scales[i] = 1.0 / scale_factor
        wandb.log({"Loss scales": self.loss_layer_scales})

    @staticmethod
    def __calculate_loss(
        gen: torch.Tensor, truth: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.mean(
            label * torch.mean((truth - gen).abs(), dim=0, keepdim=True), dim=(1, 2)
        )
        return loss

    def __get_outputs(self, inputs: torch.Tensor):
        self.feature_network(inputs)
        outputs: list = []
        for layer in self.output_feature_layers:
            outputs.append(layer.stored_output.clone())
            del layer.stored_output

        return outputs

    def forward(self, inputs: tuple):
        input_gen: torch.Tensor = inputs[0]
        input_truth: torch.Tensor = inputs[1]
        input_label: torch.Tensor = inputs[2]

        # img_losses: list = []
        this_batch_size = input_gen.shape[0]
        num_channels = 3
        num_images: int = int(input_gen.shape[1] / num_channels)
        num_label_channels = input_label.shape[1]

        batch_loss = torch.zeros(num_images, num_label_channels).float().to(self.device)

        result_truth: list = self.__get_outputs(input_truth)

        loss_contributions: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for img_no in range(num_images):
            start_channel: int = img_no * num_channels
            end_channel: int = (img_no + 1) * num_channels

            single_input: torch.Tensor = input_gen[:, start_channel:end_channel, :, :]
            result_gen: list = self.__get_outputs(single_input)

            for b in range(this_batch_size):
                # Direct Image comparison
                if self.use_loss_output_image:
                    input_loss: torch.Tensor = PerceptualLossNetwork.__calculate_loss(
                        single_input[b], input_truth[b], input_label[0]
                    )

                    # / single_input[b].numel()
                    # loss_contributions[-1] += input_loss.item()
                    batch_loss[img_no] += input_loss / self.loss_layer_scales[-1]

                # VGG feature layer output comparisons
                for i in range(len(self.output_feature_layers)):
                    label_shape: tuple = tuple(result_truth[i][b].shape[1:])
                    label_interpolate = torch.nn.functional.interpolate(
                        input=input_label, size=label_shape, mode="nearest"
                    )

                    layer_loss: torch.Tensor = PerceptualLossNetwork.__calculate_loss(
                        result_gen[i][b], result_truth[i][b], label_interpolate[0],
                    )

                    # * (1.0 / result_truth[i][b].numel())
                    # self.loss_layer_history[i].update(res)
                    # loss_contributions[i] += res.item()
                    batch_loss[img_no] += layer_loss / self.loss_layer_scales[i]

        del result_gen

        # total loss reduction = mean
        # img_losses.append(total_loss / batch_size)
        # total_loss = 0
        # plt.show()
        del result_truth
        # print(batch_loss.detach().cpu().numpy())
        min_loss, _ = torch.min(batch_loss, dim=0)
        # print(min_loss.detach().cpu().numpy())

        total_loss: torch.Tensor = (min_loss * 0.999) + (batch_loss.mean(dim=0) * 0.001)

        # loss_contributions = [x / this_batch_size for x in loss_contributions]
        # for i, val in enumerate(loss_contributions):
        #     self.loss_layer_history[i].update(val)

        del loss_contributions
        # total loss reduction = mean
        return torch.sum(total_loss / this_batch_size)
