import torch
import torch.nn as nn
import torch.nn.modules as modules
import torchvision
from Helper_Stuff import *
import wandb

from torchvision.transforms import Resize
from copy import copy


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
    def __init__(self, input_image_size: tuple, history_len: int):
        super(PerceptualLossNetwork, self).__init__()
        with torch.no_grad():
            self.vgg = torchvision.models.vgg19(
                pretrained=True, progress=True
            )
        self.vgg.eval()
        del self.vgg.classifier, self.vgg.avgpool
        torch.cuda.empty_cache()
        # for i in self.vgg.features:
        #     i.requires_grad = False

        # self.softmin = torch.nn.Softmin(dim=0)

        self.norm = torch.nn.modules.normalization

        self.loss_layer_numbers: tuple = (2, 7, 12, 21, 30)

        self.loss_layer_history: list = []
        # Values taken from official source code, no idea how they got them
        self.loss_layer_scales: list = [1.0, 1.6, 2.3, 1.8, 2.8, 0.08]

        # History
        for i in range(len(self.loss_layer_scales)):
            self.loss_layer_history.append(CircularList(history_len))

        # Loss layer coefficient base calculations
        for i, num in enumerate(self.loss_layer_numbers):
            self.vgg.features[num].register_forward_hook(get_layer_values)

    def update_lambdas(self) -> None:
        avg_list: list = [
            self.loss_layer_history[i].mean()
            for i in range(len(self.loss_layer_history))
        ]
        avg_total: float = sum(avg_list) / len(avg_list)

        for i, val in enumerate(avg_list):
            scale_factor: float = val / avg_total
            self.loss_layer_scales[i] = 1.0 / scale_factor
        no_except(wandb.log, {"Loss scales": self.loss_layer_scales})

    def forward(self, inputs: tuple):
        input: torch.Tensor = inputs[0]
        truth: torch.Tensor = inputs[1]

        device = self.vgg.features[0].weight.device

        # img_losses: list = []
        this_batch_size = input.shape[0]
        num_channels = 3
        num_images: int = int(input.shape[1] / num_channels)

        batch_loss = torch.zeros(num_images).float().to(device)

        result_truth: list = []
        result_gen: list = []

        self.vgg.features(truth)
        for num in self.loss_layer_numbers:
            result_truth.append(self.vgg.features[num].stored_output)

        loss_contributions: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for img_no in range(num_images):
            start_channel: int = img_no * num_channels
            end_channel: int = (img_no + 1) * num_channels

            single_input: torch.Tensor = input[:, start_channel:end_channel, :, :]
            self.vgg.features(single_input)
            for num in self.loss_layer_numbers:
                result_gen.append(self.vgg.features[num].stored_output)

        for img_no in range(num_images):
            start_channel: int = img_no * num_channels
            end_channel: int = (img_no + 1) * num_channels

            single_input: torch.Tensor = input[:, start_channel:end_channel, :, :]

            # result_gen: list = []
            # self.vgg.features(single_input)
            # for num in self.loss_layer_numbers:
            #     result_gen.append(self.vgg.features[num].stored_output)

            for b in range(this_batch_size):
                # Direct Image comparison
                input_loss: torch.Tensor = (
                    truth[b] - single_input[b]
                ).norm()  # / single_input[b].numel()
                # loss_contributions[-1] += input_loss.item()
                batch_loss[img_no] += input_loss / self.loss_layer_scales[-1]

                # VGG feature layer output comparisons
                for i in range(len(self.loss_layer_numbers)):
                    res: torch.Tensor = (
                        result_truth[i][b]
                        - result_gen[i + (img_no * len(self.loss_layer_numbers))][b]
                    ).norm()  # * (1.0 / result_truth[i][b].numel())
                    # self.loss_layer_history[i].update(res)
                    # loss_contributions[i] += res.item()
                    batch_loss[img_no] += res / self.loss_layer_scales[i]

        del result_gen

        # total loss reduction = mean
        # img_losses.append(total_loss / batch_size)
        # total_loss = 0
        # plt.show()
        del result_truth
        # print(batch_loss.detach().cpu().numpy())
        min_loss, _ = torch.min(batch_loss, dim=0, keepdim=True)
        # print(min_loss.detach().cpu().numpy())

        total_loss: torch.Tensor = (min_loss * 0.999) + (batch_loss.mean() * 0.001)

        # loss_contributions = [x / this_batch_size for x in loss_contributions]
        # for i, val in enumerate(loss_contributions):
        #     self.loss_layer_history[i].update(val)

        del loss_contributions
        # total loss reduction = mean
        return total_loss / this_batch_size
