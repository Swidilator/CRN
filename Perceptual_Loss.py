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
    # output is a Tensor. output.data is the Tensor we are interested
    self.stored_output = output


class PerceptualLossNetwork(modules.Module):
    def __init__(self, input_image_size: tuple, history_len: int):
        super(PerceptualLossNetwork, self).__init__()

        self.vgg: torchvision.models.VGG = torchvision.models.vgg19(
            pretrained=True, progress=True
        )
        self.vgg.eval()
        del self.vgg.classifier, self.vgg.avgpool
        for i in self.vgg.features:
            i.requires_grad = False

        self.norm = torch.nn.modules.normalization

        self.loss_layer_numbers: tuple = (2, 7, 12, 21, 30)

        self.loss_layer_history: list = []
        self.loss_layer_scales: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # History
        for i in range(len(self.loss_layer_scales)):
            self.loss_layer_history.append(CircularList(history_len))

        # Loss layer coefficient base calculations
        for i, num in enumerate(self.loss_layer_numbers):
            self.vgg.features[num].register_forward_hook(get_layer_values)

    def update_lambdas(self):
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

        self.vgg.features(input)
        result_gen: list = []
        for i, num in enumerate(self.loss_layer_numbers):
            result_gen.append(self.vgg.features[num].stored_output)

        self.vgg.features(truth)
        result_truth: list = []
        for i, num in enumerate(self.loss_layer_numbers):
            result_truth.append(self.vgg.features[num].stored_output)

        device: torch.device = self.vgg.features[0].weight.device
        total_loss: torch.Tensor = torch.zeros(1).float().to(device)

        loss_contributions: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        batch_size = input.shape[0]
        for b in range(batch_size):

            # TODO Implement correct loss
            for i in range(len(result_gen)):
                res: torch.Tensor = (result_truth[i][b] - result_gen[i][b]).norm() * (
                    1.0 / result_truth[i][b].numel()
                )
                # self.loss_layer_history[i].update(res)
                loss_contributions[i] += res.item()
                total_loss += res / self.loss_layer_scales[i]
                del res

            input_loss: torch.Tensor = (input[b] - truth[b]).norm() / input[b].numel()
            # input_loss.requires_grad = False
            # self.loss_layer_history[-1].update(input_loss)
            loss_contributions[-1] += input_loss.item()
            total_loss += input_loss / self.loss_layer_scales[-1]
            del input_loss

        loss_contributions = [x / batch_size for x in loss_contributions]
        for i, val in enumerate(loss_contributions):
            self.loss_layer_history[i].update(val)

        del result_gen, result_truth, loss_contributions
        # total loss reduction = mean
        return total_loss / batch_size
