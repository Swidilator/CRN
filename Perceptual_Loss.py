import torch
import torch.nn.modules as modules
import torchvision

import wandb


class CircularList:
    def __init__(self, list_len: int):
        self.len = list_len
        self.data: list = [1.0] * list_len
        self.pointer: int = 0

    def update(self, input_value: float) -> None:
        self.data[self.pointer] = input_value
        if self.pointer + 1 == self.len:
            self.pointer = 0
        else:
            self.pointer += 1

    def sum(self) -> float:
        return sum(self.data)

    def mean(self) -> float:
        return sum(self.data) / self.len


def get_layer_values(
    self: torch.nn.modules.conv.Conv2d, input_data: tuple, output_data: torch.Tensor
) -> None:
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested in
    self.stored_output = output_data
    pass


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
            self.feature_network = vgg.features[0:32]
            # del vgg
        elif self.base_model == "MobileNet":
            mobile_net = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
            self.feature_network = mobile_net.features[0:17]
            # del mobile_net

        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()
        torch.cuda.empty_cache()

        self.norm = torch.nn.modules.normalization

        # loss_layer_numbers: tuple = (2, 7, 12, 21, 30)
        if self.base_model == "VGG":
            loss_layer_numbers: tuple = (3, 8, 13, 22, 31)
            for i in loss_layer_numbers:
                self.output_feature_layers.append(self.feature_network[i])
        elif self.base_model == "MobileNet":
            # Todo Check these are indeed targeting relu layers
            loss_layer_numbers: tuple = (3, 5, 7, 12, 16)
            for i in loss_layer_numbers:
                self.output_feature_layers.append(self.feature_network[i].conv[2])

        self.loss_layer_history: list = []

        # Values taken from official source code, no idea how they got them
        self.loss_layer_scales = [1.6, 2.3, 1.8, 2.8, 0.08, 1.0]

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
        gen: torch.Tensor, truth: torch.Tensor, label_images: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.mean(
            label_images * torch.mean(torch.abs(truth - gen), dim=1).unsqueeze(1),
            dim=(2, 3),
        )
        return loss

    @staticmethod
    def __calculate_single_image_loss(
        gen: torch.Tensor, truth: torch.Tensor, label_images: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.mean(torch.abs(truth - gen))
        return loss

    def __get_outputs(self, inputs: torch.Tensor):
        self.feature_network(inputs)
        outputs: list = []
        for layer in self.output_feature_layers:
            outputs.append(layer.stored_output.clone())
            # del layer.stored_output

        return outputs

    def forward(
        self,
        input_gen: torch.Tensor,
        input_truth: torch.Tensor,
        input_label: torch.Tensor,
    ):

        # img_losses: list = []
        this_batch_size = input_gen.shape[0]

        # Loss function requires multiple images per image, so 5D
        input_truth = input_truth.unsqueeze(1)
        input_label = input_label.unsqueeze(1)

        # Use diversity loss if more than one output image is used
        if input_gen.shape[1] > 1:
            loss: torch.Tensor = torch.zeros(
                this_batch_size,
                input_gen.shape[1],
                input_label.shape[2],
                device=self.device,
            )

            calculate_loss = self.__calculate_loss
        else:
            loss: torch.Tensor = torch.zeros(
                this_batch_size,
                1,
                device=self.device,
            )
            calculate_loss = self.__calculate_single_image_loss

        for b in range(this_batch_size):
            result_truth: list = self.__get_outputs(input_truth[b])
            result_gen: list = self.__get_outputs(input_gen[b])

            if self.use_loss_output_image:
                input_loss: torch.Tensor = calculate_loss(
                    input_gen[b], input_truth[b], input_label[b]
                )

                loss[b] += input_loss / self.loss_layer_scales[-1]

            # VGG feature layer output comparisons
            for i in range(len(self.output_feature_layers)):
                label_shape: tuple = tuple(result_truth[i][b].shape[1:])
                label_interpolate = torch.nn.functional.interpolate(
                    input=input_label[b], size=label_shape, mode="nearest"
                )

                layer_loss: torch.Tensor = calculate_loss(
                    result_gen[i], result_truth[i], label_interpolate,
                )

                loss[b] += layer_loss / self.loss_layer_scales[i]

        if input_gen.shape[1] > 1:
            min_loss, _ = torch.min(loss, dim=1)
            # print(min_loss.detach().cpu().numpy())

            a = min_loss.sum(dim=1, keepdim=True) * 0.999
            b = loss.mean(dim=1).sum(dim=1, keepdim=True) * 0.001

            total_loss: torch.Tensor = a + b

            return torch.mean(total_loss, dim=0)
        else:
            return torch.mean(loss, dim=0)
