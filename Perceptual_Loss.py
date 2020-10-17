import torch
import torch.nn.modules as modules
import torchvision

import wandb

class FeatureNet(modules.Module):
    def __init__(self, model: str):
        super(FeatureNet, self).__init__()
        if model == "VGG":
            feature_network: torch.nn.Sequential = torchvision.models.vgg19(
                pretrained=True, progress=True
            ).features
            loss_layer_numbers: tuple = (4, 9, 14, 23, 32)
        elif model == "MobileNet":
            feature_network: torch.nn.Sequential = torchvision.models.mobilenet_v2(
                pretrained=True, progress=True
            ).features
            loss_layer_numbers: tuple = (4, 6, 8, 13, 17)
        else:
            raise ValueError
        # fmt: off
        self.seq_1 = feature_network[0:loss_layer_numbers[0]].eval()
        self.seq_2 = feature_network[loss_layer_numbers[0]:loss_layer_numbers[1]].eval()
        self.seq_3 = feature_network[loss_layer_numbers[1]:loss_layer_numbers[2]].eval()
        self.seq_4 = feature_network[loss_layer_numbers[2]:loss_layer_numbers[3]].eval()
        self.seq_5 = feature_network[loss_layer_numbers[3]:loss_layer_numbers[4]].eval()
        # fmt: on
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor):
        out1 = self.seq_1(image)
        out2 = self.seq_2(out1)
        out3 = self.seq_3(out2)
        out4 = self.seq_4(out3)
        out5 = self.seq_5(out4)
        return out1, out2, out3, out4, out5


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

        self.feature_network = FeatureNet(self.base_model)

        # Values taken from official source code, no idea how they got them
        self.loss_layer_scales = [1.6, 2.3, 1.8, 2.8, 0.08, 1.0]

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
                this_batch_size, 1, device=self.device,
            )
            calculate_loss = self.__calculate_single_image_loss

        for b in range(this_batch_size):
            result_truth: tuple = self.feature_network(input_truth[b])
            result_gen: tuple = self.feature_network(input_gen[b])

            if self.use_loss_output_image:
                input_loss: torch.Tensor = calculate_loss(
                    input_gen[b], input_truth[b], input_label[b]
                )

                loss[b] += input_loss / self.loss_layer_scales[-1]

            # VGG feature layer output comparisons
            for i in range(len(result_gen)):
                label_shape: tuple = tuple(result_truth[i][b].shape[1:])
                label_interpolate = torch.nn.functional.interpolate(
                    input=input_label[b], size=label_shape, mode="nearest"
                )

                layer_loss: torch.Tensor = calculate_loss(
                    result_gen[i], result_truth[i].detach(), label_interpolate,
                )

                loss[b] += layer_loss / self.loss_layer_scales[i]

        if input_gen.shape[1] > 1:
            min_loss, _ = torch.min(loss, dim=1)

            min_component = min_loss.sum(dim=1, keepdim=True) * 0.999
            mean_component = loss.mean(dim=1).sum(dim=1, keepdim=True) * 0.001

            total_loss: torch.Tensor = min_component + mean_component

            return torch.mean(total_loss, dim=0)
        else:
            return torch.mean(loss, dim=0)
