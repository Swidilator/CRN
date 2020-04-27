import torch
from torchvision import transforms
from typing import Tuple, List, Any
import random
import wandb
from tqdm import tqdm
import os
from PIL import ImageFile

from CRN.CRN_Dataset import CRNDataset
from CRN.Perceptual_Loss import PerceptualLossNetwork
from CRN.CRN_Network import CRN
from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CRNFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        input_image_height_width: tuple,
        batch_size_slice: int,
        batch_size_total: int,
        num_classes: int,
        num_loader_workers: int,
        subset_size: int,
        should_flip_train: bool,
        use_tanh: bool,
        use_input_noise: bool,
        sample_only: bool,
        **kwargs,
    ):
        super(CRNFramework, self).__init__(
            device,
            data_path,
            input_image_height_width,
            batch_size_slice,
            batch_size_total,
            num_classes,
            num_loader_workers,
            subset_size,
            should_flip_train,
            use_tanh,
            use_input_noise,
            sample_only,
        )
        self.model_name: str = "CRN"

        self.max_data_loader_batch_size: int = 16

        try:
            assert "input_tensor_size" in kwargs
            assert "num_output_images" in kwargs
            assert "num_inner_channels" in kwargs
            assert "history_len" in kwargs
        except AssertionError as e:
            print("Missing argument: {error}".format(error=e))
            raise SystemExit

        self.input_tensor_size: tuple = kwargs["input_tensor_size"]
        self.num_output_images: int = kwargs["num_output_images"]
        self.num_inner_channels: int = kwargs["num_inner_channels"]
        self.history_len: int = kwargs["history_len"]

        self.__set_data_loader__()

        self.__set_model__()

    @property
    def wandb_trainable_model(self) -> tuple:
        return (self.crn,)

    @classmethod
    def from_model_settings_manager(
        cls, manager: ModelSettingsManager
    ) -> "CRNFramework":

        model_frame_args: dict = {
            "device": manager.device,
            "data_path": manager.args["dataset_path"],
            "input_image_height_width": manager.args["input_image_height_width"],
            "batch_size_slice": manager.args["batch_size_pair"][0],
            "batch_size_total": manager.args["batch_size_pair"][1],
            "num_classes": manager.args["num_classes"],
            "num_loader_workers": manager.args["num_workers"],
            "subset_size": manager.args["training_subset"],
            "should_flip_train": manager.args["flip_training_images"],
            "use_tanh": not manager.args["no_tanh"],
            "use_input_noise": manager.args["input_image_noise"],
            "sample_only": manager.args["sample_only"],
        }

        settings = {
            "input_tensor_size": (
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_HEIGHT"],
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_WIDTH"],
            ),
            "num_output_images": manager.model_conf["CRN_NUM_OUTPUT_IMAGES"],
            "num_inner_channels": manager.model_conf["CRN_NUM_INNER_CHANNELS"],
            "history_len": manager.model_conf["CRN_HISTORY_LEN"],
        }
        return cls(**model_frame_args, **settings)

    def __set_data_loader__(self, **kwargs):

        if self.batch_size_total > 16:
            self.medium_batch_size: int = 16
        else:
            self.medium_batch_size: int = self.batch_size_total

        self.__data_set_train__ = CRNDataset(
            max_input_height_width=self.input_image_height_width,
            root=self.data_path,
            split="train",
            should_flip=self.should_flip_train,
            subset_size=self.subset_size,
            noise=self.use_input_noise,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=self.batch_size_slice,
            shuffle=True,
            num_workers=self.num_loader_workers,
        )

        self.__data_set_val__ = CRNDataset(
            max_input_height_width=self.input_image_height_width,
            root=self.data_path,
            split="val",
            should_flip=False,
            subset_size=0,
            noise=False,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.batch_size_slice,
            shuffle=True,
            num_workers=self.num_loader_workers,
        )

    def __set_model__(self, **kwargs) -> None:

        num_image_channels = 3

        self.crn: CRN = CRN(
            use_tanh=self.use_tanh,
            input_tensor_size=self.input_tensor_size,
            final_image_size=self.input_image_height_width,
            num_output_images=self.num_output_images,
            num_classes=self.num_classes,
            num_inner_channels=self.num_inner_channels,
        )

        # self.crn = nn.DataParallel(self.crn, device_ids=device_ids)
        self.crn = self.crn.to(self.device)

        if not self.sample_only:

            self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
                (
                    num_image_channels,
                    self.input_image_height_width[0],
                    self.input_image_height_width[1],
                ),
                self.history_len,
            )

            # self.loss_net = nn.DataParallel(self.loss_net, device_ids=device_ids)
            self.loss_net = self.loss_net.to(self.device)

            # self.optimizer = torch.optim.SGD(self.crn.parameters(), lr=0.01, momentum=0.9)

            self.optimizer = torch.optim.Adam(
                self.crn.parameters(),
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
            )

    def save_model(self, model_dir: str, epoch: int = -1) -> None:
        super().save_model(model_dir)

        try:
            assert not self.sample_only
        except AssertionError as e:
            raise AssertionError("Cannot save model in 'sample_only mode'")

        save_dict: dict = {
            "dict_crn": self.crn.state_dict(),
            "loss_layer_scales": self.loss_net.loss_layer_scales,
            "loss_history": self.loss_net.loss_layer_history,
        }

        if epoch >= 0:
            # Todo add support for manager.args["model_save_prefix"]
            epoch_file_name: str = os.path.join(
                model_dir, self.model_name + "_Epoch_{epoch}.pt".format(epoch=epoch)
            )
            torch.save(save_dict, epoch_file_name)

        latest_file_name: str = os.path.join(model_dir, self.model_name + "_Latest.pt")
        torch.save(save_dict, latest_file_name)

    def load_model(self, model_dir: str, model_file_name: str) -> None:
        super().load_model(model_dir, model_file_name)

        checkpoint = torch.load(
            os.path.join(model_dir, model_file_name), map_location=self.device
        )
        self.crn.load_state_dict(checkpoint["dict_crn"])
        self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
        self.loss_net.loss_layer_history = checkpoint["loss_history"]

    def train(self, **kwargs) -> Tuple[float, Any]:
        self.crn.train()
        torch.cuda.empty_cache()

        current_epoch: int = kwargs["current_epoch"]

        loss_ave: float = 0.0
        loss_total: float = 0.0

        # Number of times the medium batch should be looped over, given the slice size
        mini_batch_per_medium_batch: int = self.medium_batch_size // self.batch_size_slice

        current_big_batch: int = 0

        # Increments as mini batches are processed, should be equal to big batch eventually
        this_big_batch_size: int = 0

        final_medium_batch: bool = False

        if "update_lambdas" in kwargs and kwargs["update_lambdas"]:
            self.loss_net.update_lambdas()

        for batch_idx, (img_total, msk_total, _) in enumerate(
            tqdm(self.data_loader_train, desc="Training")
        ):
            this_medium_batch_size: int = img_total.shape[0]

            if (this_medium_batch_size < self.medium_batch_size) or (
                (batch_idx + 1) * self.medium_batch_size
                == len(self.data_loader_train.dataset)
            ):
                final_medium_batch = True
            self.crn.zero_grad()

            # Loop over medium batch
            for i in range(mini_batch_per_medium_batch):
                img: torch.Tensor = img_total[
                    i * self.batch_size_slice : (i + 1) * self.batch_size_slice
                ].to(self.device)
                msk: torch.Tensor = msk_total[
                    i * self.batch_size_slice : (i + 1) * self.batch_size_slice
                ].to(self.device)

                this_mini_batch_size: int = msk.shape[0]

                # Add the mini batch size to the big batch size
                this_big_batch_size += this_mini_batch_size

                if (
                    this_mini_batch_size == 0
                ):  # Empty mini batch, medium batch is last in epoch
                    # final_mini_batch = True
                    break
                # noise: torch.Tensor = torch.randn(
                #     this_batch_size,
                #     1,
                #     self.input_tensor_size[0],
                #     self.input_tensor_size[1],
                #     device=self.device,
                # )
                # noise = noise.to(self.device)

                # out: torch.Tensor = self.crn(inputs=(msk, noise, self.batch_size))
                out: torch.Tensor = self.crn(inputs=(msk, None))

                img = CRNFramework.__normalise__(img)
                out = CRNFramework.__normalise__(out)

                loss: torch.Tensor = self.loss_net((out, img, msk))
                loss.backward()
                loss_ave += loss.item()
                loss_total += loss.item()
                del msk, img, loss

            if (this_big_batch_size == self.batch_size_total) or final_medium_batch:
                # print(current_big_batch)
                i: torch.nn.Parameter
                for i in self.crn.parameters():
                    if i.grad is not None:
                        i.grad = i.grad / (
                            self.batch_size_total / self.batch_size_slice
                        )
                # print("Stepping")
                self.optimizer.step()

                # Step big batch count
                current_big_batch += 1

                # # Normalise this accumulated error
                # output_scaling_factor: float = (
                #     this_big_batch_size / self.batch_size_slice
                # )

                batch_loss_val: float = loss_ave * self.batch_size_total

                wandb.log(
                    {
                        "Epoch_Fraction": current_epoch
                        + (batch_idx / len(self.data_loader_train.dataset)),
                        "Batch Loss": batch_loss_val,
                    }
                )
                loss_ave = 0.0
                this_big_batch_size = 0

            del msk_total, img_total
        del loss_ave
        return loss_total, None

    def eval(self) -> Tuple[float, Any]:
        self.crn.eval()
        with torch.no_grad():
            loss_total: torch.Tensor = torch.Tensor([0.0]).to(self.device)
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

            out: torch.Tensor = self.crn(inputs=(msk, None))

            img = CRNFramework.__normalise__(img)
            out = CRNFramework.__normalise__(out)

            loss: torch.Tensor = self.loss_net((out, img))
            loss_total = loss_total + loss
            del loss, msk, img
        return loss_total.item(), None

    def sample(self, image_number: int, **kwargs: dict) -> dict:
        self.crn.eval()

        # noise: torch.Tensor = torch.randn(
        #     1,
        #     1,
        #     self.input_tensor_size[0],
        #     self.input_tensor_size[1],
        #     device=self.device,
        # )
        transform: transforms.ToPILImage = transforms.ToPILImage()

        (original_img, msk, msk_colour,) = self.__data_set_val__[image_number]
        msk = msk.to(self.device).unsqueeze(0)

        img_out: torch.Tensor = self.crn(inputs=(msk, None))

        split_images: list = []
        # print(img_out.shape)
        for img_no in range(self.num_output_images):
            start_channel: int = img_no * 3
            end_channel: int = (img_no + 1) * 3
            img_out_single: torch.Tensor = img_out[0, start_channel:end_channel].cpu()
            split_images.append(transform(img_out_single))

            # Bring images to cpu
        original_img = original_img.squeeze(0).cpu()
        msk = msk.squeeze(0).argmax(0, keepdim=True).float().cpu()
        msk_colour = msk_colour.float().cpu()

        output_img_dict: dict = {
            "output_img_{i}".format(i=i): img
            for i, img in enumerate(split_images)
        }

        output_dict: dict = {
            "image_index": image_number,
            "original_img": transform(original_img),
            "msk_colour": transform(msk_colour),
            "output_img_dict": output_img_dict
        }

        return output_dict

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
