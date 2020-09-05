import os
from contextlib import nullcontext
from itertools import chain
from typing import Tuple, Any, Union, Optional

import torch
import wandb
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm

from CRN.CRN_Network import CRN
from CRN.Perceptual_Loss import PerceptualLossNetwork
from support_scripts.components import CityScapesDataset, FeatureEncoder
from support_scripts.utils import MastersModel, ModelSettingsManager

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CRNFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        dataset_path: str,
        input_image_height_width: tuple,
        batch_size: int,
        use_all_classes: bool,
        num_data_workers: int,
        training_subset_size: int,
        flip_training_images: bool,
        use_tanh: bool,
        use_input_image_noise: bool,
        sample_only: bool,
        use_amp: Union[str, bool],
        log_every_n_steps: int,
        model_save_dir: str,
        image_save_dir: str,
        **kwargs,
    ):
        super(CRNFramework, self).__init__(
            device,
            dataset_path,
            input_image_height_width,
            batch_size,
            use_all_classes,
            num_data_workers,
            training_subset_size,
            flip_training_images,
            use_tanh,
            use_input_image_noise,
            sample_only,
            use_amp,
            log_every_n_steps,
            model_save_dir,
            image_save_dir,
        )
        self.model_name: str = "CRN"

        try:
            assert "input_tensor_size" in kwargs
            assert "num_output_images" in kwargs
            assert "num_inner_channels" in kwargs
            assert "history_len" in kwargs
            assert "perceptual_base_model" in kwargs
            assert "use_feature_encodings" in kwargs
            assert "use_loss_output_image" in kwargs
            assert "layer_norm_type" in kwargs
        except AssertionError as e:
            print("Missing argument: {error}".format(error=e))
            raise SystemExit

        # fmt: off
        self.input_tensor_size: tuple = kwargs["input_tensor_size"]
        self.num_output_images: int = kwargs["num_output_images"]
        self.num_inner_channels: int = kwargs["num_inner_channels"]
        self.history_len: int = kwargs["history_len"]
        self.perceptual_base_model: str = kwargs["perceptual_base_model"]
        self.use_feature_encodings: bool = kwargs["use_feature_encodings"]
        self.use_loss_output_image: bool = kwargs["use_loss_output_image"]
        self.layer_norm_type: str = kwargs["layer_norm_type"]
        self.use_saved_feature_encodings: bool = kwargs["use_saved_feature_encodings"]
        # fmt: on

        self.__set_data_loader__()

        self.__set_model__()

    @property
    def data_set_train(self) -> torch.utils.data.Dataset:
        return self.__data_set_train__

    @property
    def data_set_val(self) -> torch.utils.data.Dataset:
        return self.__data_set_val__

    @property
    def wandb_trainable_model(self) -> tuple:
        models = [self.crn]
        if self.use_feature_encodings:
            models.append(self.feature_encoder)
        return tuple(models)

    @classmethod
    def from_model_settings_manager(
        cls, manager: ModelSettingsManager
    ) -> "CRNFramework":

        # model_frame_args: dict = {
        #     "device": manager.device,
        #     "data_path": manager.args["dataset_path"],
        #     "input_image_height_width": manager.args["input_image_height_width"],
        #     "batch_size": manager.args["batch_size"],
        #     "use_all_classes": manager.args["use_all_classes"],
        #     "num_loader_workers": manager.args["num_workers"],
        #     "subset_size": manager.args["training_subset"],
        #     "should_flip_train": manager.args["flip_training_images"],
        #     "use_tanh": not manager.args["no_tanh"],
        #     "use_input_noise": manager.args["input_image_noise"],
        #     "sample_only": manager.args["sample_only"],
        #     "use_amp": manager.args["use_amp"],
        #     "log_every_n_steps": manager.args["log_every_n_steps"],
        #     "model_save_dir": manager.args["model_save_dir"],
        # }
        #
        # fmt: off
        settings = {
            "input_tensor_size": (
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_HEIGHT"],
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_WIDTH"],
            ),
            "num_output_images": manager.model_conf["CRN_NUM_OUTPUT_IMAGES"],
            "num_inner_channels": manager.model_conf["CRN_NUM_INNER_CHANNELS"],
            "history_len": manager.model_conf["CRN_HISTORY_LEN"],
            "perceptual_base_model": manager.model_conf["CRN_PERCEPTUAL_BASE_MODEL"],
            "use_feature_encodings": manager.model_conf["CRN_USE_FEATURE_ENCODINGS"],
            "use_loss_output_image": manager.model_conf["CRN_USE_LOSS_OUTPUT_IMAGE"],
            "layer_norm_type": manager.model_conf["CRN_LAYER_NORM_TYPE"],
            "use_saved_feature_encodings": manager.model_conf["CRN_USE_SAVED_FEATURE_ENCODINGS"],
        }
        # fmt: on

        return cls(**manager.args, **settings)

    def __set_data_loader__(self, **kwargs):

        dataset_features_dict: dict = {
            "instance_map": True,
            "instance_map_processed": False,
            # "feature_extractions": {"use": False, "file_path": None},
        }

        self.__data_set_train__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="train",
            should_flip=self.flip_training_images,
            subset_size=self.training_subset_size,
            noise=self.use_input_image_noise,
            dataset_features=dataset_features_dict,
            specific_model="CRN",
            use_all_classes=self.use_all_classes,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_workers,
            pin_memory=True,
        )

        self.__data_set_val__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="val",
            should_flip=False,
            subset_size=0,
            noise=False,
            dataset_features=dataset_features_dict,
            specific_model="CRN",
            use_all_classes=self.use_all_classes,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_workers,
        )

        self.num_classes = self.__data_set_train__.num_output_classes

    def __set_model__(self, **kwargs) -> None:

        # Feature Encoder
        if self.use_feature_encodings:
            self.feature_encoder: FeatureEncoder = FeatureEncoder(
                3,
                3,
                4,
                self.device,
                self.model_save_dir,
                self.use_saved_feature_encodings,
            )
            self.feature_encoder = self.feature_encoder.to(self.device)
            if self.use_saved_feature_encodings:
                self.feature_encoder.eval()
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.feature_encoder.parameters():
                    param.requires_grad = True

        self.crn: CRN = CRN(
            use_tanh=self.use_tanh,
            input_tensor_size=self.input_tensor_size,
            final_image_size=self.input_image_height_width,
            num_output_images=self.num_output_images,
            num_classes=self.num_classes,
            num_inner_channels=self.num_inner_channels,
            use_feature_encoder=self.use_feature_encodings,
            layer_norm_type=self.layer_norm_type,
        )

        # self.crn = nn.DataParallel(self.crn, device_ids=device_ids)
        self.crn = self.crn.to(self.device)

        if not self.sample_only:

            self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
                self.history_len,
                self.perceptual_base_model,
                self.device,
                self.use_loss_output_image,
            )

            # self.loss_net = nn.DataParallel(self.loss_net, device_ids=device_ids)
            self.loss_net = self.loss_net.to(self.device)

            # self.optimizer = torch.optim.SGD(self.crn.parameters(), lr=0.01, momentum=0.9)

            # Create params depending on what needs to be trained
            params = self.crn.parameters()

            if self.use_feature_encodings and not self.use_saved_feature_encodings:
                params = chain(params, self.feature_encoder.parameters(),)

            self.optimizer = torch.optim.Adam(
                params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
            )

            self.normalise = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            # Mixed precision
            if self.use_amp == "torch":
                from torch.cuda import amp as torch_amp

                self.torch_gradient_scaler: torch_amp.GradScaler() = torch_amp.GradScaler()
                self.torch_amp_autocast = torch_amp.autocast
            else:
                self.torch_amp_autocast = nullcontext

        # Empty cache for cleanup
        torch.cuda.empty_cache()

    def save_model(self, epoch: int = -1) -> None:
        super().save_model()

        try:
            assert not self.sample_only
        except AssertionError as e:
            raise AssertionError("Cannot save model in 'sample_only mode'")

        save_dict: dict = {
            "dict_crn": self.crn.state_dict(),
            "loss_layer_scales": self.loss_net.loss_layer_scales,
            "loss_history": self.loss_net.loss_layer_history,
        }
        if self.use_feature_encodings:
            save_dict.update(
                {"dict_encoder_decoder": self.feature_encoder.state_dict()}
            )

        if epoch >= 0:
            # Todo add support for manager.args["model_save_prefix"]
            epoch_file_name: str = os.path.join(
                self.model_save_dir,
                self.model_name + "_Epoch_{epoch}.pt".format(epoch=epoch),
            )
            torch.save(save_dict, epoch_file_name)

        latest_file_name: str = os.path.join(
            self.model_save_dir, self.model_name + "_Latest.pt"
        )
        torch.save(save_dict, latest_file_name)

    def load_model(self, model_file_name: str) -> None:
        super().load_model(model_file_name)

        # Create final model file path and output
        load_path: str = os.path.join(self.model_save_dir, model_file_name)
        print("Loading model:")
        print(load_path)

        checkpoint = torch.load(load_path, map_location=self.device)
        self.crn.load_state_dict(checkpoint["dict_crn"], strict=False)
        if self.use_feature_encodings:
            self.feature_encoder.load_state_dict(checkpoint["dict_encoder_decoder"])
        if not self.sample_only:
            self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
            self.loss_net.loss_layer_history = checkpoint["loss_history"]

    def train(self, **kwargs) -> Tuple[float, Any]:
        self.crn.train()

        current_epoch: int = kwargs["current_epoch"]

        if "update_lambdas" in kwargs and kwargs["update_lambdas"]:
            self.loss_net.update_lambdas()

        loss_total: float = 0.0

        for batch_idx, input_dict in enumerate(
            tqdm(self.data_loader_train, desc="Training")
        ):
            this_batch_size: int = input_dict["img"].shape[0]

            if this_batch_size == 0:
                break

            self.crn.zero_grad()

            img = input_dict["img"].to(self.device)
            msk = input_dict["msk"].to(self.device)
            instance = input_dict["inst"].to(self.device)

            with self.torch_amp_autocast():
                feature_encoding: Optional[torch.Tensor]
                if self.use_feature_encodings:
                    feature_encoding: torch.Tensor = self.feature_encoder(
                        img,
                        instance,
                        input_dict["img_id"]
                        if self.use_saved_feature_encodings
                        else None,
                        input_dict["img_flipped"],
                    )
                else:
                    feature_encoding = None

                out: torch.Tensor = self.crn(msk, feature_encoding)

                for b in range(img.shape[0]):
                    img[b] = self.normalise(img[b])

                for b in range(out.shape[0]):
                    for out_img in range(out.shape[1]):
                        out[b, out_img] = self.normalise(out[b, out_img])

                loss: torch.Tensor = self.loss_net(out, img, msk)
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            if self.use_amp == "torch":
                self.torch_gradient_scaler.scale(loss).backward()
                self.torch_gradient_scaler.step(self.optimizer)
                self.torch_gradient_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            loss_total += loss.item() * self.batch_size

            if batch_idx % self.log_every_n_steps == 0 or batch_idx == (
                len(self.data_loader_train) - 1
            ):
                batch_loss_val: float = loss.item()

                wandb.log(
                    {
                        "Epoch_Fraction": current_epoch
                        + (
                            (batch_idx * self.batch_size)
                            / len(self.data_loader_train.dataset)
                        ),
                        "Batch Loss": batch_loss_val,
                    }
                )

        return loss_total, None

    def eval(self) -> Tuple[float, Any]:
        # self.crn.eval()
        # with torch.no_grad():
        #     loss_total: torch.Tensor = torch.Tensor([0.0]).to(self.device)
        # for batch_idx, (img, msk) in enumerate(self.data_loader_val):
        #     img: torch.Tensor = img.to(self.device)
        #     msk: torch.Tensor = msk.to(self.device)
        #
        #     out: torch.Tensor = self.crn(inputs=(msk, None))
        #
        #     img = CRNFramework.__normalise__(img)
        #     out = CRNFramework.__normalise__(out)
        #
        #     loss: torch.Tensor = self.loss_net((out, img))
        #     loss_total = loss_total + loss.detach()
        #     del loss, msk, img
        # return loss_total.item(), None
        pass

    def sample(self, image_number: int, **kwargs: dict) -> dict:

        # Set CRN to eval mode
        self.crn.eval()
        if self.use_feature_encodings:
            self.feature_encoder.eval()

        # noise: torch.Tensor = torch.randn(
        #     1,
        #     1,
        #     self.input_tensor_size[0],
        #     self.input_tensor_size[1],
        #     device=self.device,
        # )
        transform: transforms.ToPILImage = transforms.ToPILImage()

        # img, msk, msk_colour, instance, instance_processed, feature_selection
        input_dict = self.__data_set_val__[image_number]

        msk = input_dict["msk"].to(self.device).unsqueeze(0)
        instance_original = input_dict["inst"].to(self.device).unsqueeze(0)
        original_img = input_dict["img"].to(self.device).unsqueeze(0)

        feature_encoding: Optional[torch.Tensor]
        if self.use_feature_encodings:
            if self.use_saved_feature_encodings:
                feature_encoding = self.feature_encoder.sample_using_means(
                    instance_original
                )
            else:
                feature_encoding = self.feature_encoder(original_img, instance_original)
        else:
            feature_encoding = None

        img_out: torch.Tensor = self.crn(msk, feature_encoding)

        # Clamp image to within correct bounds
        img_out = img_out.clamp(0.0, 1.0)

        # Drop batch dimension
        img_out = img_out.squeeze(0).cpu()

        split_images = [transform(single_img) for single_img in img_out]

        # Bring images to CPU
        original_img = original_img.squeeze(0).cpu()
        # msk = msk.squeeze(0).argmax(0, keepdim=True).float().cpu()
        msk_colour = input_dict["msk_colour"].float().cpu()

        output_img_dict: dict = {
            "output_img_{i}".format(i=i): img for i, img in enumerate(split_images)
        }
        if self.use_feature_encodings:
            feature_encoding = feature_encoding.squeeze(0).cpu()
            output_img_dict.update({"feature_selection": transform(feature_encoding)})

        output_dict: dict = {
            "image_index": image_number,
            "original_img": transform(original_img),
            "msk_colour": transform(msk_colour),
            "output_img_dict": output_img_dict,
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
    def __normalise__(input_tensor: torch.Tensor) -> torch.Tensor:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if len(input_tensor.shape) == 4:
            for i in range(input_tensor.shape[0]):
                input_tensor[i] = CRNFramework.__single_image_normalise__(
                    input_tensor[i], mean, std
                )
        else:
            input_tensor = CRNFramework.__single_image_normalise__(
                input_tensor, mean, std
            )
        return input_tensor
