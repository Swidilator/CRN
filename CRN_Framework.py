import torch
from torchvision import transforms
from typing import Tuple, List, Any, Union
import wandb
from tqdm import tqdm
import os
from PIL import ImageFile
from contextlib import nullcontext

from CRN.Perceptual_Loss import PerceptualLossNetwork
from CRN.CRN_Network import CRN
from support_scripts.utils import MastersModel, ModelSettingsManager
from support_scripts.components import CityScapesDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CRNFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        input_image_height_width: tuple,
        batch_size: int,
        num_classes: int,
        num_loader_workers: int,
        subset_size: int,
        should_flip_train: bool,
        use_tanh: bool,
        use_input_noise: bool,
        sample_only: bool,
        use_amp: Union[str, bool],
        log_every_n_steps: int,
        **kwargs,
    ):
        super(CRNFramework, self).__init__(
            device,
            data_path,
            input_image_height_width,
            batch_size,
            num_classes,
            num_loader_workers,
            subset_size,
            should_flip_train,
            use_tanh,
            use_input_noise,
            sample_only,
            use_amp,
            log_every_n_steps,
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

        self.input_tensor_size: tuple = kwargs["input_tensor_size"]
        self.num_output_images: int = kwargs["num_output_images"]
        self.num_inner_channels: int = kwargs["num_inner_channels"]
        self.history_len: int = kwargs["history_len"]
        self.perceptual_base_model: str = kwargs["perceptual_base_model"]
        self.use_feature_encodings: bool = kwargs["use_feature_encodings"]
        self.use_loss_output_image: bool = kwargs["use_loss_output_image"]
        self.layer_norm_type: bool = kwargs["layer_norm_type"]

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
            "batch_size": manager.args["batch_size"],
            "num_classes": manager.args["num_classes"],
            "num_loader_workers": manager.args["num_workers"],
            "subset_size": manager.args["training_subset"],
            "should_flip_train": manager.args["flip_training_images"],
            "use_tanh": not manager.args["no_tanh"],
            "use_input_noise": manager.args["input_image_noise"],
            "sample_only": manager.args["sample_only"],
            "use_amp": manager.args["use_amp"],
            "log_every_n_steps": manager.args["log_every_n_steps"],
        }

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
        }
        return cls(**model_frame_args, **settings)

    def __set_data_loader__(self, **kwargs):

        dataset_features_dict: dict = {
            "instance_map": True,
            "instance_map_processed": False,
            "feature_extractions": {"use": False, "file_path": None},
        }

        self.__data_set_train__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.data_path,
            split="train",
            should_flip=self.should_flip_train,
            subset_size=self.subset_size,
            noise=self.use_input_noise,
            dataset_features=dataset_features_dict,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loader_workers,
            pin_memory=True,
        )

        self.__data_set_val__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.data_path,
            split="val",
            should_flip=False,
            subset_size=0,
            noise=False,
            dataset_features=dataset_features_dict,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loader_workers,
        )

    def __set_model__(self, **kwargs) -> None:

        self.crn: CRN = CRN(
            use_tanh=self.use_tanh,
            input_tensor_size=self.input_tensor_size,
            final_image_size=self.input_image_height_width,
            num_output_images=self.num_output_images,
            num_classes=self.num_classes,
            num_inner_channels=self.num_inner_channels,
            use_feature_encoder=self.use_feature_encodings,
            layer_norm_type=self.layer_norm_type
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

            self.optimizer = torch.optim.Adam(
                self.crn.parameters(),
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
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
        if not self.sample_only:
            self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
            self.loss_net.loss_layer_history = checkpoint["loss_history"]

    def train(self, **kwargs) -> Tuple[float, Any]:
        self.crn.train()

        current_epoch: int = kwargs["current_epoch"]

        if "update_lambdas" in kwargs and kwargs["update_lambdas"]:
            self.loss_net.update_lambdas()

        loss_total: float = 0.0

        for batch_idx, (img, msk, _, instance, _, _) in enumerate(
            tqdm(self.data_loader_train, desc="Training")
        ):
            this_batch_size: int = img.shape[0]

            if this_batch_size == 0:
                break

            self.crn.zero_grad()

            img: torch.Tensor = img.to(self.device)
            msk: torch.Tensor = msk.to(self.device)
            instance: torch.Tensor = instance.to(self.device)

            with self.torch_amp_autocast():
                out: torch.Tensor = self.crn(inputs=(msk, img, instance, None))
                # transform = transforms.ToPILImage()
                # image_output = out[0].detach().cpu()
                # tf_image = transform(torch.nn.functional.tanh(image_output))
                # # plt.imshow((((output[0] + output[0].min()) / (output[0].max() / 2)) - 1).cpu().permute(1,2,0).detach())
                # plt.imshow(tf_image)
                # plt.show()

                img = self.normalise(img.squeeze(dim=0)).unsqueeze(0)

                for i in range(out.shape[0]):
                    for j in range(out.shape[1]):
                        out[i, j] = self.normalise(out[i, j])

                loss: torch.Tensor = self.loss_net((out, img, msk))
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

            if batch_idx % self.log_evey_n_steps == 0 or batch_idx == (
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
        self.crn.eval()
        with torch.no_grad():
            loss_total: torch.Tensor = torch.Tensor([0.0]).to(self.device)
        for batch_idx, (img, msk) in enumerate(self.data_loader_val):
            img: torch.Tensor = img.to(self.device)
            msk: torch.Tensor = msk.to(self.device)

            out: torch.Tensor = self.crn(inputs=(msk, None))

            img = CRNFramework.__normalise__(img)
            out = CRNFramework.__normalise__(out)

            loss: torch.Tensor = self.loss_net((out, img))
            loss_total = loss_total + loss
            del loss, msk, img
        return loss_total.item(), None

    def sample(self, image_number: int, **kwargs: dict) -> dict:
        # Retrieve setting from kwargs
        use_extracted_features: bool = bool(kwargs["use_extracted_features"])

        # Set CRN to eval mode
        self.crn.eval()

        # noise: torch.Tensor = torch.randn(
        #     1,
        #     1,
        #     self.input_tensor_size[0],
        #     self.input_tensor_size[1],
        #     device=self.device,
        # )
        transform: transforms.ToPILImage = transforms.ToPILImage()

        # img, msk, msk_colour, instance, instance_processed, feature_selection
        (
            original_img,
            msk,
            msk_colour,
            instance_original,
            _,
            feature_selection,
        ) = self.__data_set_val__[image_number]

        msk = msk.to(self.device).unsqueeze(0)
        instance_original = instance_original.to(self.device).unsqueeze(0)
        original_img = original_img.to(self.device).unsqueeze(0)

        if use_extracted_features:
            feature_selection = feature_selection.to(self.device).unsqueeze(0)
        else:
            if self.use_feature_encodings:
                feature_selection: torch.Tensor = self.crn.feature_encoder(
                    original_img, instance_original
                )
            else:
                feature_selection = None

        img_out: torch.Tensor = self.crn.sample_using_extracted_features(
            msk, feature_selection, None
        )

        # Drop batch dimension
        img_out = img_out.squeeze(0).cpu()

        split_images = [transform(single_img) for single_img in img_out]

        # Bring images to CPU
        original_img = original_img.squeeze(0).cpu()
        # msk = msk.squeeze(0).argmax(0, keepdim=True).float().cpu()
        msk_colour = msk_colour.float().cpu()
        if self.use_feature_encodings:
            feature_selection = feature_selection.squeeze(0).cpu()

        output_img_dict: dict = {
            "output_img_{i}".format(i=i): img for i, img in enumerate(split_images)
        }
        if self.use_feature_encodings:
            output_img_dict.update({"feature_selection": transform(feature_selection)})

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
    def __normalise__(input: torch.Tensor) -> torch.Tensor:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if len(input.shape) == 4:
            for i in range(input.shape[0]):
                input[i] = CRNFramework.__single_image_normalise__(input[i], mean, std)
        else:
            input = CRNFramework.__single_image_normalise__(input, mean, std)
        return input
