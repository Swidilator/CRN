import os
from contextlib import nullcontext
from itertools import chain
from typing import Tuple, Any, Union, Optional, List

import torch
import wandb
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm

from CRN.CRN_Network import CRN
from support_scripts.components import (
    FeatureEncoder,
    FullDiscriminator,
    feature_matching_error,
    PerceptualLossNetwork,
)
from support_scripts.utils import MastersModel, ModelSettingsManager, CityScapesDataset
from support_scripts.sampling import SampleDataHolder

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
        starting_epoch: int,
        num_frames_per_video: int,
        num_prior_frames: int,
        use_optical_flow: bool,
        prior_frame_seed_type: str,
        use_mask_for_instances: bool,
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
            starting_epoch,
            num_frames_per_video,
            num_prior_frames,
            use_optical_flow,
            prior_frame_seed_type,
            use_mask_for_instances,
            **kwargs,
        )
        self.model_name: str = "CRN"

        try:
            assert "input_tensor_size" in kwargs
            assert "num_output_images" in kwargs
            assert "num_inner_channels" in kwargs
            assert "perceptual_base_model" in kwargs
            assert "use_feature_encodings" in kwargs
            assert "use_loss_output_image" in kwargs
            assert "loss_scaling_method" in kwargs
            assert "layer_norm_type" in kwargs
            assert "use_saved_feature_encodings" in kwargs
            assert "use_resnet_rms" in kwargs
            assert "num_resnet_processing_rms" in kwargs
            assert "use_edge_map" in kwargs
            assert "use_discriminators" in kwargs
            assert "use_sigmoid_discriminator" in kwargs
            assert "num_discriminators" in kwargs
            assert "use_perceptual_loss" in kwargs

        except AssertionError as e:
            print("Missing argument: {error}".format(error=e))
            raise SystemExit

        # fmt: off
        self.input_tensor_size: tuple = kwargs["input_tensor_size"]
        self.num_output_images: int = kwargs["num_output_images"]
        self.num_inner_channels: int = kwargs["num_inner_channels"]
        self.perceptual_base_model: str = kwargs["perceptual_base_model"]
        self.use_feature_encodings: bool = kwargs["use_feature_encodings"]
        self.use_loss_output_image: bool = kwargs["use_loss_output_image"]
        self.loss_scaling_method: str = kwargs["loss_scaling_method"]
        self.layer_norm_type: str = kwargs["layer_norm_type"]
        self.use_saved_feature_encodings: bool = kwargs["use_saved_feature_encodings"]
        self.use_resnet_rms: bool = kwargs["use_resnet_rms"]
        self.num_resnet_processing_rms: int = kwargs["num_resnet_processing_rms"]
        self.use_edge_map: bool = kwargs["use_edge_map"]
        self.use_discriminators: bool = kwargs["use_discriminators"]
        self.use_sigmoid_discriminator: bool = kwargs["use_sigmoid_discriminator"]
        self.num_discriminators: int = kwargs["num_discriminators"]
        self.use_perceptual_loss: bool = kwargs["use_perceptual_loss"]
        # fmt: on

        self.__set_data_loader__()

        self.__set_model__()

        # Compatibility: current model uses blocks in rms. Can be set to False when loading old saves
        self.compat_using_block_rms = True

    @property
    def data_set_train(self) -> torch.utils.data.Dataset:
        return self.__data_set_train__

    @property
    def data_set_val(self) -> torch.utils.data.Dataset:
        return self.__data_set_val__

    @property
    def data_set_video(self) -> torch.utils.data.Dataset:
        return self.__data_set_video__

    @property
    def wandb_trainable_model(self) -> tuple:
        models = [self.crn]
        if self.use_feature_encodings:
            models.append(self.feature_encoder)
        if self.use_discriminators:
            models.append(self.image_discriminator)
        return tuple(models)

    @classmethod
    def from_model_settings_manager(
        cls, manager: ModelSettingsManager
    ) -> "CRNFramework":
        # fmt: off
        settings = {
            "input_tensor_size": (
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_HEIGHT"],
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_WIDTH"],
            ),
            "num_output_images": manager.model_conf["CRN_NUM_OUTPUT_IMAGES"],
            "num_inner_channels": manager.model_conf["CRN_NUM_INNER_CHANNELS"],
            "perceptual_base_model": manager.model_conf["CRN_PERCEPTUAL_BASE_MODEL"],
            "use_feature_encodings": manager.model_conf["CRN_USE_FEATURE_ENCODINGS"],
            "use_loss_output_image": manager.model_conf["CRN_USE_LOSS_OUTPUT_IMAGE"],
            "loss_scaling_method": manager.model_conf["CRN_LOSS_SCALING_METHOD"],
            "layer_norm_type": manager.model_conf["CRN_LAYER_NORM_TYPE"],
            "use_saved_feature_encodings": manager.model_conf["CRN_USE_SAVED_FEATURE_ENCODINGS"],
            "use_resnet_rms": manager.model_conf["CRN_USE_RESNET_RMS"],
            "num_resnet_processing_rms": manager.model_conf["CRN_NUM_RESNET_PROCESSING_RMS"],
            "use_edge_map": manager.model_conf["CRN_USE_EDGE_MAP"],
            "use_discriminators": manager.model_conf["CRN_USE_DISCRIMINATORS"],
            "use_sigmoid_discriminator": manager.model_conf["CRN_USE_SIGMOID_DISCRIMINATOR"],
            "num_discriminators": manager.model_conf["CRN_NUM_DISCRIMINATORS"],
            "use_perceptual_loss": manager.model_conf["CRN_USE_PERCEPTUAL_LOSS"],
        }
        # fmt: on

        return cls(**manager.args, **settings)

    def __set_data_loader__(self, **kwargs):

        self.__data_set_train__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="train",
            should_flip=self.flip_training_images,
            subset_size=self.training_subset_size,
            noise=self.use_input_image_noise,
            specific_model="CRN",
            generated_data=False,
        )

        self.data_loader_train: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                self.__data_set_train__,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_data_workers,
                pin_memory=True,
            )
        )

        self.__data_set_val__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="val",
            should_flip=False,
            subset_size=0,
            noise=False,
            specific_model="CRN",
            generated_data=False,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_workers,
        )

        self.__data_set_video__ = CityScapesDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="demoVideo",
            should_flip=False,
            subset_size=0,
            noise=False,
            specific_model="CRN",
            generated_data=True,
        )

        self.data_loader_video: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                self.__data_set_video__,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_data_workers,
                pin_memory=True,
            )
        )

        self.num_classes = self.__data_set_train__.num_output_segmentation_classes

    def __set_model__(self, **kwargs) -> None:
        # Useful channel count variables
        num_image_channels: int = 3
        num_edge_map_channels: int = self.use_edge_map * 1
        num_feature_encoding_channels: int = self.use_feature_encodings * 3

        # Feature Encoder
        if self.use_feature_encodings:
            self.feature_encoder: FeatureEncoder = FeatureEncoder(
                num_image_channels,
                num_feature_encoding_channels,
                4,
                self.device,
                self.model_save_dir,
                self.use_saved_feature_encodings,
                self.use_mask_for_instances,
                self.num_classes,
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
            use_resnet_rms=self.use_resnet_rms,
            num_resnet_processing_rms=self.num_resnet_processing_rms,
            use_edge_map=self.use_edge_map,
        )
        self.crn = self.crn.to(self.device)

        if not self.sample_only:

            # Create params depending on what needs to be trained
            params = self.crn.parameters()

            if self.use_feature_encodings and not self.use_saved_feature_encodings:
                params = chain(
                    params,
                    self.feature_encoder.parameters(),
                )

            self.optimizer_crn = torch.optim.Adam(
                params,
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
            )

            # Perceptual Loss
            if self.use_perceptual_loss:
                self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
                    self.perceptual_base_model,
                    self.device,
                    self.use_loss_output_image,
                    self.loss_scaling_method,
                )
                self.loss_net = self.loss_net.to(self.device)
                self.normalise = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

            # Discriminator
            if self.use_discriminators:
                assert (
                    self.num_output_images == 1
                ), "self.use_discriminators is True, but self.num_output_images > 1."
                self.criterion_D = torch.nn.MSELoss()

                self.image_discriminator_input_channel_count: int = (
                    self.num_classes + num_edge_map_channels + num_image_channels
                )

                self.image_discriminator: FullDiscriminator = FullDiscriminator(
                    self.device,
                    self.image_discriminator_input_channel_count,
                    self.num_discriminators,
                    self.use_sigmoid_discriminator,
                )
                self.image_discriminator = self.image_discriminator.to(self.device)

                self.optimizer_D_image: torch.optim.Adam = torch.optim.Adam(
                    self.image_discriminator.parameters(),
                    lr=0.0001,
                    betas=(0.5, 0.999),
                    # eps=1e-08,
                    # weight_decay=0,
                )

            # Mixed precision
            if self.use_amp == "torch":
                from torch.cuda import amp as torch_amp

                self.torch_gradient_scaler: torch_amp.GradScaler() = (
                    torch_amp.GradScaler()
                )
                self.torch_amp_autocast = torch_amp.autocast
            else:
                self.torch_amp_autocast = nullcontext

        # Empty cache for cleanup
        torch.cuda.empty_cache()

    def save_model(self, epoch: int = -1) -> None:
        super().save_model()

        if self.sample_only:
            raise RuntimeError("Cannot save model in 'sample_only' mode.")

        crn_state_dict: dict = self.crn.state_dict()
        if not self.compat_using_block_rms:
            crn_state_dict = {
                x: crn_state_dict[x]
                for x in crn_state_dict.keys()
                if "block" in x or "final" in x
            }

        save_dict: dict = {
            "dict_crn": crn_state_dict,
            "args": self.args,
            "kwargs": self.kwargs,
            "compat_using_block_rms": True,
        }

        if self.use_feature_encodings:
            save_dict.update(
                {"dict_encoder_decoder": self.feature_encoder.state_dict()}
            )
        if self.use_discriminators:
            for i in range(self.num_discriminators):
                save_dict.update(
                    {
                        "dict_discriminator_{num}".format(
                            num=i
                        ): self.image_discriminator.discriminators[i].state_dict()
                    }
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
        if "compat_using_block_rms" not in checkpoint:
            self.compat_using_block_rms = False
        self.crn.load_state_dict(checkpoint["dict_crn"], strict=False)
        if self.use_feature_encodings:
            self.feature_encoder.load_state_dict(checkpoint["dict_encoder_decoder"])
        if self.use_discriminators:
            if "dict_discriminator" in checkpoint:
                self.image_discriminator.load_state_dict(
                    checkpoint["dict_discriminator"], strict=False
                )
            else:
                for i in range(self.num_discriminators):
                    if "dict_discriminator_{num}".format(num=i) in checkpoint:
                        self.image_discriminator.discriminators[i].load_state_dict(
                            checkpoint["dict_discriminator_{num}".format(num=i)]
                        )

    @classmethod
    def load_model_with_embedded_settings(cls, manager: ModelSettingsManager):
        load_path: str = os.path.join(
            manager.args["model_save_dir"], manager.args["load_saved_model"]
        )
        checkpoint = torch.load(load_path)
        args: dict = checkpoint["args"]
        kwargs: dict = checkpoint["kwargs"]

        model_frame: CRNFramework = cls(**args, **kwargs)
        model_frame.load_model(manager.args["load_saved_model"])
        return model_frame

    def train(self, **kwargs) -> Tuple[float, Any]:
        self.crn.train()
        if self.use_feature_encodings:
            self.feature_encoder.train()
        if self.use_discriminators:
            self.image_discriminator.train()

        current_epoch: int = kwargs["current_epoch"]

        loss_total: float = 0.0

        # Discriminator labels
        real_label: float = 1.0
        fake_label: float = 0.0
        real_label_gan: float = 1.0

        for batch_idx, input_dict in enumerate(
            tqdm(self.data_loader_train, desc="Training")
        ):
            this_batch_size: int = input_dict["img"].shape[0]

            log_this_batch: bool = (batch_idx % self.log_every_n_steps == 0) or (
                batch_idx == (len(self.data_loader_train) - 1)
            )

            if this_batch_size == 0:
                break

            self.crn.zero_grad()

            real_img: torch.Tensor = input_dict["img"].to(self.device)
            msk: torch.Tensor = input_dict["msk"].to(self.device)
            instance: torch.Tensor = input_dict["inst"].to(self.device)
            edge_map: Optional[torch.Tensor] = input_dict["edge_map"].to(self.device)

            with self.torch_amp_autocast():

                # Losses
                loss_img: torch.Tensor = torch.zeros(1, device=self.device)
                loss_d: torch.Tensor = torch.zeros(1, device=self.device)
                loss_g: torch.Tensor = torch.zeros(1, device=self.device)

                feature_encoding: Optional[torch.Tensor]
                if self.use_feature_encodings:
                    feature_encoding: torch.Tensor = self.feature_encoder(
                        real_img,
                        instance,
                        input_dict["img_id"]
                        if self.use_saved_feature_encodings
                        else None,
                        input_dict["img_flipped"],
                        msk,
                    )
                else:
                    feature_encoding = None

                fake_img: torch.Tensor = self.crn(
                    msk, feature_encoding, edge_map if self.use_edge_map else None
                )

                if self.use_discriminators:
                    # Image discriminator
                    output_d_fake: torch.Tensor
                    output_d_fake, _ = self.image_discriminator(
                        (
                            msk,
                            edge_map if self.use_edge_map else None,
                            fake_img[:, 0].detach(),
                        )
                    )
                    loss_d_fake: torch.Tensor = self.image_discriminator.calculate_loss(
                        output_d_fake, fake_label, self.criterion_D
                    )

                    output_d_real: torch.Tensor
                    (output_d_real, output_d_real_extra,) = self.image_discriminator(
                        (
                            msk,
                            edge_map if self.use_edge_map else None,
                            real_img,
                        )
                    )
                    loss_d_real: torch.Tensor = self.image_discriminator.calculate_loss(
                        output_d_real, real_label, self.criterion_D
                    )

                    # Generator
                    output_g: torch.Tensor
                    output_g, output_g_extra = self.image_discriminator(
                        (
                            msk,
                            edge_map,
                            fake_img[:, 0],
                        )
                    )
                    loss_g_gan: torch.Tensor = self.image_discriminator.calculate_loss(
                        output_g, real_label_gan, self.criterion_D
                    )

                    loss_g_fm: torch.Tensor = feature_matching_error(
                        output_d_real_extra,
                        output_g_extra,
                        10,
                        self.num_discriminators,
                    )

                    # Prepare for backwards pass
                    loss_d = (loss_d_fake + loss_d_real) * 0.5
                    loss_g = loss_g_gan + loss_g_fm

                if self.use_perceptual_loss:
                    # Normalise image data for use in perceptual loss
                    real_img_normalised = real_img.clone()
                    for b in range(real_img.shape[0]):
                        real_img_normalised[b] = self.normalise(real_img[b].clone())

                    fake_img_normalised = fake_img.clone()
                    for b in range(fake_img.shape[0]):
                        fake_img_normalised[b] = self.normalise(fake_img[b].clone())

                    # Calculate loss on final network output image
                    loss_img: torch.Tensor = self.loss_net(
                        fake_img_normalised.unsqueeze(1), real_img_normalised, msk
                    )

                # Add losses for CRNVideo together, no discriminator loss
                loss: torch.Tensor = loss_img + loss_g

            if self.use_amp == "torch":
                self.optimizer_crn.zero_grad()
                self.torch_gradient_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.crn.parameters(), 30)
                self.torch_gradient_scaler.step(self.optimizer_crn)
                self.torch_gradient_scaler.update()

                if self.use_discriminators:
                    self.optimizer_D_image.zero_grad()
                    self.torch_gradient_scaler.scale(loss_d).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.image_discriminator.parameters(), 30
                    )
                    self.torch_gradient_scaler.step(self.optimizer_D_image)
                    self.torch_gradient_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.crn.parameters(), 30)
                self.optimizer_crn.step()

                if self.use_discriminators:
                    self.optimizer_D_image.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.image_discriminator.parameters(), 30
                    )
                    self.optimizer_D_image.step()

            loss_total += loss.item() * self.batch_size

            if log_this_batch:
                wandb_log_dict: dict = {
                    "Epoch_Fraction": current_epoch
                    + (
                        (batch_idx * self.batch_size)
                        / len(self.data_loader_train.dataset)
                    ),
                    "Batch Loss": loss.item(),
                    "Batch Loss Final Image": loss_img.item(),
                }
                if self.use_discriminators:
                    wandb_log_dict.update(
                        {
                            "Batch Loss Discriminator Image": loss_d.item(),
                            "Batch Loss Generator Image": loss_g_gan.item(),
                            "Batch Loss Feature Matching Image": loss_g_fm.item(),
                            "Output D Fake Image": output_d_fake.mean().item(),
                            "Output D Real Image": output_d_real.mean().item(),
                        }
                    )
                wandb.log(wandb_log_dict)

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

    def sample(
        self, image_numbers: Union[int, tuple], video_dataset: bool = False
    ) -> Union[dict, List[dict]]:

        # Set CRN to eval mode
        self.crn.eval()
        if self.use_feature_encodings:
            self.feature_encoder.eval()
        if self.use_discriminators:
            self.image_discriminator.train()

        with torch.no_grad():
            transform: transforms.ToPILImage = transforms.ToPILImage()

            if isinstance(image_numbers, int):
                image_numbers = (image_numbers,)

            batch_size: int = len(image_numbers)

            first_img: bool = True
            msk_total: Optional[torch.Tensor] = None
            msk_colour_total: Optional[torch.Tensor] = None
            instance_original_total: Optional[torch.Tensor] = None
            edge_map_total: Optional[torch.Tensor] = None
            original_img_total: Optional[torch.Tensor] = None

            for image_no in image_numbers:
                if not video_dataset:
                    input_dict = self.__data_set_val__[image_no]
                else:
                    input_dict = self.__data_set_video__[image_no]

                msk = input_dict["msk"].to(self.device).unsqueeze(0)
                msk_colour = input_dict["msk_colour"].float().unsqueeze(0)
                instance_original = input_dict["inst"].to(self.device).unsqueeze(0)
                edge_map = input_dict["edge_map"].to(self.device).unsqueeze(0)
                original_img = input_dict["img"].to(self.device).unsqueeze(0)

                if first_img:
                    msk_total = msk
                    msk_colour_total = msk_colour
                    instance_original_total = instance_original
                    edge_map_total = edge_map
                    original_img_total = original_img

                    first_img = False
                else:
                    msk_total = torch.cat((msk_total, msk), dim=0)
                    msk_colour_total = torch.cat((msk_colour_total, msk_colour), dim=0)
                    instance_original_total = torch.cat(
                        (instance_original_total, instance_original), dim=0
                    )
                    edge_map_total = torch.cat((edge_map_total, edge_map), dim=0)
                    original_img_total = torch.cat(
                        (original_img_total, original_img), dim=0
                    )

            feature_encoding: Optional[torch.Tensor]
            if self.use_feature_encodings:
                if self.use_saved_feature_encodings:
                    feature_encoding_total = self.feature_encoder.sample_using_means(
                        instance_original_total, msk_total
                    )
                else:
                    feature_encoding_total = self.feature_encoder(
                        original_img_total, instance_original_total
                    )
            else:
                feature_encoding_total = None

            img_out_total: torch.Tensor = self.crn(
                msk_total,
                feature_encoding_total,
                edge_map_total if self.use_edge_map else None,
            )

            # Clamp image to within correct bounds
            img_out_total = img_out_total.clamp(0.0, 1.0)

            # # Drop batch dimension
            # img_out = img_out.squeeze(0).cpu()

            # Bring images to CPU
            img_out_total = img_out_total.cpu()
            original_img_total = original_img_total.cpu()
            msk_colour_total = msk_colour_total.cpu()
            if self.use_feature_encodings:
                feature_encoding_total = feature_encoding_total.cpu()

            output_data_holders: list = []

            for batch_no in range(batch_size):

                split_images = [
                    transform(single_img) for single_img in img_out_total[batch_no]
                ]

                feature_selection: list = (
                    [transform(feature_encoding_total[batch_no])]
                    if self.use_feature_encodings
                    else []
                )

                output_data_holder: SampleDataHolder = SampleDataHolder(
                    image_index=image_numbers[batch_no],
                    video_sample=False,
                    reference_image=[transform(original_img_total[batch_no])],
                    mask_colour=[transform(msk_colour_total[batch_no])],
                    output_image=split_images,
                    feature_selection=feature_selection,
                )

                output_data_holders.append(output_data_holder)

            if len(output_data_holders) == 1:
                return output_data_holders[0]
            else:
                return output_data_holders

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
