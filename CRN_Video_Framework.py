import os
from contextlib import nullcontext
from itertools import chain
from typing import Tuple, Any, Union, Optional, List

import torch
import wandb
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm

import flowiz as fz

from CRN.CRN_Video_Network import CRNVideo
from CRN.Perceptual_Loss import PerceptualLossNetwork
from support_scripts.components import FeatureEncoder, FlowNetWrapper, FullDiscriminator, feature_matching_error
from support_scripts.sampling import SampleDataHolder
from support_scripts.utils import (
    MastersModel,
    ModelSettingsManager,
    CityScapesDemoVideoDataset,
    CityScapesVideoDataset,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# noinspection PyUnresolvedReferences
class CRNVideoFramework(MastersModel):
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
        **kwargs,
    ):
        super(CRNVideoFramework, self).__init__(
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
            **kwargs,
        )
        self.model_name: str = "CRNVideo"

        try:
            assert "input_tensor_size" in kwargs
            assert "num_inner_channels" in kwargs
            assert "perceptual_base_model" in kwargs
            assert "use_feature_encodings" in kwargs
            assert "use_loss_output_image" in kwargs
            assert "loss_scaling_method" in kwargs
            assert "layer_norm_type" in kwargs
            assert "use_saved_feature_encodings" in kwargs
            assert "use_resnet_rms" in kwargs
            assert "flownet_save_path" in kwargs
            assert "num_resnet_processing_rms" in kwargs
            assert "use_edge_map" in kwargs
            assert "use_twin_network" in kwargs
            assert "use_discriminators" in kwargs

        except AssertionError as e:
            print("Missing argument: {error}".format(error=e))
            raise SystemExit

        # fmt: off
        self.input_tensor_size: tuple = kwargs["input_tensor_size"]
        self.num_inner_channels: int = kwargs["num_inner_channels"]
        self.perceptual_base_model: str = kwargs["perceptual_base_model"]
        self.use_feature_encodings: bool = kwargs["use_feature_encodings"]
        self.use_loss_output_image: bool = kwargs["use_loss_output_image"]
        self.loss_scaling_method: str = kwargs["loss_scaling_method"]
        self.layer_norm_type: str = kwargs["layer_norm_type"]
        self.use_saved_feature_encodings: bool = kwargs["use_saved_feature_encodings"]
        self.use_resnet_rms: bool = kwargs["use_resnet_rms"]
        self.flownet_save_path: str = kwargs["flownet_save_path"]  # This is in args
        self.num_resnet_processing_rms: int = kwargs["num_resnet_processing_rms"]
        self.use_edge_map: bool = kwargs["use_edge_map"]
        self.use_twin_network: bool = kwargs["use_twin_network"]
        self.use_discriminators: bool = kwargs["use_discriminators"]
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
    def data_set_video(self) -> torch.utils.data.Dataset:
        return self.__data_set_video__

    @property
    def wandb_trainable_model(self) -> tuple:
        models = [self.crn_video]
        if self.use_feature_encodings:
            models.append(self.feature_encoder)
        return tuple(models)

    @classmethod
    def from_model_settings_manager(
        cls, manager: ModelSettingsManager
    ) -> "CRNVideoFramework":
        # fmt: off
        settings = {
            "input_tensor_size": (
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_HEIGHT"],
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_WIDTH"],
            ),
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
            "use_twin_network": manager.model_conf["CRN_USE_TWIN_NETWORK"],
            "use_discriminators": manager.model_conf["CRN_USE_DISCRIMINATORS"],
        }
        # fmt: on

        return cls(**manager.args, **settings)

    def __set_data_loader__(self, **kwargs):

        self.__data_set_train__ = CityScapesVideoDataset(
            root=self.dataset_path + "/sequence",
            split="train",
            should_flip=False,
            subset_size=self.training_subset_size,
            output_image_height_width=self.input_image_height_width,
            num_frames=self.num_frames_per_video,
            frame_offset="random",
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

        self.__data_set_val__ = CityScapesVideoDataset(
            root=self.dataset_path + "/sequence",
            split="val",
            should_flip=False,
            subset_size=0,
            output_image_height_width=self.input_image_height_width,
            num_frames=self.num_frames_per_video,
            frame_offset="random",
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            pin_memory=True,
        )

        self.__data_set_video__ = CityScapesDemoVideoDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="demoVideo",
            should_flip=False,
            subset_size=0,
            noise=False,
            specific_model="CRN",
            num_frames=16,
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
        # Feature Encoder
        if self.use_feature_encodings:
            self.feature_encoder: FeatureEncoder = FeatureEncoder(
                3,
                3,
                4,
                self.device,
                self.model_save_dir,
                self.use_saved_feature_encodings,
                use_masks_as_instances=True,
                num_semantic_classes=self.num_classes,
            )
            self.feature_encoder = self.feature_encoder.to(self.device)
            if self.use_saved_feature_encodings:
                self.feature_encoder.eval()
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.feature_encoder.parameters():
                    param.requires_grad = True

        self.crn_video: CRNVideo = CRNVideo(
            use_tanh=self.use_tanh,
            input_tensor_size=self.input_tensor_size,
            final_image_size=self.input_image_height_width,
            num_classes=self.num_classes,
            num_inner_channels=self.num_inner_channels,
            use_feature_encoder=self.use_feature_encodings,
            layer_norm_type=self.layer_norm_type,
            use_resnet_rms=self.use_resnet_rms,
            num_resnet_processing_rms=self.num_resnet_processing_rms,
            num_prior_frames=self.num_prior_frames,
            use_optical_flow=self.use_optical_flow,
            use_edge_map=self.use_edge_map,
            use_twin_network=self.use_twin_network
        )
        print(self.crn_video)

        # self.crn_video = nn.DataParallel(self.crn_video, device_ids=device_ids)
        self.crn_video = self.crn_video.to(self.device)

        if not self.sample_only:
            if self.use_discriminators:
                self.discriminator: FullDiscriminator = FullDiscriminator(
                    self.device,
                    self.num_classes,
                    1,
                    False,
                    self.use_edge_map,
                )
                self.discriminator = self.discriminator.to(self.device)

                self.optimizer_D: torch.optim.Adam = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=0.0001,
                    betas=(0.5, 0.999),
                    # eps=1e-08,
                    # weight_decay=0,
                )
                self.criterion_D = torch.nn.MSELoss()

            self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
                self.perceptual_base_model,
                self.device,
                self.use_loss_output_image,
                self.loss_scaling_method,
            )
            self.loss_net = self.loss_net.to(self.device)

            self.flow_criterion = torch.nn.L1Loss()

            # Create params depending on what needs to be trained
            params = self.crn_video.parameters()

            if self.use_feature_encodings and not self.use_saved_feature_encodings:
                params = chain(
                    params,
                    self.feature_encoder.parameters(),
                )

            self.optimizer = torch.optim.Adam(
                params,
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
            )

            self.normalise = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            # Flownet for video training
            if self.use_optical_flow:
                self.flownet = FlowNetWrapper(self.flownet_save_path)
                self.flownet = self.flownet.to(self.device)

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
        network: torch.nn.Module

        for network in self.wandb_trainable_model:
            network.eval()
            network.zero_grad()

        try:
            assert not self.sample_only
        except AssertionError as e:
            raise AssertionError("Cannot save model in 'sample_only mode'")

        save_dict: dict = {
            "dict_crn": self.crn_video.state_dict(),
            "args": self.args,
            "kwargs": self.kwargs,
        }
        if self.use_feature_encodings:
            save_dict.update(
                {"dict_encoder_decoder": self.feature_encoder.state_dict()}
            )
        if self.use_discriminators:
            save_dict.update(
                {"dict_discriminator": self.discriminator.state_dict()}
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
        self.crn_video.load_state_dict(checkpoint["dict_crn"], strict=False)
        if self.use_feature_encodings:
            self.feature_encoder.load_state_dict(checkpoint["dict_encoder_decoder"])
        if self.use_discriminators:
            self.discriminator.load_state_dict(checkpoint["dict_discriminator"])

    @classmethod
    def load_model_with_embedded_settings(cls, manager: ModelSettingsManager):
        load_path: str = os.path.join(
            manager.args["model_save_dir"], manager.args["load_saved_model"]
        )
        checkpoint = torch.load(load_path)
        args: dict = checkpoint["args"]
        kwargs: dict = checkpoint["kwargs"]

        model_frame: CRNVideoFramework = cls(**args, **kwargs)
        model_frame.load_model(manager.args["load_saved_model"])
        return model_frame

    def train(self, **kwargs) -> Tuple[float, Any]:
        self.crn_video.train()
        # If sampling from saved feature encodings, and using a single set of settings
        #  all sampling for a single epoch will have the same settings, but refresh between epochs
        if self.use_feature_encodings and self.use_saved_feature_encodings:
            self.feature_encoder.feature_extractions_sampler.update_single_setting_class_list()

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
            num_frames: int = input_dict["img"].shape[1]

            log_this_batch: bool = (batch_idx % self.log_every_n_steps == 0) or (
                batch_idx == (len(self.data_loader_train) - 1)
            )

            if this_batch_size == 0:
                break

            # prev_image: torch.Tensor = torch.zeros_like(input_dict["img"][:,0]).to(self.device)
            prior_image_list: list = [
                torch.zeros_like(input_dict["img"][:, 0], device=self.device)
            ] * self.num_prior_frames
            prior_msk_list: list = [
                torch.zeros_like(input_dict["msk"][:, 0], device=self.device)
            ] * self.num_prior_frames

            if self.num_prior_frames > 0:
                if self.prior_frame_seed_type == "real":
                    prior_image_list[0] = input_dict["img"][:, 0].to(self.device)
                    prior_msk_list[0] = input_dict["msk"][:, 0].to(self.device)

            # Loss holders
            video_loss: float = 0.0
            video_loss_img: float = 0.0
            video_loss_h: float = 0.0
            video_loss_flow: float = 0.0
            video_loss_warp: float = 0.0

            video_loss_d: float = 0.0
            video_loss_g: float = 0.0
            video_loss_g_fm: float = 0.0
            video_output_d_real_mean: float = 0.0
            video_output_d_fake_mean: float = 0.0

            skip_first_frame: int = self.prior_frame_seed_type == "real"

            for i in range(skip_first_frame, num_frames):
                self.crn_video.zero_grad()
                if self.use_discriminators:
                    self.discriminator.zero_grad()

                real_img: torch.Tensor = input_dict["img"][:, i].to(self.device)
                msk: torch.Tensor = input_dict["msk"][:, i].to(self.device)
                instance: torch.Tensor = input_dict["inst"][:, i].to(self.device)
                edge_map: torch.Tensor = input_dict["edge_map"][:, i].to(self.device)

                with self.torch_amp_autocast():
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

                    fake_img: torch.Tensor
                    fake_flow: torch.Tensor
                    (
                        fake_img,
                        fake_img_h,
                        fake_img_w,
                        fake_flow,
                        fake_flow_mask,
                    ) = self.crn_video(
                        msk,
                        feature_encoding,
                        edge_map if self.use_edge_map else None,
                        torch.cat(prior_image_list, dim=1)
                        if self.num_prior_frames > 0
                        else None,
                        torch.cat(prior_msk_list, dim=1)
                        if self.num_prior_frames > 0
                        else None,
                    )

                    if self.use_discriminators:
                        # Discriminator
                        output_d_fake: torch.Tensor
                        output_d_fake, _ = self.discriminator(
                            (msk, edge_map if self.use_edge_map else None, fake_img.detach())
                        )
                        loss_d_fake = self.discriminator.calculate_loss(
                            output_d_fake, fake_label, self.criterion_D
                        )

                        output_d_real: torch.Tensor
                        output_d_real, output_d_real_extra = self.discriminator(
                            (msk, edge_map if self.use_edge_map else None, real_img)
                        )
                        loss_d_real = self.discriminator.calculate_loss(
                            output_d_real, real_label, self.criterion_D
                        )

                        # Generator
                        output_g, output_g_extra = self.discriminator((msk, edge_map, fake_img))
                        loss_g_gan = self.discriminator.calculate_loss(
                            output_g, real_label_gan, self.criterion_D
                        )

                        loss_g_fm = feature_matching_error(
                            output_d_real_extra,
                            output_g_extra,
                            10,
                            1,
                        )

                        # Prepare for backwards pass
                        loss_d = (loss_d_fake + loss_d_real) * 0.5
                        loss_g = loss_g_gan + loss_g_fm
                    else:
                        loss_d = torch.zeros(1, device=self.device)
                        loss_g = torch.zeros(1, device=self.device)

                    # Normalise image data for use in perceptual loss
                    real_img_normalised = real_img.clone()
                    for b in range(real_img.shape[0]):
                        real_img_normalised[b] = self.normalise(real_img[b].clone())

                    fake_img_normalised = fake_img.clone()
                    for b in range(fake_img.shape[0]):
                        fake_img_normalised[b] = self.normalise(fake_img[b].clone())

                    loss_warp = torch.zeros(1, device=self.device)
                    loss_flow = torch.zeros(1, device=self.device)
                    loss_img_h = torch.zeros(1, device=self.device)
                    # Previous outputs stored for input later
                    if self.num_prior_frames > 0:

                        prior_image_list = [
                            fake_img.detach().clone().clamp(0.0, 1.0),
                            *prior_image_list[1 : self.num_prior_frames],
                        ]
                        prior_msk_list = [
                            msk.detach(),
                            *prior_msk_list[1 : self.num_prior_frames],
                        ]

                        if self.use_optical_flow:
                            # Normalise generated image and calculate loss
                            fake_img_h_normalised = fake_img_h.clone()
                            for b in range(fake_img.shape[0]):
                                fake_img_h_normalised[b] = self.normalise(
                                    fake_img_h[b].clone()
                                )

                            loss_img_h: torch.Tensor = self.loss_net(
                                fake_img_h_normalised.unsqueeze(1),
                                real_img_normalised,
                                msk,
                            )

                            # Generate reference optical flow
                            real_flow: torch.Tensor = (
                                self.flownet(
                                    real_img,
                                    input_dict["img"][:, i - 1].to(self.device),
                                )
                                .detach()
                                .permute(0, 2, 3, 1)
                            )

                            # Warp prior reference image and compare to warped prior image
                            warped_real_prev_image: torch.Tensor = (
                                FlowNetWrapper.resample(
                                    prior_image_list[1].detach(),
                                    fake_flow,
                                    self.crn_video.grid,
                                )
                            )
                            loss_warp_scaling_factor: float = 10.0
                            loss_warp: torch.Tensor = (
                                self.flow_criterion(
                                    warped_real_prev_image, real_img[:, 0:3].detach()
                                )
                                * loss_warp_scaling_factor
                            )

                            # Calculate direct comparison optical flow loss
                            loss_flow_scaling_factor: float = 10.0
                            loss_flow: torch.Tensor = (
                                self.flow_criterion(
                                    fake_flow.permute(0, 2, 3, 1), real_flow
                                )
                                * loss_flow_scaling_factor
                            )

                    # Calculate loss on final network output image
                    loss_img: torch.Tensor = self.loss_net(
                        fake_img_normalised.unsqueeze(1), real_img_normalised, msk
                    )

                    # Add losses together
                    loss: torch.Tensor = loss_img + loss_img_h + loss_warp + loss_flow + loss_g

                if self.use_amp == "torch":
                    self.optimizer.zero_grad()
                    self.torch_gradient_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.crn_video.parameters(), 10)
                    self.torch_gradient_scaler.step(self.optimizer)
                    self.torch_gradient_scaler.update()

                    if self.use_discriminators:
                        self.optimizer_D.zero_grad()
                        self.torch_gradient_scaler.scale(loss_d).backward()
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10)
                        self.torch_gradient_scaler.step(self.optimizer_D)
                        self.torch_gradient_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.crn_video.parameters(), 10)
                    self.optimizer.step()

                    if self.use_discriminators:
                        self.optimizer_D.zero_grad()
                        loss_d.backward()
                        self.optimizer_D.step()

                loss_total += loss.item() * self.batch_size / (num_frames - skip_first_frame)
                video_loss += loss.item() * self.batch_size / (num_frames - skip_first_frame)

                video_loss_img += loss_img.item() * self.batch_size / (num_frames - skip_first_frame)
                video_loss_h += loss_img_h.item() * self.batch_size / (num_frames - skip_first_frame)
                video_loss_flow += loss_flow.item() * self.batch_size / (num_frames - skip_first_frame)
                video_loss_warp += loss_warp.item() * self.batch_size / (num_frames - skip_first_frame)

                if self.use_discriminators:
                    video_loss_d += loss_d.item() * self.batch_size / (num_frames - skip_first_frame)
                    video_loss_g += loss_g.item() * self.batch_size / (num_frames - skip_first_frame)
                    video_loss_g_fm += loss_g_fm.item() * self.batch_size / (num_frames - skip_first_frame)
                    video_output_d_real_mean += output_d_real.mean().item() * self.batch_size / (num_frames - skip_first_frame)
                    video_output_d_fake_mean += output_d_fake.mean().item() * self.batch_size / (num_frames - skip_first_frame)

            if log_this_batch:
                wandb.log(
                    {
                        "Epoch_Fraction": current_epoch
                        + (
                            (batch_idx * self.batch_size)
                            / len(self.data_loader_train.dataset)
                        ),
                        "Batch Loss Video": video_loss,
                        "Batch Loss Combined": video_loss_img,
                        "Batch Loss Hallucinated": video_loss_h,
                        "Batch Loss Warp": video_loss_warp,
                        "Batch Loss Flow": video_loss_flow,
                        "Batch Loss Discriminator": video_loss_d,
                        "Batch Loss Generator": video_loss_g,
                        "Batch Loss Feature matching": video_loss_g_fm,
                        "Output D Fake": video_output_d_fake_mean,
                        "Output D Real": video_output_d_real_mean,
                    }
                )

        return loss_total, None

    def eval(self) -> Tuple[float, Any]:
        pass

    def sample(
        self, image_numbers: Union[int, tuple], video_dataset: bool = False
    ) -> Union[dict, List[dict]]:

        assert (
            type(image_numbers) is int or len(image_numbers) == 1
        ), "Video networks only support 1 image at a time"

        # Set CRN to eval mode
        self.crn_video.eval()
        if self.use_feature_encodings:
            self.feature_encoder.eval()

        with torch.no_grad():

            transform: transforms.ToPILImage = transforms.ToPILImage()

            if isinstance(image_numbers, int):
                image_numbers = (image_numbers,)

            output_data_holders: list = []

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

            prior_image_list: list = [
                torch.zeros_like(original_img_total[:, 0], device=self.device)
            ] * self.num_prior_frames
            prior_msk_list: list = [
                torch.zeros_like(msk_total[:, 0], device=self.device)
            ] * self.num_prior_frames

            if self.num_prior_frames > 0:
                if self.prior_frame_seed_type == "real":
                    prior_image_list[0] = original_img_total[:, 0].to(self.device)
                    prior_msk_list[0] = msk_total[:, 0].to(self.device)

            reference_image_list: list = []
            mask_colour_list: list = []
            output_image_list: list = []
            feature_selection_list: list = []
            hallucinated_image_list: list = []
            warped_image_list: list = []
            combination_weights_list: list = []
            output_flow_list: list = []
            reference_flow_list: list = []

            for frame_no in range(
                (self.prior_frame_seed_type == "real"), self.num_frames_per_video
            ):
                self.crn_video.zero_grad()

                real_img: torch.Tensor = original_img_total[:, frame_no]
                msk: torch.Tensor = msk_total[:, frame_no]
                instance: torch.Tensor = instance_original_total[:, frame_no].to(
                    self.device
                )
                # feature_encoding: torch.Tensor = feature_encoding_total[:, frame_no].to(self.device)
                edge_map: torch.Tensor = edge_map_total[:, frame_no].to(self.device)

                feature_encoding: Optional[torch.Tensor]
                if self.use_feature_encodings:
                    if self.use_saved_feature_encodings:
                        feature_encoding = self.feature_encoder.sample_using_means(
                            instance, msk, fixed_class_lists=True
                        )
                    else:
                        feature_encoding = self.feature_encoder(
                            real_img, instance, mask=msk
                        )
                else:
                    feature_encoding = None

                fake_img: torch.Tensor
                fake_flow: torch.Tensor
                (
                    fake_img,
                    fake_img_h,
                    fake_img_w,
                    fake_flow,
                    fake_flow_mask,
                ) = self.crn_video(
                    msk,
                    feature_encoding,
                    edge_map if self.use_edge_map else None,
                    torch.cat(prior_image_list, dim=1)
                    if self.num_prior_frames > 0
                    else None,
                    torch.cat(prior_msk_list, dim=1)
                    if self.num_prior_frames > 0
                    else None,
                )

                # Previous outputs stored for input later
                if self.num_prior_frames > 0:
                    prior_image_list = [
                        fake_img.detach().clone().clamp(0.0, 1.0),
                        *prior_image_list[1 : self.num_prior_frames],
                    ]
                    prior_msk_list = [
                        msk.detach(),
                        *prior_msk_list[1 : self.num_prior_frames],
                    ]

                if self.use_optical_flow:
                    real_flow: torch.Tensor = (
                        self.flownet(
                            real_img,
                            original_img_total[:, frame_no - 1].to(self.device),
                        )
                        .detach()
                        .permute(0, 2, 3, 1)
                    )

                    fake_flow_viz: torch.Tensor = (
                        torch.tensor(
                            fz.convert_from_flow(
                                fake_flow.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                            )
                        )
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                    )

                    real_flow_viz: torch.Tensor = (
                        torch.tensor(
                            fz.convert_from_flow(real_flow.squeeze().cpu().numpy())
                        )
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                    )
                    hallucinated_image_list.append(
                        transform(fake_img_h.squeeze().clamp(0.0, 1.0).cpu())
                    )
                    warped_image_list.append(
                        transform(fake_img_w.squeeze().clamp(0.0, 1.0).cpu())
                    )
                    combination_weights_list.append(
                        transform(fake_flow_mask[0, 0].cpu())
                    )
                    output_flow_list.append(transform(fake_flow_viz))
                    reference_flow_list.append(transform(real_flow_viz))

                reference_image_list.append(transform(real_img.squeeze().cpu()))
                mask_colour_list.append(transform(msk_colour_total[0, frame_no]))
                output_image_list.append(
                    transform(fake_img.squeeze().clamp(0.0, 1.0).cpu())
                )
                if self.use_feature_encodings:
                    feature_selection_list.append(
                        transform(feature_encoding.squeeze().cpu())
                    )

            output_data_holder: SampleDataHolder = SampleDataHolder(
                image_index=image_no,
                video_sample=True,
                reference_image=reference_image_list,
                mask_colour=mask_colour_list,
                output_image=output_image_list,
                hallucinated_image=hallucinated_image_list,
                warped_image=warped_image_list,
                combination_weights=combination_weights_list,
                output_flow=output_flow_list,
                reference_flow=reference_flow_list,
                feature_selection=feature_selection_list,
            )

            output_data_holders.append(output_data_holder)

            if len(output_data_holders) == 1:
                return output_data_holders[0]
            else:
                return output_data_holders
