import os
import sys
from contextlib import nullcontext
from itertools import chain
from typing import Tuple, Any, Union, Optional, List

import torch
import wandb
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm

from CRN.CRN_Video_Network import CRNVideo
from CRN.Perceptual_Loss import PerceptualLossNetwork
from support_scripts.components import FeatureEncoder, FlowNetWrapper
from support_scripts.utils import (
    MastersModel,
    ModelSettingsManager,
    CityScapesDataset,
    CityScapesDemoVideoDataset,
)
from support_scripts.utils.datasets.custom_video_datasets import CityScapesVideoDataset2

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        }
        # fmt: on

        return cls(**manager.args, **settings)

    def __set_data_loader__(self, **kwargs):

        self.__data_set_train__ = CityScapesVideoDataset2(
            root=self.dataset_path + "/sequence",
            split="train",
            should_flip=False,
            subset_size=self.training_subset_size,
            output_image_height_width=self.input_image_height_width,
            num_frames=6,
            frame_offset="random",
        )

        self.data_loader_train: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                self.__data_set_train__,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_data_workers,
                pin_memory=True,
            )
        )

        self.__data_set_val__ = CityScapesDemoVideoDataset(
            output_image_height_width=self.input_image_height_width,
            root=self.dataset_path,
            split="val",
            should_flip=False,
            subset_size=0,
            noise=False,
            specific_model="CRN",
            num_frames=16,
            use_all_classes=self.use_all_classes,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_workers,
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
            use_all_classes=True,  # Since it only contains the correct amount of classes in the dataset
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
            num_resnet_processing_rms=self.num_resnet_processing_rms
        )

        # self.crn_video = nn.DataParallel(self.crn_video, device_ids=device_ids)
        self.crn_video = self.crn_video.to(self.device)

        if not self.sample_only:

            self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
                self.perceptual_base_model,
                self.device,
                self.use_loss_output_image,
                self.loss_scaling_method,
            )

            # self.loss_net = nn.DataParallel(self.loss_net, device_ids=device_ids)
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

        current_epoch: int = kwargs["current_epoch"]

        loss_total: float = 0.0

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
            prev_image = torch.cat(
                (
                    input_dict["img"][:, 0],
                    torch.zeros_like(input_dict["img"][:, 0]),
                ),
                dim=1,
            ).to(self.device)
            prev_msk = torch.cat(
                (
                    input_dict["msk"][:, 0],
                    torch.zeros_like(input_dict["msk"][:, 0]),
                ),
                dim=1,
            ).to(self.device)

            # Loss holders
            video_loss: float = 0.0
            video_loss_img: float = 0.0
            video_loss_h: float = 0.0
            video_loss_flow: float = 0.0
            video_loss_warp: float = 0.0

            for i in range(1, num_frames):
                self.crn_video.zero_grad()

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
                        edge_map if self.use_feature_encodings else None,
                        prev_image,
                        prev_msk,
                    )

                    # Previous outputs stored for input later
                    prev_image = torch.cat(
                        (fake_img.detach().clone().clamp(0.0, 1.0), prev_image[:, 0:3]),
                        dim=1,
                    )
                    prev_msk = torch.cat(
                        (msk.detach(), prev_msk[:, 0 : self.num_classes]), dim=1
                    )

                    real_flow: torch.Tensor = (
                        self.flownet(
                            real_img, input_dict["img"][:, i - 1].to(self.device)
                        )
                        .detach()
                        .permute(0, 2, 3, 1)
                    )

                    real_flow_mask = (
                        real_flow.permute(0, 3, 1, 2).abs().max(dim=1, keepdim=True)[0]
                    )
                    real_flow_mask = real_flow_mask - real_flow_mask.min()
                    real_flow_mask = real_flow_mask / real_flow_mask.max()

                    show_images: bool = True
                    if show_images:
                        from matplotlib import pyplot as plt
                        import flowiz as fz

                        fake_flow_viz: torch.Tensor = fz.convert_from_flow(
                            fake_flow.detach()
                            .permute(0, 2, 3, 1)
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                        real_flow_viz: torch.Tensor = fz.convert_from_flow(
                            real_flow.detach().squeeze().cpu().numpy()
                        )

                        fig, axs = plt.subplots(4, 2)
                        axs[0, 0].imshow(
                            prev_image[:, 0:3]
                            .squeeze()
                            .permute(1, 2, 0)
                            .cpu()
                            .clamp(0.0, 1.0)
                            .numpy()
                        )

                        axs[0, 1].imshow(
                            real_img[:, 0:3]
                            .squeeze()
                            .permute(1, 2, 0)
                            .cpu()
                            .clamp(0.0, 1.0)
                            .numpy()
                        )

                        axs[1, 0].imshow(
                            fake_img_h.detach()
                            .squeeze()
                            .permute(1, 2, 0)
                            .cpu()
                            .clamp(0.0, 1.0)
                            .numpy()
                        )

                        axs[1, 1].imshow(
                            fake_img_w.detach()
                            .squeeze()
                            .permute(1, 2, 0)
                            .cpu()
                            .clamp(0.0, 1.0)
                            .numpy()
                        )

                        # plt.title("Image")
                        fake_flow_mask_viz = (
                            fake_flow_mask.detach()
                            .squeeze()
                            .cpu()
                            .clamp(0.0, 1.0)
                            .numpy()
                        )
                        axs[2, 0].imshow(fake_flow_mask_viz)

                        real_flow_mask_viz = (
                            real_flow_mask.detach()
                            .squeeze()
                            .cpu()
                            .clamp(0.0, 1.0)
                            .numpy()
                        )
                        axs[2, 1].imshow(real_flow_mask_viz)

                        axs[3, 0].imshow(fake_flow_viz)
                        # plt.title("Flow")

                        axs[3, 1].imshow(real_flow_viz)

                        # plt.title("Real Flow")
                        plt.show()

                    warped_real_prev_image: torch.Tensor = FlowNetWrapper.resample(
                        prev_image[:, 3:6].detach(), fake_flow, self.crn_video.grid
                    )
                    loss_warp_scaling_factor: float = 10.0
                    loss_warp: torch.Tensor = (
                        self.flow_criterion(
                            warped_real_prev_image, real_img[:, 0:3].detach()
                        )
                        * loss_warp_scaling_factor
                    )

                    loss_flow_scaling_factor: float = 10.0
                    loss_flow: torch.Tensor = (
                        self.flow_criterion(fake_flow.permute(0, 2, 3, 1), real_flow)
                        * loss_flow_scaling_factor
                    )

                    # Normalise image data for use in perceptual loss
                    real_img_normalised = real_img.clone()
                    for b in range(real_img.shape[0]):
                        real_img_normalised[b] = self.normalise(real_img[b].clone())

                    fake_img_normalised = fake_img.clone()
                    for b in range(fake_img.shape[0]):
                        fake_img_normalised[b] = self.normalise(fake_img[b].clone())

                    fake_img_h_normalised = fake_img_h.clone()
                    for b in range(fake_img.shape[0]):
                        fake_img_h_normalised[b] = self.normalise(fake_img_h[b].clone())

                    loss_img: torch.Tensor = self.loss_net(
                        fake_img_normalised.unsqueeze(1), real_img_normalised, msk
                    )

                    loss_img_h: torch.Tensor = self.loss_net(
                        fake_img_h_normalised.unsqueeze(1), real_img_normalised, msk
                    )

                    loss: torch.Tensor = loss_warp + loss_flow + loss_img_h + loss_img

                if self.use_amp == "torch":
                    self.torch_gradient_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.crn_video.parameters(), 10)
                    self.torch_gradient_scaler.step(self.optimizer)
                    self.torch_gradient_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_total += loss.item() * self.batch_size / (num_frames - 1)
                video_loss += loss.item() * self.batch_size / (num_frames - 1)

                video_loss_img += loss_img.item() * self.batch_size / (num_frames - 1)
                video_loss_h += loss_img_h.item() * self.batch_size / (num_frames - 1)
                video_loss_flow += loss_flow.item() * self.batch_size / (num_frames - 1)
                video_loss_warp += loss_warp.item() * self.batch_size / (num_frames - 1)

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
                    }
                )

        return loss_total, None

    def eval(self) -> Tuple[float, Any]:
        pass

    def sample(
        self, image_numbers: Union[int, tuple], video_dataset: bool = False
    ) -> Union[dict, List[dict]]:

        # Set CRN to eval mode
        self.crn_video.eval()
        if self.use_feature_encodings:
            self.feature_encoder.eval()

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
                # img, msk, msk_colour, instance, instance_processed, feature_selection

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

            img_out_total: torch.Tensor = self.crn_video(
                msk_total, feature_encoding_total, edge_map_total
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

            output_dicts: list = []

            for batch_no in range(batch_size):

                split_images = [
                    transform(single_img) for single_img in img_out_total[batch_no]
                ]

                output_img_dict: dict = {
                    "output_img_{i}".format(i=i): img
                    for i, img in enumerate(split_images)
                }
                if self.use_feature_encodings:
                    output_img_dict.update(
                        {
                            "feature_selection": transform(
                                feature_encoding_total[batch_no]
                            )
                        }
                    )

                output_dict: dict = {
                    "image_index": image_numbers[batch_no],
                    "original_img": transform(original_img_total[batch_no]),
                    "msk_colour": transform(msk_colour_total[batch_no]),
                    "output_img_dict": output_img_dict,
                }
                output_dicts.append(output_dict)

            if len(output_dicts) == 1:
                return output_dicts[0]
            else:
                return output_dicts

