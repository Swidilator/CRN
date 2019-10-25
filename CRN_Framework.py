import torch
from torchvision import transforms
from typing import Tuple, List, Any
import random
import wandb


from CRN.CRN_Dataset import CRNDataset
from CRN.Perceptual_Loss import PerceptualLossNetwork
from CRN.CRN_Network import CRN
from Training_Framework import MastersModel


class CRNFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        batch_size_slice: int,
        batch_size_total: int,
        num_loader_workers: int,
        subset_size: int,
        use_tanh: bool,
        use_input_noise: bool,
        settings: dict,
    ):
        super(CRNFramework, self).__init__(
            device,
            data_path,
            batch_size_slice,
            batch_size_total,
            num_loader_workers,
            subset_size,
            use_tanh,
            use_input_noise,
            settings,
        )
        self.model_name: str = "CRN"

        self.max_data_loader_batch_size: int = 16

        self.input_tensor_size: Tuple[int, int] = settings["input_tensor_size"]
        max_input_height_width: Tuple[int, int] = settings["max_input_height_width"]
        num_output_images: int = settings["num_output_images"]
        num_inner_channels: int = settings["num_inner_channels"]
        num_classes: int = settings["num_classes"]
        history_len: int = settings["history_len"]

        self.__set_data_loader__(
            data_path,
            batch_size_total,
            num_loader_workers,
            subset_size,
            use_input_noise,
            settings={
                "max_input_height_width": max_input_height_width,
                "num_classes": num_classes,
            },
        )

        self.__set_model__(
            settings={
                "max_input_height_width": max_input_height_width,
                "num_classes": num_classes,
                "input_tensor_size": self.input_tensor_size,
                "num_output_images": num_output_images,
                "num_inner_channels": num_inner_channels,
                "history_len": history_len,
            }
        )

    @property
    def wandb_trainable_model(self) -> tuple:
        return tuple([self.crn])

    def __set_data_loader__(
        self,
        data_path,
        batch_size_total,
        num_loader_workers,
        subset_size,
        use_input_noise,
        settings,
    ):
        max_input_height_width = settings["max_input_height_width"]
        num_classes: int = settings["num_classes"]

        if batch_size_total > 16:
            batch_size: int = 16
        else:
            batch_size: int = batch_size_total

        self.__data_set_train__ = CRNDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="train",
            num_classes=num_classes,
            should_flip=True,
            subset_size=subset_size,
            noise=use_input_noise,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_test__ = CRNDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="test",
            num_classes=num_classes,
            should_flip=False,
            subset_size=0,
            noise=False,
        )

        self.data_loader_test: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_test__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_val__ = CRNDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="val",
            num_classes=num_classes,
            should_flip=False,
            subset_size=0,
            noise=False,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

    def __set_model__(self, settings) -> None:

        input_tensor_size = settings["input_tensor_size"]
        max_input_height_width = settings["max_input_height_width"]
        self.num_output_images = settings["num_output_images"]
        num_classes = settings["num_classes"]
        num_inner_channels = settings["num_inner_channels"]
        history_len = settings["history_len"]

        IMAGE_CHANNELS = 3

        self.crn: CRN = CRN(
            use_tanh=self.use_tanh,
            input_tensor_size=input_tensor_size,
            final_image_size=max_input_height_width,
            num_output_images=self.num_output_images,
            num_classes=num_classes,
            num_inner_channels=num_inner_channels,
        )

        # self.crn = nn.DataParallel(self.crn, device_ids=device_ids)
        self.crn = self.crn.to(self.device)

        self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
            (IMAGE_CHANNELS, max_input_height_width[0], max_input_height_width[1]),
            history_len,
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

    def save_model(self, model_dir: str, snapshot: bool = False) -> None:
        super().save_model(model_dir, snapshot)

        model_snapshot: str = self.__get_model_snapshot_name__()

        if snapshot:
            torch.save(
                {
                    "model_state_dict": self.crn.state_dict(),
                    "loss_layer_scales": self.loss_net.loss_layer_scales,
                    "loss_history": self.loss_net.loss_layer_history,
                },
                model_dir + model_snapshot,
            )
        torch.save(
            {
                "model_state_dict": self.crn.state_dict(),
                "loss_layer_scales": self.loss_net.loss_layer_scales,
                "loss_history": self.loss_net.loss_layer_history,
            },
            model_dir + self.model_name + "_Latest.pt",
        )

    def load_model(self, model_dir: str, model_snapshot: str = None) -> None:
        super().load_model(model_dir, model_snapshot)

        if model_snapshot is not None:
            checkpoint = torch.load(
                model_dir + model_snapshot, map_location=self.device
            )
            self.crn.load_state_dict(checkpoint["model_state_dict"])
            self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
            self.loss_net.loss_layer_history = checkpoint["loss_history"]
        else:
            checkpoint = torch.load(
                model_dir + self.model_name + "_Latest.pt", map_location=self.device
            )
            self.crn.load_state_dict(checkpoint["model_state_dict"])
            self.loss_net.loss_layer_scales = checkpoint["loss_layer_scales"]
            self.loss_net.loss_layer_history = checkpoint["loss_history"]

    def train(self, update_lambdas: bool) -> Tuple[float, Any]:
        self.crn.train()
        torch.cuda.empty_cache()

        loss_ave: float = 0.0
        loss_total: float = 0.0

        if update_lambdas:
            self.loss_net.update_lambdas()

        # Logic for big batch, whereby we have a large value for a batch, but dataloader provides smaller batch
        mini_batch_per_batch: int = int(
            self.batch_size_total / self.max_data_loader_batch_size
        )
        if mini_batch_per_batch < 1:
            mini_batch_per_batch = 1

        current_mini_batch: int = 0

        this_batch_size: int = 0

        for batch_idx, (img_total, msk_total) in enumerate(self.data_loader_train):
            self.optimizer.zero_grad()
            current_mini_batch += 1

            if current_mini_batch % mini_batch_per_batch == 0:
                this_batch_size = 0

            # Number of times the medium batch should be looped over, given the slice size
            if self.batch_size_total > self.max_data_loader_batch_size:
                batch_size_loops: int = int(
                    self.max_data_loader_batch_size / self.batch_size_slice
                )
            else:
                batch_size_loops: int = int(
                    self.batch_size_total / self.batch_size_slice
                )

            # Loop over medium batch
            for i in range(batch_size_loops):
                img: torch.Tensor = img_total[
                    i * self.batch_size_slice: (i + 1) * self.batch_size_slice
                ].to(self.device)
                msk: torch.Tensor = msk_total[
                    i * self.batch_size_slice: (i + 1) * self.batch_size_slice
                ].to(self.device)

                mini_batch_size: int = msk.shape[0]
                if mini_batch_size == 0:
                    continue
                else:
                    this_batch_size += mini_batch_size
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
                del msk, img
                del loss

            if current_mini_batch % mini_batch_per_batch == 0:
                if (self.batch_size_total > 8) or (
                    batch_idx * self.batch_size_total % 120 == 112
                ):
                    batch_loss_val: float = (
                        loss_ave / this_batch_size
                    ) * self.batch_size_total

                    print(
                        "Batch: {batch}\nLoss: {loss_val}".format(
                            batch=int(batch_idx / mini_batch_per_batch),
                            loss_val=batch_loss_val,
                        )
                    )
                    # WandB logging, if WandB disabled this should skip the logging without error
                    wandb.log({"Batch Loss": batch_loss_val})
                    loss_ave = 0.0

                for i in self.crn.parameters():
                    i: torch.nn.Parameter = i
                    i.grad = i.grad / (self.batch_size_total / self.batch_size_slice)
                # print("Stepping")
                self.optimizer.step()
            # del loss, msk, noise, img
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

    def sample(self, k: int) -> List[Any]:
        self.crn.eval()
        sample_list: list = random.sample(range(len(self.__data_set_test__)), k)
        outputs: List[Any] = []
        # noise: torch.Tensor = torch.randn(
        #     1,
        #     1,
        #     self.input_tensor_size[0],
        #     self.input_tensor_size[1],
        #     device=self.device,
        # )
        transform: transforms.ToPILImage = transforms.ToPILImage()
        for i, val in enumerate(sample_list):
            img, msk = self.__data_set_test__[val]
            msk = msk.to(self.device).unsqueeze(0)
            img_out: torch.Tensor = self.crn(inputs=(msk, None))
            # print(img_out.shape)
            for img_no in range(self.num_output_images):
                start_channel: int = img_no * 3
                end_channel: int = (img_no + 1) * 3
                img_out_single: torch.Tensor = img_out[
                    0, start_channel:end_channel
                ].cpu()
                outputs.append(transform(img_out_single))
            del img, msk
        return outputs

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
