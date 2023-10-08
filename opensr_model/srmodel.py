import pathlib
from typing import Union

import requests
import torch
from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from skimage.exposure import match_histograms


class SRLatentDiffusion(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device] = "cpu"):
        super().__init__()

        # Parameters used in the autoencoder
        first_stage_config = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 4,
            "out_ch": 4,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }

        # Parameters used in the denoiser
        cond_stage_config = {
            "image_size": 64,
            "in_channels": 8,
            "model_channels": 160,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": [16, 8],
            "channel_mult": [1, 2, 2, 4],
            "num_head_channels": 32,
        }

        # Set up the model
        self.model = LatentDiffusion(
            first_stage_config,
            cond_stage_config,
            timesteps=1000,
            unet_config=cond_stage_config,
            linear_start=0.0015,
            linear_end=0.0155,
            concat_mode=True,
            cond_stage_trainable=False,
            first_stage_key="image",
            cond_stage_key="LR_image",
        )

        # Apply normalization
        self.mean = [
            -0.7383498348772526,
            -0.7252872248232365,
            -0.7637851044356823,
            -0.6586044501721859,
        ]
        self.std = [
            0.0726721865855623,
            0.06286528447978199,
            0.050181950839143244,
            0.07026348426636543,
        ]

        # Set up the model for inference
        self.device = device
        self.model.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def _create_batch(self, X: torch.Tensor):
        # Normalize to [-1, 1]
        X = X * 2 - 1

        # Apply means and stds to each band
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - self.mean[i]) / self.std[i]

        # Same device as the model
        X = X.to(self.device)

        # Create a reference image
        X_ref = torch.nn.functional.interpolate(
            torch.zeros_like(X), scale_factor=4, mode="nearest"
        )

        return {"LR_image": X, "image": X_ref}

    @torch.no_grad()
    def forward(
        self,
        X: torch.Tensor,
        custom_steps: int = 100,
        temperature: float = 1.0,
        spectral_correction: bool = True,
    ):
        """Obtain the super resolution of the given image.

        Args:
            X (torch.Tensor): If a Sentinel-2 L2A image with reflectance values
                in the range [0, 1] and shape CxWxH, the super resolution of the image
                is returned. If a batch of images with shape BxCxWxH is given, a batch
                of super resolved images is returned.
            custom_steps (int, optional): Number of steps to run the denoiser. Defaults
                to 100.
            temperature (float, optional): Temperature to use in the denoiser.
                Defaults to 1.0. The higher the temperature, the more stochastic
                the denoiser is.
            spectral_correction (bool, optional): Apply spectral correction to the SR
                image, using the LR image as reference. Defaults to True.

        Returns:
            torch.Tensor: The super resolved image or batch of images with a shape of
                Cx(Wx4)x(Hx4) or BxCx(Wx4)x(Hx4).
        """
        # Clone the input
        X = X.clone()

        # If X is a CxWxH tensor, add batch dimension
        if len(X.shape) == 3:
            X = X.unsqueeze(0)

        # If X is not a float tensor, convert it
        if X.dtype != torch.float32:
            X = X.float()

        # Create a batch from the input
        batch = self._create_batch(X)

        # Run the model
        Xsr = self.model.compute(
            batch, custom_steps=custom_steps, temperature=temperature
        )

        # Denormalize
        for i in range(Xsr.shape[1]):
            Xsr[:, i] = Xsr[:, i] * self.std[i] + self.mean[i]
        Xsr = (Xsr + 1) / 2.0

        # Apply spectral correction
        if spectral_correction:
            for i in range(Xsr.shape[0]):
                Xsr[i] = self.hq_histogram_matching(Xsr[i], X[i])

        # If the value is negative, set it to 0
        Xsr[Xsr < 0] = 0

        return Xsr.squeeze()

    def hq_histogram_matching(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> torch.Tensor:
        """Lazy implementation of histogram matching

        Args:
            image1 (torch.Tensor): The low-resolution image (C, H, W).
            image2 (torch.Tensor): The super-resolved image (C, H, W).

        Returns:
            torch.Tensor: The super-resolved image with the histogram of
                the target image.
        """

        # Go to numpy
        np_image1 = image1.detach().cpu().numpy()
        np_image2 = image2.detach().cpu().numpy()

        if np_image1.ndim == 3:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
        elif np_image1.ndim == 2:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
        else:
            raise ValueError("The input image must have 2 or 3 dimensions.")

        # Go back to torch
        image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

        return image1_hat

    def load_pretrained(self, weights_file: str):
        """
        Loads the pretrained model from the given path.

        Args:
            path (str): The path to the pretrained model.
            device (str): The device to use.
        """

        # download pretrained model
        hf_model = "https://huggingface.co/isp-uv-es/opensr-model/resolve/main/sr_checkpoint.ckpt"

        # download pretrained model
        if not pathlib.Path(weights_file).exists():
            print("Downloading pretrained weights from: ", hf_model)
            with open(weights_file, "wb") as f:
                f.write(requests.get(hf_model).content)

        weights = torch.load(weights_file, map_location=self.device)["state_dict"]

        # Remote perceptual tensors from weights
        for key in list(weights.keys()):
            if "loss" in key:
                del weights[key]

        self.model.load_state_dict(weights, strict=True)
        print("Loaded pretrained weights from: ", weights_file)
