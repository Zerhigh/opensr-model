import time
import requests
import numpy as np
import torch
import opensr_model
import safetensors.torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import random


# Load the model --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# set the type of model, 4x10m or 6x20m
model_type = "10m"
assert model_type in ["10m","20m"], "model_type must be either 10m or 20m"

if model_type == "10m": # if 10m, create according model and load ckpt
    model = opensr_model.SRLatentDiffusion(bands=model_type,device=device) # 10m
    model.load_pretrained("opensr_10m_v4_v2.ckpt") # 10m

model = model.eval()

# define batch loader
def get_random_image(idx=None):
    import os
    import torch
    input_path = "/data3/final_s2naip_simon/val/LR/none/"
    input_path_HR = "/data3/final_s2naip_simon/val/HR/"
    #read all files in path
    files = os.listdir(input_path)
    random.shuffle(files)
    if idx==None:
        idx = np.random.randint(0, len(files)-1)
    #print(idx,"\tof\t",len(files))
    #return path to file
    file_path_LR = os.path.join(input_path,files[idx])
    file_path_HR = os.path.join(input_path_HR,files[idx])
    # assert files exist
    assert os.path.exists(file_path_LR), "LR doesnt exist"
    assert os.path.exists(file_path_HR), "HR doesnt exist"
    # load images
    lr = torch.load(file_path_LR).unsqueeze(0)
    hr = torch.load(file_path_HR).unsqueeze(0)
    lr = lr/10000
    hr = hr/10000
    return(lr,hr)

def stack_batches(n=1):
    lrs,hrs = [],[]
    for i in range(n):
        lr,hr = get_random_image()
        lrs.append(lr)
        hrs.append(hr)
    lrs = torch.stack(lrs)
    hrs = torch.stack(hrs)
    lrs,hrs = lrs.squeeze(),hrs.squeeze()
    return({"LR_image":lrs,"image":hrs})


no_batches = 100
no_variations = 20

batch = stack_batches(n=no_batches)
batch["image"],batch["LR_image"] = batch["image"].cuda(),batch["LR_image"].cuda()

# get dataloader
#from opensr_model.utils import get_dataloader

# create 20 examples for each SR
variations_total = []

for v,b in tqdm(enumerate(batch["LR_image"])):
    b = b.unsqueeze(0)

    variations = []
    for e in range(no_variations):
        sr = model(b)
        sr = sr.squeeze(0)
        variations.append(sr)
    variations = torch.stack(variations)
    variations_total.append(variations)
    if v==999:
        break # do only 1st image of batch
variations_total = torch.stack(variations_total)
batch["SR_variations"] = variations_total

# denorm LR/HR
#batch["image"] = linear_transform_4b(batch["image"],stage="denorm")
#batch["LR_image"] = linear_transform_4b(batch["LR_image"],stage="denorm")

def convention_stretch_sen2(t):
    # assuming range of t=0..1
    # times 10000 to get to the Sen2 range and then /3000 by convention:
    # https://github.com/google/dynamicworld/blob/master/single_image_runner.ipynb
    t = t * (10 / 4)
    t = t.clamp(0,1)
    return(t)

def minmax_stretch(t):
    t = t - t.min()
    t = t / t.max()
    return(t)

def stre(t):
    return(t*1.5)


for v,(hr,lr,srs) in enumerate(zip(batch["image"],batch["LR_image"],batch["SR_variations"])):

    lr,hr = convention_stretch_sen2(lr),convention_stretch_sen2(hr)
    new_srs = []
    for x in srs:
        im_1 = convention_stretch_sen2(x)
        new_srs.append(im_1)
    srs = torch.stack(new_srs)

    # calculate mean and std of tensor
    srs_mean = srs.mean(dim=0)
    srs_stdev = srs.std(dim=0)

    lower_bound = srs_mean-srs_stdev
    upper_bound = srs_mean+srs_stdev

    error = torch.abs(hr - srs_mean)
    interval_size = srs_stdev*2

    # crop corner
    im_size = 100
    offset=15
    lr = lr[:,offset//4:im_size//4,offset//4:im_size//4]
    hr = hr[:,offset:im_size,offset:im_size]
    srs = srs[:,:,offset:im_size,offset:im_size]
    srs_mean = srs_mean[:,offset:im_size,offset:im_size]
    srs_stdev = srs_stdev[:,offset:im_size,offset:im_size]
    lower_bound = lower_bound[:,offset:im_size,offset:im_size]
    upper_bound = upper_bound[:,offset:im_size,offset:im_size]
    error = error[:,offset:im_size,offset:im_size]
    interval_size = interval_size[:,offset:im_size,offset:im_size]

    # stretch for viz
    lr,hr = stre(lr),stre(hr)
    lower_bound,upper_bound = stre(lower_bound),stre(upper_bound)
    srs[0] = stre(srs[0])

    error = error.mean(dim=0)
    error = minmax_stretch(error).unsqueeze(0)
    interval_size = interval_size.mean(dim=0).unsqueeze(0)
    interval_size = minmax_stretch(interval_size)

    # plot images
    fig, ax = plt.subplots(1, 7, figsize=(30, 5))

    # LR
    ax[0].imshow(rearrange(lr, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[0].set_title("LR Input")
    ax[0].axis('off')

    # SR example
    ax[1].imshow(rearrange(srs[0], 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[1].set_title("SR Example")
    ax[1].axis('off')

    # Lower Bound
    ax[2].imshow(rearrange(lower_bound, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[2].set_title("Lower Bound")
    ax[2].axis('off')

    # Upper Bound
    ax[3].imshow(rearrange(upper_bound, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[3].set_title("Upper Bound")
    ax[3].axis('off')

    # Ground Truth
    ax[4].imshow(rearrange(hr, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[4].set_title("Ground Truth")
    ax[4].axis('off')

    # Error
    ax[5].imshow(rearrange(error, 'c h w -> h w c').cpu().numpy()[:,:,:3],cmap="gray")
    ax[5].set_title("Error")
    ax[5].axis('off')

    # Interval Size
    ax[6].imshow(rearrange(interval_size, 'c h w -> h w c').cpu().numpy()[:,:,:3],cmap="gray")
    ax[6].set_title("Interval Size")
    ax[6].axis('off')
    
    plt.subplots_adjust(wspace=0.025, hspace=0.025)

    plt.savefig(f"example_{v+1}.png",dpi=300)
    plt.close()
    


if False:
    # ------------------------------------------------------------------------------
    # Do CRPS
    import numpy as np
    import properscoring as ps

    def calculate_crps_for_tensors(observation: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Calculates the CRPS score for a tensor of observations and predictions.

        Args:
            observation (np.ndarray): Tensor of observations (C, H, W)
            predictions (np.ndarray): Tensor of predictions (T, C, H, W)

        Returns:
            np.ndarray: Tensor of CRPS scores (C, H, W)
        """

        C, H, W = observation.shape
        crps_scores = np.zeros((C, H, W))
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    obs = observation[c, h, w]
                    fcst = predictions[:, c, h, w]
                    crps_score = ps.crps_ensemble(obs, fcst)
                    crps_scores[c, h, w] = crps_score
        return crps_scores



    for v,(hr,lr,srs) in enumerate(zip(batch["image"],batch["LR_image"],batch["SR_variations"])):
        # LR - Observation (C, H, W)
        observation = torch.clone(lr).cpu()

        # SR - Predictions (T, C, H, W)
        predictions = torch.clone(srs).cpu()

        # Calculate CRPS for the entire tensor
        crps_score = calculate_crps_for_tensors(observation, predictions)

        # Calculate the MAE
        mae_score = np.abs(observation - predictions.mean())

        print("Batch:", v+1)
        print("Average CRPS for the entire tensor:", crps_score.mean())
        print("Average MAE for the entire tensor:", mae_score.mean().item())