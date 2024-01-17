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


# Load the model --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# set the type of model, 4x10m or 6x20m
model_type = "10m"
assert model_type in ["10m","20m"], "model_type must be either 10m or 20m"

if model_type == "10m": # if 10m, create according model and load ckpt
    model = opensr_model.SRLatentDiffusion(bands=model_type,device=device) # 10m
    model.load_pretrained("opensr_10m_v4_v2.ckpt") # 10m

model = model.eval()

batch = torch.load("tensor_dict_denorm.pth")
batch["image"],batch["LR_image"] = batch["image"].cuda(),batch["LR_image"].cuda()
batch["image"] = rearrange(batch["image"],"b h w c -> b c h w")
batch["LR_image"] = rearrange(batch["LR_image"],"b h w c -> b c h w")

# create 100 examples for each SR
from opensr_model.utils import linear_transform_4b
variations_total = []
for v,b in enumerate(batch["LR_image"]):
    b = b.unsqueeze(0)

    variations = []
    for e in range(20):
        sr = model(b)
        sr = sr.squeeze(0)
        variations.append(sr)
    variations = torch.stack(variations)
    variations_total.append(variations)
    if v==3:
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
    return(t*3)

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
    lr = lr[:,:im_size//4,:im_size//4]
    hr = hr[:,:im_size,:im_size]
    srs = srs[:,:,:im_size,:im_size]
    srs_mean = srs_mean[:,:im_size,:im_size]
    srs_stdev = srs_stdev[:,:im_size,:im_size]
    lower_bound = lower_bound[:,:im_size,:im_size]
    upper_bound = upper_bound[:,:im_size,:im_size]
    error = error[:,:im_size,:im_size]
    interval_size = interval_size[:,:im_size,:im_size]

    # stretch for viz
    #lr,hr = minmax_stretch(lr),minmax_stretch(hr)
    #error = minmax_stretch(error)
    #interval_size = minmax_stretch(interval_size)
    #lower_bound,upper_bound = minmax_stretch(lower_bound),minmax_stretch(upper_bound)
    error = error.mean(dim=0)
    error = minmax_stretch(error).unsqueeze(0)
    interval_size = interval_size.mean(dim=0).unsqueeze(0)
    interval_size = minmax_stretch(interval_size)

    # plot images
    fig, ax = plt.subplots(1, 7, figsize=(35, 5))

    # LR
    ax[0].imshow(rearrange(lr, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[0].set_title("LR Input")

    # SR example
    ax[1].imshow(rearrange(srs[0], 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[1].set_title("SR Example")

    # Lower Bound
    ax[2].imshow(rearrange(lower_bound, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[2].set_title("Lower Bound")

    # Upper Bound
    ax[3].imshow(rearrange(upper_bound, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[3].set_title("Upper Bound")

    # Ground Truth
    ax[4].imshow(rearrange(hr, 'c h w -> h w c').cpu().numpy()[:,:,:3])
    ax[4].set_title("Ground Truth")

    # Error
    ax[5].imshow(rearrange(error, 'c h w -> h w c').cpu().numpy()[:,:,:3],cmap="gray")
    ax[5].set_title("Error")

    # Interval Size
    ax[6].imshow(rearrange(interval_size, 'c h w -> h w c').cpu().numpy()[:,:,:3],cmap="gray")
    ax[6].set_title("Interval Size")

    plt.savefig(f"example_{v+1}.png",dpi=300)
    plt.close()
    
