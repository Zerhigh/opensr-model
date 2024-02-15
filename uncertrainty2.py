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
model = model.cuda()




import torch
from tqdm import tqdm
import numpy as np
import properscoring as ps
import os
import matplotlib.pyplot as plt

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
    return(t*1.9)

def plot_images(lr,hr,srs,crps):
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
    crps = crps[:,offset:im_size,offset:im_size]
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
    crps = minmax_stretch(crps)

    # plot images
    fig, ax = plt.subplots(1, 8, figsize=(35, 5))

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

    # CRPS
    ax[7].imshow(rearrange(crps, 'c h w -> h w c')[:,:,:3],cmap="gray")
    ax[7].set_title("CRPS")
    ax[7].axis('off')
    
    plt.subplots_adjust(wspace=0.025, hspace=0.025)

    # get current second in unix time
    import time
    now = str(int(time.time()))
    plt.savefig("images/"+now+".png",dpi=300)
    plt.close()


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



model = model.cuda()


from opensr_model.utils import linear_transform_4b
# list all pth files in directory
import os
import torch
import glob
from tqdm import tqdm
crps_ls = []
directory_path = "/data2/simon/xai_data/"
files = [file for file in os.listdir(directory_path) if file.endswith('.pt')]
for f in tqdm(files):
    try:
        if os.path.exists(directory_path+f):
            batch = torch.load(directory_path+f)
    except:
        continue

    variations = []
    for lr,hr in zip(batch["LR_image"],batch["image"]):
        lr = rearrange(lr,"h w c -> c h w")
        hr = rearrange(hr,"h w c -> c h w")

        variations_image = []
        for x in range(2):
            sr = model(lr.unsqueeze(0).cuda()).squeeze(0)
            variations_image.append(sr.cpu())
        variations_image = torch.stack(variations_image)
        variations.append(variations_image)
    variations = torch.stack(variations)

    batch["variations"] = variations

    # now for each batch calculate stuff and create images
    #batch["LR_image"] = linear_transform_4b(batch["LR_image"],stage="denorm")
    #batch["image"] = linear_transform_4b(batch["image"],stage="denorm")

    for lr,hr,vars in zip(batch["LR_image"],batch["image"],batch["variations"]):
        lr = rearrange(lr,"h w c -> c h w")
        hr = rearrange(hr,"h w c -> c h w")
        crps = calculate_crps_for_tensors(hr.cpu().numpy(),vars.cpu().numpy())
        crps_ls.append(crps.mean())

        # do image
        plot_images(lr[:3,:,:],hr[:3,:,:],vars[:,:3,:,:],crps[:3,:,:])

    # save list to have something to work with
    torch.save(torch.Tensor(crps_ls),"crps_ls.pt")


# create  Histogram
ls = list(torch.load("crps_ls.pt"))
def h(values,b="auto"):
    # Calculate the mean
    mean_value = np.mean(values)

    # Create histogram
    #bins = np.arange(min(values), max(values) + 1.5) - 0.5  # Adjust bin edges if necessary
    plt.hist(values, bins=b, alpha=0.7, color='blue', edgecolor='black')

    # Add a line for the mean
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_value, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color = 'red')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of CRPS')

    # Show plot
    plt.show()
