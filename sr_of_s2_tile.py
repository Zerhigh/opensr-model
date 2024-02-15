# This script is an example how opensr-utils can be used in unison with opensr-model.

# Import and Instanciate SR Model
import opensr_model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = opensr_model.SRLatentDiffusionLightning(bands="10m",device=device) 
model.load_pretrained("opensr_10m_v4_v2.ckpt")

# perform SR with opensr-utils
from tqdm import tqdm
from opensr_utils.main import windowed_SR_and_saving
path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
sr_obj = windowed_SR_and_saving(folder_path=path, window_size=(128, 128), factor=4, keep_lr_stack=True)
sr_obj.start_super_resolution(band_selection="10m",model=model,forward_call="forward",custom_steps=100,overlap=40, eliminate_border_px=20) # start


