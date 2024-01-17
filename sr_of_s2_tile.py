# general imports
import torch

# import our opensr_utils package for S2 handling
from opensr_utils.main import windowed_SR_and_saving 
# import our opensr_model package for SR
import opensr_model
# -----------------------------------------------------------------------------


# Define Device  --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:",device)
# create SR object ------------------------------------------------------------
file_path = "/data2/simon/test_s2/S2A_MSIL2A_20230729T100031_N0509_R122_T33TUG_20230729T134559.SAFE/"
sr_obj = windowed_SR_and_saving(file_path)

# SR 20m bands ----------------------------------------------------------------
model = opensr_model.SRLatentDiffusion(bands="20m",device=device) # 20m
model.load_pretrained("opensr_20m_v1.ckpt") # 20m
model = model.eval()
sr_obj.start_super_resolution(band_selection="20m",model=model,forward_call="forward",custom_steps=100) # start

# SR 10m bands ----------------------------------------------------------------
model = opensr_model.SRLatentDiffusion(bands="10m",device=device) # 10m
model.load_pretrained("opensr_10m_v4_v2.ckpt") # 10m
model = model.eval()
sr_obj.start_super_resolution(band_selection="10m",model=model,forward_call="forward",custom_steps=100) # start


