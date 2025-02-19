# Latent Diffusion Super-Resolution - Sentinel 2 (LDSR-S2)
This repository contains the code of the paper [Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion](https://ieeexplore.ieee.org/abstract/document/10887321). In order to embed this model in your workflow, please check out [SuperS2](https://github.com/IPL-UV/supers2) under [Section 4 - Diffusion Model](https://github.com/IPL-UV/supers2?tab=readme-ov-file#4-diffusion-model) , which implements many SR models including this one and provides supplementary code

## Description
This package contains the latent-diffusion model to super-resolute 10 and 20m bands of Sentinel-2. This repository contains the bare model. It can be embedded in the "opensr-utils" package in order to be applied to Sentinel-2 Imagery. 
## Results Preview
Some example Sr scenes can be found as [super-resoluted tiffs](https://drive.google.com/drive/folders/1OBgYS6c8Kpe_JuGzWOQwOK6UYwhm-3Vh?usp=drive_link) on Doogle Drive. Scenes available:
- Buenos Aires, Argentina  
- Blue Mountains, Australia  
- Louisville, USA  
- Kutahya, Türkyie  
- Catalunya, Spain  

# Citation
If you use this model in your work, please cite  
```tex
@ARTICLE{10887321,
  author={Donike, Simon and Aybar, Cesar and Gomez-Chova, Luis and Kalaitzis, Freddie},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Superresolution;Remote sensing;Training;Diffusion models;Measurement;Spatial resolution;Image reconstruction;Uncertainty;Adaptation models;European Space Agency;Super-Resolution;Remote Sensing;Sentinel-2;Deep Learning;Latent Diffusion;Model Uncertainty},
  doi={10.1109/JSTARS.2025.3542220}}
```

# Status
This is a work in progress and published explicitly as a research preview. This repository will leave the experimental stage with the publication of v1.0.0. 

[![PyPI Downloads](https://static.pepy.tech/badge/opensr-model)](https://pepy.tech/projects/opensr-model)
