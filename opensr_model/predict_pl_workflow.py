
"""
def predict_pl_workflow(input_file,**kwargs):
    band_selection = kwargs.get('band_selection', "20m")
    weights_file = kwargs.get('weights_file', None)
    overlap = kwargs.get('overlap', 40)
    eliminate_border_px = kwargs.get('eliminate_border_px', 20)
    num_workers = kwargs.get('num_workers', 64)
    batch_size = kwargs.get('batch_size', 24)
    prefetch_factor = kwargs.get('prefetch_factor', 4)
    accelerator = kwargs.get('accelerator', "gpu")
    devices = kwargs.get('devices', -1)
    strategy = kwargs.get('strategy', "ddp")
    custom_steps = kwargs.get('custom_steps', 100)
    
    # -----------------------------------------------------------------------------
    # Create PyTorch Lighnting Workflow for Multi-GPU processing
    import torch
    torch.set_float32_matmul_precision('medium')
    from opensr_utils.main import windowed_SR_and_saving_dataset 
    import opensr_model

    # create DataLoader object from opensr_utils
    from opensr_utils.main import windowed_SR_and_saving_dataset
    from torch.utils.data import Dataset, DataLoader
    ds = windowed_SR_and_saving_dataset(folder_path=input_file, band_selection=band_selection,
                                        overlap=overlap,eliminate_border_px=eliminate_border_px,
                                        keep_lr_stack=False)
    dl = DataLoader(ds,num_workers=num_workers, batch_size=batch_size,prefetch_factor=prefetch_factor)

    # Create custom writer that writes pl_model outputs to placeholder
    from opensr_model import CustomWriter
    writer_callback = CustomWriter(ds)

    # create Trainer - here for initialization of multi-GPU processing
    from pytorch_lightning import Trainer
    trainer = Trainer(accelerator=accelerator, devices=devices,strategy=strategy,callbacks=[writer_callback])

    # initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_model = opensr_model.SRLatentDiffusionLightning(bands=band_selection,device=device,custom_steps=custom_steps)
    pl_model.load_pretrained(weights_file) # 20m

    # run prediction
    trainer.predict(pl_model, dataloaders=dl,return_predictions=False)

    
"""