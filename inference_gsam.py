import argparse
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import (
    UNet2DConditionModel,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    load_image,
)
from huggingface_hub import hf_hub_download

from gsampipeline import StableDiffusionInstructPix2PixGSAMPipeline
from utils import get_val_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_model_name_or_path", help="Path to the pretrained model", type=str, default="jay-jnp/F-ViTA_KAIST", )
    parser.add_argument("save_name", help="Name of the save directory", type=str, default="samples-inference", )
    parser.add_argument("dataset_name", help="Name of the dataset/folder inside the datasets directory", type=str, default="samples")
    args = parser.parse_args()
    
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    save_name = args.save_name
    dataset_name = args.dataset_name

    num_inference_steps = 100
    return_text = True
    return_boxes = False
    return_masks = True

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=None,
    )

    ema_unet = EMAModel(
                unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
    )

    #added linear projection
    added_linear = torch.nn.Linear(1024, 768)
    if os.path.isdir(pretrained_model_name_or_path):
        _linear_path = os.path.join(pretrained_model_name_or_path, 'added_linear.pth')
    else:
        _linear_path = hf_hub_download(pretrained_model_name_or_path, 'added_linear.pth')
    added_linear.load_state_dict(torch.load(_linear_path, weights_only=True))
    pipeline = StableDiffusionInstructPix2PixGSAMPipeline(
        unet = unet,
        added_linear = added_linear,
        return_boxes=return_boxes,
        return_masks=return_masks,
        return_text=return_text
    )

    #get list of images for validation
    images, gts, names = get_val_images(dataset_name)
    preds = []
    save_path = os.path.join("predictions", dataset_name, save_name)
    os.makedirs(save_path, exist_ok=True)
    for i,im in enumerate(images):
        if os.path.exists(os.path.join(save_path, names[i])):
            continue
        init_image = load_image(im)
        with torch.autocast(
                            'cuda',
                            enabled=True,
        ):
            pred_im = pipeline("Create an infrared version of the given image. Make it long wave infrared", image=init_image, im_path=im, num_inference_steps=num_inference_steps).images[0]

        pred_im.save(os.path.join(save_path, names[i]))
        # preds.append(os.path.join(save_path, names[i]))