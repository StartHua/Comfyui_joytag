
import os
import datetime
import numpy as np
import folder_paths
import comfy.model_base
from pathlib import Path
from collections import defaultdict
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
import nodes

from .Models import VisionModel
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")
checkpoints =os.path.join(custom_nodes_path,"Comfyui_joytag","checkpoints")
top_tags_path = os.path.join(checkpoints,"top_tags.txt")

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    
    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    return image_tensor

class CXH_JoyTag:
   
    def __init__(self):
        self.top_tags  = None
        self.model = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {   
                "image":("IMAGE", {"default": "","multiline": False}),    
                "THRESHOLD":("FLOAT", {"default":0.2, "min": 0.1, "max": 1}),  
                "addTag":("STRING", {"default": "","multiline": True}), 
                "removeTag":("STRING", {"default": "","multiline": True}), 
                }
        }

    RETURN_TYPES = ("STRING","INT")
    RETURN_NAMES = ("tags","count")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"
    

    def sample(self,image,THRESHOLD,addTag,removeTag):
        tag_string = ""
        if self.top_tags == None:
            with open(top_tags_path, 'r') as f:
                self.top_tags = [line.strip() for line in f.readlines() if line.strip()]
            
        if self.model == None:
            model = VisionModel.load_model(checkpoints)
            model.eval()
            model = model.to('cuda')

        image = tensor2pil(image)
        image = prepare_image(image, model.image_size)
        batch = {
            'image': image.unsqueeze(0).to('cuda'),
        }
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            preds = model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        
        result = removeTag.split(",")  
  
        scores = {self.top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
        predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD and (tag not in result) ]
        tag_string = ', '.join(predicted_tags)
        # 添加addTag
        tag_string =addTag +"," +tag_string 
        return (tag_string,len(predicted_tags))