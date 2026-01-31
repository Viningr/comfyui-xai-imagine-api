import os
from xai_sdk import Client
from PIL import Image
import requests
import torch
import numpy as np
# ... other imports

class XAI_Image_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["grok-imagine-image"], {"default": "grok-imagine-image"}),
                # Add: n (INT, 1-10), aspect_ratio (COMBO list from docs), image_format ("url", "base64"), etc.
            },
            "optional": {
                # image edit params, etc.
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "xAI/Imagine"

    def generate(self, api_key, prompt, model="grok-imagine-image", **kwargs):
        if not api_key: api_key = os.getenv("XAI_API_KEY")
        client = Client(api_key=api_key)
        response = client.image.sample(model=model, prompt=prompt, **kwargs)  # or sample_batch
        # Download/convert URL or base64 to tensor (standard Comfy pattern: requests.get → PIL → np → torch)
        img = Image.open(requests.get(response.url, stream=True).raw).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
        return (img_tensor,)

class XAI_Video_Generate:  # Similar; use client.video.generate (auto-polls)
    # INPUT_TYPES: prompt, model="grok-imagine-video", duration (INT 1-15), aspect_ratio (COMBO), resolution (COMBO), image_url (STRING optional), video_url/object for edit
    RETURN_TYPES = ("STRING",)  # URL
    # In execute: response = client.video.generate(...); return (response.url, )  # or handle start + manual poll with timeout

NODE_CLASS_MAPPINGS = {
    "XAI_Image_Generate": XAI_Image_Generate,
    "XAI_Video_Generate": XAI_Video_Generate,
    # Add more (edit modes)
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k.replace("_", " ") for k in NODE_CLASS_MAPPINGS}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
