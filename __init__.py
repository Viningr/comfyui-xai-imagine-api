import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64

WEB_DIRECTORY = "./dist"

class XAIImagineImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful futuristic cityscape at sunset"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"], {"default": "16:9"}),
            },
            "optional": {
                "response_format": (["url", "b64_json"], {"default": "url"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "xAI/Imagine"

    def generate_image(self, prompt: str, api_key: str, n: int = 1, aspect_ratio: str = "16:9", response_format: str = "url"):
        if not api_key or not api_key.strip():
            raise ValueError("xAI API key is required. Obtain from https://x.ai/console or similar.")

        api_url = "https://api.x.ai/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-imagine-image",
            "prompt": prompt,
            "n": n,
            "aspect_ratio": aspect_ratio,
            "response_format": response_format
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"xAI API request failed: {str(e)}") from e

        images = []
        for item in data.get("data", []):
            if response_format == "url" and "url" in item:
                img_url = item["url"]
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                pil_img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            elif response_format == "b64_json" and "b64_json" in item:
                img_bytes = base64.b64decode(item["b64_json"])
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            else:
                raise ValueError(f"Unexpected response format or missing image data: {response_format}")

            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            images.append(img_tensor)

        if not images:
            raise ValueError("No images returned by xAI API.")

        # Stack into batch (B, H, W, 3)
        batch_tensor = torch.stack(images, dim=0)
        return (batch_tensor,)

NODE_CLASS_MAPPINGS = {
    "XAI_Imagine_Image": XAIImagineImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XAI_Imagine_Image": "xAI Imagine Image Generation"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
