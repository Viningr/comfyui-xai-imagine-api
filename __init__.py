import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import json

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

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "status",)
    FUNCTION = "generate_image"
    CATEGORY = "xAI/Imagine"
    OUTPUT_NODE = False  # Not an output-only node

    def generate_image(self, prompt: str, api_key: str, n: int = 1, aspect_ratio: str = "16:9", response_format: str = "url"):
        status_msg = "Unknown status"

        if not api_key or not api_key.strip():
            status_msg = "Error: xAI API key is required. Get one from https://console.x.ai"
            return (torch.empty(0), status_msg)

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
            resp = requests.post(api_url, headers=headers, json=payload, timeout=90)  # Slightly longer timeout for safety
            if resp.status_code != 200:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", f"HTTP {resp.status_code}")
                except json.JSONDecodeError:
                    err_msg = resp.text.strip() or f"HTTP {resp.status_code} - No details"
                
                status_msg = f"API Error ({resp.status_code}): {err_msg}"
                return (torch.empty(0), status_msg)

            data = resp.json()
            images = []
            revised_prompts = []  # Optional future use

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
                    raise ValueError("Unexpected response: missing image data")

                # Collect any revised_prompt if present (some APIs return it)
                if "revised_prompt" in item:
                    revised_prompts.append(item["revised_prompt"])

                img_np = np.array(pil_img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)[None, ...]  # Add batch dim per image
                images.append(img_tensor)

            if not images:
                status_msg = "API returned no images"
                return (torch.empty(0), status_msg)

            batch_tensor = torch.cat(images, dim=0)  # Proper batch (n, H, W, 3)

            count = len(images)
            status_msg = f"Generated {count} image(s) successfully"
            if revised_prompts:
                status_msg += f" | Revised prompt(s): {', '.join(revised_prompts)}"

            return (batch_tensor, status_msg)

        except requests.exceptions.RequestException as e:
            status_msg = f"Request failed: {str(e)}"
            return (torch.empty(0), status_msg)
        except Exception as e:
            status_msg = f"Unexpected error: {str(e)}"
            return (torch.empty(0), status_msg)


NODE_CLASS_MAPPINGS = {
    "XAI_Imagine_Image": XAIImagineImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XAI_Imagine_Image": "xAI Imagine Image Generation"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
