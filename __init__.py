import torch
import numpy as np
from PIL import Image
from io import BytesIO
import urllib.request
import base64
from typing import Optional

from xai_sdk import Client

class XAIImagineImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful futuristic cityscape at sunset"}),
                "model": (["grok-imagine-image"], {"default": "grok-imagine-image"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "image_format": (["url", "base64"], {"default": "url"}),
                "aspect_ratio": (aspect_ratios, {"default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),  # For edit mode
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "xAI/Imagine"

    def generate(self, prompt: str, model: str, n: int, image_format: str, aspect_ratio: str = "", image: Optional[torch.Tensor] = None, api_key: str = ""):
        # Client setup
        if api_key.strip():
            client = Client(api_key=api_key.strip())
        else:
            client = Client()  # Uses XAI_API_KEY env var

        try:
            kwargs = {
                "model": model,
                "prompt": prompt,
                "image_format": image_format,
            }
            if aspect_ratio:
                kwargs["aspect_ratio"] = aspect_ratio

            if image is not None and image.shape[0] > 0:
                # Edit mode (MVP: single image only)
                img_tensor = image[0]  # [H, W, 3] float32 0-1
                pil_img = self.tensor_to_pil(img_tensor)
                buffered = BytesIO()
                pil_img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                kwargs["image_url"] = f"data:image/png;base64,{img_b64}"
                # Force n=1 for edit in MVP
                n = 1
                response = client.image.sample(**kwargs)
                responses = [response]
            else:
                # Generation
                if n == 1:
                    response = client.image.sample(**kwargs)
                    responses = [response]
                else:
                    responses = list(client.image.sample_batch(**kwargs, n=n))

            # Process to ComfyUI IMAGE batch
            tensors = []
            for resp in responses:
                if image_format == "url":
                    url = getattr(resp, 'url', None) or resp.url if hasattr(resp, 'url') else None
                    if not url:
                        raise ValueError("No URL in response")
                    with urllib.request.urlopen(url) as http_resp:
                        img_data = http_resp.read()
                    pil_img = Image.open(BytesIO(img_data)).convert("RGB")
                else:  # base64
                    img_bytes = getattr(resp, 'image', None)
                    if not img_bytes:
                        # Fallback if b64_json string
                        b64_str = getattr(resp, 'b64_json', '') or ''
                        img_bytes = base64.b64decode(b64_str)
                    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

                tensor = self.pil_to_tensor(pil_img)
                tensors.append(tensor)

            if tensors:
                batch_tensor = torch.cat(tensors, dim=0)  # [B, H, W, 3]
            else:
                # Dummy fallback (rare)
                batch_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

            return (batch_tensor,)

        except Exception as e:  # grpc.RpcError, timeouts, refusals, invalid args, etc.
            error_msg = str(e)
            # Optional: log specifics (e.g., if "refusal" or safety in msg)
            raise RuntimeError(f"xAI Imagine API error (check API key, prompt safety, quotas): {error_msg}") from e

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        # [H, W, 3] float32 0-1 -> PIL RGB
        array = (tensor.clamp(0, 1) * 255).to(torch.uint8).numpy()
        return Image.fromarray(array)

    @staticmethod
    def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
        # PIL RGB -> [1, H, W, 3] float32 0-1
        array = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array)
        return tensor.unsqueeze(0)  # Add batch dim for later cat


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "XAIImagineImage": XAIImagineImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XAIImagineImage": "xAI Imagine Image Generation (MVP)"
}
