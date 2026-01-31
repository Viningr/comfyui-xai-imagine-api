import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import json  # for nicer error dumping if needed

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
    RETURN_NAMES = ("image", "status_info",)
    FUNCTION = "generate_image"
    CATEGORY = "xAI/Imagine"
    OUTPUT_NODE = False  # still a generator, not pure output

    def generate_image(self, prompt: str, api_key: str, n: int = 1, aspect_ratio: str = "16:9", response_format: str = "url"):
        if not api_key or not api_key.strip():
            return (None, "Error: xAI API key is required. Get one from console.x.ai")

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

        status_msg = "Unknown status"

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=90)  # slightly longer timeout for safety
            resp.raise_for_status()  # raises on 4xx/5xx

            data = resp.json()

            # Check for error field first (common in OpenAI-style APIs)
            if "error" in data:
                err_msg = data["error"].get("message", "Unknown API error")
                err_type = data["error"].get("type", "unknown")
                return (None, f"API Error ({err_type}): {err_msg}\nFull response: {json.dumps(data, indent=2)}")

            images_data = data.get("data", [])
            if not images_data:
                # Likely a soft refusal or empty result
                revised = images_data[0].get("revised_prompt", "") if images_data else ""
                refusal_hint = " (possible content policy refusal or empty result)" if not revised else ""
                return (None, f"No images generated.{refusal_hint}\nResponse: {json.dumps(data, indent=2)}")

            images = []
            for item in images_data:
                if response_format == "url" and "url" in item:
                    img_url = item["url"]
                    img_resp = requests.get(img_url, timeout=30)
                    img_resp.raise_for_status()
                    pil_img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                elif response_format == "b64_json" and "b64_json" in item:
                    img_bytes = base64.b64decode(item["b64_json"])
                    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                else:
                    raise ValueError(f"Missing expected image data in item: {item}")

                img_np = np.array(pil_img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)[None, ...]  # add batch dim per image for clarity
                images.append(img_tensor)

            if not images:
                return (None, "No valid images after processing.")

            # Stack batch: (B, H, W, C)
            batch_tensor = torch.cat(images, dim=0)

            # Build informative status
            revised_prompts = [item.get("revised_prompt", "No revision") for item in images_data]
            status_msg = f"Success: {len(images)} image(s) generated.\n"
            if any(r != "No revision" for r in revised_prompts):
                status_msg += "Revised prompt(s):\n" + "\n".join(revised_prompts)

            return (batch_tensor, status_msg)

        except requests.exceptions.HTTPError as e:
            try:
                err_data = resp.json()
                err_detail = err_data.get("error", {}).get("message", str(e))
            except:
                err_detail = str(e)
            return (None, f"HTTP Error {resp.status_code}: {err_detail}")

        except requests.exceptions.RequestException as e:
            return (None, f"Network/Request failed: {str(e)}")

        except Exception as e:
            return (None, f"Unexpected error during processing: {str(e)}\nCheck prompt, key, or API status.")

NODE_CLASS_MAPPINGS = {
    "XAI_Imagine_Image": XAIImagineImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XAI_Imagine_Image": "xAI Imagine Image Generation"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
