from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch


class ReportGenerator:
    def __init__(self, model_name, system_prompt_path, user_prompt_path):
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path

        print(f"\n\nTesting model: {model_name}")
        
        # Determine dtype and attention implementation based on model
        # LLaVA models use float16, others use bfloat16
        if "llava" in model_name.lower():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16
            
        # SmolVLM has issues with flash_attention_2, use eager instead
        if "smolvlm" in model_name.lower():
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"
        
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                device_map="auto",
            )
        except Exception as e:
            # Fallback to eager attention if flash_attention_2 fails
            print(f"Warning: Failed to load with {attn_implementation}, falling back to eager")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="eager",
                device_map="auto",
            )

        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        except:
            self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Using processor: {self.processor.__class__.__name__}")

    def get_chat_template(self):
        with open(self.system_prompt_path, 'r') as f:
            system_prompt = f.read()
        with open(self.user_prompt_path, 'r') as f:
            user_prompt = f.read()
            
        messages = [
            {
              "role": "system",
              "content": f"{system_prompt}"
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        return messages

    def generate_report(self, patient):
        image_inputs = patient['images']
        gt_report = patient.get('report', None)
        
        pil_images = []
        for img in image_inputs:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                pil_images.append(Image.fromarray(img))
        
        with open(self.system_prompt_path, 'r') as f:
            system_prompt = f.read().strip()
        with open(self.user_prompt_path, 'r') as f:
            user_prompt = f.read().strip()

        user_content = []
        for img in pil_images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_prompt})
        
        # Check if model is LLaVA-Next (doesn't support system role)
        if "llava" in self.model_name.lower():
            # LLaVA models don't support system role, combine system and user prompts
            combined_text = f"{system_prompt}\n\n{user_prompt}"
            user_content = []
            for img in pil_images:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": combined_text})
            
            messages = [
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=pil_images,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )
        
        generated_text = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        assistant_markers = ["assistant\n", "ASSISTANT:", "Assistant:", "assistant:", "ASSISTANT\n", "model\n", "[/INST]"]
        
        for marker in assistant_markers:
            if marker in generated_text:
                parts = generated_text.split(marker, 1)
                if len(parts) > 1:
                    generated_text = parts[1].strip()
                    break

        return generated_text
