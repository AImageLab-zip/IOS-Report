import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # or a model class that exposes vision parts
from qwen_vl_utils import process_vision_info

# 1. load model + processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",  # or whichever variant you use
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

model.eval()

image_paths = [
    "https://121clicks.com/wp-content/uploads/2023/07/cursed-images-funny-photos-instagram-01.jpg",
    "https://121clicks.com/wp-content/uploads/2023/07/cursed-images-funny-photos-instagram-01.jpg"
]

messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": p} for p in image_paths],
    }
]

image_tensors, video_tensors, video_kwargs = process_vision_info(
    messages,
    image_patch_size=processor.image_processor.patch_size,
    return_video_kwargs=True,
    return_video_metadata=True
)

# 4. Pass through model’s vision encoder (this part depends on model implementation)

