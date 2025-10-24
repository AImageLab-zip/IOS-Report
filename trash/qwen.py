import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def run(image_path: str, question: str):
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    # model_id = "Qwen/Qwen3-VL-8B-Instruct"
    # model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare multimodal chat input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Format for model input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate model output
    generated_ids = model.generate(inputs, max_new_tokens=256, temperature=1)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


if __name__ == "__main__":
    image = "https://i.chzbgr.com/full/9273236480/h80C607AB/banana-wwwpak101com"
    question = "What is in this image?"
    output = run(image, question)
    print("Model response:", output)
