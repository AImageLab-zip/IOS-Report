from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model  
MODEL_NAME = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)

# Simple test without outlines first
prompt = """<|im_start|>user
Il paziente presenta una contrazione del mascellare superiore in assenza di cross bite,  
mentre dal punto di vista verticale è presente un morso profondo.  
Sagittalmente è una classe II. Le linee mediane sono coincidenti tra loro.   
Curve di spee e di Wilson normali.  
Affollamento presente in arcata superiore ed inferiore con la presenza di spazi   
in entrambe le arcate ed alterazione nel bolton con incisivi superiori laterali piccoli.  
La paziente presenta una buona igiene orale e non sono presenti lesioni cariose.  

Scrivi un report ortodontico strutturato con i seguenti campi:
- bite_type: tipo di morso verticale
- bites: classificazione sagittale  
- median_line: allineamento delle linee mediane
- spee_wilson: curve di Spee e Wilson
- affoll: presenza di affollamento
<|im_end|>
<|im_start|>assistant
"""

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:")
print(response[len(prompt):])
