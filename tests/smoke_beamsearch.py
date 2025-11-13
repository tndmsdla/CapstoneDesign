from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "gpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

prompt = "The quick brown fox"
inputs = tok(prompt, return_tensors="pt").to(device)

out = model.generate(
    **inputs,
    max_new_tokens=20,
    num_beams=5,
    num_return_sequences=5,
    early_stopping=True,
    do_sample=False
)

print("=== N-best (beam=5) ===")
for i, seq in enumerate(out, 1):
    print(f"[{i}] ", tok.decode(seq, skip_special_tokens=True))
