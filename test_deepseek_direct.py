from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Set offline environment variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

model_dir = "/opt/llm/deepseek-r1-14b"

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

# Prepare prompt
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Solve 27×36. Show steps then box the answer."}],
    tokenize=False, add_generation_prompt=True
)

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("Generating response...")
output = model.generate(**inputs, max_new_tokens=128, temperature=0.6)

# Decode and print
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nModel response:\n", response) 