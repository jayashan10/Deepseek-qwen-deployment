# Deploying DeepSeek-R1-Distill-Qwen-14B Fully Offline (No-Phone-Home)

This guide details every step, command, and learning required to deploy the DeepSeek-R1-Distill-Qwen-14B model in a fully air-gapped, no-phone-home environment. It covers both direct Python usage and serving via vLLM (OpenAI-compatible API).

---

## 1. **Prepare Model Files (on Internet-Connected Machine)**

### Download the Model
```bash
pip install --upgrade "huggingface_hub>=0.20.0"
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --local-dir deepseek-r1-14b --local-dir-use-symlinks False
```

### (Optional) Archive the Model Directory for Transfer
```bash
tar -czf deepseek-r1-14b.tgz deepseek-r1-14b
```

Transfer the `deepseek-r1-14b` directory (or the tarball) to your air-gapped server.

---

## 2. **Prepare Python Environment (on Air-Gapped Server)**

### Create Conda Environment
```bash
conda create -n deepseek python=3.10 -y
conda activate deepseek
```

### Install Required Packages
```bash
pip install torch transformers accelerate tiktoken bitsandbytes vllm
```

---

## 3. **Place Model Files**

If you transferred a tarball:
```bash
mkdir -p /opt/llm
# If using tarball:
tar -xzf deepseek-r1-14b.tgz -C /opt/llm
# If you transferred the folder directly:
cp -r /path/to/deepseek-r1-14b /opt/llm/
```

---

## 4. **Set Offline Environment Variables**

These ensure no outbound connections or telemetry:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

---

## 5. **Test the Model with Direct Python Script**

Create a file `test_deepseek_direct.py`:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

model_dir = "/opt/llm/deepseek-r1-14b"

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

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Solve 27×36. Show steps then box the answer."}],
    tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("Generating response...")
output = model.generate(**inputs, max_new_tokens=128, temperature=0.6)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nModel response:\n", response)
```

Run the script:
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate deepseek
python3 test_deepseek_direct.py
```

---

## 6. **Serve the Model via vLLM (OpenAI-Compatible API)**

Create a script `serve_deepseek_vllm.sh`:
```bash
#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate deepseek
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
vllm serve /opt/llm/deepseek-r1-14b \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

Make it executable and run:
```bash
chmod +x serve_deepseek_vllm.sh
./serve_deepseek_vllm.sh &> vllm_server.log &
```

Check the log to ensure the server is running:
```bash
tail -40 vllm_server.log
```

---

## 7. **Send Requests to the vLLM API**

> **Note:** Always use the exact model name as returned by the `/v1/models` endpoint. For this deployment, the model name is likely `/opt/llm/deepseek-r1-14b`.
> You can check available models with:
> ```bash
> curl http://localhost:8000/v1/models
> ```

### Using `curl`:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/llm/deepseek-r1-14b",
    "messages": [
      {"role": "user", "content": "What do you know about the deepseek-r1-14b model?"}
    ],
    "max_tokens": 128,
    "temperature": 0.6
  }'
```

### Using Python (`openai` library):
```python
import openai

openai.base_url = "http://localhost:8000/v1"
openai.api_key = "EMPTY"  # vLLM does not require a real key by default

response = openai.chat.completions.create(
    model="/opt/llm/deepseek-r1-14b",
    messages=[{"role": "user", "content": "Solve 27×36. Show steps then box the answer."}],
    max_tokens=128,
    temperature=0.6,
)
print(response.choices[0].message.content)
```

---

## 8. **Learnings and Best Practices**
- Always use the correct `huggingface-cli download` command for model acquisition.
- Tarring the model directory is optional but helps with atomic transfer and integrity.
- Always set offline environment variables to prevent any outbound calls or telemetry.
- Use conda environments for reproducibility and isolation.
- vLLM provides an OpenAI-compatible API for easy integration with existing tools.
- Always check logs (`vllm_server.log`) to confirm successful startup and troubleshoot issues.
- For compliance, make `/opt/llm` read-only and audit network activity as needed.
- **When using the vLLM API, always use the exact model name as returned by the `/v1/models` endpoint.**
  - For example, the model name may be `/opt/llm/deepseek-r1-14b` instead of just `deepseek-r1-14b`.
  - If you get an error like `The model ... does not exist.`, query `/v1/models` to find the correct model name.

---

## 9. **Troubleshooting**
- If the model does not load, check for missing or corrupted files in `/opt/llm/deepseek-r1-14b`.
- If vLLM is not listening on port 8000, check `vllm_server.log` for errors.
- Ensure your GPU has enough VRAM for the chosen precision/quantization.

---

## 10. **References**
- [DeepSeek Model on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)

---

**This guide is tested and ready for secure, reproducible, and fully offline DeepSeek deployments.** 