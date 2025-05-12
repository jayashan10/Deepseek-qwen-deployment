# DeepSeek-R1-Distill-Qwen-14B Offline Deployment

This repository provides everything you need to deploy the DeepSeek-R1-Distill-Qwen-14B large language model **fully offline** (no internet, no telemetry, no phone-home) on your own GPU hardware.

## What's Included

- `DEPLOY_DEEPSEEK_OFFLINE.md` — **Comprehensive step-by-step guide** for secure, reproducible, and air-gapped deployment, including troubleshooting and best practices.
- `test_deepseek_direct.py` — Example Python script to load and use the model directly (no server needed).
- `serve_deepseek_vllm.sh` — Shell script to serve the model via vLLM, exposing an OpenAI-compatible API endpoint.
- `.gitignore` — Ensures only essential scripts and documentation are tracked (no large model files, logs, or outputs).

## Quick Start

1. **Read the full guide:**
   - See [`DEPLOY_DEEPSEEK_OFFLINE.md`](./DEPLOY_DEEPSEEK_OFFLINE.md) for all steps, requirements, and troubleshooting.

2. **Test the model directly:**
   ```bash
   source /root/miniconda3/etc/profile.d/conda.sh
   conda activate deepseek
   python3 test_deepseek_direct.py
   ```

3. **Serve the model via API:**
   ```bash
   ./serve_deepseek_vllm.sh &> vllm_server.log &
   # Then send requests as shown in the guide
   ```

## Notes
- **Model weights are not included** in this repo. Follow the guide to download and place them securely.
- **No logs or outputs** are tracked in git—generate your own as needed.

## References
- [DEPLOY_DEEPSEEK_OFFLINE.md](./DEPLOY_DEEPSEEK_OFFLINE.md) — Main deployment and usage guide
- [DeepSeek Model on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [vLLM Documentation](https://vllm.readthedocs.io/)

---

**This repo is designed for secure, compliant, and fully offline LLM deployments.** 