# Qwen 3.5 9B on the HPC

Goal: Qwen 3.5 9B runs on a GPU node of the uni HPC and answers like an OpenAI API. Can uses it from his laptop through an SSH tunnel. The game loop stays on the laptop, so the HPC only serves the model.

Things in `<angle brackets>` depend on our HPC account. The HPC documentation has the exact names.

## 1. Environment and model

On the login node (compute nodes often have no internet):

```bash
python -m venv ~/qwen-env
source ~/qwen-env/bin/activate
pip install vllm

hf download Qwen/Qwen3.5-9B --local-dir <models dir>/Qwen3.5-9B
```

Older versions of the Hugging Face tools use `huggingface-cli download` instead of `hf download`. Put the model in a project or scratch folder, since it is about 20 GB.

Create a key so that other users of the cluster cannot use our server:

```bash
openssl rand -hex 16 > ~/.qwen_key
chmod 600 ~/.qwen_key
```

## 2. Job script

Save as `qwen_server.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen-server
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=<longest allowed>
#SBATCH --output=qwen-%j.log

source ~/qwen-env/bin/activate
echo "Node: $(hostname)"

vllm serve <models dir>/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 --port 8000 \
  --api-key "$(cat ~/.qwen_key)" \
  --max-model-len 32768
```

Start it with `sbatch qwen_server.sh`. The node name is at the top of `qwen-<job id>.log`, and the server is ready when the log says it is listening on port 8000.

GPU memory: the model needs roughly 20 GB in its normal precision. On a 16 GB GPU, use a quantized version of Qwen 3.5 9B instead. If the cluster only has old GPUs (for example V100), recent vLLM may not run on them. In that case stop and tell Can, and we switch to Ollama or llama.cpp.

If memory allows, raise `--max-model-len` (65536 or more is useful for us).

## 3. Test

From the login node, with the node name from the log:

```bash
curl http://<node>:8000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.qwen_key)" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-9b", "messages": [{"role": "user", "content": "Say hello"}]}'
```

Then the same request with thinking switched off, by adding this field next to `messages`:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

Both should answer. The second one should answer without a reasoning part.

## 4. Tunnel from Can's laptop

```bash
ssh -N -L 8000:<node>:8000 <user>@<login host>
```

While this runs, the model is at `http://localhost:8000/v1` on the laptop. The server must not be opened to the internet in any other way.

## What to send back

- the final `qwen_server.sh`
- the exact tunnel command, with the real login host and port
- the key from `~/.qwen_key` (send it privately, not in the repo or a group chat)
- GPU type, maximum job time, and how long the model takes to load
- whether switching thinking off works

Target date: 27 November, when the pilot runs start.
