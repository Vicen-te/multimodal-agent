# Ollama

Ollama runs open-weight language models locally and exposes them through a small
HTTP API, by default on port 11434. You pull a model once with `ollama pull
<name>`, and it is then available for chat and generation without any cloud API
or per-token cost. This makes it a practical backend for demos and for workloads
where data must stay on the machine.

Ollama serves both text and multimodal models. A text model such as qwen2.5
answers chat prompts and, for models trained for it, can emit structured tool
calls that an agent framework executes. A vision-language model such as qwen2.5vl
accepts images alongside the prompt: each message can carry an `images` field
holding base64-encoded pictures, and the model describes or reasons about them in
text.

Because everything runs on local hardware, model size is the main constraint. A
two-to-three billion parameter model runs on CPU, slowly but acceptably for a
demo, while larger models really want a GPU. Choosing a small vision model keeps
the whole system runnable on a free CPU-only host.
