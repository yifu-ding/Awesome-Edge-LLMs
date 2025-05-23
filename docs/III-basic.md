# III. Models

<p align="left">
<a href="../README.md#table-of-contents">👈🏻Back to Main Content</a>
</p>

## Contents

<!-- [III. Models](III-basic.md) -->
- [III.A. Typical Model Families](III-basic.md#iiia-typical-examples-of-edge-llms)
  - [Fast Links 🚀](III-basic.md#fast-links)
  - [Sort by Time 📅](III-basic.md#sort-by-time)
    - [2020](III-basic.md#2020)
    - [2021](III-basic.md#2021)
    - [2022](III-basic.md#2022)
    - [2023](III-basic.md#2023)
    - [2024](III-basic.md#2024)
    - [2025](III-basic.md#2025)
- [III.B. Capability on Multimodal Tasks 📷](III-basic.md#iiib-capability-on-multimodal-tasks)

## III.A. Typical Model Families

### Fast Links
| Model | Organization | Links  |
|-|-|-|
| GPT3 | OpenAI | <a href="https://arxiv.org/abs/2005.14165" target="_blank"> <img src="https://img.shields.io/badge/arxiv-20.05-b31b1b" alt="badge"/></a> <a href="https://platform.openai.com/docs/models" target="_blank"> <img src="https://img.shields.io/badge/official-openai-A9EA7A" alt="badge"/></a> |
| PanGu-α | Huawei | <a href="https://arxiv.org/abs/2104.12369" target="_blank"> <img src="https://img.shields.io/badge/arxiv-21.04-b31b1b" alt="badge"/></a> <a href="https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-α" target="_blank"> <img src="https://img.shields.io/badge/git-huawei-6BACF8" alt="badge"/></a> |
| OPT | Meta | <a href="https://arxiv.org/abs/2205.01068" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.05-b31b1b" alt="badge"/></a> <a href="https://github.com/facebookresearch/metaseq" target="_blank"> <img src="https://img.shields.io/badge/git-meta-6BACF8" alt="badge"/></a> |
| BLOOM | BigScience | <a href="https://arxiv.org/abs/2211.05100" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.11-b31b1b" alt="badge"/></a> <a href="https://github.com/bigscience-workshop/bigscience" target="_blank"> <img src="https://img.shields.io/badge/git-bigscience-6BACF8" alt="badge"/></a> |
| LLaMA Series | Meta | <a href="https://arxiv.org/abs/2302.13971" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.02-b31b1b" alt="badge"/></a> <a href="https://github.com/facebookresearch/llama" target="_blank"> <img src="https://img.shields.io/badge/git-meta-6BACF8" alt="badge"/></a> |
| ChatGLM Series | Zhipu AI | <a href="https://arxiv.org/abs/2210.02414" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.10-b31b1b" alt="badge"/></a> <a href="https://github.com/THUDM/ChatGLM-6B" target="_blank"> <img src="https://img.shields.io/badge/git-zhipu-6BACF8" alt="badge"/></a> |
| LLaVA Series | UW-Madison | <a href="https://arxiv.org/abs/2304.08485" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.04-b31b1b" alt="badge"/></a> <a href="https://github.com/haotian-liu/LLaVA" target="_blank"> <img src="https://img.shields.io/badge/git-llava-6BACF8" alt="badge"/></a> |
| MPT | Mosaic ML | <a href="https://www.mosaicml.com/blog/mpt-7b" target="_blank"> <img src="https://img.shields.io/badge/official-mosaic-A9EA7A" alt="badge"/></a> <a href="https://github.com/mosaicml/llm-foundry" target="_blank"> <img src="https://img.shields.io/badge/git-mosaic-6BACF8" alt="badge"/></a> |
| Falcon Series | TII | <a href="https://falconllm.tii.ae/" target="_blank"> <img src="https://img.shields.io/badge/official-tii-A9EA7A" alt="badge"/></a> <a href="https://huggingface.co/tiiuae/falcon-40b" target="_blank"> <img src="https://img.shields.io/badge/hf-falcon-F8D44E" alt="badge"/></a> |
| Phi Series | Microsoft | <a href="https://arxiv.org/abs/2309.05463" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.09-b31b1b" alt="badge"/></a> <a href="https://huggingface.co/microsoft/phi-2" target="_blank"> <img src="https://img.shields.io/badge/hf-phi-F8D44E" alt="badge"/></a> |
| Mistral Series | Mistral | <a href="https://mistral.ai/" target="_blank"> <img src="https://img.shields.io/badge/official-mistral-A9EA7A" alt="badge"/></a> <a href="https://github.com/mistralai/mistral-src" target="_blank"> <img src="https://img.shields.io/badge/git-mistral-6BACF8" alt="badge"/></a> |
| Baichuan Series | Baichuan | <a href="https://www.baichuan-ai.com/" target="_blank"> <img src="https://img.shields.io/badge/official-baichuan-A9EA7A" alt="badge"/></a> <a href="https://github.com/baichuan-inc/Baichuan-7B" target="_blank"> <img src="https://img.shields.io/badge/git-baichuan-6BACF8" alt="badge"/></a> |
| Qwen Series | Alibaba | <a href="https://arxiv.org/abs/2309.16609" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.09-b31b1b" alt="badge"/></a> <a href="https://github.com/QwenLM/Qwen" target="_blank"> <img src="https://img.shields.io/badge/git-qwen-6BACF8" alt="badge"/></a> |
| Yi Series | 01.AI | <a href="https://01.ai/" target="_blank"> <img src="https://img.shields.io/badge/official-01ai-A9EA7A" alt="badge"/></a> <a href="https://github.com/01-ai/Yi" target="_blank"> <img src="https://img.shields.io/badge/git-yi-6BACF8" alt="badge"/></a> |
| AndesGPT | OPPO | <a href="https://www.oppo.com/en/technology/andesgpt/" target="_blank"> <img src="https://img.shields.io/badge/official-oppo-A9EA7A" alt="badge"/></a> |
| BlueLM Series | VIVO | <a href="https://www.vivo.com/" target="_blank"> <img src="https://img.shields.io/badge/official-vivo-A9EA7A" alt="badge"/></a> <a href="https://github.com/vivo-ai-lab/BlueLM" target="_blank"> <img src="https://img.shields.io/badge/git-bluelm-6BACF8" alt="badge"/></a> |
| Gemini Series | Google | <a href="https://deepmind.google/technologies/gemini/" target="_blank"> <img src="https://img.shields.io/badge/official-google-A9EA7A" alt="badge"/></a> <a href="https://blog.google/technology/ai/google-gemini-ai/" target="_blank"> <img src="https://img.shields.io/badge/blog-gemini-A9EA7A" alt="badge"/></a> |
| InternLM Series | Shanghai AI Lab | <a href="https://arxiv.org/abs/2312.00752" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge"/></a> <a href="https://github.com/InternLM/InternLM" target="_blank"> <img src="https://img.shields.io/badge/git-internlm-6BACF8" alt="badge"/></a> |
| MinCPM Series | ModelBest | <a href="https://github.com/OpenBMB/CPM" target="_blank"> <img src="https://img.shields.io/badge/git-cpm-6BACF8" alt="badge"/></a> |
| Octopus Series | NEXA AI | <a href="https://www.nexaai.com/" target="_blank"> <img src="https://img.shields.io/badge/official-nexa-A9EA7A" alt="badge"/></a> |
| MM/OpenELM Series | Apple | <a href="https://arxiv.org/abs/2402.05100" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge"/></a> |
| MiLM Series | Xiaomi | <a href="https://xiaomi.com/" target="_blank"> <img src="https://img.shields.io/badge/official-xiaomi-A9EA7A" alt="badge"/></a> |
| DCLM | Apple | <a href="https://arxiv.org/abs/2401.06692" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.01-b31b1b" alt="badge"/></a> |
| SmolLM | Hugging Face | <a href="https://huggingface.co/smol-ai/SmolLM" target="_blank"> <img src="https://img.shields.io/badge/hf-smol--ai-F8D44E" alt="badge"/></a> |
| Deepseek Series | DeepSeek-AI | <a href="https://deepseek.ai/" target="_blank"> <img src="https://img.shields.io/badge/official-deepseek-A9EA7A" alt="badge"/></a> <a href="https://github.com/deepseek-ai/DeepSeek-LLM" target="_blank"> <img src="https://img.shields.io/badge/git-deepseek-6BACF8" alt="badge"/></a> |

### Sort by Time

#### 2020
Model | Organization | Size | Edge | Release Date
-|-|-|-|-
GPT3 | openAI | 175B | ✗ | 2020.05

#### 2021
Model | Organization | Size | Edge | Release Date
-|-|-|-|-
PanGu-$\alpha$ | huawei | 2.6B、13B、200B |✔️| 2021.04

#### 2022
Model | Organization | Size | Edge | Release Date
-|-|-|-|-
OPT | meta | 125M、350M、1.3B、2.7B、6.7B、13B、30B、66B、175B |✗ | 2022.05
BLOOM | BigScience | 3B、7.1B、176B |✗  | 2022.05

#### 2023
Model | Organization | Size | Edge | Release Date
-|-|-|-|-
LLaMA | meta | 7B、13B、30B、65B |✗ | 2023.02
ChatGLM | Zhipu AI | 6B |✔️| 2023.03
LLaVA1.0 | University of Wisconsin-Madison | 13B |?| 2023.04
MPT | Mosaic ML | 7B |✔️| 2023.05
Falcon | TII | 7B、40B、180B |✔️| 2023.05
Phi 1 | Microsoft | 1.3B |✔️| 2023.06
ChatGLM2 | Zhipu AI | 6B |✔️| 2023.06
LLaMA2 | meta | 7B、13B、70B |✗  | 2023.07
Mistral 7B | Mistral | 7B |✔️| 2023.09
BaiChuan | BaiChuan | 7B、13B | ? | 2023.09
Qwen | Alibaba | 1.8B、7B、14B |✔️| 2023.09
Phi 1.5 | Microsoft | 1.5B |✔️| 2023.09
ChatGLM3 | Zhipu AI | 6B |✔️| 2023.10
LLaVA1.5 | University of Wisconsin-Madison | 6.7B、13B | ? | 2023.10
AndesGPT | OPPO | 7B、160B | OPPO find x7 | 2023.10
Yi | 01.AI | 6B、34B | 3090、4090、A10、A30 | 2023.11
BlueLM | VIVO | 3B、7B、70B | VIVO X100/X200 | 2023.11
Mistral 8x7B | Mistral | 47B | ✗ | 2023.12
Phi 2 | Microsoft | 2.7B |✔️| 2023.12
Gemini | Google | nano(1.8B、3.25B)、pro、ultra | Pixel 8 Pro/9 | 2023.12

#### 2024
Model | Organization | Size | Edge | Release Date
-|-|-|-|-
InternLM2 | Shanghai AI Laboratory | 1.8B、7B、20B |✔️| 2024.01
Yi VL | 01.AI | 6B、34B |✔️| 2024.01
MAP Neo | M-A-P | 7B |✔️| 2024.01
MinCPM | ModelBest | 2B |✔️| 2024.02
Qwen1.5 | Alibaba | 0.5B、1.8B、4B、7B、14B、72B |✔️| 2024.02
Gemma1 | Google | 2B、7B | ? | 2024.03
Octopus v2 | NEXA AI | 0.5B、2B |✔️| 2024.03
MM1 | Apple | 270M、450M、1.1B、3B、30B | iPhone 16 | 2024.03
Octopus v3 | NEXA AI | 1B | Raspberry Pi、Rabbit R1 | 2024.04
OpenELM | apple | 0.27B、0.45B、1.08B、3.04B | MacBook Pro M2 Max | 2024.04
MinCPM V2.0 | ModelBest | 2.8B |✔️| 2024.04
MinCPM V3.0 | ModelBest | 4B |✔️| 2024.04
Phi 3 | Microsoft | 3.8B、7B、14B | iPhone A16 | 2024.04
Ferret-UI | Apple | 1.1B | iPhone | 2024.04
Llama3-Chat |✔️| 3B、7B、8B | Snapdragon 8 Gen3/8 Elite | 2024.04
Yi 1.5 | 01.AI | 6B、9B、20B、34B | 3090、4090、A10、A30 | 2024.05
MiLM | xiaomi | 1.3B、6.4B | xiaomi 14/Pro | 2024.05
MinCPM Llama3 V2.5 | ModelBest | 8B |✔️| 2024.05
Falcon 2 | TII | 11B |✔️| 2024.05
ChatGLM4 | Zhipu AI | 9B、32B |✔️| 2024.05
LLaVA NeXT | University of Wisconsin-Madison | 7B |?| 2024.06
Gemma2 | Google | 9B、27B | TPU | 2024.06
SmolLM | Hugging Face | 134M、360M、1.7B |✔️| 2024.06
DCLM | Apple | 412M、1.4B、2.8B、6.9B |✔️| 2024.06
Qwen2 | Alibaba | 0.5B、1.5B、7B、57B、72B |✔️| 2024.07
LLaMA3 | meta | 8B、70B、405B |✗ | 2024.07
InternLM2.5 | Shanghai AI Laboratory | 7B |✔️| 2024.07
BaiChuan2 | Baichuan | 7B、13B | ? | 2024.09
MiLM2 | xiaomi | 0.3B、0.7B、1.3B、2.4B、4B、6B、13B、30B |✔️| 2024.11
BlueLM-V-3B | VIVO、MMLab | 3B | VIVO X100 | 2024.11
Qwen2.5 | Alibaba | 0.5B、1.5B、3B、7B、14B、32B、72B |✔️| 2024.12
Falcon 3 | TII | 1B、7B、10B |✔️| 2024.12
Deepseek-V3 | DeepSeek-AI | 671B | ✗ | 2024.12

#### 2025
Model | Organization | Size | Edge | Release Date
-|-|-|-|-
Phi 4 | Microsoft | 14B | iPhone 12 Pro | 2025.01
InternLM3 | Shanghai AI Laboratory | 8B |✔️| 2025.01
Gemma3 | Google | 1B、4B、12B、27B |✔️| 2025.03
Qwen3 | Alibaba | 0.6B、1.7B、4B、8B、14B、32B、30B、235B |✔️| 2025.04
LLaMA4 | meta | Scout(17B)、Maverick(17B)、Behemoth(288B) |✗ | 2025.04



## III.B. Capability on Multimodal Tasks

| Model | Capability | Links |
|-|-|-|
| Phi-3.5-Vision | Image |<a href="https://arxiv.org/pdf/2404.14219" target="_blank"><img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge"/></a> <a href="https://github.com/instill-ai/models/blob/main/phi-3-5-vision/README.md" target="_blank"><img src="https://img.shields.io/badge/git-instill_ai-6BACF8" alt="badge"/></a> <a href="https://huggingface.co/microsoft/Phi-3.5-vision-instruct" target="_blank"><img src="https://img.shields.io/badge/hf-phi_3.5_vision-F8D44E" alt="badge" /></a> |
| Gemma-3 | Image |<a href="https://arxiv.org/abs/2503.19786" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.03-b31b1b" alt="badge"/></a>  <a href="https://blog.google/technology/developers/gemma-3/" target="_blank"> <img src="https://img.shields.io/badge/google-developers-A9EA7A" alt="badge"/></a>  |
| Gemini Nano | Image, Audio, Video | <a href="https://arxiv.org/abs/2312.11805" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge"/></a>  <a href="https://deepmind.google/models/gemini/nano/" target="_blank"> <img src="https://img.shields.io/badge/deepmind-gemini-A9EA7A" alt="badge"/></a> |
| Octopus | Image | <a href="https://arxiv.org/abs/2404.01549" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge"/></a> <a href="https://github.com/dongyh20/Octopus" target="_blank"> <img src="https://img.shields.io/badge/git-octopus-6BACF8" alt="badge"/></a> <a href="https://huggingface.co/NexaAIDev/Octopus-v2" target="_blank"> <img src="https://img.shields.io/badge/hf-octopus-F8D44E" alt="badge"/></a> |
| GLM-Edge-V | Image | <a href="https://github.com/THUDM/GLM-Edge" target="_blank"> <img src="https://img.shields.io/badge/git-glm_edge-6BACF8" alt="badge"/></a> <a href="https://huggingface.co/spaces/THUDM-HF-SPACE/GLM-Edge-V-5B-Space" target="_blank"> <img src="https://img.shields.io/badge/hf-glm_edge-F8D44E" alt="badge"/></a> |
| MiniCPM-V | Image, Audio, Video | <a href="https://arxiv.org/abs/2408.01800" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.08-b31b1b" alt="badge"/></a> <a href="https://github.com/OpenBMB/MiniCPM-o" target="_blank"> <img src="https://img.shields.io/badge/git-minicpm-6BACF8" alt="badge"/></a> <a href="https://huggingface.co/openbmb/MiniCPM-V-2" target="_blank"> <img src="https://img.shields.io/badge/hf-minicpm-F8D44E" alt="badge"/></a> |
<!-- | Ferret-v2 | Image | <a href="https://arxiv.org/abs/2404.07973" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge"/></a> | -->