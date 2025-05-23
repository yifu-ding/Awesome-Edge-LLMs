# Awesome-Edge-LLMs

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/yifu-ding/Awesome-Edge-LLMs/main?logo=github) ![Static Badge](https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat) 

A repository accompanying the survey *Edge AI Meets LLM* (coming soon), containing a comprehensive list of papers, codebases, toolchains, and open-source frameworks. It is intended to serve as a handbook for researchers and developers interested in Edge/Mobile LLMs. 


## 📢 News
**May 23 2025**: Upload a comprehensive collection about frameworks & benchmarks, commercial products & applications, models, and add papers in frameworks section. 



<p align="center">
<img src="images/timeline-emergence-llms.png" width="80%" alt="LLM Emergence Timeline">
<br>
<i>Figure: Timeline showcasing the evolution and emergence of Edge/Mobile LLMs, highlighting key milestones and developments in the field.</i>
</p>


<!-- ## 📊 Taxonomy

<p align="center">
<img src="images/intro-content.png" width="100%" alt="Overall Content">
</p>
*Overview of the technical scope and content structure covered in this repository, encompassing research directions at the intersection of Edge AI and Large Language Models.* -->

<!-- <a href="https://arxiv.org/abs/2310.16836" target="_blank">
  <img src="https://img.shields.io/badge/arxiv-23.10-b31b1b" alt="badge" />
</a> -->

## 🌈 Tags Convention

***🔗 Hyper links***: 
**Normal Paper:** <a href="https://link/to/paper" target="_blank"><img src="https://img.shields.io/badge/Publisher-YY.MM-b31b1b" alt="badge"/></a> 
**Official website:** <a href="https://link/to/official/website" target="_blank"><img src="https://img.shields.io/badge/official-organization-A9EA7A" alt="badge"/></a> 
**Github repo**: <a href="https://link/to/git/repo" target="_blank"><img src="https://img.shields.io/badge/git-user--name-6BACF8" alt="badge"/></a> 
**Huggingface link:** <a href="https://link/to/hf" target="_blank"><img src="https://img.shields.io/badge/hf-user--name-F8D44E" alt="badge"/></a> 

***💡 Highlights***: **Short name:** <img src="https://img.shields.io/badge/Short--Name-7875DF" alt="badge"/> 
**Survey**: <img src="https://img.shields.io/badge/Survey-568A37" alt="badge"/>  


## Contents

**🔨 Deployment Frameworks**

- [I. Open Source Frameworks and Benchmarks](#i-open-source-frameworks-and-benchmarks)
    - [I.A. End-to-End Frameworks](#ia-end-to-end-frameworks)
        - [I.A.1. Open Source Frameworks](#open-source-frameworks)
        - [I.A.2. Native Deployment Frameworks by Vendors](#native-deployment-frameworks-by-vendors)
    - [I.B. Performance Benchmarks](#ib-performance-benchmarks)
        - [I.B.1. General Benchmarks for Edge LLM](#ib1-general-benchmarks-for-edge-llm)
        - [I.B.2. LLM Compression Benchmarks](#ib2-llm-compression-benchmarks)
    - [I.C. Model Export Format](#ic-model-export-format)
<!-- - [More Collections](#more-collections) -->

**📱 Commercial Products and Applications**

- [II. Commercial Cases](docs/II-commercial-products.md)
    - [II.A. Downstream Applications](docs/II-commercial-products.md#iib-applications-and-ai-agents)
        - [Text Generation](docs/II-commercial-products.md#text-generation)
        - [Image Generation](docs/II-commercial-products.md#image-generation) 
        - [Intelligent Assistant](docs/II-commercial-products.md#intelligent-assistant)
    - [II.B. Accelerators and AI Chips](docs/II-commercial-products.md#iia-accelerators-and-ai-chips)
- [III. Models](docs/III-models.md)
    - [III.A. Typical Model Families](docs/III-basic.md#iiia-typical-examples-of-edge-llms)
    - [III.B. Capability on Multimodal Tasks](docs/III-basic.md#iiib-capability-on-multimodal-tasks)


<!-- - [Paper Lists](#paper-lists) -->

**[📑 Paper Lists](#paper-lists)**

- [IV. Algorithms (TBC)](docs/IV-algorithms.md)
- [V. Frameworks](docs/V-frameworks.md)
- [VI. Hardware (TBC)](docs/VI-hardware.md)



## I. Open Source Frameworks and Benchmarks
### I.A. End-to-End Frameworks

#### I.A.1. Open Source Frameworks

| Framework | Backend | Device Support | Model Family | Model Size | Organization |
|-----------|---------|----------------|--------------|------------|--------------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | CUDA, HIP, SYCL<br>OpenCL, MUSA, Vulkan<br>RPC, BLAS, BLIS<br>CANN, Metal | CPU: x86_64, ARM<br>GPU: Intel, Nvidia, MTT, Adreno, AMD<br>NPU: Ascend<br>Apple Silicon | Phi, Gemma, Qwen<br>OpenELM, MiniCPM<br>GLM-edge | 0.5B, 1.5B | ggml |
| [ollama](https://github.com/ollama/ollama)<br>(based on [llama.cpp](https://github.com/ggml-org/llama.cpp))  | CUDA,Metal |  CPU:x86_64, Apple-M  | DeepSeek-R1, Gemma<br>LLaMA, Phi, Mistral<br>LLaVA, QwQ  |  1B, 3B, 3.8B, 4B, 7B | ollama  |
| [vLLM](https://github.com/vllm-project/vllm) | CUDA, HIP<br>SYCL, AWS Neuron | CPU: AMD, Intel, PowerPC<br>GPU: Nvidia, AMD, Intel<br>TPU | Gemma, Qwen, Phi, MiniCPM | 1B, 1.2B | UC Berkeley |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | CUDA, Vulkan<br>OpenCL, Metal | CPU: x86_64, ARM<br>GPU: Nvidia<br>Apple Silicon | LLaMA | 3B | MLC |
| [MNN-LLM](https://github.com/alibaba/MNN) | HIAI, CoreML<br>OpenCL, CUDA<br>Vulkan, Metal | CPU: x86_64, ARM<br>GPU: Nvidia<br>NPU: Ascend, ANE, Apple Silicon | Qwen, Zhipu, Baichuan | 0.5B, 1B, 1.5B, 2B | Alibaba |
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | CUDA, Metal | CPU: x86_64<br>GPU: Nvidia<br>Apple Silicon | Falcon, Bamboo | 7B | Shanghai Jiao Tong University |
| [ExecuTorch](https://pytorch.ac.cn/executorch) | XNNPACK, Vulkan<br>ARM Ethos-U, CoreML<br>MediaTek, MPS<br>CUDA, Qualcomm AI Engine Direct SDK | CPU: ARM<br>GPU: Nvidia<br>NPU: ANE | LLaMA | 1B, 3B | PyTorch |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | CUDA | CPU: x86_64, ARM<br>GPU: Nvidia | Gemma, Falcon, Phi<br>StableLM | 1B, 2B | Google |
| [OpenPPL](https://github.com/OpenPPL/ppl.nn) | CUDA, CANN | CPU: x86_64, ARM<br>GPU: Nvidia<br>NPU: Ascend, Hexagon, Cambricon | ChatGLM, Baichuan, InternLM | 7B | SenseTime |
| [OpenVino](https://docs.openvino.ai) | CUDA | CPU, GPU, NPU, FPGA | Phi, Gemma, Qwen<br>MiniCPM, GLM-edge | 0.5B, 1B | Intel |
| [ONNX Runtime](https://onnxruntime.ai) | CUDA | CPU, GPU, FPGA | Phi, LLaMA | 1B | Microsoft |
| [mllm-NPU](https://github.com/UbiquitousLearning/mllm) | CUDA, QNN | CPU: x86_64, ARM<br>GPU: Nvidia<br>NPU | Phi, Gemma, Qwen<br>MiniCPM, OpenELM | 0.5B, 1B, 1.1B, 1.5B | BUPT, PKU |
| [FastLLM](https://github.com/ServiceNow/Fast-LLM) | CUDA | CPU: x86_64, ARM<br>GPU: Nvidia | Qwen, LLaMA | 1B | ServiceNow |


#### I.A.2. Native Deployment Frameworks by Vendors

| Framework | Organization | Core Features | Links |
|-|-|-|-|
| [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) | Qualcomm | Backend: CPU(Kryo), GPU(Adreno), DSP(Hexagon)<br>Device: Snapdragon 8 Gen2/3/Elite<br>Features: Support 130+ model deployment, auto model conversion, support PyTorch/ONNX | <a href="https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk" target="_blank"> <img src="https://img.shields.io/badge/official-qualcomm-A9EA7A" alt="badge"/></a> |
| [NeuroPilot](https://www.mediatek.com/innovations/artificial-intelligence) | MediaTek | Backend: CPU, GPU, APU<br>Device: Dimensity series<br>Features: Support mainstream AI frameworks, complete toolchain, support 1B-33B parameter models | <a href="https://www.mediatek.com/innovations/artificial-intelligence" target="_blank"> <img src="https://img.shields.io/badge/official-mediatek-A9EA7A" alt="badge"/></a> |
| [MLX](https://github.com/ml-explore/mlx) | Apple | Backend: Metal<br>Device: M series chips<br>Features: Unified memory architecture, support text/image generation, low power consumption | <a href="https://github.com/ml-explore/mlx" target="_blank"> <img src="https://img.shields.io/badge/git-mlx-6BACF8" alt="badge"/></a>  |
| [Google AI Edge SDK](https://ai.google.dev/tutorials/android_edge_sdk_quickstart) | Google | Backend: TPU<br>Device: Tensor G series<br>Features: Fast integration of AI capabilities | <a href="https://ai.google.dev/tutorials/android_edge_sdk_quickstart" target="_blank"> <img src="https://img.shields.io/badge/official-google-A9EA7A" alt="badge"/></a> |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA | Backend: CUDA<br>Device: Jetson series<br>Features: Dynamic batching, paged KV cache, quantization, speculative decoding | <a href="https://github.com/NVIDIA/TensorRT-LLM" target="_blank"> <img src="https://img.shields.io/badge/git-tensorrt_llm-6BACF8" alt="badge"/></a> |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | Intel | Backend: CPU, GPU, VPU<br>Device: Intel processors/graphics<br>Features: Hardware-algorithm co-optimization | <a href="https://github.com/openvinotoolkit/openvino" target="_blank"> <img src="https://img.shields.io/badge/git-openvino-6BACF8" alt="badge"/></a> <a href="https://docs.openvino.ai" target="_blank"> <img src="https://img.shields.io/badge/official-openvino-A9EA7A" alt="badge"/></a> |


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

### I.B. Performance Benchmarks 

#### I.B.1. General Benchmarks for Edge LLM
- Open LLM Leaderboard for Edge Devices <a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?params=0%2C3" target="_blank"> <img src="https://img.shields.io/badge/hf-open_llm-F8D44E" alt="badge" /> </a>

- Open LLM Leaderboard for Consumers <a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?params=3%2C7" target="_blank"> <img src="https://img.shields.io/badge/hf-open_llm-F8D44E" alt="badge" /> </a>

#### I.B.2. LLM Compression Benchmarks
- LLM Compression Benchmark <a href="https://github.com/Picovoice/llm-compression-benchmark" target="_blank"> <img src="https://img.shields.io/badge/git-picovoice-6BACF8" alt="badge" /> </a>
- LLMCBench <a href="https://github.com/AboveParadise/LLMCBench/" target="_blank"> <img src="https://img.shields.io/badge/git-llmcbench-6BACF8" alt="badge"/></a> <a href="https://arxiv.org/abs/2410.21352" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.10-b31b1b" alt="badge" /> </a>

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

### I.C. Model Export Format


| Format | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16/Fp8 |
|--------|------|-----------|--------------|-------------------|----------------|-------------|---------------|
| GGUF (ggml-org) <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md" target="_blank"> <img src="https://img.shields.io/badge/git-ggml_org-6BACF8" alt="badge"/></a>            <a href="https://huggingface.co/docs/hub/en/gguf" target="_blank"> <img src="https://img.shields.io/badge/official-huggingface-D85A45" alt="badge" />  </a>  | ✔️ | ✔️ | ✔️ | ~ | ~ | ✔️ | ✔️ | 
| pickle (PyTorch) <a href="https://docs.python.org/3/library/pickle.html" target="_blank"> <img src="https://img.shields.io/badge/official-python-D85A45" alt="badge"/></a> | ✗ | ✗ | ✗ | ✔️ | ✗ | ✔️ | ✔️ |
| H5 (Tensorflow) <a href="https://www.tensorflow.org/guide/keras/serialization_and_saving" target="_blank"> <img src="https://img.shields.io/badge/official-tensorflow-D85A45" alt="badge"/></a> | ✔️ | ✗ | ✔️ | ✔️ | ~ | ~ | ✗ |
| SavedModel (Tensorflow) <a href="https://www.tensorflow.org/guide/saved_model" target="_blank"> <img src="https://img.shields.io/badge/official-tensorflow-D85A45" alt="badge"/></a> | ✔️ | ✗ | ✗ | ✔️ | ✔️ | ✗ | ✔️ |
| MsgPack (flax) <a href="https://msgpack.org/" target="_blank"> <img src="https://img.shields.io/badge/official-msgpack-D85A45" alt="badge"/></a> | ✔️ | ✔️ | ✗ | ✔️ | ✗ | ✗ | ✔️ |
| Protobuf (ONNX) <a href="https://github.com/onnx/onnx" target="_blank"> <img src="https://img.shields.io/badge/git-onnx-6BACF8" alt="badge"/></a> <a href="https://onnx.ai/onnx/api/serialization.html" target="_blank"> <img src="https://img.shields.io/badge/official-onnx-D85A45" alt="badge"/></a> | ✔️ | ✗ | ✗ | ✗ | ✗ | ✗ | ✔️ |
| Cap'n'Proto <a href="https://github.com/capnproto/capnproto" target="_blank"> <img src="https://img.shields.io/badge/git-capnproto-6BACF8" alt="badge"/></a> <a href="https://capnproto.org/" target="_blank"> <img src="https://img.shields.io/badge/official-capnproto-D85A45" alt="badge"/></a> | ✔️ | ✔️ | ~ | ✔️ | ✔️ | ~ | ✗ |
| llamafile (Mozilla) <a href="https://github.com/Mozilla-Ocho/llamafile" target="_blank"> <img src="https://img.shields.io/badge/git-mozilla-6BACF8" alt="badge"/></a> |  ✔️| ✗ | ✗ | ✗ | ~ | ~ | ✔️ |
| Numpy (npy,npz) <a href="https://numpy.org/" target="_blank"> <img src="https://img.shields.io/badge/official-numpy-D85A45" alt="badge"/></a> <a href="https://github.com/numpy/numpy" target="_blank"> <img src="https://img.shields.io/badge/git-numpy-6BACF8" alt="badge"/></a> | ✔️ | ? | ? | ✗ | ✔️ | ✗ | ✗ |
| pdparams (Paddle) <a href="https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/Model_en.html" target="_blank"> <img src="https://img.shields.io/badge/official-paddle-D85A45" alt="badge"/></a> | ✗ | ✗ | ✗ | ✔️ | ✗ | ✔️ | ✔️ |
| SafeTensors <a href="https://github.com/huggingface/safetensors" target="_blank"> <img src="https://img.shields.io/badge/git-huggingface-6BACF8" alt="badge"/></a> <a href="https://huggingface.co/docs/safetensors/en/index" target="_blank"> <img src="https://img.shields.io/badge/official-huggingface-D85A45" alt="badge"/></a> | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✗ | ✔️ |

> 💡 **Note**: This table is taken from [safetensors git repository](https://github.com/huggingface/safetensors?tab=readme-ov-file#yet-another-format-), and more detailed information can be found there.

<p align="center">
<img src="./images/model-export-format.png" width="80%" alt="model export format">
<br>
<i>Figure: File format illustrations reference <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/safetensors/safetensors-format.svg">safetensors</a> and <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png">GGUF</a>.</i>
</p>


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

<!-- ## Commercial Products and Applications

[II. Commercial Products and Cases](docs/II-commercial-products.md)
  - [II.A. Applications and AI Agents](docs/II-commercial-products.md#iib-applications-and-ai-agents)
    - [Text Generation](docs/II-commercial-products.md#text-generation)
    - [Image Generation](docs/II-commercial-products.md#image-generation) 
    - [Intelligent Assistant](docs/II-commercial-products.md#intelligent-assistant)
  - [II.B. Accelerators and AI Chips](docs/II-commercial-products.md#iia-accelerators-and-ai-chips)

[III. Models](docs/III-models.md)
- [III.A. Typical Model Families](docs/III-basic.md#iiia-typical-examples-of-edge-llms)
- [III.B. Capability on Multimodal Tasks](docs/III-basic.md#iiib-capability-on-multimodal-tasks)

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p> -->

## Paper Lists

[IV. Algorithms (TBC)](docs/IV-algorithms.md)
<!-- 
[IV. Algorithms](docs/IV-algorithms.md)
- [IV.A. Model Compression Techniques](docs/IV-algorithms.md#iva-model-compression-techniques)
  - [IV.A.1. Sparsification](docs/IV-algorithms.md#iva1-sparsification)
  - [IV.A.2. Quantization](docs/IV-algorithms.md#iva2-quantization)
  - [IV.A.3. Low-rank Decomposition](docs/IV-algorithms.md#iva3-low-rank-decomposition)
- [IV.B. Meta-Architecture Design](docs/IV-algorithms.md#ivb-meta-architecture-design)
  - [IV.B.1. RNN-based](docs/IV-algorithms.md#ivb1-rnn-based)
  - [IV.B.2. Mamba](docs/IV-algorithms.md#ivb2-mamba)
  - [IV.B.3. TTT-based](docs/IV-algorithms.md#ivb3-ttt-based) -->

[V. Frameworks](docs/V-frameworks.md)
- [V.A. High-Speed Computation Kernels](docs/V-frameworks.md#va-high-speed-computation-kernels)
  - [V.A.1. Quantization Strategies and Customized Kernels](docs/V-frameworks.md#va1-quantization-strategies-and-customized-kernels)
  - [V.A.2. Sparse Storage and Computation](docs/V-frameworks.md#va2-sparse-storage-and-computation)
- [V.B. Graph Optimization](docs/V-frameworks.md#vb-graph-optimization)
  - [V.B.1. Atomic Operators Fusion](docs/V-frameworks.md#vb1-atomic-operators-fusion)
  - [V.B.2. Reuse and Sharing](docs/V-frameworks.md#vb2-reuse-and-sharing)
  - [V.B.3. Automatic Graph Generation](docs/V-frameworks.md#vb3-automatic-graph-generation)
- [V.C. Memory Optimization](docs/V-frameworks.md#vc-memory-optimization)
  - [V.C.1. Memory Reuse](docs/V-frameworks.md#vc1-memory-reuse)
  - [V.C.2. Data Locality and Access Pattern](docs/V-frameworks.md#vc2-data-locality-and-access-pattern)
  - [V.C.3. Storage Hierarchy and Offloading](docs/V-frameworks.md#vc3-storage-hierarchy-and-offloading)
- [V.D. Pipeline Optimization](docs/V-frameworks.md#vd-pipeline-optimization)
  - [V.D.1. Double Buffering](docs/V-frameworks.md#vd1-double-buffering)
  - [V.D.2. Multi-core Workload Balancing](docs/V-frameworks.md#vd2-multi-core-workload-balancing)
- [V.E. Multi-device Collaboration](docs/V-frameworks.md#ve-multi-device-collaboration)
  - [V.E.1. Heterogeneous Platforms](docs/V-frameworks.md#ve1-heterogeneous-platforms)
  - [V.E.2. Heterogeneous Computing](docs/V-frameworks.md#ve2-heterogeneous-computing)
- [V.F. Cloud-Edge Collaboration](docs/V-frameworks.md#vf-cloud-edge-collaboration)
  - [V.F.1. Separate Model Deployment](docs/V-frameworks.md#vf31-separate-model-deployment)
  - [V.F.2. Partitioned Model Deployment](docs/V-frameworks.md#vf32-partitioned-model-deployment)

[VI. Hardware (TBC)](docs/VI-hardware.md)
<!-- 
[VI. Hardware](docs/VI-hardware.md)
- [VI.A. ASIC & FPGA](docs/VI-hardware.md#via-asic--fpga)
- [VI.B. PIM](docs/VI-hardware.md#vib-pim) -->


## 📄 License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>
