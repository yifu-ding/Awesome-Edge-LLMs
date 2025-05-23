# V. Frameworks

<p align="left">
<a href="../README.md#table-of-contents">👈🏻Back to Main Content</a>
</p>

## Contents

[V. Frameworks](V-frameworks.md)
- [V.A. High-Speed Computation Kernels](V-frameworks.md#va-high-speed-computation-kernels)
  - [V.A.1. Quantization Strategies and Customized Kernels](V-frameworks.md#va1-quantization-strategies-and-customized-kernels)
  - [V.A.2. Sparse Storage and Computation](V-frameworks.md#va2-sparse-storage-and-computation)
- [V.B. Graph Optimization](V-frameworks.md#vb-graph-optimization)
  - [V.B.1. Atomic Operators Fusion](V-frameworks.md#vb1-atomic-operators-fusion)
  - [V.B.2. Reuse and Sharing](V-frameworks.md#vb2-reuse-and-sharing)
  - [V.B.3. Automatic Graph Generation](V-frameworks.md#vb3-automatic-graph-generation)
- [V.C. Memory Optimization](V-frameworks.md#vc-memory-optimization)
  - [V.C.1. Memory Reuse](V-frameworks.md#vc1-memory-reuse)
  - [V.C.2. Data Locality and Access Pattern](V-frameworks.md#vc2-data-locality-and-access-pattern)
  - [V.C.3. Storage Hierarchy and Offloading](V-frameworks.md#vc3-storage-hierarchy-and-offloading)
- [V.D. Pipeline Optimization](V-frameworks.md#vd-pipeline-optimization)
  - [V.D.1. Double Buffering](V-frameworks.md#vd1-double-buffering)
  - [V.D.2. Multi-core Workload Balancing](V-frameworks.md#vd2-multi-core-workload-balancing)
- [V.E. Multi-device Collaboration](V-frameworks.md#ve-multi-device-collaboration)
  - [V.E.1. Heterogeneous Platforms](V-frameworks.md#ve1-heterogeneous-platforms)
  - [V.E.2. Heterogeneous Computing](V-frameworks.md#ve2-heterogeneous-computing)
  - [V.E.3. Cloud-Edge Collaboration](V-frameworks.md#ve3-cloud-edge-collaboration)


## V.A. High-Speed Computation Kernels

### V.A.1. Quantization

- Towards Efficient LUT-based PIM: A Scalable and Low-Power Approach for Modern Workloads <a href="https://arxiv.org/abs/2502.02142" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.02-b31b1b" alt="badge" /> </a> 
- PowerInfer-2: Fast Large Language Model Inference on a Smartphone <a href="https://arxiv.org/abs/2406.06282" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.06-b31b1b" alt="badge" /> </a>
- MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices <a href="https://dl.acm.org/doi/10.1145/3700410.3702126" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.12-b31b1b" alt="badge" /> </a>
- PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference <a href="https://dl.acm.org/doi/10.1145/3676641.3716267" target="_blank"> <img src="https://img.shields.io/badge/ACM-25.03-b31b1b" alt="badge" /> </a>
- Empowering 1000 tokens/second on-device llm prefilling with mllm-npu <a href="https://arxiv.org/html/2407.05858v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge" /> </a>
- Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices <a href="https://arxiv.org/abs/2504.08242" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.03-b31b1b" alt="badge" /> </a>
- RTP-LLM: Alibaba's high-performance LLM inference engine for diverse applications <a href="https://github.com/alibaba/rtp-llm" target="_blank"> <img src="https://img.shields.io/badge/git-rtp_llm-6BACF8" alt="badge" /> </a>
- TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs <a href="https://arxiv.org/abs/2504.16266" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.04-b31b1b" alt="badge" /> </a>
- Minicache: Kv cache compression in depth dimension for large language models <a href="https://arxiv.org/abs/2405.14366" target="_blank"> <img src="https://img.shields.io/badge/NeurIPS-2024-b31b1b" alt="badge" /> </a>
- Energon: Toward efficient acceleration of transformers using dynamic sparse attention <a href="https://arxiv.org/abs/2110.09310" target="_blank"> <img src="https://img.shields.io/badge/arxiv-21.10-b31b1b" alt="badge" /> </a>
- Transformer-lite: High-efficiency deployment of large language models on mobile phone gpus <a href="https://arxiv.org/abs/2403.20041" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge" /> </a>

<!-- - ZeroQuant-V2: Exploring Post-Training Quantization in LLMs from Comprehensive Study to Low Rank Compensation <a href="https://arxiv.org/abs/2303.08302" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.03-b31b1b" alt="badge" /> </a>

- LUT-GEMM: Quantized Matrix Multiplication Based on LUTs for Efficient Inference in Large-Scale Generative Language Models <a href="https://arxiv.org/abs/2312.11514" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge" /> </a>

- Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs <a href="https://arxiv.org/abs/2402.10517" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge" /> </a>

- A Speed Odyssey for Deployable Quantization of LLMs <a href="https://arxiv.org/abs/2311.09550" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge" /> </a> -->



### V.A.2. Sparsification

- Spatten: Efficient sparse attention architecture with cascade token and head pruning <a href="https://arxiv.org/abs/2012.09852" target="_blank"> <img src="https://img.shields.io/badge/arxiv-20.12-b31b1b" alt="badge" /> </a>
- Hottiles: Accelerating spmm with heterogeneous accelerator architectures <a href="https://iacoma.cs.uiuc.edu/iacoma-papers/hpca24_2.pdf" target="_blank"> <img src="https://img.shields.io/badge/HPCA-24.02-b31b1b" alt="badge" /> </a>
- Learning sparse matrix row permutations for efficient spmm on gpu architectures <a href="https://www.seas.upenn.edu/~leebcc/documents/mehrabi21-ispass.pdf" target="_blank"> <img src="https://img.shields.io/badge/ISPASS-21.03-b31b1b" alt="badge" /> </a>
- Toward Efficient Permutation for Hierarchical N:M Sparsity on GPUs <a href="https://arxiv.org/html/2407.20496v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge" /> </a>
- HC-SpMM: Accelerating Sparse Matrix-Matrix Multiplication for Graphs with Hybrid GPU Cores <a href="https://arxiv.org/abs/2412.08902" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.12-b31b1b" alt="badge" /> </a>
- Endor: Hardware-Friendly Sparse Format for Offloaded LLM Inference <a href="https://arxiv.org/abs/2406.11674" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.06-b31b1b" alt="badge" /> </a>
- SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs <a href="https://www.cse.ust.hk/~weiwa/papers/eurosys25-fall-spinfer.pdf" target="_blank"> <img src="https://img.shields.io/badge/EuroSys-25.03-b31b1b" alt="badge" /> </a>
- Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity <a href="https://arxiv.org/abs/2309.10285" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.09-b31b1b" alt="badge" /> </a>


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>


## V.B. Graph Optimization

### V.B.1. Atomic Operators Fusion

- MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices <a href="https://dl.acm.org/doi/10.1145/3700410.3702126" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.12-b31b1b" alt="badge" /> </a>
- Deepspeed-inference: Enabling efficient inference of transformer models at unprecedented scale <a href="https://arxiv.org/abs/2207.00032" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.07-b31b1b" alt="badge" /> </a>
- Transformer-lite: High-efficiency deployment of large language models on mobile phone gpus <a href="https://arxiv.org/abs/2403.20041" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge" /> </a>
- SmartMem: Layout transformation elimination and adaptation for efficient dnn execution on mobile <a href="https://arxiv.org/abs/2404.13528" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge" /> </a>
- SystemML: Declarative machine learning on MapReduce <a href="https://vikas.sindhwani.org/systemML.pdf" target="_blank"> <img src="https://img.shields.io/badge/VLDB-11.08-b31b1b" alt="badge" /> </a>

### V.B.2. Reuse and Sharing

- Empowering 1000 tokens/second on-device llm prefilling with mllm-npu <a href="https://arxiv.org/html/2407.05858v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge" /> </a>


### V.B.3. Automatic Graph Generation


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>


## V.C. Memory Optimization

### V.C.1. Memory Reuse

### V.C.2. Data Locality and Access Pattern

### V.C.3. Storage Hierarchy and Offloading

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

## V.D. Pipeline Optimization

### V.D.1. Double Buffering

### V.D.2. Multi-core Workload Balancing

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

## V.E. Multi-device Collaboration

### V.E.1. Heterogeneous Platforms

### V.E.2. Heterogeneous Computing

### V.E.3. Cloud-Edge Collaboration

---

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>
