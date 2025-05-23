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
[V.F. Cloud-Edge Collaboration](V-frameworks.md#vf-cloud-edge-collaboration)
  - [V.F.3.1. Separate Model Deployment](V-frameworks.md#vf31-separate-model-deployment)
  - [V.F.3.2. Partitioned Model Deployment](V-frameworks.md#vf32-partitioned-model-deployment)


## V.A. High-Speed Computation Kernels

### V.A.1. Quantization

- Towards Efficient LUT-based PIM: A Scalable and Low-Power Approach for Modern Workloads <a href="https://arxiv.org/abs/2502.02142" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.02-b31b1b" alt="badge"/></a> 
- PowerInfer-2: Fast Large Language Model Inference on a Smartphone <a href="https://arxiv.org/abs/2406.06282" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.06-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/PowerInfer--2-568A37" alt="badge" />
- MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices <a href="https://dl.acm.org/doi/10.1145/3700410.3702126" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.12-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/MNN--LLM-568A37" alt="badge" />
- PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference <a href="https://dl.acm.org/doi/10.1145/3676641.3716267" target="_blank"> <img src="https://img.shields.io/badge/ACM-25.03-b31b1b" alt="badge" /> </a>
- Empowering 1000 Tokens/Second On-Device LLM Prefilling With mllm-NPU <a href="https://arxiv.org/html/2407.05858v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge"/></a>  <img src="https://img.shields.io/badge/mllm--NPU-568A37" alt="badge" /> 
- Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices <a href="https://arxiv.org/abs/2504.08242" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.03-b31b1b" alt="badge" /> </a>
- RTP-LLM: Alibaba's high-performance LLM inference engine for diverse applications <a href="https://github.com/alibaba/rtp-llm" target="_blank"> <img src="https://img.shields.io/badge/git-rtp_llm-6BACF8" alt="badge" /> </a>
- TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs <a href="https://arxiv.org/abs/2504.16266" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.04-b31b1b" alt="badge" /> </a>
- Minicache: Kv cache compression in depth dimension for large language models <a href="https://arxiv.org/abs/2405.14366" target="_blank"> <img src="https://img.shields.io/badge/NeurIPS-2024-b31b1b" alt="badge" /> </a>
- Energon: Toward efficient acceleration of transformers using dynamic sparse attention <a href="https://arxiv.org/abs/2110.09310" target="_blank"> <img src="https://img.shields.io/badge/arxiv-21.10-b31b1b" alt="badge" /> </a>
- Transformer-lite: High-efficiency deployment of large language models on mobile phone gpus <a href="https://arxiv.org/abs/2403.20041" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge"/></a>  <img src="https://img.shields.io/badge/Transformer--Lite-568A37" alt="badge" />


<!-- - ZeroQuant-V2: Exploring Post-Training Quantization in LLMs from Comprehensive Study to Low Rank Compensation <a href="https://arxiv.org/abs/2303.08302" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.03-b31b1b" alt="badge" /> </a>

- LUT-GEMM: Quantized Matrix Multiplication Based on LUTs for Efficient Inference in Large-Scale Generative Language Models <a href="https://arxiv.org/abs/2312.11514" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge" /> </a>

- Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs <a href="https://arxiv.org/abs/2402.10517" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge" /> </a>

- A Speed Odyssey for Deployable Quantization of LLMs <a href="https://arxiv.org/abs/2311.09550" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge"/></a> -->



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

<!-- ### V.B.1. Atomic Operators Fusion -->
- MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices <a href="https://dl.acm.org/doi/10.1145/3700410.3702126" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.12-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/MNN--LLM-568A37" alt="badge" />
- Deepspeed-inference: Enabling efficient inference of transformer models at unprecedented scale <a href="https://arxiv.org/abs/2207.00032" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.07-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/Deepspeed--inference-568A37" alt="badge" />
- Transformer-lite: High-efficiency deployment of large language models on mobile phone gpus <a href="https://arxiv.org/abs/2403.20041" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/Transformer--Lite-568A37" alt="badge" />
- SmartMem: Layout transformation elimination and adaptation for efficient dnn execution on mobile <a href="https://arxiv.org/abs/2404.13528" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge"/></a> <!-- ### V.B.2. Reuse and Sharing -->
- SystemML: Declarative machine learning on MapReduce <a href="https://vikas.sindhwani.org/systemML.pdf" target="_blank"> <img src="https://img.shields.io/badge/VLDB-11.08-b31b1b" alt="badge" /> </a>
- Empowering 1000 Tokens/Second On-Device LLM Prefilling With mllm-NPU <a href="https://arxiv.org/html/2407.05858v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge"/></a>   <img src="https://img.shields.io/badge/mllm--NPU-568A37" alt="badge" /> <!-- V.B.3. Automatic Graph Generation -->
- QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models <a href="https://arxiv.org/abs/2307.03738" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.07-b31b1b" alt="badge" /> </a>


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>


## V.C. Memory Optimization

### V.C.1. Memory Reuse
- Head-wise Shareable Attention for Large Language Models <a href="https://aclanthology.org/2024.findings-emnlp.144/" target="_blank"> <img src="https://img.shields.io/badge/EMNLP-24.01-b31b1b" alt="badge" /> </a>
- Enhancing Scalability of Pre-trained Language Models via Efficient Parameter Sharing <a href="https://aclanthology.org/2023.findings-emnlp.920/" target="_blank"> <img src="https://img.shields.io/badge/EMNLP-23.12-b31b1b" alt="badge" /> </a>
- LightFormer: Light-weight Transformer Using SVD-based Weight Transfer and Parameter Sharing <a href="https://aclanthology.org/2023.findings-acl.656/" target="_blank"> <img src="https://img.shields.io/badge/ACL-23.07-b31b1b" alt="badge" /> </a>
- EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models <a href="https://arxiv.org/abs/2407.21325" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge" /> </a>
- High-throughput Generative Inference of Large Language Models with a Single GPU <a href="https://api.semanticscholar.org/CorpusID:257495837" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.05-b31b1b" alt="badge" /> </a>
- FlashDecoding++: Faster Large Language Model Inference on GPUs <a href="https://arxiv.org/abs/2311.01282" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/FlashDecoding++-568A37" alt="badge" />
- MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases <a href="https://arxiv.org/abs/2402.14905" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge" /> </a>
- MECLA: Memory-Compute-Efficient LLM Accelerator with Scaling Sub-matrix Partition <a href="https://www.computer.org/csdl/proceedings-article/isca/2024/265800b032/1Z3pCEBnapO" target="_blank"> <img src="https://img.shields.io/badge/ISCA-24.06-b31b1b" alt="badge" /> </a>
- Transformer-Lite: High-Efficiency Deployment of Large Language Models on Mobile Phone GPUs <a href="https://arxiv.org/abs/2403.20041" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/Transformer--Lite-568A37" alt="badge" />

### V.C.2. Data Locality and Access Pattern
#### Tensor Reorder
<!-- - GPU Merge Path: A GPU Merging Algorithm <a href="https://doi.org/10.1145/2304576.2304621" target="_blank"> <img src="https://img.shields.io/badge/ACM-12.06-b31b1b" alt="badge"/></a> -->
<!-- - An Overview on Loop Tiling Techniques for Code Generation <a href="https://ieeexplore.ieee.org/document/8308298" target="_blank"> <img src="https://img.shields.io/badge/IEEE-18.03-b31b1b" alt="badge"/></a> -->
- HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs Platform with Heterogeneous AI Accelerators <a href="https://arxiv.org/abs/2501.14794" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.01-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/HeteroLLM-568A37" alt="badge" />
- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time <a href="https://arxiv.org/abs/2310.17157" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.10-b31b1b" alt="badge" /> </a>
- T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge <a href="https://arxiv.org/abs/2407.00088" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge" /> </a>

#### Memory Fragments Elimination
- LLM in a Flash: Efficient Large Language Model Inference with Limited Memory <a href="https://arxiv.org/abs/2312.11514" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge" /> </a>
- SpeedLoader: An I/O Efficient Scheme for Heterogeneous and Distributed LLM Operation <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d3a9e085540c65dd3e5731361f9320e-Abstract-Conference.html" target="_blank"> <img src="https://img.shields.io/badge/NeurIPS-24.12-b31b1b" alt="badge" /> </a>
- Fast Inference of Mixture-of-Experts Language Models with Offloading <a href="https://arxiv.org/abs/2312.17238" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge" /> </a>
- Orca: A Distributed Serving System for Transformer-Based Generative Models <a href="#" target="_blank"> <img src="https://img.shields.io/badge/OSDI-22.07-b31b1b" alt="badge" /> </a>
- Efficient Memory Management for Large Language Model Serving with PagedAttention <a href="https://arxiv.org/abs/2309.06180" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.09-b31b1b" alt="badge" /> </a>
- Ripple: Accelerating LLM Inference on Smartphones with Correlation-Aware Neuron Management <a href="https://arxiv.org/html/2410.19274v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.10-b31b1b" alt="badge" /> </a>

### V.C.3. Storage Hierarchy and Offloading
#### Cost Model and Search Policy
- High-throughput Generative Inference of Large Language Models with a Single GPU <a href="https://api.semanticscholar.org/CorpusID:257495837" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.05-b31b1b" alt="badge" /> </a>
- EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models <a href="https://arxiv.org/abs/2407.21325" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge"/></a> 

#### KV Cache Offloading
- MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices <a href="https://dl.acm.org/doi/10.1145/3700410.3702126" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.12-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/MNN--LLM-568A37" alt="badge" />
- LLM as a System Service on Mobile Devices <a href="https://arxiv.org/abs/2403.11805" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge" /> </a>
- Energon: Toward Efficient Acceleration of Transformers Using Dynamic Sparse Attention <a href="https://arxiv.org/abs/2110.09310" target="_blank"> <img src="https://img.shields.io/badge/arxiv-21.10-b31b1b" alt="badge" /> </a>

#### Layer-Level Offloading
- STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining <a href="https://arxiv.org/abs/2207.05022" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.07-b31b1b" alt="badge" /> </a>
- ZeRO-Inference: Democratizing Massive Model Inference <a href="https://www.deepspeed.ai/2022/09/09/zero-inference.html" target="_blank"> <img src="https://img.shields.io/badge/blog-22.09-b31b1b" alt="badge" /> </a>

#### MoE Offloading
- Fast Inference of Mixture-of-Experts Language Models with Offloading <a href="https://arxiv.org/abs/2312.17238" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge" /> </a>
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference <a href="https://arxiv.org/abs/2308.12066" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.08-b31b1b" alt="badge" /> </a>


<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

## V.D. Pipeline Optimization

### V.D.1. Double Buffering

- SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping <a href="https://dl.acm.org/doi/10.1145/3373376.3378530" target="_blank"> <img src="https://img.shields.io/badge/ACM-20.04-b31b1b" alt="badge" /> </a>
- PowerInfer-2: Fast Large Language Model Inference on a Smartphone <a href="https://arxiv.org/abs/2406.06282" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.06-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/PowerInfer--2-568A37" alt="badge" />
- SpeedLoader: An I/O Efficient Scheme for Heterogeneous and Distributed LLM Operation <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d3a9e085540c65dd3e5731361f9320e-Abstract-Conference.html" target="_blank"> <img src="https://img.shields.io/badge/NeurIPS-24.12-b31b1b" alt="badge" /> </a>
- LLM as a System Service on Mobile Devices <a href="https://arxiv.org/abs/2403.11805" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge" /> </a>



### V.D.2. Multi-core Workload Balancing

- MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices <a href="https://dl.acm.org/doi/10.1145/3700410.3702126" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.12-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/MNN--LLM-568A37" alt="badge" />
- Deepspeed-inference: Enabling efficient inference of transformer models at unprecedented scale <a href="https://arxiv.org/abs/2207.00032" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.07-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/Deepspeed--inference-568A37" alt="badge" />



<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>

## V.E. Multi-device Collaboration

### V.E.1. Heterogeneous Platforms

**Existing Memory Architectures**

- Compute Express Link (CXL): Enabling Heterogeneous Data-Centric Computing with Heterogeneous Memory Hierarchy <a href="https://dl.acm.org/doi/abs/10.1109/MM.2022.3228561" target="_blank"> <img src="https://img.shields.io/badge/IEEE-22.12-b31b1b" alt="badge" /> </a>
- Pathfinding Future PIM Architectures by Demystifying a Commercial PIM Technology <a href="https://arxiv.org/abs/2308.00846" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.08-b31b1b" alt="badge" /> </a>
- An LPDDR-Based CXL-PNM Platform for TCO-Efficient Inference of Transformer-Based Large Language Models <a href="https://ieeexplore.ieee.org/document/10476443" target="_blank"> <img src="https://img.shields.io/badge/Illinois-23.10-b31b1b" alt="badge" /> </a>
- Understanding the Trade-offs in Multi-Level Cell ReRAM Memory Design <a href="https://dl.acm.org/doi/10.1145/2463209.2488867" target="_blank"> <img src="https://img.shields.io/badge/ACM-13.06-b31b1b" alt="badge" /> </a>
- Towards Efficient LUT-Based PIM: A Scalable and Low-Power Approach for Modern Workloads <a href="https://arxiv.org/abs/2502.02142" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge" /> </a>
- PIM-AI: A Novel Architecture for High-Efficiency LLM Inference <a href="https://arxiv.org/abs/2411.17309" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/Samsung's_HBM--PIM-568A37" alt="badge" />
- A Scalable and Energy-Efficient Processing-In-Memory Architecture for Gen-AI <a href="https://ieeexplore.ieee.org/document/10985893" target="_blank"> <img src="https://img.shields.io/badge/IEEE-24.02-b31b1b" alt="badge" /> </a>
- The Landscape of Compute-Near-Memory and Compute-In-Memory: A Research and Commercial Overview <a href="https://arxiv.org/abs/2401.14428" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.01-b31b1b" alt="badge" /> </a>

**Interconnection Bandwidth in Memory**

- Compute Express Link (CXL)-Cache/MEM Protocol Interface (CPI) Specification <a href="https://cdrdv2-public.intel.com/644330/644330_CPI%20Specification_Rev1p0.pdf" target="_blank"> <img src="https://img.shields.io/badge/Intel-23.10-b31b1b" alt="badge" /> </a>
- An Introduction to the Compute Express Link (CXL) Interconnect <a href="https://arxiv.org/abs/2306.11227" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.06-b31b1b" alt="badge" /> </a>
- Samsung Electronics Semiconductor Unveils Cutting-Edge Memory Technology to Accelerate Next-Generation AI <a href="https://semiconductor.samsung.com/news-events/tech-blog/samsung-electronics-semiconductor-unveils-cutting-edge-memory-technology-to-accelerate-next-generation-ai/" target="_blank"> <img src="https://img.shields.io/badge/Samsung-tech_blog-b31b1b" alt="badge" /> </a>
- SK Hynix Presents AI Memory Solutions at CXL DevCon 2024 <a href="https://news.skhynix.com/sk-hynix-presents-ai-memory-solutions-at-cxl-devcon-2024" target="_blank"> <img src="https://img.shields.io/badge/SKHynix-2024-b31b1b" alt="badge" /> </a>
- CXL Memory Expansion: A Closer Look on Actual Platform <a href="https://micron.com/content/dam/micron/global/public/products/white-paper/cxl-memory-expansion-a-close-look-on-actual-platform.pdf" target="_blank"> <img src="https://img.shields.io/badge/Micron-white_paper-b31b1b" alt="badge" /> </a>
- The Breakthrough Memory Solutions for Improved Performance on LLM Inference <a href="https://dl.acm.org/doi/abs/10.1109/MM.2024.3375352" target="_blank"> <img src="https://img.shields.io/badge/IEEE-24.01-b31b1b" alt="badge" /> </a>
- Low-Overhead General-Purpose Near-Data Processing in CXL Memory Expanders <a href="https://arxiv.org/abs/2404.19381" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge" /> </a>
- Sapphire Rapids: The Next-Generation Intel Xeon Scalable Processor <a href="https://ieeexplore.ieee.org/document/9731107" target="_blank"> <img src="https://img.shields.io/badge/IEEE-22.03-b31b1b" alt="badge" /> </a>
- AMD Delivers Breakthrough Memory Performance with DDR5 DRAM and Compute Express Link (CXL) Support <a href="https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/231963000-A_en_AMD-EPYC-9004-Series-Processors-Memory-and-CXL-Advances-White-Paper_pdf.pdf" target="_blank"> <img src="https://img.shields.io/badge/AMD-white_paper-b31b1b" alt="badge" /> </a>
- Exploring Performance and Cost Optimization with ASIC-Based CXL Memory <a href="https://openreview.net/pdf?id=cJOoD0jx6b" target="_blank"> <img src="https://img.shields.io/badge/OpenReview-24.01-b31b1b" alt="badge" /> </a>
- Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM <a href="https://arxiv.org/abs/2502.16963" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge" /> </a>
- Smart-Infinity: Fast Large Language Model Training Using Near-Storage Processing on a Real System <a href="https://arxiv.org/abs/2403.06664" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge" /> </a>
- PIMnet: A Domain-Specific Network for Efficient Collective Communication in Scalable PIM <a href="https://www.computer.org/csdl/proceedings-article/hpca/2025/064700b557/25Ko7PcpTfG" target="_blank"> <img src="https://img.shields.io/badge/HPCA-25.03-b31b1b" alt="badge" /> </a>
- Low-Overhead General-Purpose Near-Data Processing in CXL Memory Expanders <a href="https://arxiv.org/abs/2404.19381" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge" /> </a>
- Exploring and Evaluating Real-World CXL: Use Cases and System Adoption <a href="https://arxiv.org/abs/2405.14209v3" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.05-b31b1b" alt="badge" /> </a>

### V.E.2. Heterogeneous Computing
- End-Edge-Cloud Collaborative Computing for Deep Learning: A Comprehensive Survey <a href="https://ieeexplore.ieee.org/document/10508191" target="_blank"> <img src="https://img.shields.io/badge/Survey-568A37" alt="badge" /> </a>
- PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference <a href="https://arxiv.org/abs/2502.07578" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.02-b31b1b" alt="badge"/></a> 

**Drove by Data Movement Burden**

- Exploiting Intel Advanced Matrix Extensions (AMX) for Large Language Model Inference <a href="https://ieeexplore.ieee.org/document/10538369" target="_blank"> <img src="https://img.shields.io/badge/IEEE-24.02-b31b1b" alt="badge" /> </a>
- Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM <a href="https://arxiv.org/abs/2502.16963" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_PIMs-ECE8EE" alt="badge" />
- AttAcc! Unleashing the power of PIM for batched transformer-based generative model inference <a href="https://dl.acm.org/doi/10.1145/3620665.3640422" target="_blank"> <img src="https://img.shields.io/badge/ACM-23.12-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_PIMs-ECE8EE" alt="badge" />
- Neupims: Npu-pim heterogeneous acceleration for batched llm inferencing <a href="https://arxiv.org/abs/2403.00579" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_PIMs-ECE8EE" alt="badge" />
- PIM-AI: A Novel Architecture for High-Efficiency LLM Inference <a href="https://arxiv.org/abs/2411.17309" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/Samsung's_HBM--PIM-568A37" alt="badge" /> <img src="https://img.shields.io/badge/on_PIMs-ECE8EE" alt="badge" />
- PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System <a href="https://arxiv.org/abs/2502.15470" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.02-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_PIMs-ECE8EE" alt="badge" />
- Monde: Mixture of near-data experts for large-scale sparse models <a href="https://arxiv.org/abs/2405.18832" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.05-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_NDPs-ECE8EE" alt="badge" />
- LM-Offload: Performance Model-Guided Generative Inference of Large Language Models with Parallelism Control <a href="https://pasalabs.org/papers/2024/llm_offload_2024.pdf" target="_blank"> <img src="https://img.shields.io/badge/PASA-24.01-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_CPUs-ECE8EE" alt="badge" />
- Exploiting Intel Advanced Matrix Extensions (AMX) for Large Language Model Inference <a href="https://ieeexplore.ieee.org/document/10538369" target="_blank"> <img src="https://img.shields.io/badge/IEEE-24.02-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_CPUs-ECE8EE" alt="badge" />
- FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines <a href="https://arxiv.org/abs/2403.11421" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.03-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/on_CPUs-ECE8EE" alt="badge" />

**Drove by Arithmetic Intensity**

- Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective <a href="https://arxiv.org/abs/2410.04466" target="_blank"> <img src="https://img.shields.io/badge/Survey-568A37" alt="badge" /> </a>
- PowerInfer-2: Fast Large Language Model Inference on a Smartphone <a href="https://arxiv.org/abs/2406.06282" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.06-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/PowerInfer--2-568A37" alt="badge" />
- HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs Platform with Heterogeneous AI Accelerators <a href="https://arxiv.org/abs/2501.14794" target="_blank"> <img src="https://img.shields.io/badge/arxiv-25.01-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/HeteroLLM-568A37" alt="badge" />
- Empowering 1000 Tokens/Second On-Device LLM Prefilling With mllm-NPU <a href="https://arxiv.org/html/2407.05858v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge"/></a> <img src="https://img.shields.io/badge/mllm--NPU-568A37" alt="badge" />
- Cambricon-llm: A chiplet-based hybrid architecture for on-device inference of 70b llm <a href="https://arxiv.org/abs/2409.15654" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.09-b31b1b" alt="badge" /> </a>

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>


### V.F. Cloud-Edge Collaboration

#### V.F.3.1 Separate Model Deployment
- PerLLM: Personalized Inference Scheduling with Edge-Cloud Collaboration for Diverse LLM Services <a href="https://arxiv.org/abs/2405.14636" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.05-b31b1b" alt="badge" /> </a>
- Collaborative Learning of On-Device Small Model and Cloud-Based Large Model: Advances and Future Directions <a href="https://arxiv.org/abs/2504.15300" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.04-b31b1b" alt="badge" /> </a>
- Cloud-Device Collaborative Learning for Multimodal Large Language Models <a href="https://arxiv.org/abs/2312.16279" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.12-b31b1b" alt="badge" /> </a>
- Edge Intelligence Optimization for Large Language Model Inference with Batching and Quantization <a href="https://arxiv.org/abs/2405.07140v1" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.05-b31b1b" alt="badge" /> </a>

**Draft-Verify**
- Hybrid SLM and LLM for Edge-Cloud Collaborative Inference <a href="https://dl.acm.org/doi/pdf/10.1145/3662006.3662067" target="_blank"> <img src="https://img.shields.io/badge/ACM-23.12-b31b1b" alt="badge" /> </a>
- Edge and Terminal Cooperation Enabled LLM Deployment Optimization in Wireless Network <a href="https://ieeexplore.ieee.org/document/10693742" target="_blank"> <img src="https://img.shields.io/badge/IEEE-24.01-b31b1b" alt="badge" /> </a>

**Easy-Hard**
- Generative AI as a Service in 6G Edge-Cloud: Generation Task Offloading by In-Context Learning <a href="https://arxiv.org/abs/2408.02549" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.08-b31b1b" alt="badge" /> </a>
- Confident Adaptive Language Modeling <a href="https://arxiv.org/abs/2207.07061" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.07-b31b1b" alt="badge" /> </a>
- Progressive Feature Transmission for Split Classification at the Wireless Edge <a href="https://ieeexplore.ieee.org/document/9955582" target="_blank"> <img src="https://img.shields.io/badge/IEEE-22.11-b31b1b" alt="badge" /> </a>
- Tabi: An Efficient Multi-Level Inference System for Large Language Models <a href="https://cse.hkust.edu.hk/~kaichen/papers/tabi-eurosys23.pdf" target="_blank"> <img src="https://img.shields.io/badge/EuroSys-23.05-b31b1b" alt="badge" /> </a>
- CE-CoLLM: Efficient and Adaptive Large Language Models Through Cloud-Edge Collaboration <a href="https://arxiv.org/abs/2411.02829" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge" /> </a>

**Prompt-Supplement**
- Enhancing On-Device LLM Inference with Historical Cloud-Based LLM Interactions <a href="https://dl.acm.org/doi/10.1145/3637528.3671679" target="_blank"> <img src="https://img.shields.io/badge/ACM-24.01-b31b1b" alt="badge" /> </a>
- NetGPT: An AI-Native Network Architecture for Provisioning Beyond Personalized Generative Services <a href="https://arxiv.org/abs/2307.06148" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.07-b31b1b" alt="badge" /> </a>
- PrivateLoRA For Efficient Privacy Preserving LLM <a href="https://arxiv.org/abs/2311.14030" target="_blank"> <img src="https://img.shields.io/badge/arxiv-23.11-b31b1b" alt="badge" /> </a>

**Abstract-Details**
- Large Language Models (LLMs) Inference Offloading and Resource Allocation in Cloud-Edge Networks: An Active Inference Approach <a href="https://dl.acm.org/doi/abs/10.1109/TMC.2024.3415661" target="_blank"> <img src="https://img.shields.io/badge/IEEE_TMC-24.01-b31b1b" alt="badge" /> </a>
- Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations <a href="https://arxiv.org/abs/2202.07800" target="_blank"> <img src="https://img.shields.io/badge/arxiv-22.02-b31b1b" alt="badge" /> </a>
- GKT: A Novel Guidance-Based Knowledge Transfer Framework For Efficient Cloud-edge Collaboration LLM Deployment <a href="https://aclanthology.org/2024.findings-acl.204.pdf" target="_blank"> <img src="https://img.shields.io/badge/ACL-24.05-b31b1b" alt="badge" /> </a>
- PICE: A Semantic-Driven Progressive Inference System for LLM Serving in Cloud-Edge Networks <a href="https://arxiv.org/abs/2501.09367" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.01-b31b1b" alt="badge" /> </a>

#### V.F.3.2 Partitioned Model Deployment
- Poster: PipeLLM: Pipeline LLM Inference on Heterogeneous Devices with Sequence Slicing <a href="https://dl.acm.org/doi/10.1145/3603269.3610856" target="_blank"> <img src="https://img.shields.io/badge/ACM-23.10-b31b1b" alt="badge" /> </a>
- EdgeShard: Efficient LLM Inference via Collaborative Edge Computing <a href="https://arxiv.org/abs/2405.14371" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.05-b31b1b" alt="badge" /> </a>
- Mobile Edge Intelligence for Large Language Models: A Contemporary Survey <a href="https://arxiv.org/abs/2407.18921" target="_blank"> <img src="https://img.shields.io/badge/Survey-568A37" alt="badge" /> </a>



---

<p align="left">
<a href="#contents">👆🏻Back to Contents</a>
</p>
