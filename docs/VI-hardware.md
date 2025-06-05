# VI. Hardware

<p align="left">
<a href="../README.md#table-of-contents">👈🏻Back to Main Content</a>
</p>

## Contents

[VI. Hardware](VI-hardware.md)
- [VI.A. ASIC & FPGA](VI-hardware.md#via-asic-&-fpga)
  - [VI.A.1. Quantization](VI-hardware.md#via1-quantization)
  - [VI.A.2. Sparsity](VI-hardware.md#via2-sparsity)
  - [VI.A.3. Operator](VI-hardware.md#via3-operator)
  - [VI.A.4. Architecture](VI-hardware.md#via4-architecture)
- [VI.B. PIM]()

## VI.A. ASIC & FPGA

### VI.A.1. Quantization
- GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference <a href="https://microarch.org/micro53/papers/738300a811.pdf" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2020-b31b1b" alt="badge"/></a>
- FIGNA: Integer Unit-Based Accelerator Design for FP-INT GEMM Preserving Numerical Accuracy <a href="https://ieeexplore.ieee.org/document/10476470" target="_blank"> <img src="https://img.shields.io/badge/HPCA-2024-b31b1b" alt="badge"/></a>
- Edgellm: A highly efficient cpu-fpga heterogeneous edge accelerator for large language models <a href="https://arxiv.org/abs/2407.21325" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.07-b31b1b" alt="badge"/></a>
- Mokey: Enabling narrow fixed-point inference for out-of-the-box floating-point transformer models <a href="https://dl.acm.org/doi/10.1145/3470496.3527438" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2022-b31b1b" alt="badge"/></a>
- Olive: Accelerating large language models via hardware-friendly outlier-victim pair quantization <a href="https://dl.acm.org/doi/abs/10.1145/3579371.3589038" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2023-b31b1b" alt="badge"/></a>
- OPAL: Outlier-Preserved Microscaling Quantization Accelerator for Generative Large Language Models <a href="https://arxiv.org/abs/2409.05902" target="_blank"> <img src="https://img.shields.io/badge/DAC-2024-b31b1b" alt="badge"/></a>
- Spark: Scalable and precision-aware acceleration of neural networks via efficient encoding <a href="https://ieeexplore.ieee.org/document/10476472" target="_blank"> <img src="https://img.shields.io/badge/HPCA-2024-b31b1b" alt="badge"/></a>
- Tender: Accelerating Large Language Models via Tensor Decomposition and Runtime Requantization <a href="https://ieeexplore.ieee.org/document/10609625" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2024-b31b1b" alt="badge"/></a>
- Quantization and hardware architecture co-design for matrix-vector multiplications of large language models <a href="https://ieeexplore.ieee.org/abstract/document/10400181" target="_blank"> <img src="https://img.shields.io/badge/TCSI-2024-b31b1b" alt="badge"/></a>
- A fast and flexible fpga-based accelerator for natural language processing neural networks <a href="https://dl.acm.org/doi/10.1145/3564606" target="_blank"> <img src="https://img.shields.io/badge/TACO-2023-b31b1b" alt="badge"/></a>
- HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis <a href="https://arxiv.org/abs/2405.00738" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.05-b31b1b" alt="badge"/></a>
- Designing efficient LLM accelerators for edge devices <a href="https://arxiv.org/abs/2408.00462" target="_blank"> <img src="https://img.shields.io/badge/arxiv-24.08-b31b1b" alt="badge"/></a>
- Understanding the potential of fpga-based spatial acceleration for large language model inference <a href="https://dl.acm.org/doi/10.1145/3656177" target="_blank"> <img src="https://img.shields.io/badge/TRETS-2024-b31b1b" alt="badge"/></a>
- Flightllm: Efficient large language model inference with a complete mapping flow on fpgas <a href="https://dl.acm.org/doi/10.1145/3626202.3637562" target="_blank"> <img src="https://img.shields.io/badge/FPGA-2024-b31b1b" alt="badge"/></a> <a href="https://github.com/FlightLLM/flightllm_test_demo" target="_blank"> <img src="https://img.shields.io/badge/github-6BACF8" alt="badge"/></a>
- Bridging the Gap Between LLMs and LNS with Dynamic Data Format and Architecture Codesign <a href="https://ieeexplore.ieee.org/document/10764686" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2024-b31b1b" alt="badge"/></a>

### VI.A.2. Sparsity
- A^3: Accelerating attention mechanisms in neural networks with approximation <a href="https://ieeexplore.ieee.org/document/9065498" target="_blank"> <img src="https://img.shields.io/badge/HPCA-2020-b31b1b" alt="badge"/></a>
- ELSA: Hardware-software co-design for efficient, lightweight self-attention mechanism in neural networks <a href="https://ieeexplore.ieee.org/document/9499860" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2021-b31b1b" alt="badge"/></a>
- Sanger: A co-design framework for enabling sparse attention using reconfigurable architecture <a href="https://dl.acm.org/doi/fullHtml/10.1145/3466752.3480125" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2021-b31b1b" alt="badge"/></a> <a href="https://github.com/hatsu3/Sanger" target="_blank"> <img src="https://img.shields.io/badge/github-6BACF8" alt="badge"/></a>
- Spatten: Efficient sparse attention architecture with cascade token and head pruning <a href="https://ieeexplore.ieee.org/abstract/document/9407232" target="_blank"> <img src="https://img.shields.io/badge/HPCA-2021-b31b1b" alt="badge"/></a> <a href="https://github.com/mit-han-lab/spatten" target="_blank"> <img src="https://img.shields.io/badge/git-mit-6BACF8" alt="badge"/></a>
- Fact: Ffn-attention co-optimized transformer architecture with eager correlation prediction <a href="https://dl.acm.org/doi/epdf/10.1145/3579371.3589057" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2023-b31b1b" alt="badge"/></a>
- Hardware-software co-design enabling static and dynamic sparse attention mechanisms <a href="https://ieeexplore.ieee.org/document/10460307" target="_blank"> <img src="https://img.shields.io/badge/TCAD-2024-b31b1b" alt="badge"/></a> <a href="https://github.com/sjtu-zhao-lab/SALO" target="_blank"> <img src="https://img.shields.io/badge/github-6BACF8" alt="badge"/></a>
- ASADI: Accelerating sparse attention using diagonal-based in-situ computing <a href="https://ieeexplore.ieee.org/abstract/document/10476432" target="_blank"> <img src="https://img.shields.io/badge/HPCA-2024-b31b1b" alt="badge"/></a>
- Alisa: Accelerating large language model inference via sparsity-aware kv caching <a href="https://ieeexplore.ieee.org/document/10609626" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2024-b31b1b" alt="badge"/></a>
- SOFA: A compute-memory optimized sparsity accelerator via cross-stage coordinated tiling <a href="https://ieeexplore.ieee.org/document/10764509" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2024-b31b1b" alt="badge"/></a>
- TF-MVP: Novel sparsity-aware transformer accelerator with mixed-length vector pruning <a href="https://ieeexplore.ieee.org/document/10247799" target="_blank"> <img src="https://img.shields.io/badge/DAC-2023-b31b1b" alt="badge"/></a>
- Edge-llm: Enabling efficient large language model adaptation on edge devices via unified compression and adaptive layer voting <a href="https://dl.acm.org/doi/10.1145/3649329.3658473" target="_blank"> <img src="https://img.shields.io/badge/DAC-2024-b31b1b" alt="badge"/></a>
- Feasta: A flexible and efficient accelerator for sparse tensor algebra in machine learning <a href="https://dl.acm.org/doi/10.1145/3620666.3651336" target="_blank"> <img src="https://img.shields.io/badge/ASPLOS-2024-b31b1b" alt="badge"/></a>

### VI.A.3. Operator
- 8-bit Transformer Inference and Fine-tuning for Edge Accelerators <a href="https://dl.acm.org/doi/10.1145/3620666.3651368" target="_blank"> <img src="https://img.shields.io/badge/ASPLOS-2024-b31b1b" alt="badge"/></a>
- Atalanta: A bit is worth a “thousand” tensor values <a href="https://dl.acm.org/doi/10.1145/3620665.3640356" target="_blank"> <img src="https://img.shields.io/badge/ASPLOS-2024-b31b1b" alt="badge"/></a>
- Cambricon-C: Efficient 4-Bit Matrix Unit via Primitivization <a href="https://ieeexplore.ieee.org/document/10764444" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2024-b31b1b" alt="badge"/></a>
- A Latency Processing Unit: A Latency-Optimized and Highly Scalable Processor for Large Language Model Inference <a href="https://ieeexplore.ieee.org/abstract/document/10591630" target="_blank"> <img src="https://img.shields.io/badge/MM-2024-b31b1b" alt="badge"/></a>
- Consmax: Hardware-friendly alternative softmax with learnable parameters <a href="https://dl.acm.org/doi/abs/10.1145/3676536.3676766" target="_blank"> <img src="https://img.shields.io/badge/ICCAD-2024-b31b1b" alt="badge"/></a>

### VI.A.4. Architecture
- MECLA: Memory-Compute-Efficient LLM Accelerator with Scaling Sub-matrix Partition <a href="https://ieeexplore.ieee.org/document/10609710" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2024-b31b1b" alt="badge"/></a>
- BBS: Bi-directional bit-level sparsity for deep learning acceleration <a href="https://ieeexplore.ieee.org/document/10764496" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2024-b31b1b" alt="badge"/></a>
- Trapezoid: A Versatile Accelerator for Dense and Sparse Matrix Multiplications <a href="https://ieeexplore.ieee.org/document/10609623" target="_blank"> <img src="https://img.shields.io/badge/ISCA-2024-b31b1b" alt="badge"/></a>
- Base-2 softmax function: Suitability for training and efficient hardware implementation <a href="https://ieeexplore.ieee.org/document/9851522" target="_blank"> <img src="https://img.shields.io/badge/TCSI-2022-b31b1b" alt="badge"/></a>
- Cambricon-llm: A chiplet-based hybrid architecture for on-device inference of 70b llm <a href="https://ieeexplore.ieee.org/document/10764574" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2024-b31b1b" alt="badge"/></a>
- Dfx: A low-latency multi-fpga appliance for accelerating transformer-based text generation <a href="https://dl.acm.org/doi/10.1109/MICRO56248.2022.00051" target="_blank"> <img src="https://img.shields.io/badge/MICRO-2022-b31b1b" alt="badge"/></a>

## VI.B. PIM
