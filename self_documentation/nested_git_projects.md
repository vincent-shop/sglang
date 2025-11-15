# Nested Git Projects in SGLang

This document describes the git projects nested within the current SGLang repository, what they are, and how SGLang uses them. They do not come bundled, but SGLang pulls and uses code from them, either as dependencies or manual code synced.

## flash-attention

**What it is:** FlashAttention is an efficient implementation of fast and memory-efficient exact attention with IO-awareness, providing FlashAttention-2 and FlashAttention-3 optimizations for NVIDIA GPUs (Ampere, Ada, Hopper). It offers custom attention kernels that dramatically reduce memory usage and increase speed through better memory access patterns.

**How SGLang uses it:** SGLang integrates FlashAttention kernels as one of its attention backend options to provide high-performance attention computation during LLM inference, enabling faster serving with reduced memory footprint.

## flashinfer

**What it is:** FlashInfer is a kernel library for LLM serving that provides high-performance implementations of attention kernels (FlashAttention, SparseAttention, PageAttention), sampling operations, and other LLM-specific operators. It focuses specifically on LLM inference serving scenarios with features like load-balanced scheduling, cascade attention for hierarchical KV-Cache, and customizable attention variants.

**How SGLang uses it:** SGLang uses FlashInfer as a primary attention backend and kernel provider for efficient batch attention operations, paged KV cache management, and optimized sampling during LLM inference serving.

## vllm

**What it is:** vLLM is a fast and easy-to-use library for LLM inference and serving developed originally at UC Berkeley's Sky Computing Lab. It features PagedAttention for efficient KV cache management, continuous batching, and state-of-the-art serving throughput.

**How SGLang uses it:** SGLang references vLLM's codebase for implementation patterns, architecture design, and kernel integrations, building upon and extending many of vLLM's core concepts for its own serving framework.

## TensorRT-LLM

**What it is:** TensorRT-LLM is NVIDIA's library providing an easy-to-use Python API to define and optimize Large Language Models for efficient inference on NVIDIA GPUs. It offers state-of-the-art optimizations including custom attention kernels, inflight batching, paged KV caching, quantization (FP8/FP4/INT4/INT8), and is architected on PyTorch.

**How SGLang uses it:** SGLang references TensorRT-LLM implementations for NVIDIA-specific optimizations, attention backends, and deployment patterns, particularly for understanding how to integrate TensorRT optimizations into the serving stack.

## TensorRT-Model-Optimizer

**What it is:** NVIDIA TensorRT Model Optimizer (ModelOpt) is a library of state-of-the-art model optimization techniques including quantization, distillation, pruning, and sparsity for accelerating model inference. It provides Python APIs to optimize Hugging Face, PyTorch, or ONNX models and export optimized checkpoints deployable on TensorRT-LLM, vLLM, and SGLang.

**How SGLang uses it:** SGLang supports loading and serving models quantized with TensorRT Model Optimizer (particularly FP8, FP4, and INT4 quantized models), enabling users to deploy highly optimized models generated through ModelOpt's quantization workflows.
