# mini-distributed

A personal project to learn how distributed training works by implementing some of the ideas and components used in libraries like Megatron-LM and Picotron.

I'm following parts of the [Picotron tutorial](https://github.com/huggingface/picotron_tutorial) and the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), with the goal of better understanding how models are trained across multiple GPUs, how different parallelism strategies work, and where communication is needed.

This is mainly a learning project, not intended to be a production training framework.

## Current progress

- [x] LLaMA-style transformer model with FlashAttention, RoPE, RMSNorm, and SwiGLU MLP
- [x] Basic training loop with AdamW and gradient accumulation
- [x] Tokenization and microbatch dataloader
- [x] Process group setup for data, tensor, and pipeline parallelism
- [ ] Tensor parallelism (in progress)
  - Column-parallel and row-parallel linear layers
  - Custom autograd functions for all-reduce and all-gather
  - Integrating sharded layers into the transformer
- [ ] Data parallelism
  - Gradient synchronization across replicas
  - Communication overlap
- [ ] Pipeline parallelism
  - Split transformer layers across GPUs
  - GPipe and 1F1B scheduling
- [ ] Combine DP, TP, and PP for 3D parallel training
- [ ] Benchmark communication overhead, memory usage, and training throughput

## Implementation

The code is organized into a few components:

| File | Description |
|---|---|
| `model.py` | LLaMA-style transformer |
| `train.py` | Training loop, gradient accumulation, and logging |
| `dataloader.py` | Tokenization and microbatch loading |
| `process_group_manager.py` | Process groups for DP, TP, and PP |
| `tensor_parallel.py` | Tensor-parallel layers and communication (WIP) |
| `utils.py` | Miscellaneous utilities |

## References

- [Picotron](https://github.com/huggingface/picotron_tutorial)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)