# moe-transformer-from-scratch

A GPT-style transformer with a hand-built Mixture-of-Experts layer: gating network, top-k token routing, capacity-limited dispatch, and load-balancing losses, implemented in raw PyTorch and benchmarked against a parameter-matched dense baseline.

## Highlights

- No `nn.Transformer`, no HuggingFace model classes. Attention, feed-forward, and the MoE layer are all written from primitives.
- Routing is inspectable: which expert handled which token, per layer, is logged and plotted, not just assumed to work.
- Capacity-limited dispatch drops tokens under load, the same tradeoff Switch Transformer and GShard make in production, rather than treating compute as unbounded.
- Dense and MoE models are parameter-matched so the comparison between them means something.
- Runs on a CPU or a free-tier GPU (Kaggle, Colab). No cluster required.

## Why this exists

Mixture-of-Experts is the architecture behind most large-scale language models in production today: a router sends each token to a small subset of expert feed-forward layers instead of one large shared one, so total parameters can grow without a matching increase in compute per token. The interesting problems show up in what happens when the router misbehaves. It can collapse onto one or two experts and ignore the rest, a single popular expert can be handed more tokens than a batch was sized for, and the routing decision itself is discrete, so gradients need a path around it rather than through it. This project implements the standard production fixes for each of those (a load-balancing auxiliary loss, capacity-based token dropping, and a router z-loss for stability) and includes the tooling to check whether they worked, rather than taking that on faith.

## Architecture

```
Input tokens
     |
     v
Token Embedding          (position info: RoPE, applied inside attention below)
     |
     v
Transformer Block x N
  |
  |-- RMSNorm -> Causal Self-Attention (RoPE on Q/K) --+ (residual)
  |
  |-- RMSNorm -> Router (top-k over E experts)
  |                 |
  |          selected experts (SwiGLU FFN each)
  |                 |
  |          weighted combine --+ (residual)
     |
     v
RMSNorm -> Linear head (tied to embedding) -> logits
```

RMSNorm + RoPE + SwiGLU is the Llama/Mixtral-style recipe rather than 2019 GPT-2's LayerNorm + learned positional embeddings + GELU: it's what current production MoE models (Mixtral, DeepSeek-MoE, Qwen-MoE) actually use, and RoPE avoids a fixed-context-length position table.

Each MoE layer also produces an auxiliary loss (load-balancing + router z-loss), summed across layers and added to the main cross-entropy loss during training.

## Project structure

```
moe-transformer-from-scratch/
├── pyproject.toml                          # installable package (src/ layout) + pytest config
├── requirements.txt
├── configs/config.yaml                     # all hyperparameters, overridable from the CLI
├── data/
│   ├── raw/                                 # source corpus (TinyShakespeare for dev)
│   └── processed/                            # tokenized .bin files (generated, gitignored)
├── src/moe_transformer/
│   ├── config.py                             # ModelConfig
│   ├── init.py                                # GPT-2-style (std=0.02) weight init
│   ├── data/
│   │   ├── tokenizer.py                        # tiktoken (GPT-2 BPE) wrapper
│   │   └── dataset.py                           # memory-mapped windowed Dataset
│   ├── model/
│   │   ├── norm.py                              # RMSNorm
│   │   ├── rope.py                               # rotary position embeddings
│   │   ├── attention.py                           # causal self-attention (manual, no fused kernel)
│   │   ├── feedforward.py                          # SwiGLU FFN -- also the per-expert module
│   │   ├── block.py                                 # transformer block (dense)
│   │   └── moe.py                                    # router, top-k dispatch, capacity, aux losses
│   ├── models.py                              # DenseGPT and MoEGPT (embedding, blocks, head)
│   ├── train.py                               # training loop
│   ├── routing_analysis.py                    # expert utilization and per-token routing plots
│   └── compare.py                             # dense vs. MoE: loss, params, inference speed
├── scripts/
│   ├── prepare_data.py                        # tokenize a raw corpus into train/val .bin files
│   └── download_data.py                       # fetch a larger corpus (TinyStories, WikiText)
├── tests/                                    # pytest suite, alongside each module above
├── checkpoints/                              # saved model weights (.safetensors)
└── outputs/                                  # generated charts and reports
```

## Installation

```bash
git clone <this-repo>
cd moe-transformer-from-scratch
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Requires Python 3.11+. Works on CPU; a GPU speeds up training but isn't required for the default config.

## Usage

```bash
# train the MoE model
python -m moe_transformer.train model.kind=moe
# train the dense baseline for comparison
python -m moe_transformer.train model.kind=dense
# override any hyperparameter from the CLI
python -m moe_transformer.train training.max_steps=5000 model.num_experts=8 model.top_k=2
# inspect routing behavior of a trained MoE model
python -m moe_transformer.routing_analysis
# compare dense vs. MoE: loss curves, parameter counts, inference speed
python -m moe_transformer.compare
```

Everything writes to `outputs/` (charts, JSON reports) and `checkpoints/` (`.safetensors` weights).

## What to look at

- `outputs/loss_comparison.png`: dense vs. MoE validation loss for the same training budget.
- `outputs/expert_utilization.png`: per-layer bar chart of how evenly tokens are spread across experts, with a collapse score (0 = balanced, 1 = fully collapsed onto one expert).
- `outputs/routing_heatmap.png` / `.html`: which expert handled each token in a real sentence, at every layer.
- `outputs/comparison_summary.json`: total vs. active parameters, final loss, and inference throughput for both models.

## Tech stack

| Purpose | Tool |
|---|---|
| Framework | PyTorch |
| Tokenization | tiktoken (GPT-2 BPE) |
| Config | Hydra |
| Checkpoints | safetensors |
| Visualization | matplotlib, plotly |
| Testing | pytest |

The expert dispatch in `src/moe_transformer/model/moe.py` gathers and scatters tokens with plain PyTorch ops, functionally identical to what Switch Transformer and GShard describe. Production systems replace that with fused GPU kernels (Megablocks, Tutel, DeepSpeed-MoE) because a per-expert Python loop is slow. `python -m moe_transformer.compare` makes this visible directly: the MoE model does fewer active FLOPs per token than the dense model but can still run slower in wall-clock time, which is exactly the gap those kernel libraries close.

## Testing

```bash
pytest tests/ -v
```

Tests target the parts of MoE that are easy to get subtly wrong: routing weights renormalize to sum to 1, capacity limits actually drop tokens under load, gradients reach the router through the soft routing probabilities, and a deliberately collapsed router scores worse on the load-balancing loss than a balanced one does.

## Scaling to a full run

The default config is sized to train in minutes on a CPU. For a larger result:

```bash
python scripts/download_data.py --dataset tinystories --n_docs 50000
python -m moe_transformer.train model.kind=dense model.n_embd=384 model.n_layer=6 training.max_steps=10000
python -m moe_transformer.train model.kind=moe model.n_embd=384 model.n_layer=6 model.num_experts=8 training.max_steps=10000
python -m moe_transformer.routing_analysis model.n_embd=384 model.n_layer=6 model.num_experts=8
python -m moe_transformer.compare model.n_embd=384 model.n_layer=6 model.num_experts=8
```

At this scale, expert routing tends to show real specialization (different experts activating for punctuation, dialogue, or rare tokens, for instance) rather than the near-uniform routing a small, briefly-trained model produces.

## Author

Meeta Dave, M.Sc. Web Engineering student at TU Chemnitz

- LinkedIn: [linkedin.com/in/meetadave](https://linkedin.com/in/meetadave)
- Email: [davemeeta12@gmail.com](mailto:davemeeta12@gmail.com)
