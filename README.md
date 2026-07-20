# ComfyUI-AnimaTurbo

Anima / cosmos-predict2 int8 inference speedups for ComfyUI, packaged as a single
self-contained `custom_nodes` directory. No comfy-kitchen source fork required: the
one component that started as a comfy-kitchen branch patch (a faster ConvRot
quantize kernel) is reproduced here as a standalone CUDA extension plus a runtime
shim, so the whole pack works against a stock comfy-kitchen install.

## What it does

- **AnimaFuseQKV** (node): fuses each quantized Anima/cosmos-predict2 block's
  self-attention q/k/v Linears into one concatenated-weight Linear via an object
  patch. Cuts self-attn from 3 quantize+GEMM calls to 1, and — unlike the two
  eager-mode caching patches below — the saving survives `torch.compile` because
  it changes model structure before the graph is traced.
- **sage_pv_fp16_patch.py** (auto-applies): redirects SageAttention's dispatch to
  the CUDA `pv=fp16` kernel with `smooth_v=True` (`qk_quant_gran="per_warp"`),
  which is faster than the Triton kernel SageAttention's own auto-dispatch would
  pick on SM86, without the fp16 PV-accumulator overflow that `smooth_v=False`
  hits on this workload.
- **int8_shared_quant_patch.py** (auto-applies): caches the last row-wise-quantized
  activation so that self-attn/cross-attn q/k/v calls sharing the exact same input
  tensor object (by Python identity) skip redundant re-quantization, for plain
  (non-ConvRot) `TensorWiseINT8Layout` int8 Linears.
- **int8_convrot_shared_quant_patch.py** (auto-applies): the same idea for ConvRot
  (`convrot=True`, groupsize 256) int8 Linears — caches the last
  rotated+quantized activation so shared q/k/v inputs skip redundant
  rotate+quantize.
- **warp_fht/** (auto-applies): JIT-builds a standalone CUDA extension
  reproducing a warp-per-group fast-Hadamard-transform ConvRot quantize kernel,
  and installs a runtime shim over
  `comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64` that routes
  eligible calls (`group_size=256`, non-stochastic, `K % 1024 == 0`,
  `1024 <= K <= 32768`, dtype fp32/fp16/bf16) to it. Ineligible calls, and any
  build/install failure, fall through to comfy-kitchen's own handler unchanged.

## Install

```
git clone <repo-url> custom_nodes/ComfyUI-AnimaTurbo
```

Restart ComfyUI. A CUDA toolkit (`nvcc`) matching your installed PyTorch's CUDA
version is needed for the `warp_fht` kernel's one-time JIT build; without one,
that component logs a warning and the rest of the pack (AnimaFuseQKV, the sage
patch, both shared-quant patches) still works exactly the same. If your system
compiler is newer than your CUDA toolkit supports, set `CC`/`CXX`/`CUDAHOSTCXX`
to a compiler version your toolkit accepts before launching ComfyUI; otherwise
the JIT build fails the same way any `torch.utils.cpp_extension` build would,
and the pack degrades gracefully in that case too.

## Usage

- Add the **AnimaFuseQKV** node after any LoRA loaders and before
  `TorchCompileModel` in your workflow. LoRAs applied downstream of this node
  will not affect the fused q/k/v — the node fuses only when q/k/v are
  identically-shaped, unpatched, quantized weights.
- Everything else (sage dispatch, both shared-quant caches, the warp-FHT kernel
  shim) applies automatically as soon as the pack loads — no workflow changes
  needed.
- Recommended launch flags for the full speedup:
  `--use-sage-attention --fast fp8_matrix_mult cublas_ops autotune`

## Measured results (RTX 3090, SM86)

All numbers below are for `anima-base-v1.0-int8convrot.safetensors`, 1536x1536,
`er_sde`/`simple`, cfg 4, under the launch flags above.

| Measurement | Value | Config |
|---|---|---|
| E2E steady-state, compiled, AnimaFuseQKV + TorchCompileModel | **0.69-0.70 s/it** | 10-30 step runs |
| E2E steady-state, compiled, without AnimaFuseQKV (shared-quant + sage + warp-kernel patches only) | 0.7066 s/it | 30 steps |
| Fully stock (no patches) | ~0.735 s/it | uninstrumented |
| ConvRot quantize kernel, warp-FHT vs. the prior (non-warp) fused kernel | **1.61x @ K=2048**, **1.98x @ K=8192** | isolated microbenchmark, M=18432 |
| AnimaFuseQKV: quantize/GEMM calls per step | 280 -> 224 (**-56 calls/step**) | compiled graph, 28-block DiT |
| AnimaFuseQKV: peak allocated VRAM | 4301.6 MiB -> 4639.3 MiB (**+338 MiB**) | compiled, 30 steps |
| SageAttention dispatch, CUDA pv=fp16 (smooth_v=True) vs. default Triton dispatch | ~0.74 s/it vs. 0.80-0.85 s/it | see sage_pv_fp16_patch.py |

The E2E numbers were measured through the real ComfyUI custom-node loader
(`nodes.init_extra_nodes()`) with only this pack providing patches — the loose,
superseded `convrot_fht_triton_patch.py` and `anima_bf16_residual_patch.py`
files, if present in `custom_nodes/`, were excluded so the numbers reflect this
pack alone.

## Env kill-switches

- `ANIMA_TURBO_NO_KERNEL=1` — skips the `warp_fht` kernel build and shim install
  entirely (this pack's own switch). Everything else in the pack still applies.
- `COMFY_KITCHEN_DISABLE_CUTLASS=1` — comfy-kitchen's own switch (not defined by
  this pack). Both shared-quant-cache patches respect it per-call: with it set,
  their fast paths never engage and every call falls through to whatever handler
  is installed beneath them, so the caching patches become no-ops without needing
  to be removed.
- AnimaFuseQKV, `sage_pv_fp16_patch.py`, `int8_shared_quant_patch.py`, and
  `int8_convrot_shared_quant_patch.py` have **no dedicated env-var switch** of
  their own. AnimaFuseQKV is opt-in by workflow placement (just don't add the
  node). The other three self-disable automatically and log a warning if their
  target isn't importable or an unexpected error occurs, but there is no env var
  to force them off short of removing the pack.

## Compatibility notes

- Tested against **comfy-kitchen 0.2.22**. `comfy_kitchen` has no `__version__`
  attribute in that release, so the pack reads the version via
  `importlib.metadata` for its untested-version log message. The shim installs
  on any version — an untested version only logs an info-level notice, it never
  hard-refuses — and self-disables (restores the original handler) if a future
  comfy-kitchen changes the `_C.quantize_int8_rowwise_convrot64` call signature
  in a way that breaks the shim's argument unpacking.
- The `warp_fht` JIT build targets whatever CUDA architecture
  `torch.utils.cpp_extension.load()` detects for the current GPU (no hardcoded
  arch list), so it is not limited to SM86 — it should build for any SM80+ CUDA
  architecture PyTorch's toolchain supports. Tested on SM86 (RTX 3090).
- SageAttention is optional: `sage_pv_fp16_patch.py` is wrapped in a top-level
  `try/except` and, if `sageattention` isn't installed (or the patch fails for any
  other reason), it logs a warning and leaves ComfyUI's default SageAttention
  dispatch (or lack thereof) completely untouched — verified by reading the
  patch's own code path, which imports `sageattention` inside the `try` block
  before touching anything.
