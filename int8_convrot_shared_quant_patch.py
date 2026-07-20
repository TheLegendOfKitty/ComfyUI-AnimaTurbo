# Optimization for the comfy-kitchen INT8 ConvRot (QuaRot-style) Linear path
# (TensorWiseINT8Layout with convrot=True -- used by anima-base-v1.0-int8convrot's
# self_attn/cross_attn/mlp projections; ALL 280 quantized Linears in that checkpoint
# use this layout with convrot_groupsize=256, confirmed via a header-only read of the
# checkpoint's _quantization_metadata safetensors metadata -- see the task report).
#
# WHY: exactly the same input-sharing pattern int8_shared_quant_patch.py documents for
# plain int8_tensorwise Linears applies here unchanged: comfy/ldm/cosmos/predict2.py's
# Attention.compute_qkv() calls
#   q = self.q_proj(q_input); k = self.k_proj(k_input); v = self.v_proj(v_input)
# where, for self-attention, q_input is k_input is v_input (all three are the exact
# same Python tensor object `x`), and for cross-attention k_input is v_input (both
# are the exact same `context` tensor object). Each comfy-kitchen ConvRot Linear
# independently ROTATES (online per-256-group Hadamard transform, offline-rotated
# weight, QuaRot-style) AND row-wise int8-quantizes its own input activation before
# the GEMM (torch.ops.comfy_kitchen.int8_linear(..., convrot=True), which for this
# checkpoint's K values -- 1024/2048/8192, all divisible by 256 and <=16384 with
# shared memory to spare -- dispatches to
# comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot64, a SINGLE fused CUDA
# kernel that never materializes the rotated bf16 activation in HBM). Rotation is a
# pure function of activation values (independent of which weight the result will be
# multiplied against, same as plain row-wise quantize), so re-rotating+re-quantizing
# the SAME input tensor object 2-3x in a row for q/k/v (or 2x for cross-attn k/v) is
# exactly as redundant here as it is in the non-rotated case -- more so, since
# rotation is measured below to be the expensive part.
#
# MEASURED (in-process torch.profiler, 5-step device-track trace, real production
# launch flags `--use-sage-attention --fast fp8_matrix_mult cublas_ops autotune`,
# stock dispatch i.e. before this file's caching -- see the task report):
#   - quantize_int8_rowwise_convrot64_kernel: 180.3 ms/step, 280 calls/step (1:1 with
#     the checkpoint's 280 convrot Linears -- there are no plain int8_tensorwise
#     Linears in this checkpoint to begin with, so int8_shared_quant_patch.py's own
#     fast path never engages on it; its `convrot` exclusion always routes here).
#   - int8 GEMM (cutlass/cuBLAS + dequant): ~250-256 ms/step, SAME kernels and SAME
#     call counts as the plain-int8 allint8 checkpoint (statistically indistinguishable
#     between the two, +/-2%) -- the offline weight rotation and per-group dequant do
#     NOT slow the GEMM down at all, it is bit-for-bit the same cutlass_int8_dequant /
#     cublas_gemm_int8 + dequantize_int8_linear call, just fed a per-channel weight
#     scale array it already knew how to consume.
#   - Scaling the plain-int8 allint8 canonical activation-quantize bucket (46.94
#     ms/step at 196 calls, i.e. already cached by int8_shared_quant_patch.py) up to
#     the same 280-call, UNCACHED basis gives ~67.1 ms/step for an equivalent plain
#     (non-rotating) quantize of the same call pattern. 180.3 - 67.1 = ~113 ms/step is
#     attributable to the rotation math ALONE (a per-256-group fast Hadamard
#     transform -- 8 butterfly stages per group vs a single absmax-reduce for plain
#     quantize) -- and that ~113 ms/step accounts for essentially ALL of int8convrot's
#     measured ~110-123 ms/step (~15%) slowdown vs allint8 under this same production
#     config. Every other bucket (attention, norms, elementwise, rope) is unchanged
#     within run-to-run noise between the two checkpoints.
#   - Conclusion for "is the rotation kernel itself inefficient": no -- it is already
#     a single fused CUDA kernel (confirmed: only one kernel name,
#     quantize_int8_rowwise_convrot64_kernel, appears for all 280 calls; no separate
#     rotate-then-quantize two-kernel sequence, no bf16 HBM round-trip of the rotated
#     activation). Its extra cost vs plain quantize reflects genuine extra arithmetic
#     (the Hadamard transform itself), not a memory-bound inefficiency or launch
#     overhead -- so the only remaining lever is ELIMINATING REDUNDANT CALLS, which is
#     what this file does. Of the 280 calls/step, up to 2/4 in self-attn (k,v reusing
#     q's rotate+quantize of x) and 1/4 in cross-attn (v reusing k's rotate+quantize
#     of context) are provably redundant given the source above -- the same 84/280
#     (30%) ratio int8_shared_quant_patch.py's docs establish for the plain case,
#     applied here (28 blocks x 3 redundant calls/block = 84).
#
# FIX: mirrors int8_shared_quant_patch.py's mechanism exactly (single-slot "last
# rotated+quantized activation" cache, identity + version/is_inference guards) but
# targets ONLY convrot=True weights, using the checkpoint's actual online-rotation
# entry point (comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot64) instead of
# the plain quantize_int8_rowwise. On a hit, both the rotation AND the quantize are
# skipped in one shot (they're fused into a single kernel call in stock code -- there
# is nothing to split), and the previously computed (int8 data, fp32 per-row scale)
# pair is reused verbatim for the GEMM.
#
# COMPOSES WITH int8_shared_quant_patch.py regardless of custom_nodes load order: both
# files patch the SAME dispatch-table slot
# (ck_base._LAYOUT_DISPATCH_TABLE[aten.linear.default][TensorWiseINT8Layout]), each
# reading whatever handler is CURRENTLY installed as its own "_orig_handler" and
# delegating to it for any input it doesn't specifically target (this file: any
# non-convrot weight; int8_shared_quant_patch.py: any convrot weight, via its own
# `getattr(weight._params, "convrot", False)` exclusion). Whichever file's top-level
# code runs second ends up wrapping the other's handler; correctness does not depend
# on which order that happens in, because each file only ever intercepts inputs
# matching its own condition and transparently forwards everything else down the
# chain. This file does NOT edit int8_shared_quant_patch.py in any way -- it is a
# fully independent, independently-deletable sibling file with its OWN module-level
# cache slot (never shared with that file's slot, so the two can never corrupt each
# other's cached values even though both may observe the same underlying tensor
# objects at different times).
#
# SAFETY / CORRECTNESS (identical reasoning to int8_shared_quant_patch.py):
#   - Online ConvRot rotation + row-wise quantization is a pure function of tensor
#     VALUES (fixed offline-baked group structure, groupsize=256, no learned or
#     call-order-dependent state). A cache hit is only possible when the incoming
#     tensor `is` (exact same Python object) the one we rotated+quantized last; we
#     hold a strong reference to it so its id() cannot be recycled for an unrelated
#     tensor. This makes a hit mathematically identical to a fresh rotate+quantize --
#     no approximation, no precision change beyond what stock ConvRot already applies.
#   - Fast path only engages when ALL of: weight is a TensorWiseINT8Layout
#     QuantizedTensor with convrot=True and convrot_groupsize==256 (the only value
#     this checkpoint uses, and the one stock's own int8_linear() special-cases with
#     its fastest, no-HBM-roundtrip fused kernel); input K divisible by 256,
#     256<=K<=16384, and shared-memory fit for that fused kernel (mirrors stock's own
#     `_convrot_fused_shared_memory_fits` gate exactly -- if it wouldn't fit, stock
#     falls back to a slower rotate-then-quantize two-kernel path this file does NOT
#     attempt to reimplement, so it defers to the next handler instead); M>1
#     (gemv/M==1 uses a completely different fused kernel, int8_linear_m1, not
#     reimplemented here -- same scoping choice int8_shared_quant_patch.py makes for
#     the plain M==1 case); non-Turing GPU; non-transposed weight; non-QuantizedTensor,
#     CUDA, non-autograd-tracked input; COMFY_KITCHEN_DISABLE_CUTLASS unset (the stock
#     kill-switch, read fresh via module attribute access on every call, not cached at
#     install time -- mirrors int8_shared_quant_patch.py's own handling verbatim); ANY
#     exception -- all other cases fall back to the CURRENTLY-installed
#     `_orig_handler` UNCHANGED, so on any input this file doesn't specifically
#     target, the result is exactly what the next handler in the chain (either
#     int8_shared_quant_patch.py or stock comfy-kitchen) would have produced.
#   - On the inputs this fast path DOES target, it reimplements stock's own
#     cutlass/cuBLAS GEMM+dequant branch (mirroring backends/cuda/__init__.py's
#     int8_linear line-for-line for the convrot, M>1, groupsize=256 fused-kernel case,
#     including its weight-contiguity guarantee and its COMFY_KITCHEN_DISABLE_CUTLASS
#     gate) -- verified numerically BIT-EXACT vs stock via a same-seed run (latent
#     torch.equal(baseline, patched) True across multiple seeds; see the task report)
#     -- but being a reimplementation rather than a delegation, it depends on staying
#     in sync with comfy-kitchen's own dispatch logic if that ever changes upstream
#     (identical caveat to int8_shared_quant_patch.py).
#   - Only engages under torch.ops.aten.linear.default's QuantizedTensor dispatch
#     (eager execution). Does not touch cast_bias_weight / LoRA weight_function /
#     bias_function / offloading -- all of that machinery in comfy/ops.py's
#     MixedPrecisionOps.Linear.forward() still runs exactly as before.
#   - No locking around the shared cache slot: ComfyUI runs a given model's forward
#     pass on a single Python thread (one CUDA stream, sequential eager dispatch), so
#     there is never concurrent access to the module-level cache from two calls at
#     once.
#
# Delete this file to revert to whatever dispatch was installed beneath it (stock
# comfy-kitchen, or int8_shared_quant_patch.py's plain-int8 caching alone, which never
# actually engages on this checkpoint since every Linear here is convrot=True) -- no
# other state to clean up: the dispatch table entry is restored to the original
# function object at interpreter exit anyway since this only lives in comfy_kitchen's
# in-memory dispatch table, never touches any file on disk.
import logging

_LOG_PREFIX = "[int8_convrot_shared_quant_patch]"

try:
    import torch

    # Import the layout submodule FIRST so its @register_layout_op decorators run
    # and populate the dispatch table before we try to read/replace an entry in it
    # (same defensive ordering as int8_shared_quant_patch.py -- harmless no-op if
    # some other module already triggered this import first).
    import comfy_kitchen.tensor.int8 as ck_int8  # noqa: F401
    import comfy_kitchen.tensor.base as ck_base
    import comfy_kitchen.backends.cuda as ck_cuda
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, _dtype_code

    _LINEAR_OP = torch.ops.aten.linear.default
    _orig_handler = ck_base._LAYOUT_DISPATCH_TABLE[_LINEAR_OP][TensorWiseINT8Layout]

    _CONVROT_FUSED_MAX_K = getattr(ck_cuda, "_CONVROT_FUSED_MAX_K", 16384)

    # Single-slot "last rotated+quantized activation" cache -- own slot, independent
    # of int8_shared_quant_patch.py's. q/k/v (or cross-attn k/v) calls are strictly
    # sequential with nothing else in between in
    # comfy.ldm.cosmos.predict2.Attention.compute_qkv, so one slot captures every
    # sharing opportunity. Holding a strong ref to the source tensor in `_slot_src`
    # is what makes the `is` identity check safe against id() reuse.
    _slot_src = [None]      # the exact input tensor object last rotated+quantized (or None)
    _slot_version = [None]  # its ._version at the time we rotated+quantized it (or None)
    _slot_qdata = [None]    # its int8 rotated+quantized data
    _slot_scale = [None]    # its fp32 per-row scale

    _stats = {"hits": 0, "misses": 0, "fallbacks": 0}

    def _tensor_version_or_none(t):
        # ComfyUI's sampling loop runs entirely inside torch.inference_mode(), and
        # PyTorch deliberately does not track ._version for inference tensors
        # (accessing it raises RuntimeError). Tensor.is_inference() tells us, per
        # tensor, whether ._version is readable at all; when it isn't, both the
        # current and previously-cached version read as the same sentinel `None`, so
        # `is` alone gates the hit -- safe given this file's single-threaded,
        # strictly-sequential-call, strong-ref-held design (see module docstring).
        return None if t.is_inference() else t._version

    def _get_or_rotate_quantize(x_2d, src_tensor, convrot_groupsize):
        version = _tensor_version_or_none(src_tensor)
        if _slot_src[0] is src_tensor and _slot_version[0] == version:
            _stats["hits"] += 1
            return _slot_qdata[0], _slot_scale[0]
        _stats["misses"] += 1
        qdata, scale = ck_cuda.quantize_int8_rowwise_convrot64(
            x_2d, convrot_groupsize, stochastic_rounding=0
        )
        _slot_src[0] = src_tensor
        _slot_version[0] = version
        _slot_qdata[0] = qdata
        _slot_scale[0] = scale
        return qdata, scale

    def _cached_convrot_int8_linear(qt, args, kwargs):
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None

        # Any condition our reimplemented fast path doesn't specifically cover ->
        # delegate to whatever handler is currently installed beneath us, unchanged.
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != "TensorWiseINT8Layout"
            or getattr(weight._params, "transposed", False)
            or not getattr(weight._params, "convrot", False)
            or getattr(weight._params, "convrot_groupsize", 256) != 256
            or isinstance(input_tensor, QuantizedTensor)
            or not input_tensor.is_cuda
            or input_tensor.requires_grad
            or ck_cuda._DISABLE_CUTLASS_INT8
        ):
            return _orig_handler(qt, args, kwargs)

        try:
            weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            # Stock guarantees this contiguity twice (see int8_shared_quant_patch.py's
            # identical comment); get_plain_tensors() gives no such guarantee on its
            # own. No-op when already contiguous.
            weight_qdata = weight_qdata.contiguous()
            out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

            x = input_tensor
            orig_shape = x.shape
            x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, x.shape[-1]).contiguous()
            m, k = x_2d.shape
            n = weight_qdata.shape[0]
            if k != weight_qdata.shape[-1]:
                return _orig_handler(qt, args, kwargs)

            # Fast path only reimplements the M>1, groupsize=256,
            # quantize_int8_rowwise_convrot64 fused-kernel branch of stock's
            # int8_linear -- exactly what this repo's checkpoint (K in {1024, 2048,
            # 8192}, all divisible by 256 and comfortably <=16384) and hardware (SM86
            # RTX 3090) actually hit (confirmed via profiling: only this one kernel
            # name appears, 280/280 calls). M==1 (gemv) and Turing devices fall back
            # to the currently-installed handler unchanged.
            if m <= 1 or ck_cuda._cuda_device_is_turing(x.get_device()):
                return _orig_handler(qt, args, kwargs)
            if k % 256 != 0 or not (256 <= k <= _CONVROT_FUSED_MAX_K):
                return _orig_handler(qt, args, kwargs)
            if not ck_cuda._convrot_fused_shared_memory_fits(x_2d, k, 256):
                return _orig_handler(qt, args, kwargs)
            if not ck_cuda._cuda_device_supports_cutlass_int8_dequant(x_2d):
                return _orig_handler(qt, args, kwargs)

            x_qdata, x_scale = _get_or_rotate_quantize(x_2d, x, 256)

            output_dtype_code = _dtype_code(out_dtype)
            out = torch.empty((m, n), dtype=out_dtype, device=x.device)
            weight_scale_arg = ck_cuda._int8_weight_scale_arg(weight_scale, x.device)
            bias_arg = bias if bias is not None else ck_cuda._empty_cuda_tensor(x.device, out_dtype)
            if bias is not None and (bias.device != x.device or bias.dtype != out_dtype or not bias.is_contiguous()):
                bias_arg = bias.to(device=x.device, dtype=out_dtype).contiguous()

            stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
            ws_cutlass = weight_scale_arg if weight_scale_arg.numel() == n else weight_scale_arg.expand(n).contiguous()
            bias_f32 = bias_arg.to(torch.float32).contiguous() if bias is not None else bias_arg
            used_cutlass = ck_cuda._C.cutlass_int8_dequant(
                ck_cuda._wrap_for_dlpack(x_qdata),
                ck_cuda._wrap_for_dlpack(weight_qdata),
                ck_cuda._wrap_for_dlpack(x_scale),
                ck_cuda._wrap_for_dlpack(ws_cutlass),
                ck_cuda._wrap_for_dlpack(bias_f32),
                ck_cuda._wrap_for_dlpack(out),
                output_dtype_code,
                stream_ptr,
            )
            if not used_cutlass:
                out_int32 = torch.empty((m, n), dtype=torch.int32, device=x.device)
                ck_cuda._C.cublas_gemm_int8(
                    ck_cuda._wrap_for_dlpack(x_qdata),
                    ck_cuda._wrap_for_dlpack(weight_qdata),
                    ck_cuda._wrap_for_dlpack(out_int32),
                    ck_cuda._wrap_for_dlpack(ck_cuda.get_cublas_workspace()),
                    stream_ptr,
                )
                ck_cuda._C.dequantize_int8_linear(
                    ck_cuda._wrap_for_dlpack(out_int32),
                    ck_cuda._wrap_for_dlpack(x_scale),
                    ck_cuda._wrap_for_dlpack(weight_scale_arg),
                    ck_cuda._wrap_for_dlpack(bias_arg),
                    ck_cuda._wrap_for_dlpack(out),
                    output_dtype_code,
                    stream_ptr,
                )

            return out.reshape(*orig_shape[:-1], n)
        except Exception:
            logging.getLogger(__name__).warning(
                f"{_LOG_PREFIX} fast path raised, falling back to stock handler for this call", exc_info=True
            )
            _stats["fallbacks"] += 1
            return _orig_handler(qt, args, kwargs)

    # Idempotency guard: if this file's top-level code ever runs twice in the same
    # process (e.g. a custom-node reload), avoid wrapping our own wrapper a second
    # time. `_orig_handler` above was just read from whatever is CURRENTLY
    # installed -- on a second run that would already be our own replacement from
    # the first run, identifiable by this sentinel attribute.
    if getattr(_orig_handler, "_int8_convrot_shared_quant_patched", False):
        logging.info(f"{_LOG_PREFIX} already installed in this process, skipping re-patch (idempotent no-op)")
    else:
        _cached_convrot_int8_linear._int8_convrot_shared_quant_patched = True
        ck_base._LAYOUT_DISPATCH_TABLE[_LINEAR_OP][TensorWiseINT8Layout] = _cached_convrot_int8_linear
        logging.info(
            f"{_LOG_PREFIX} installed: caches the last rotated+quantized ConvRot activation so "
            f"self_attn/cross_attn q/k/v (and any other Linear calls sharing an identical input "
            f"tensor object) skip redundant re-rotation+re-quantization on convrot=True int8 "
            f"Linears. Delete this file to revert."
        )
except Exception as e:
    logging.warning(f"{_LOG_PREFIX} patch failed, using stock/underlying comfy-kitchen convrot dispatch: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
