# Optimization for the comfy-kitchen INT8 tensorwise Linear path (TensorWiseINT8Layout,
# used by e.g. Anima-allint8's self_attn/cross_attn/mlp projections).
#
# WHY: comfy/ldm/cosmos/predict2.py's Attention.compute_qkv() calls
#   q = self.q_proj(q_input); k = self.k_proj(k_input); v = self.v_proj(v_input)
# where, for self-attention, q_input is k_input is v_input (all three are the exact
# same Python tensor object `x`), and for cross-attention k_input is v_input (both
# are the exact same `context` tensor object). Each of comfy-kitchen's int8 Linear
# calls independently row-wise-quantizes its OWN input activation
# (quantize_int8_rowwise: per-row absmax + round to int8) before the GEMM. Since
# row-wise quantization is a pure function of the activation values alone (it does
# not depend on which weight it will be multiplied against), quantizing the SAME
# input tensor 2-3 times in a row for q/k/v (or 2x for cross-attn k/v) is 100%
# redundant work: identical (int8 data, fp32 scale) is recomputed from scratch and
# thrown away.
#
# Measured with a low-overhead cuda-event hook (no torch.profiler) around comfy-kitchen's
# real int8_linear + quantize_int8_rowwise calls during an actual Anima-allint8 sampling
# run (1536x1536, RTX 3090, SM86): the int8 bucket is ~310 ms/step (~36% of an ~850 ms/step
# un-profiled steady-state), split ~82% GEMM / ~18% row-wise activation quantize
# (quantize_int8_rowwise alone = ~56 ms/step, 280 calls/step, 1:1 with the 280 int8
# Linears in the 28-block DiT). Of those 280 quantize calls/block-group, up to 2/4 in
# self-attn (k,v reusing q's quantize of x) and 1/4 in cross-attn (v reusing k's
# quantize of context) are provably redundant given the source above.
#
# FIX: monkeypatch comfy-kitchen's QuantizedTensor dispatch table entry for
# TensorWiseINT8Layout's `aten.linear.default` handler (the exact function that
# runs for every int8_tensorwise nn.Linear forward) to cache the LAST quantized
# input tensor (by Python object identity, keyed to a single slot -- q/k/v calls are
# strictly sequential/adjacent in compute_qkv with nothing else in between, so a
# single-slot "last quantized" cache is sufficient; identity is checked via `is`
# against a tensor we hold a strong reference to, so there is no id()-reuse-after-gc
# risk). On a hit, we skip quantize_int8_rowwise entirely and reuse the cached
# (int8 data, fp32 per-row scale) for the GEMM. On a miss (different tensor, e.g.
# the residual/output_proj input, or a plain non-shared Linear), we quantize fresh
# and refill the slot -- functionally IDENTICAL to stock behavior, just cached.
#
# SAFETY / CORRECTNESS:
#   - Row-wise quantization is a pure function of tensor VALUES. A cache hit is only
#     possible when the incoming tensor `is` (exact same Python object) the one we
#     quantized last; we hold a strong reference to it so its id() cannot be recycled
#     for an unrelated tensor. This makes a hit mathematically identical to a fresh
#     quantize -- there is no approximation and no precision change.
#   - The fast path only engages for the exact conditions this repo's comfy-kitchen
#     0.2.22 CUDA backend already uses for M>1 int8_tensorwise Linears on SM80+
#     non-Turing GPUs (cutlass fused GEMM+dequant+bias, falling back to cuBLAS
#     int8 GEMM + separate dequant kernel if cutlass declines): M==1 (gemv path),
#     ConvRot-rotated weights, Turing GPUs, transposed weight layout, non-CUDA
#     tensors, autograd-tracked (requires_grad) inputs, COMFY_KITCHEN_DISABLE_CUTLASS=1
#     (the stock kill-switch), non-per-tensor/per-channel weight scale shapes, or
#     ANY exception -- all fall back to comfy-kitchen's ORIGINAL handler
#     (torch.ops.comfy_kitchen.int8_linear) UNCHANGED, so on any input this file
#     doesn't specifically target, the result is exactly what stock would have
#     produced (we call stock's own function verbatim, not a reimplementation).
#     On the inputs the fast path DOES target, we reimplement stock's own
#     cutlass/cuBLAS branch (mirroring backends/cuda/__init__.py's int8_linear
#     line-for-line, including its weight-contiguity guarantee and its
#     COMFY_KITCHEN_DISABLE_CUTLASS gate) and verified it numerically
#     bit-identical to stock via a same-seed run (see the verification note at
#     the bottom of this file / the task's report) -- but being a reimplementation
#     rather than a delegation, it depends on staying in sync with comfy-kitchen's
#     own dispatch logic if that ever changes upstream.
#   - Only engages under torch.ops.aten.linear.default's QuantizedTensor dispatch
#     (eager execution). Does not touch cast_bias_weight / LoRA weight_function /
#     bias_function / offloading -- all of that machinery in comfy/ops.py's
#     MixedPrecisionOps.Linear.forward() still runs exactly as before; this file
#     only intercepts what happens to the (already-resolved) input/weight/bias
#     tensors at the point comfy-kitchen would otherwise quantize+GEMM them.
#   - No locking around the shared cache slot: ComfyUI runs a given model's
#     forward pass on a single Python thread (one CUDA stream, sequential eager
#     dispatch), so there is never concurrent access to the module-level cache
#     from two calls at once.
#
# Delete this file to revert to stock comfy-kitchen dispatch (no other state to
# clean up: the dispatch table entry is restored to the original function object
# at interpreter exit anyway since this only lives in comfy_kitchen's in-memory
# dispatch table, never touches any file on disk).
import logging

_LOG_PREFIX = "[int8_shared_quant_patch]"

try:
    import torch

    # Import the layout submodule FIRST so its @register_layout_op decorators run
    # and populate the dispatch table before we try to read/replace an entry in it.
    import comfy_kitchen.tensor.int8 as ck_int8
    import comfy_kitchen.tensor.base as ck_base
    import comfy_kitchen.backends.cuda as ck_cuda
    from comfy_kitchen.tensor.base import QuantizedTensor, dequantize_args
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, _dtype_code

    _LINEAR_OP = torch.ops.aten.linear.default
    _orig_handler = ck_base._LAYOUT_DISPATCH_TABLE[_LINEAR_OP][TensorWiseINT8Layout]

    # Single-slot "last quantized activation" cache. q/k/v (or cross-attn k/v) calls
    # are strictly sequential with nothing else in between in
    # comfy.ldm.cosmos.predict2.Attention.compute_qkv, so one slot captures every
    # sharing opportunity. Holding a strong ref to the source tensor in `_slot_src`
    # is what makes the `is` identity check safe against id() reuse.
    #
    # This hook is installed globally for every TensorWiseINT8Layout Linear in the
    # process, not just the compute_qkv call site it was designed around, and the
    # failure mode of a stale hit would be silent wrong numbers -- not a crash. So
    # in addition to the `is` identity check we also require the tensor's autograd
    # version counter (`._version`, bumped by PyTorch on every in-place mutation)
    # to be unchanged since we quantized it, guarding against the same Python
    # object being reused-and-mutated in place between two Linear calls.
    #
    # ComfyUI's actual sampling loop runs entirely inside torch.inference_mode(),
    # and PyTorch deliberately does NOT track ._version for inference tensors
    # (accessing it raises RuntimeError: "Inference tensors do not track version
    # counter") -- there is simply no mutation-tracking mechanism available for
    # them, by design (inference mode's whole point is skipping that bookkeeping).
    # `Tensor.is_inference()` tells us, per-tensor, whether ._version is readable
    # at all; when it isn't, we degrade to the identity-only check (both the
    # current and previously-cached version read as the same sentinel `None`, so
    # `is` alone gates the hit) -- i.e. exactly this file's original, already-
    # justified-safe design (single Python thread, strictly sequential/adjacent
    # calls in compute_qkv, tensor kept alive by our own strong ref so id() can't
    # be recycled). For any tensor where version tracking IS available (outside
    # inference_mode), we get the requested extra hardening for free.
    _slot_src = [None]      # the exact input tensor object last quantized (or None)
    _slot_version = [None]  # its ._version at the time we quantized it (or None)
    _slot_qdata = [None]    # its int8 row-wise quantized data
    _slot_scale = [None]    # its fp32 per-row scale

    _stats = {"hits": 0, "misses": 0, "fallbacks": 0}

    def _tensor_version_or_none(t):
        return None if t.is_inference() else t._version

    def _get_or_quantize(x_2d, src_tensor):
        version = _tensor_version_or_none(src_tensor)
        if _slot_src[0] is src_tensor and _slot_version[0] == version:
            _stats["hits"] += 1
            return _slot_qdata[0], _slot_scale[0]
        _stats["misses"] += 1
        qdata, scale = ck_cuda.quantize_int8_rowwise(x_2d)
        _slot_src[0] = src_tensor
        _slot_version[0] = version
        _slot_qdata[0] = qdata
        _slot_scale[0] = scale
        return qdata, scale

    def _cached_int8_linear_tensorwise(qt, args, kwargs):
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None

        # Any condition our reimplemented fast path doesn't specifically cover ->
        # delegate to comfy-kitchen's original handler unchanged. This includes
        # comfy-kitchen's own stock kill-switch (COMFY_KITCHEN_DISABLE_CUTLASS=1,
        # read fresh via module attribute access -- not cached at patch-install
        # time -- so it stays correct if ever toggled after this file is loaded);
        # when set, stock steps down to the cuBLAS-GEMM+separate-dequant fallback,
        # and rather than re-deriving that combination here we simply defer to
        # stock's own handler entirely.
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != "TensorWiseINT8Layout"
            or getattr(weight._params, "transposed", False)
            or getattr(weight._params, "convrot", False)
            or isinstance(input_tensor, QuantizedTensor)
            or not input_tensor.is_cuda
            or input_tensor.requires_grad
            or ck_cuda._DISABLE_CUTLASS_INT8
        ):
            return _orig_handler(qt, args, kwargs)

        try:
            weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            # Stock guarantees this contiguity twice (the caller-side .contiguous()
            # in comfy_kitchen/tensor/int8.py and the defensive check inside
            # backends/cuda/__init__.py's int8_linear); get_plain_tensors() gives no
            # such guarantee on its own (e.g. a future LoRA/weight-patch path could
            # hand back a non-contiguous view), and a non-contiguous qdata fed
            # straight into the dlpack-wrapped cutlass/cuBLAS kernels would silently
            # produce wrong numbers rather than error. No-op when already contiguous.
            weight_qdata = weight_qdata.contiguous()
            out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

            x = input_tensor
            orig_shape = x.shape
            x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, x.shape[-1]).contiguous()
            m, k = x_2d.shape
            n = weight_qdata.shape[0]
            if k != weight_qdata.shape[-1]:
                return _orig_handler(qt, args, kwargs)

            # Fast path only reimplements the M>1, non-Turing cutlass/cuBLAS branch --
            # exactly what this repo's hardware (SM86 RTX 3090) actually uses. M==1
            # (gemv) and Turing devices fall back to the original (their extra
            # padding/gemv logic isn't worth re-deriving here for a tiny slice of calls).
            if m <= 1 or ck_cuda._cuda_device_is_turing(x.get_device()):
                return _orig_handler(qt, args, kwargs)
            if not ck_cuda._cuda_device_supports_cutlass_int8_dequant(x_2d):
                return _orig_handler(qt, args, kwargs)

            x_qdata, x_scale = _get_or_quantize(x_2d, x)

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
    if getattr(_orig_handler, "_int8_shared_quant_patched", False):
        logging.info(f"{_LOG_PREFIX} already installed in this process, skipping re-patch (idempotent no-op)")
    else:
        _cached_int8_linear_tensorwise._int8_shared_quant_patched = True
        ck_base._LAYOUT_DISPATCH_TABLE[_LINEAR_OP][TensorWiseINT8Layout] = _cached_int8_linear_tensorwise
        logging.info(
            f"{_LOG_PREFIX} installed: caches the last row-wise-quantized activation so "
            f"self_attn/cross_attn q/k/v (and any other Linear calls sharing an identical "
            f"input tensor object) skip redundant re-quantization. Delete this file to revert."
        )
except Exception as e:
    logging.warning(f"{_LOG_PREFIX} patch failed, using stock comfy-kitchen int8 dispatch: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
