# Optimization for sm_86 (RTX 3090): SageAttention auto-dispatch uses the Triton kernel.
# The CUDA int8 / pv=fp16 kernel is faster but OVERFLOWS on one attn layer/step -> black image.
# Fix: smooth_v=True mean-centers V so the fp16 PV accumulator stays in range (verified: 0 NaN
# across 1680 calls, output matches fp16+fp32 reference). Delete this file to revert.
#
# Re-verified 2026-07-19 against the Anima allint8 workload (1536x1536, 30 steps, RTX 3090,
# --use-sage-attention --fast fp8_matrix_mult cublas_ops autotune):
#   - CORRECTION (2026-07-19): an earlier version of this note claimed the DiT runs its
#     attention activations in torch.float16 under this exact flag set. That claim was WRONG --
#     it was based on a profiling run that used a bare `--fast` (no explicit ops list), which
#     enables PerformanceFeature.Fp16Accumulation, which sets model_management.PRIORITIZE_FP16
#     = True and pushes the model into fp16 compute. That is a DIFFERENT regime from this
#     file's actual documented flags (`--fast fp8_matrix_mult cublas_ops autotune`, which
#     excludes fp16_accumulation). Re-checked directly against a live server started with the
#     flags above (plus independently via in-process instrumentation of
#     comfy.ldm.cosmos.predict2.Block.forward and a hooked torch.Tensor.to() call-site tally):
#     under these flags the model computes in bfloat16 end-to-end ("model weight dtype
#     torch.bfloat16, manual cast: torch.bfloat16"), and q/k/v at the sageattn call boundary
#     are bfloat16, not float16. There is no bf16<->fp16 casting anywhere in this code path,
#     with or without this patch, exactly as originally stated -- that part was correct.
#   - The fp32-residual/fp16-compute block design in comfy/ldm/cosmos/predict2.py (shared by
#     Anima) is gated by `if x.dtype == torch.float16: x = x.float()` in MiniTrainDIT._forward.
#     Under bf16 compute that condition is never true (the residual stream arrives already in
#     bf16, via comfy/model_base.py's unconditional cast of the input latent to the model's
#     inference dtype before diffusion_model.forward is even called), so that upcast is dead
#     code for this deployment and the per-block .to(compute_dtype)/.to(residual_dtype) calls
#     are same-dtype no-ops. A 5-step torch.profiler trace under the real flags confirms this:
#     genuine dtype-converting copy kernels total ~0.02 ms/step (attributable only to the
#     once-per-step whole-latent input cast, unrelated to the 28-block residual stream), not
#     the ~46 ms/step (~5% of step time) previously attributed here. That ~46 ms/step figure is
#     real, but only for the fp16-compute regime (e.g. bare `--fast`, or an explicit
#     --force-fp16) -- it does not apply to this file's documented launch flags. Either way,
#     that cost lives entirely in predict2.py's block design and is unrelated to attention
#     backend choice or this patch.
#   - Benchmarked pv_accum_dtype="fp16"+smooth_v=True (this file, current) against
#     pv_accum_dtype="fp32", pv_accum_dtype="fp16+fp32", stock Triton dispatch (patch removed),
#     and pv_accum_dtype="fp16" with smooth_v=False: this config is the fastest of all of them
#     that produces a correct image (0.74 s/it vs 0.80-0.85 s/it for the alternatives). Disabling
#     smooth_v is ~7% faster (0.69 s/it) but reproduces the black-image overflow bug byte-for-byte
#     (mean=0.0, std=0.0 output) -- confirming smooth_v's cost is required correctness overhead,
#     not removable waste. No further optimization found; config below is left unchanged.
import logging
try:
    import comfy.ldm.modules.attention as _a
    from sageattention import sageattn_qk_int8_pv_fp16_cuda as _cuda_fp16

    _stock_sageattn = _a.sageattn  # keep a handle to ComfyUI's default dispatch for fallback

    def _fast_sage(q, k, v, is_causal=False, tensor_layout="HND", sm_scale=None,
                   smooth_k=False, **kw):
        # The CUDA pv=fp16 kernel used below has no attn_mask support. ComfyUI's own
        # attention_sage() currently never forwards a mask this far (it falls back to
        # attention_pytorch first when the installed sageattention build doesn't advertise
        # attn_mask support), but guard it explicitly anyway so a future ComfyUI/sageattention
        # version can't silently compute unmasked attention when a mask was requested.
        if kw.get("attn_mask") is not None:
            return _stock_sageattn(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout,
                                    sm_scale=sm_scale, smooth_k=smooth_k, **kw)
        return _cuda_fp16(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
                          sm_scale=sm_scale, smooth_k=smooth_k,
                          pv_accum_dtype="fp16", smooth_v=True, qk_quant_gran="per_warp")

    _a.sageattn = _fast_sage
    logging.info("[sage_pv_fp16_patch] sageattn -> CUDA pv=fp16 + smooth_v=True (per_warp)")
except Exception as e:
    logging.warning(f"[sage_pv_fp16_patch] patch failed, using default sage: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
