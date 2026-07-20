"""
JIT-builds a standalone CUDA extension (convrot_warp_quantize.cu) that reproduces
comfy-kitchen's warp-per-group FHT ConvRot row-wise INT8 quantize kernel, then installs
a shim over comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64 that routes
eligible calls to it. Ineligible calls, and any failure at build or install time, fall
through to comfy-kitchen's own stock behavior unchanged -- this module either speeds up
an eligible subset of calls or is a complete no-op, never a correctness risk.

Eligibility (must match every condition; anything else delegates to the original _C
entry point unchanged): group_size == 256, stochastic is False, input is a 2D CUDA
tensor with dtype in {float32, float16, bfloat16}, and K % 1024 == 0 with
1024 <= K <= 32768.

Env kill-switch: ANIMA_TURBO_NO_KERNEL=1 skips the build and shim install entirely.
"""
import logging
import os

_LOG_PREFIX = "[warp_fht]"
_SHIM_MARKER = "_anima_turbo_shim"

# comfy-kitchen version(s) this extraction was verified against. comfy_kitchen has no
# __version__ attribute in the tested release; version is read via importlib.metadata.
_TESTED_VERSIONS = {"0.2.22"}

_ELIGIBLE_DTYPES = None  # populated lazily once torch is imported


def _comfy_kitchen_version():
    try:
        import comfy_kitchen
        v = getattr(comfy_kitchen, "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version("comfy-kitchen")
    except Exception:
        return None


def _build_extension():
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "convrot_warp_quantize.cu")
    return load(
        name="anima_turbo_warp_fht",
        sources=[src],
        # Matches comfy-kitchen's own CUDA_NVCC_FLAGS (--use_fast_math): the quantize
        # divide is fp32 regardless of input dtype, and fast-math's approximate
        # reciprocal (vs IEEE division) is what makes this kernel's int8 output
        # bitwise-identical to comfy-kitchen's, not just numerically close.
        extra_cuda_cflags=["--use_fast_math"],
        verbose=False,
    )


def _default_wrap_for_dlpack(t):
    if t.requires_grad:
        t = t.detach()
    return t.__dlpack__(stream=-1)


def _install_shim(ext):
    import torch
    import comfy_kitchen.backends.cuda as ck

    global _ELIGIBLE_DTYPES
    _ELIGIBLE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

    c_module = getattr(ck, "_C", None)
    if c_module is None or not hasattr(c_module, "quantize_int8_rowwise_convrot64"):
        logging.warning(
            f"{_LOG_PREFIX} comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64 "
            f"not found, skipping shim install"
        )
        return

    orig = c_module.quantize_int8_rowwise_convrot64
    if getattr(orig, _SHIM_MARKER, False):
        logging.info(f"{_LOG_PREFIX} shim already installed in this process, skipping (idempotent no-op)")
        return

    wrap_for_dlpack = getattr(ck, "_wrap_for_dlpack", None)
    if wrap_for_dlpack is None:
        wrap_for_dlpack = _default_wrap_for_dlpack

    state = {"orig": orig, "active": True}

    def shim(*args, **kwargs):
        if not state["active"]:
            return state["orig"](*args, **kwargs)

        # Unpack in its own try/except: nothing below has touched any capsule yet,
        # so on failure the original, still-unconsumed args are safe to delegate
        # as-is.
        try:
            input_capsule, output_capsule, scales_capsule, group_size, stochastic, seed, stream_ptr = args
        except Exception:
            logging.warning(
                f"{_LOG_PREFIX} unexpected call signature from comfy_kitchen (arg unpack "
                f"failed); delegating this call to stock unchanged",
                exc_info=True,
            )
            return state["orig"](*args, **kwargs)

        # DLPack capsules are single-use (the exporter renames the capsule on
        # import) -- convert each exactly once, in positional order. If conversion
        # fails partway (e.g. a future comfy_kitchen passes a non-DLPack object in
        # one slot), the capsules already converted above are now consumed and
        # CANNOT be re-delegated as-is -- re-wrap each converted tensor fresh via
        # wrap_for_dlpack instead, and pass through the pristine original capsule
        # for any slot never reached.
        x = q = s = None
        try:
            x = torch.utils.dlpack.from_dlpack(input_capsule)
            q = torch.utils.dlpack.from_dlpack(output_capsule)
            s = torch.utils.dlpack.from_dlpack(scales_capsule)
        except Exception:
            logging.warning(
                f"{_LOG_PREFIX} capsule conversion failed; restoring stock "
                f"_C.quantize_int8_rowwise_convrot64 and disabling the shim for the "
                f"rest of this process",
                exc_info=True,
            )
            state["active"] = False
            setattr(c_module, "quantize_int8_rowwise_convrot64", state["orig"])
            recovered_input = wrap_for_dlpack(x) if x is not None else input_capsule
            recovered_output = wrap_for_dlpack(q) if q is not None else output_capsule
            recovered_scales = wrap_for_dlpack(s) if s is not None else scales_capsule
            return state["orig"](
                recovered_input, recovered_output, recovered_scales,
                group_size, stochastic, seed, stream_ptr,
            )

        eligible = (
            group_size == 256
            and not stochastic
            and x.dim() == 2
            and x.dtype in _ELIGIBLE_DTYPES
            and x.shape[1] % 1024 == 0
            and 1024 <= x.shape[1] <= 32768
        )
        if not eligible:
            return state["orig"](
                wrap_for_dlpack(x), wrap_for_dlpack(q), wrap_for_dlpack(s),
                group_size, stochastic, seed, stream_ptr,
            )

        try:
            # x/q/s are zero-copy DLPack views: writes to q/s land directly in the
            # caller's buffers, matching _C's in-place, return-None contract.
            ext.convrot_warp_quantize(x, q, s, stream_ptr)
            return None
        except Exception:
            logging.warning(
                f"{_LOG_PREFIX} kernel call failed, falling back to stock handler for this call",
                exc_info=True,
            )
            return state["orig"](
                wrap_for_dlpack(x), wrap_for_dlpack(q), wrap_for_dlpack(s),
                group_size, stochastic, seed, stream_ptr,
            )

    setattr(shim, _SHIM_MARKER, True)
    setattr(c_module, "quantize_int8_rowwise_convrot64", shim)
    logging.info(
        f"{_LOG_PREFIX} installed: quantize_int8_rowwise_convrot64 calls with group_size=256, "
        f"non-stochastic, K%1024==0, 1024<=K<=32768, dtype in {{fp32,fp16,bf16}} now route to the "
        f"standalone warp-FHT extension; all other calls fall through to comfy-kitchen's own "
        f"handler unchanged."
    )


def _main():
    if os.environ.get("ANIMA_TURBO_NO_KERNEL", "0") == "1":
        logging.info(f"{_LOG_PREFIX} ANIMA_TURBO_NO_KERNEL=1, skipping kernel build and shim install")
        return

    try:
        import comfy_kitchen  # noqa: F401
    except Exception as e:
        logging.warning(f"{_LOG_PREFIX} comfy_kitchen not importable, skipping ({e})")
        return

    version = _comfy_kitchen_version()
    if version is not None and version not in _TESTED_VERSIONS:
        logging.info(
            f"{_LOG_PREFIX} comfy_kitchen {version} is untested (verified against "
            f"{sorted(_TESTED_VERSIONS)}); installing anyway"
        )

    try:
        ext = _build_extension()
    except Exception as e:
        logging.warning(f"{_LOG_PREFIX} kernel build failed, comfy-kitchen will run stock: {e}")
        return

    try:
        _install_shim(ext)
    except Exception as e:
        logging.warning(f"{_LOG_PREFIX} shim install failed, comfy-kitchen will run stock: {e}")


_main()
