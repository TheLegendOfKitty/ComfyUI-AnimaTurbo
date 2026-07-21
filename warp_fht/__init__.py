"""
JIT-builds a standalone CUDA extension (convrot_warp_quantize.cu) that reproduces
comfy-kitchen's warp-per-group FHT ConvRot row-wise INT8 and INT4 quantize kernels,
then installs shims over comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64
and .quantize_int4_rowwise_convrot64 that route eligible calls to it. Ineligible calls,
and any failure at build or install time, fall through to comfy-kitchen's own stock
behavior unchanged -- this module either speeds up an eligible subset of calls or is a
complete no-op, never a correctness risk. Both shims share eligibility and capsule-
lifecycle handling (_make_shim); only the underlying kernel entry point differs.

Shared eligibility (must match every condition; anything else delegates to the original
_C entry point unchanged): group_size == 256, stochastic is False, input is a 2D CUDA
tensor with dtype in {float32, float16, bfloat16}, and K % 1024 == 0 with
1024 <= K <= 32768. The int8 shim uses exactly this. The int4 shim ANDs in a narrower,
microbenchmark-derived K band (see _INT4_MIN_WIN_K/_INT4_MAX_WIN_K below) -- outside
that band the standalone kernel measured slower than stock, so those calls correctly
keep delegating.

Env kill-switch: ANIMA_TURBO_NO_KERNEL=1 skips the build and shim install entirely.
"""
import logging
import os
import shutil

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


def _ensure_cuda_home():
    # torch.utils.cpp_extension.load() needs CUDA_HOME/CUDA_PATH to find nvcc's
    # toolkit tree; on distros that don't symlink /usr/local/cuda (e.g. Arch's
    # /opt/cuda) it's otherwise unset and the JIT build fails outright.
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    if os.path.isdir("/usr/local/cuda"):
        return
    if os.path.isdir("/opt/cuda"):
        os.environ["CUDA_HOME"] = "/opt/cuda"
        return
    nvcc = shutil.which("nvcc")
    if nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))


def _build_extension():
    # Must run before the cpp_extension import below: torch caches CUDA_HOME into a
    # module-level variable the first time torch.utils.cpp_extension is imported
    # anywhere in the process, so setting the env var after that import is a no-op.
    _ensure_cuda_home()
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


def _shim_eligible(x, group_size, stochastic):
    return (
        group_size == 256
        and not stochastic
        and x.dim() == 2
        and x.dtype in _ELIGIBLE_DTYPES
        and x.shape[1] % 1024 == 0
        and 1024 <= x.shape[1] <= 32768
    )


def _make_shim(c_module, attr_name, orig, wrap_for_dlpack, kernel_call, extra_eligible=None):
    """Builds the interception closure shared by the int8 and int4 _C entry points --
    same 7-arg (input, output, scales, group_size, stochastic, seed, stream_ptr) signature,
    same capsule lifecycle, same base eligibility. kernel_call(x, q, s, stream_ptr) is the
    only thing that differs between callers; it must write into q/s in place and return
    None, matching each stock entry point's own contract. extra_eligible(x), if given, is
    ANDed with the shared eligibility check -- for a caller-specific restriction (e.g. a
    measured-regression K band to exclude) without touching the other caller's criteria.
    """
    import torch

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
                f"{_LOG_PREFIX} unexpected call signature from comfy_kitchen for _C.{attr_name} "
                f"(arg unpack failed); delegating this call to stock unchanged",
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
                f"{_LOG_PREFIX} capsule conversion failed for _C.{attr_name}; restoring stock "
                f"handler and disabling this shim for the rest of this process",
                exc_info=True,
            )
            state["active"] = False
            setattr(c_module, attr_name, state["orig"])
            recovered_input = wrap_for_dlpack(x) if x is not None else input_capsule
            recovered_output = wrap_for_dlpack(q) if q is not None else output_capsule
            recovered_scales = wrap_for_dlpack(s) if s is not None else scales_capsule
            return state["orig"](
                recovered_input, recovered_output, recovered_scales,
                group_size, stochastic, seed, stream_ptr,
            )

        eligible = _shim_eligible(x, group_size, stochastic) and (extra_eligible is None or extra_eligible(x))
        if not eligible:
            return state["orig"](
                wrap_for_dlpack(x), wrap_for_dlpack(q), wrap_for_dlpack(s),
                group_size, stochastic, seed, stream_ptr,
            )

        try:
            # x/q/s are zero-copy DLPack views: writes to q/s land directly in the
            # caller's buffers, matching _C's in-place, return-None contract.
            kernel_call(x, q, s, stream_ptr)
            return None
        except Exception:
            logging.warning(
                f"{_LOG_PREFIX} kernel call failed for _C.{attr_name}, falling back to stock "
                f"handler for this call",
                exc_info=True,
            )
            return state["orig"](
                wrap_for_dlpack(x), wrap_for_dlpack(q), wrap_for_dlpack(s),
                group_size, stochastic, seed, stream_ptr,
            )

    setattr(shim, _SHIM_MARKER, True)
    return shim


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

    shim = _make_shim(
        c_module, "quantize_int8_rowwise_convrot64", orig, wrap_for_dlpack,
        lambda x, q, s, stream_ptr: ext.convrot_warp_quantize(x, q, s, stream_ptr),
    )
    setattr(c_module, "quantize_int8_rowwise_convrot64", shim)
    logging.info(
        f"{_LOG_PREFIX} installed: quantize_int8_rowwise_convrot64 calls with group_size=256, "
        f"non-stochastic, K%1024==0, 1024<=K<=32768, dtype in {{fp32,fp16,bf16}} now route to the "
        f"standalone warp-FHT extension; all other calls fall through to comfy-kitchen's own "
        f"handler unchanged."
    )


# Microbenchmarked (bf16, RTX 3090, K%1024==0 grid, idle GPU, >=50 iters): the warp
# kernel beats stock only where stock's OWN dispatch already uses its less-efficient
# block_threads_multi path (K>4096, K!=15360) -- at K<=4096 stock's dedicated
# block_threads_small path is already faster than the warp kernel (0.85-1.00x here,
# a regression at K=2048/3072/4096 specifically), and at K>=14336 the warp kernel's
# own register pressure (74 regs/thread for fp16/bf16, no half2 packing) drops
# occupancy enough to lose again (0.58-0.89x, worst at K=15360 where stock has its
# own dedicated fast PACK4 path). [5120, 12288] is the validated win band (1.12x-1.89x
# at tested points 5120/6144/8192/9216/10240/12288); this does not include the most
# frequent real shape in this model's w4a4 workload (K=2048), which correctly keeps
# delegating to stock unchanged.
_INT4_MIN_WIN_K = 5120
_INT4_MAX_WIN_K = 12288


def _int4_extra_eligible(x):
    return _INT4_MIN_WIN_K <= x.shape[1] <= _INT4_MAX_WIN_K


def _install_int4_shim(ext):
    import comfy_kitchen.backends.cuda as ck

    c_module = getattr(ck, "_C", None)
    if c_module is None or not hasattr(c_module, "quantize_int4_rowwise_convrot64"):
        logging.warning(
            f"{_LOG_PREFIX} comfy_kitchen.backends.cuda._C.quantize_int4_rowwise_convrot64 "
            f"not found, skipping int4 shim install"
        )
        return

    orig = c_module.quantize_int4_rowwise_convrot64
    if getattr(orig, _SHIM_MARKER, False):
        logging.info(f"{_LOG_PREFIX} int4 shim already installed in this process, skipping (idempotent no-op)")
        return

    wrap_for_dlpack = getattr(ck, "_wrap_for_dlpack", None)
    if wrap_for_dlpack is None:
        wrap_for_dlpack = _default_wrap_for_dlpack

    shim = _make_shim(
        c_module, "quantize_int4_rowwise_convrot64", orig, wrap_for_dlpack,
        lambda x, q, s, stream_ptr: ext.convrot_warp_quantize_int4(x, q, s, stream_ptr),
        extra_eligible=_int4_extra_eligible,
    )
    setattr(c_module, "quantize_int4_rowwise_convrot64", shim)
    logging.info(
        f"{_LOG_PREFIX} installed: quantize_int4_rowwise_convrot64 calls with group_size=256, "
        f"non-stochastic, K%1024==0, {_INT4_MIN_WIN_K}<=K<={_INT4_MAX_WIN_K}, dtype in {{fp32,fp16,bf16}} "
        f"now route to the standalone warp-FHT extension (narrower than the general K range: "
        f"outside this band the warp kernel measured slower than stock); all other calls fall "
        f"through to comfy-kitchen's own handler unchanged."
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
        logging.warning(f"{_LOG_PREFIX} int8 shim install failed, comfy-kitchen will run stock: {e}")

    try:
        _install_int4_shim(ext)
    except Exception as e:
        logging.warning(f"{_LOG_PREFIX} int4 shim install failed, comfy-kitchen will run stock: {e}")


_main()
