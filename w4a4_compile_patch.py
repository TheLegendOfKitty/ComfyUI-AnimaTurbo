"""torch.compile support for ConvRot W4A4 (int4) quantized Linears.

comfy_kitchen.tensor.convrot_w4a4 registers `aten.linear.default` /
`aten.mm.default` / `aten.addmm.default` layout handlers for
TensorCoreConvRotW4A4Layout that all bottom out in raw CUDA-extension calls
with no `torch.library` registration, so dynamo cannot fake-tensor them:
TorchCompileModel hard-crashes at the first quantized Linear ("Cannot access
data pointer of Tensor (FakeTensor)"). Under dynamo's fake-tensor tracing,
`F.linear` on a >2D input decomposes to `t()` + `mm()` before any subclass
dispatch on `aten.linear.default` itself would run, so `mm.default` (and,
for a 2D+bias input, `addmm.default`) are the ops that actually need to be
covered, not just `linear.default`.

This registers a `torch.library.custom_op` that wraps the exact same
module-level `convrot_w4a4_linear()` call the stock handlers make (identical
registry dispatch, identical backend, zero numeric change) and repoints all
three layout dispatch table entries to route through it -- the same pattern
comfy-kitchen's own `int8_linear` custom op already uses, which is why the
int8 layout compiles cleanly.

Kill switch: ANIMA_TURBO_NO_W4A4_COMPILE=1 skips this patch (stock
comfy-kitchen dispatch; eager inference is unaffected either way).
"""
import logging
import os

_LOG_PREFIX = "[w4a4_compile_patch]"

try:
    import torch

    if os.environ.get("ANIMA_TURBO_NO_W4A4_COMPILE") == "1":
        logging.info(f"{_LOG_PREFIX} disabled via ANIMA_TURBO_NO_W4A4_COMPILE=1, stock dispatch kept")
    else:
        import comfy_kitchen.tensor.base as ck_base
        import comfy_kitchen.tensor.convrot_w4a4 as ck_w4a4
        from comfy_kitchen.tensor.base import QuantizedTensor
        from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout

        _LINEAR_OP = torch.ops.aten.linear.default
        _MM_OP = torch.ops.aten.mm.default
        _ADDMM_OP = torch.ops.aten.addmm.default
        # KeyError here (unexpected comfy-kitchen dispatch-table shape) is
        # caught by the outer except -- falls through to stock, single warning.
        _orig_linear_handler = ck_base._LAYOUT_DISPATCH_TABLE[_LINEAR_OP][TensorCoreConvRotW4A4Layout]
        _orig_mm_handler = ck_base._LAYOUT_DISPATCH_TABLE[_MM_OP][TensorCoreConvRotW4A4Layout]
        _orig_addmm_handler = ck_base._LAYOUT_DISPATCH_TABLE[_ADDMM_OP][TensorCoreConvRotW4A4Layout]

        if getattr(_orig_linear_handler, "_w4a4_compile_patched", False):
            logging.info(f"{_LOG_PREFIX} already installed in this process, skipping re-patch (idempotent no-op)")
        else:
            # Registered once per process. A second import of this module (e.g.
            # custom-node reload) finds the op already present in the
            # anima_turbo namespace and skips re-registering it (torch.library
            # raises on double registration of the same qualified name).
            if not hasattr(torch.ops.anima_turbo, "convrot_w4a4_linear"):

                @torch.library.custom_op("anima_turbo::convrot_w4a4_linear", mutates_args=())
                def _op_convrot_w4a4_linear(
                    x: torch.Tensor,
                    qweight: torch.Tensor,
                    wscales: torch.Tensor,
                    bias: torch.Tensor | None,
                    convrot_groupsize: int,
                    quant_group_size: int,
                    linear_dtype: str,
                ) -> torch.Tensor:
                    # Same module-level entry point the stock (uncompiled)
                    # handlers call: registry dispatch, backend selection, and
                    # the underlying CUDA kernels are all unchanged.
                    return ck_w4a4.convrot_w4a4_linear(
                        x,
                        qweight,
                        wscales,
                        bias=bias,
                        convrot_groupsize=convrot_groupsize,
                        quant_group_size=quant_group_size,
                        linear_dtype=linear_dtype,
                    )

                @_op_convrot_w4a4_linear.register_fake
                def _op_convrot_w4a4_linear_fake(
                    x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype
                ):
                    # qweight is int4-packed [N, K/2] -- out_features = N (dim 0).
                    # Output dtype always equals the activation dtype: every
                    # branch of the real implementation threads out_dtype=x.dtype
                    # straight through to its final GEMM+dequant call.
                    return torch.empty(*x.shape[:-1], qweight.shape[0], dtype=x.dtype, device=x.device)

            def _dispatch_via_op(input_tensor, weight, bias):
                qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
                params = weight._params
                return torch.ops.anima_turbo.convrot_w4a4_linear(
                    input_tensor,
                    qweight,
                    wscales,
                    bias,
                    params.convrot_groupsize,
                    params.quant_group_size,
                    params.linear_dtype,
                )

            def _handle_linear_compileable(qt, args, kwargs):
                input_tensor, weight = args[0], args[1]
                bias = args[2] if len(args) > 2 else None
                # Non-quantized weight, or a logically-transposed one: identical
                # to what the stock handler does for these same two cases
                # (dequantize-and-fall-back), so just defer to it unchanged.
                if not isinstance(weight, QuantizedTensor) or weight._params.transposed:
                    return _orig_linear_handler(qt, args, kwargs)
                if isinstance(input_tensor, QuantizedTensor):
                    input_tensor = input_tensor.dequantize()
                return _dispatch_via_op(input_tensor, weight, bias)

            def _handle_mm_compileable(qt, args, kwargs):
                a, b = args[0], args[1]
                # Not quantized, or not the transposed W.T view the GEMM kernel
                # expects (mm's RHS convention -- the opposite of linear's):
                # defer to stock, which dequantizes or raises exactly as before.
                if not isinstance(b, QuantizedTensor) or not b._params.transposed:
                    return _orig_mm_handler(qt, args, kwargs)
                if isinstance(a, QuantizedTensor):
                    a = a.dequantize()
                return _dispatch_via_op(a, b, None)

            def _handle_addmm_compileable(qt, args, kwargs):
                bias, a, b = args[0], args[1], args[2]
                if not isinstance(b, QuantizedTensor) or not b._params.transposed:
                    return _orig_addmm_handler(qt, args, kwargs)
                if isinstance(a, QuantizedTensor):
                    a = a.dequantize()
                return _dispatch_via_op(a, b, bias)

            _handle_linear_compileable._w4a4_compile_patched = True
            ck_base._LAYOUT_DISPATCH_TABLE[_LINEAR_OP][TensorCoreConvRotW4A4Layout] = _handle_linear_compileable
            ck_base._LAYOUT_DISPATCH_TABLE[_MM_OP][TensorCoreConvRotW4A4Layout] = _handle_mm_compileable
            ck_base._LAYOUT_DISPATCH_TABLE[_ADDMM_OP][TensorCoreConvRotW4A4Layout] = _handle_addmm_compileable
            logging.info(
                f"{_LOG_PREFIX} installed: ConvRot W4A4 Linear/mm/addmm now dispatch through a "
                f"registered torch.library custom op (anima_turbo::convrot_w4a4_linear) with a "
                f"fake-tensor kernel, fixing the FakeTensor data-pointer crash torch.compile hits "
                f"on the unregistered comfy-kitchen op. Set ANIMA_TURBO_NO_W4A4_COMPILE=1 to revert "
                f"to stock dispatch."
            )
except Exception as e:
    logging.warning(f"{_LOG_PREFIX} patch failed, using stock/underlying comfy-kitchen convrot dispatch: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
