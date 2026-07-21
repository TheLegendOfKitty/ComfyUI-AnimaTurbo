"""
Fuses each quantized cosmos-predict2/Anima block's self-attention q/k/v
Linears into one concatenated-weight Linear via an object patch, so the
compiled graph runs one quantize + one GEMM per block instead of three. The
sibling eager-mode dispatch-cache patches in this directory dedupe the same
redundancy but are invisible to torch.compile; this node's fusion survives
compilation because it patches structure before the graph is traced. Place
this node AFTER any LoRA loaders and BEFORE TorchCompileModel -- LoRAs added
downstream of this node will not affect the fused q/k/v. Never mutates the
shared model: the replacement self_attn is installed via add_object_patch on
a clone, so sibling clones (and the original) are unaffected. Skips a block
if its q/k/v are not identically-shaped, identically-parameterized weights
all in the same supported layout (TensorWiseINT8Layout or
TensorCoreConvRotW4A4Layout), if a LoRA/patch targets any of them, or if it
is already fused. q_proj/k_proj/
v_proj stay registered on the replacement (unused by the fused compute_qkv)
so whole-model device moves keep them in lockstep with the rest of the block;
as a result a fused model's state_dict emits qkv_proj.* keys ALONGSIDE the
original q_proj/k_proj/v_proj.* keys, and round-trips through the stock
loader, which reports the extra qkv_proj.* keys as unexpected.
"""
import dataclasses
import logging
import types

import torch
from einops import rearrange

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor

_LOG_PREFIX = "[anima_qkv_fusion]"

# Per-instance hook registries nn.Module keeps in __dict__ (names vary slightly by
# torch version); forked in _clone_attn_shell so a future hook registration on the
# replacement can never leak into the original attn.
_HOOK_CONTAINER_NAMES = (
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


_INT8_LAYOUT = "TensorWiseINT8Layout"
_W4A4_LAYOUT = "TensorCoreConvRotW4A4Layout"
_SUPPORTED_LAYOUTS = (_INT8_LAYOUT, _W4A4_LAYOUT)

# Params fields (besides scale/orig_shape, checked separately) that must match
# across q/k/v for TensorWiseINT8Layout.
_INT8_PARAM_MATCH_FIELDS = ("convrot", "convrot_groupsize", "orig_dtype", "is_weight", "transposed")


def _patch_key(block_idx):
    return f"diffusion_model.blocks.{block_idx}.self_attn"


def _is_quantized_linear(mod):
    return (
        mod is not None
        and hasattr(mod, "weight")
        and isinstance(mod.weight, QuantizedTensor)
        and hasattr(mod, "in_features")
        and hasattr(mod, "out_features")
    )


def _normalize_scale(scale, out_features):
    """Normalize a weight scale to an explicit (out_features, 1) shape, so a
    scalar, a flat (out_features,) vector, or any other rank can never reach
    torch.cat sitting next to a differently-shaped scale from another
    projection."""
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(out_features, 1)
    return scale.reshape(out_features, 1)


def _build_fused_linear(q_proj, k_proj, v_proj):
    q_w, k_w, v_w = q_proj.weight, k_proj.weight, v_proj.weight
    q_out, k_out, v_out = q_proj.out_features, k_proj.out_features, v_proj.out_features
    in_features = q_proj.in_features
    out_features = q_out + k_out + v_out
    layout = q_w._layout_cls

    # Row order [q; k; v] -> GEMM output columns follow the same order, so a
    # plain chunk(3, dim=-1) on the fused output recovers q, k, v exactly.
    # Exact for w4a4 too: int4 packing runs along K (columns), two values per
    # byte, so concatenating whole output rows (dim 0) never splits a nibble pair.
    fused_qdata = torch.cat([q_w._qdata, k_w._qdata, v_w._qdata], dim=0)
    fused_scale = torch.cat(
        [
            _normalize_scale(q_w._params.scale, q_out),
            _normalize_scale(k_w._params.scale, k_out),
            _normalize_scale(v_w._params.scale, v_out),
        ],
        dim=0,
    )
    if layout == _W4A4_LAYOUT:
        # w4a4's GEMM validates wscales as exactly 1D -- this layout is always
        # row-wise (never per-tensor), unlike int8's kernel below, which
        # tolerates (and this fusion otherwise keeps) the (out_features, 1) shape.
        fused_scale = fused_scale.reshape(out_features)

    has_bias = q_proj.bias is not None
    fused_cls = type(q_proj)
    device = q_w._qdata.device
    compute_dtype = q_proj.factory_kwargs.get("dtype")

    fused = fused_cls(in_features, out_features, bias=has_bias, device=device, dtype=compute_dtype)

    fused_params = dataclasses.replace(q_w._params, scale=fused_scale, orig_shape=fused._orig_shape)
    fused.weight = torch.nn.Parameter(
        QuantizedTensor(fused_qdata, q_w._layout_cls, fused_params), requires_grad=False,
    )
    if has_bias:
        fused.bias = torch.nn.Parameter(
            torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0), requires_grad=False,
        )

    # Mirror the rest of what _load_quantized_module populates on a real load.
    fused.quant_format = q_proj.quant_format
    fused.layout_type = q_proj.layout_type
    fused._full_precision_mm_config = q_proj._full_precision_mm_config
    fused._full_precision_mm = q_proj._full_precision_mm
    fused.tensor_class = q_proj.tensor_class
    return fused


def _clone_attn_shell(attn):
    """Returns a new nn.Module instance carrying attn's identity/state but
    with its OWN _modules/_parameters/_buffers/_non_persistent_buffers_set
    (and hook registry) containers, so mutating the clone (adding submodules,
    registering hooks) can never reach back into attn's containers.
    Submodule/parameter/buffer VALUES are still shared references at this
    point -- only the containers are copies."""
    new = object.__new__(type(attn))
    new.__dict__ = dict(attn.__dict__)
    new._modules = dict(attn._modules)
    new._parameters = dict(attn._parameters)
    new._buffers = dict(attn._buffers)
    new._non_persistent_buffers_set = set(attn._non_persistent_buffers_set)
    for name in _HOOK_CONTAINER_NAMES:
        if name in attn.__dict__:
            setattr(new, name, type(attn.__dict__[name])(attn.__dict__[name]))
    return new


def _build_fused_attn(attn):
    """Builds a replacement self_attn object with q/k/v fused into qkv_proj.
    `attn` itself is left completely untouched."""
    fused_module = _build_fused_linear(attn.q_proj, attn.k_proj, attn.v_proj)

    new_attn = _clone_attn_shell(attn)
    new_attn.qkv_proj = fused_module
    # q_proj/k_proj/v_proj stay registered on the replacement -- same module objects
    # as attn's, shared by reference -- even though compute_qkv below never calls
    # them. Invariant: while this object patch is installed, a whole-model
    # .to()/_apply() traversal (e.g. ModelPatcher.unpatch_model calls
    # self.model.to(device_to) BEFORE restoring object_patches_backup) must still
    # reach these modules through the replacement, or they get stranded on a stale
    # device once the original attn rejoins the tree after the patch is reverted.
    new_attn._qkv_fused = True
    new_attn.compute_qkv = types.MethodType(_fused_compute_qkv, new_attn)
    return new_attn


def _fused_compute_qkv(self, x, context=None, rope_emb=None, transformer_options={}):
    """Instance-bound replacement for Attention.compute_qkv on a fused
    self-attn replacement object. Post-projection logic (reshape, norms,
    rope) is byte-for-byte predict2.py's Attention.compute_qkv; only the
    projection step changes."""
    context = x if context is None else context
    q_input = x
    k_input = context
    v_input = context

    transformer_patches = transformer_options.get("patches", {})
    patch_name = "attn1_patch" if self.is_selfattn else "attn2_patch"
    if patch_name in transformer_patches:
        extra_options = transformer_options.copy()
        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.head_dim
        for patch in transformer_patches[patch_name]:
            out = patch(q_input, k_input, v_input, pe=rope_emb, attn_mask=None, extra_options=extra_options)
            q_input = out.get("q", q_input)
            k_input = out.get("k", k_input)
            v_input = out.get("v", v_input)
            rope_emb = out.get("pe", rope_emb)

    if q_input is k_input and k_input is v_input:
        # The only case for self-attn with no attn1_patch touching q/k/v (the common,
        # fast path this fusion exists for): one shared input, one fused GEMM.
        qkv = self.qkv_proj(q_input)
        q, k, v = qkv.chunk(3, dim=-1)
    else:
        # A patch made q/k/v diverge. qkv_proj's weight rows are unchanged, so
        # projecting each input through the fused Linear and keeping only its own
        # chunk is still exact -- just without the fusion's call-count savings.
        q = self.qkv_proj(q_input).chunk(3, dim=-1)[0]
        k = self.qkv_proj(k_input).chunk(3, dim=-1)[1]
        v = self.qkv_proj(v_input).chunk(3, dim=-1)[2]

    q, k, v = map(
        lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )

    def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            q_scale, _, q_offload_stream = comfy.ops.cast_bias_weight(self.q_norm, q, offloadable=True)
            k_scale, _, k_offload_stream = comfy.ops.cast_bias_weight(self.k_norm, k, offloadable=True)
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_emb, q_scale, k_scale, self.q_norm.eps)
            comfy.ops.uncast_bias_weight(self.q_norm, q_scale, None, q_offload_stream)
            comfy.ops.uncast_bias_weight(self.k_norm, k_scale, None, k_offload_stream)
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

    return q, k, v


def _check_eligible(attn):
    """Pure eligibility check (no mutation). Returns (True, None) or
    (False, reason)."""
    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)

    for name, proj in (("q_proj", q_proj), ("k_proj", k_proj), ("v_proj", v_proj)):
        if not _is_quantized_linear(proj):
            return False, f"self_attn.{name} missing or not a quantized Linear"

    q_w, k_w, v_w = q_proj.weight, k_proj.weight, v_proj.weight
    layout = q_w._layout_cls
    if not (layout == k_w._layout_cls == v_w._layout_cls):
        return False, "mismatched layout across q/k/v"
    if layout not in _SUPPORTED_LAYOUTS:
        return False, f"unsupported layout for fusion: {layout}"

    if not (q_proj.in_features == k_proj.in_features == v_proj.in_features):
        return False, "mismatched in_features across q/k/v"

    if not (q_proj.out_features == k_proj.out_features == v_proj.out_features):
        # chunk(3, dim=-1) after the fused GEMM assumes equal output widths.
        return False, "mismatched out_features across q/k/v"

    if type(q_proj) is not type(k_proj) or type(k_proj) is not type(v_proj):
        return False, "mismatched Linear module class across q/k/v"

    if (
        len(q_proj.weight_function) or len(k_proj.weight_function) or len(v_proj.weight_function)
        or len(q_proj.bias_function) or len(k_proj.bias_function) or len(v_proj.bias_function)
    ):
        # A weight/bias-patching function is already live on the module instance --
        # fusing now would bake a stale weight and silently drop it.
        return False, "active weight_function/bias_function on q/k/v"

    biases_present = (q_proj.bias is not None, k_proj.bias is not None, v_proj.bias is not None)
    if any(biases_present) and not all(biases_present):
        return False, "bias present on some but not all of q/k/v"

    q_p, k_p, v_p = q_w._params, k_w._params, v_w._params
    if layout == _INT8_LAYOUT:
        match_fields = _INT8_PARAM_MATCH_FIELDS
    else:
        # TensorCoreConvRotW4A4Layout: every Params field except the per-row
        # scale and orig_shape, both of which legitimately differ with out_features.
        match_fields = tuple(f.name for f in dataclasses.fields(q_p) if f.name not in ("scale", "orig_shape"))
    for field in match_fields:
        qf, kf, vf = getattr(q_p, field, None), getattr(k_p, field, None), getattr(v_p, field, None)
        if not (qf == kf == vf):
            return False, f"mismatched weight param '{field}' across q/k/v"

    q_dtype = q_proj.factory_kwargs.get("dtype")
    if q_dtype != k_proj.factory_kwargs.get("dtype") or q_dtype != v_proj.factory_kwargs.get("dtype"):
        return False, "mismatched compute dtype across q/k/v"

    if q_proj.quant_format != k_proj.quant_format or q_proj.quant_format != v_proj.quant_format:
        return False, "mismatched quant_format across q/k/v"

    return True, None


def _lora_active_on_qkv(m, block_idx):
    """True if any LoRA/weight patch (comfy's standard additive-patch
    mechanism, ModelPatcher.patches) targets this block's self_attn
    q_proj/k_proj/v_proj. Those patches are applied by key against the
    ORIGINAL module names at patch_model() time; fusing here would remove
    the keys they target and silently drop them."""
    patches = getattr(m, "patches", None)
    if not patches:
        return False
    prefix = _patch_key(block_idx) + "."
    targets = (prefix + "q_proj.", prefix + "k_proj.", prefix + "v_proj.")
    return any(key.startswith(targets) for key in patches)


def _quantized_linear_bytes(mod):
    """Resident bytes of a quantized Linear's weight (+scale) and bias."""
    w = mod.weight
    n = w._qdata.numel() * w._qdata.element_size()
    n += w._params.scale.numel() * w._params.scale.element_size()
    if mod.bias is not None:
        n += mod.bias.numel() * mod.bias.element_size()
    return n


def _get_diffusion_model(m):
    base_model = getattr(m, "model", None)
    if base_model is None:
        return None
    object_patches = getattr(m, "object_patches", {})
    if "diffusion_model" in object_patches:
        return object_patches["diffusion_model"]
    return getattr(base_model, "diffusion_model", None)


def _fuse_block(m, block_idx):
    patch_key = _patch_key(block_idx)
    already_patched = patch_key in m.object_patches
    attn = m.get_model_object(patch_key)

    if already_patched:
        if getattr(attn, "_qkv_fused", False):
            return "already_fused"
        logging.debug(f"{_LOG_PREFIX} block {block_idx}: skipping (self_attn already object-patched by something else)")
        return "skipped"

    if attn is None or not hasattr(attn, "q_proj"):
        logging.debug(f"{_LOG_PREFIX} block {block_idx}: skipping (no self_attn.q_proj)")
        return "skipped"

    if _lora_active_on_qkv(m, block_idx):
        logging.debug(f"{_LOG_PREFIX} block {block_idx}: skipping (LoRA/patch targets q/k/v weight)")
        return "skipped"

    eligible, reason = _check_eligible(attn)
    if not eligible:
        logging.debug(f"{_LOG_PREFIX} block {block_idx}: skipping ({reason})")
        return "skipped"

    new_attn = _build_fused_attn(attn)
    m.add_object_patch(patch_key, new_attn)
    return "fused"


class AnimaFuseQKV:
    CATEGORY = "model_patches"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def execute(self, model):
        m = model.clone()
        diffusion_model = _get_diffusion_model(m)
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            logging.info(
                f"{_LOG_PREFIX} model has no model.diffusion_model.blocks (not a cosmos-predict2/Anima "
                f"model); leaving model unchanged"
            )
            return (m,)

        n_blocks = len(blocks)
        fused = already = skipped = 0
        fused_block_indices = []
        for i in range(n_blocks):
            try:
                status = _fuse_block(m, i)
            except Exception:
                logging.warning(f"{_LOG_PREFIX} block {i}: skipping (unexpected error during fusion)", exc_info=True)
                status = "skipped"
            if status == "fused":
                fused += 1
                fused_block_indices.append(i)
            elif status == "already_fused":
                already += 1
            else:
                skipped += 1

        if fused_block_indices:
            # object patches add resident bytes (kept q/k/v + the new qkv_proj copy)
            # that model_size()'s cache -- populated once, from the pre-fusion tree --
            # predates and never gets invalidated by add_object_patch. Read the cache
            # first (pre-fusion baseline), then correct it by the exact newly-resident
            # delta so downstream VRAM-budget decisions (e.g. partially_load) see the
            # true footprint.
            base_size = m.model_size()
            delta = sum(
                _quantized_linear_bytes(m.object_patches[_patch_key(i)].qkv_proj)
                for i in fused_block_indices
            )
            m.size = base_size + delta

        logging.info(
            f"{_LOG_PREFIX} fused {fused}/{n_blocks} self-attn block(s) "
            f"({already} already fused, {skipped} skipped)"
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaFuseQKV": AnimaFuseQKV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaFuseQKV": "Anima Fuse Self-Attn QKV",
}
