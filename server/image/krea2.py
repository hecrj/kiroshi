"""Krea 2 (``Krea2Pipeline``) single-file support.

Krea 2 models are delivered as a single transformer checkpoint (ComfyUI key
layout) in one of three forms:

- plain bf16 weights, loadable directly through ``Krea2Pipeline.from_single_file``;
- ComfyUI-quants "Int8Extended" checkpoints: int8 ``*.weight`` + f32
  ``*.weight_scale`` + ``*.comfy_quant`` JSON sidecars, with a regular
  Hadamard rotation baked into the weights (``convrot``);
- NVFP4 checkpoints: packed 4-bit ``*.weight`` (uint8, two codes per byte) +
  f8e4m3 block scales ``*.weight_scale`` + f32 tensor scale
  ``*.weight_scale_2`` + ``*.comfy_quant`` sidecars.

Quantized checkpoints are run **packed at runtime** (no dequantization step,
no dequantized copy on disk), the same way ComfyUI runs them via
``comfy-kitchen``:

- int8 layers become fused W8A8 ``Linear`` replacements
  (``comfy_kitchen.int8_linear``; CUDA sm_75+, eager fallback on CPU);
- nvfp4 layers stay packed and unpack per-forward with the fast
  ``comfy_kitchen.dequantize_nvfp4`` kernel before a full-precision GEMM
  (honoring the checkpoint's ``full_precision_matrix_mult``).

LoRA fusion needs plain ``nn.Linear`` modules, so when LoRAs are requested
the transformer is loaded dequantized instead: int8 files go through the
streaming on-disk dequantization cache (:func:`cached_dequantized_file`),
any other mix is dequantized in memory.

The int8 dequantization follows the reference layout from
``Comfy-Org/comfy-kitchen`` (``TensorWiseINT8Layout``): a normalized regular
Hadamard rotation (Kronecker powers of a 4x4 base) applied per row in groups
of ``convrot_groupsize``, then row-wise int8 quantization.
"""

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import safetensors
import torch

#: Marker keys identifying a Krea 2 transformer checkpoint (raw or
#: ``model.diffusion_model.``-prefixed ComfyUI layout).
KREA2_MARKER_KEYS = (
    "txtfusion.projector.weight",
    "model.diffusion_model.txtfusion.projector.weight",
)

_DTYPE_NAMES = {
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.bfloat16: "BF16",
}
_NAME_DTYPES = {v: k for k, v in _DTYPE_NAMES.items()}

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


# ---------------------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------------------


def _header(path: str) -> dict:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length))


def is_krea2_file(path: str) -> bool:
    """Whether a single-file checkpoint is a Krea 2 transformer."""
    return any(key in KREA2_MARKER_KEYS for key in _header(path) if key != "__metadata__")


def quantized_layers(header: dict) -> list[str]:
    """ComfyUI base keys of the quantized layers (``*.comfy_quant`` sidecars)."""
    return [key[: -len(".comfy_quant")] for key in header if key.endswith(".comfy_quant")]


def is_quantized(header: dict) -> bool:
    return any(key.endswith(".comfy_quant") for key in header if key != "__metadata__")


def _specs(path: str) -> dict[str, dict]:
    """ComfyUI base key -> parsed ``comfy_quant`` spec."""
    st = safetensors.safe_open(path, framework="pt")
    out: dict[str, dict] = {}
    for key in st.keys():
        if key.endswith(".comfy_quant"):
            out[key[: -len(".comfy_quant")]] = json.loads(st.get_tensor(key).numpy().tobytes().decode())
    return out


# ---------------------------------------------------------------------------
# int8 dequantization (reference-implementation semantics)
# ---------------------------------------------------------------------------


def _build_hadamard(size: int) -> torch.Tensor:
    """Normalized regular orthogonal Hadamard matrix (power of 4)."""
    h = _HADAMARD_CACHE.get(size)
    if h is None:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
        h4 = torch.tensor(
            [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        h = h4
        current = 4
        while current < size:
            h = torch.kron(h, h4)
            current *= 4
        h = h / size**0.5
        _HADAMARD_CACHE[size] = h
    return h


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, convrot: bool, group_size: int) -> torch.Tensor:
    """Dequantize a tensor-wise int8 weight back to floats (f32)."""
    w = q.to(torch.float32) * scale.to(torch.float32)
    if convrot:
        out_f, in_f = w.shape
        n_groups = in_f // group_size
        h = _build_hadamard(group_size)
        w = torch.matmul(w.reshape(out_f, n_groups, group_size), h).reshape(out_f, in_f)
    return w


# ---------------------------------------------------------------------------
# Packed quantized Linear (native runtime)
# ---------------------------------------------------------------------------


class QuantLinear(torch.nn.Module):
    """Drop-in ``nn.Linear`` replacement holding packed quantized weights.

    The packed/scale parameters never change dtype (device moves only, via
    :meth:`_apply`), so ``module.to("cuda")`` / offload hooks work as usual.
    ``state_dict`` exposes the packed layout (``weight``/``scale``/
    ``scale_2``); a pipeline with these modules is not meant to be saved or
    reloaded through ``from_pretrained``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        spec: dict,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format = spec.get("format", "int8_tensorwise")
        self.convrot = bool(spec.get("convrot"))
        self.group_size = int(spec.get("convrot_groupsize", 256))

        if self.format == "int8_tensorwise":
            self.weight = torch.nn.Parameter(
                torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False
            )
            self.scale = torch.nn.Parameter(torch.empty(out_features, dtype=torch.float32), requires_grad=False)
        elif self.format == "nvfp4":
            self.weight = torch.nn.Parameter(
                torch.empty(out_features, in_features // 2, dtype=torch.uint8), requires_grad=False
            )
            self.scale = torch.nn.Parameter(
                torch.empty(out_features, in_features // 16, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.scale_2 = torch.nn.Parameter(torch.empty((), dtype=torch.float32), requires_grad=False)
        else:
            raise ValueError(f"Unsupported quantization format: {self.format!r}")

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn):
        # Packed weights / fp8 block scales: move device only, never cast dtype
        # (the cast would corrupt the codes). ``fn`` is probed with a
        # zero-element tensor to discover the target device.
        for name, param in list(self._parameters.items()):
            if param is None:
                continue
            if name in ("weight", "scale", "scale_2"):
                probe = fn(param.detach().new_empty(0))
                param.data = param.data.to(probe.device)
            else:
                param.data = fn(param.data)
        for name, buf in list(self._buffers.items()):
            if buf is not None:
                self._buffers[name] = fn(buf)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck

        if self.format == "int8_tensorwise":
            return ck.int8_linear(
                x,
                self.weight,
                self.scale,
                bias=self.bias,
                out_dtype=x.dtype,
                convrot=self.convrot,
                convrot_groupsize=self.group_size,
            )
        # nvfp4: unpack to the compute dtype (fast kernel) then full-precision GEMM
        weight = ck.dequantize_nvfp4(self.weight, self.scale_2, self.scale, output_type=x.dtype)
        return torch.nn.functional.linear(x, weight, self.bias)


# ---------------------------------------------------------------------------
# Pipeline / transformer loading
# ---------------------------------------------------------------------------


def krea2_pipeline_cls():
    """``Krea2Pipeline`` with single-file loading.

    diffusers 0.41 registers ``Krea2Pipeline`` without
    ``FromSingleFileMixin``, so ``from_single_file`` is missing; mix it in
    (SDXL-style) when absent. Subclassing keeps ``isinstance`` checks
    against ``Krea2Pipeline`` working.
    """
    from diffusers import Krea2Pipeline

    if hasattr(Krea2Pipeline, "from_single_file"):
        return Krea2Pipeline

    from diffusers.loaders.single_file import FromSingleFileMixin

    class Krea2SingleFilePipeline(Krea2Pipeline, FromSingleFileMixin):
        pass

    return Krea2SingleFilePipeline


def _load_transformer(single_file: str, config_dir: str, dtype: torch.dtype, packed: bool) -> "Krea2Transformer2DModel":
    """Build the transformer from a single-file checkpoint.

    ``packed=True`` keeps quantized layers packed (:class:`QuantLinear`);
    ``packed=False`` dequantizes them in memory (needed for LoRA fusion,
    which requires plain ``nn.Linear`` modules).
    """
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_krea2_transformer_checkpoint_to_diffusers

    state = convert_krea2_transformer_checkpoint_to_diffusers(safetensors.torch.load_file(single_file))

    # diffusers linear path -> spec (the sidecars go through the converter too)
    quant = {
        key[: -len(".comfy_quant")]: json.loads(value.numpy().tobytes().decode())
        for key, value in state.items()
        if key.endswith(".comfy_quant")
    }

    with init_empty_weights():
        transformer = Krea2Transformer2DModel.from_config(
            Krea2Transformer2DModel.load_config(str(Path(config_dir) / "transformer")),
            torch_dtype=dtype,
        )

    if packed and quant:
        modules = dict(transformer.named_modules())
        for linear_path, spec in quant.items():
            module = modules.get(linear_path)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"Expected a Linear at {linear_path!r}, got {type(module).__name__}")
            parent_path, _, child = linear_path.rpartition(".")
            setattr(
                modules[parent_path],
                child,
                QuantLinear(module.in_features, module.out_features, spec, bias=module.bias is not None),
            )

        load = {}
        for key, value in state.items():
            if key.endswith(".comfy_quant"):
                continue
            if key.endswith(".weight_scale_2"):
                key = key[: -len(".weight_scale_2")] + ".scale_2"
            elif key.endswith(".weight_scale"):
                key = key[: -len(".weight_scale")] + ".scale"
                if value.dim() == 2 and value.shape[1] == 1:  # per-row [out, 1] -> [out]
                    value = value.reshape(-1)
            load[key] = value
    else:
        if quant:
            modules = dict(transformer.named_modules())
            for linear_path, spec in quant.items():
                module = modules[linear_path]
                q = state[linear_path + ".weight"]
                scale = state.pop(linear_path + ".weight_scale")
                scale_2 = state.pop(linear_path + ".weight_scale_2", None)
                if spec.get("format") == "int8_tensorwise":
                    w = dequantize_int8(
                        q, scale, bool(spec.get("convrot")), int(spec.get("convrot_groupsize", 256))
                    ).to(dtype)
                elif spec.get("format") == "nvfp4":
                    import comfy_kitchen as ck

                    w = ck.dequantize_nvfp4(q, scale_2, scale, output_type=dtype)
                    # dequant returns the (possibly 16-aligned) padded shape
                    w = w[: module.weight.shape[0], : module.weight.shape[1]]
                else:
                    raise ValueError(f"Unsupported quantization format: {spec.get('format')!r}")
                state[linear_path + ".weight"] = w
            load = {key: value for key, value in state.items() if not key.endswith(".comfy_quant")}

    transformer.load_state_dict(load, assign=True)
    return transformer


def load_krea2_pipeline(
    single_file: str,
    dtype: torch.dtype,
    config_dir: str = "templates/krea-2",
    lora: bool = False,
) -> "Krea2Pipeline":
    """Load a Krea 2 pipeline from a transformer-only single file.

    The single file contains only the transformer (ComfyUI key layout); the
    VAE, text encoder, tokenizer and scheduler come from the
    ``templates/krea-2`` reference dir (the hecrj/krea-2 submodule) and are
    loaded directly (diffusers' single-file loader cannot load an
    ``AutoencoderKLQwenImage`` component from a config dir).

    Plain bf16 files use ``Krea2Pipeline.from_single_file``. Quantized files
    are loaded with the quantized layers kept packed at runtime (no
    dequantization), unless ``lora`` is set — LoRA fusion requires plain
    ``nn.Linear`` modules, so int8 files then go through the streaming
    dequantization cache and other mixes are dequantized in memory.
    """
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
    from transformers import AutoTokenizer, Qwen3VLModel

    required = (
        Path(config_dir) / "text_encoder" / "model-00001-of-00002.safetensors",
        Path(config_dir) / "vae" / "diffusion_pytorch_model.safetensors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Krea 2 reference weights missing: {', '.join(missing)}. The server "
            "image was likely built with WITH_KREA2=0 (SDXL only); rebuild with "
            "WITH_KREA2=1 or clone hecrj/krea-2 into templates/krea-2."
        )

    vae = AutoencoderKLQwenImage.from_pretrained(
        config_dir, subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    text_encoder = Qwen3VLModel.from_pretrained(
        config_dir, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config_dir, subfolder="tokenizer", local_files_only=True)

    index = json.loads(Path(config_dir, "model_index.json").read_text())
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        str(Path(config_dir) / "scheduler"), local_files_only=True
    )

    header = _header(single_file)

    if not is_quantized(header):
        return krea2_pipeline_cls().from_single_file(
            single_file,
            config=config_dir,
            use_safetensors=True,
            torch_dtype=dtype,
            local_files_only=True,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

    specs = _specs(single_file)

    if lora and all(spec.get("format") == "int8_tensorwise" for spec in specs.values()):
        # Plain-bf16 dequantized file (streamed once, then cached on disk)
        return krea2_pipeline_cls().from_single_file(
            cached_dequantized_file(single_file, dtype=dtype),
            config=config_dir,
            use_safetensors=True,
            torch_dtype=dtype,
            local_files_only=True,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

    transformer = _load_transformer(single_file, config_dir, dtype, packed=not lora)
    return Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        text_encoder_select_layers=index.get("text_encoder_select_layers"),
        is_distilled=index.get("is_distilled", False),
        patch_size=index.get("patch_size", 2),
    )


# ---------------------------------------------------------------------------
# Streaming dequantization cache (int8 -> plain bf16/fp16 on disk)
# ---------------------------------------------------------------------------


def _torch_dtype(name: str) -> torch.dtype:
    return _NAME_DTYPES[name]


def _write_tensor(f, tensor: torch.Tensor, offset: int) -> None:
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        # numpy has no bfloat16; bit-cast to int16 (same size)
        tensor = tensor.view(torch.int16)
    f.seek(offset)
    tensor.numpy().tofile(f)


def _size(dtype: torch.dtype, shape: list[int]) -> int:
    return math.prod(shape) * torch.tensor([], dtype=dtype).element_size()


def _dequantize_checkpoint(path: str, dst: Path, dtype: torch.dtype) -> None:
    """Stream ``path`` to ``dst``, dequantizing every quantized weight.

    Only int8 (``convrot``) layouts are supported; the output file is a plain
    safetensors checkpoint (ComfyUI key layout) in ``dtype`` with the
    quantization sidecars dropped.
    """
    header = _header(path)
    entries = {k: v for k, v in header.items() if k != "__metadata__"}

    st = safetensors.safe_open(path, framework="pt")

    # base name -> (weight info, scale info)
    quantized: dict[str, dict] = {}
    for key, info in entries.items():
        if key.endswith(".comfy_quant"):
            base = key[: -len(".comfy_quant")]
            spec = json.loads(st.get_tensor(key).numpy().tobytes().decode())
            if spec.get("format") != "int8_tensorwise":
                raise ValueError(
                    f"{base} uses format {spec.get('format')!r}; only int8_tensorwise "
                    "can be dequantized to a plain checkpoint"
                )
            quantized[base] = {
                "weight": entries.get(base + ".weight"),
                "scale": entries.get(base + ".weight_scale"),
                "convrot": bool(spec.get("convrot")),
                "group_size": int(spec.get("convrot_groupsize", 256)),
            }

    # Output plan, in source order: dequantized weights first (replacing the
    # weight/scale/comfy_quant triple), then all other tensors.
    plan: list[tuple[str, torch.dtype, list[int], str | None]] = []  # (name, dtype, shape, base)
    seen: set[str] = set()
    weight_keys = {b + ".weight" for b in quantized}
    scale_keys = {b + ".weight_scale" for b in quantized}
    for key, info in entries.items():
        if key.endswith(".comfy_quant"):
            continue
        base = key[: -len(".weight")] if key in weight_keys else None
        if base is not None:
            if base in seen:
                continue
            seen.add(base)
            plan.append((base + ".weight", dtype, list(quantized[base]["weight"]["shape"]), base))
            continue
        if key in scale_keys:
            continue
        plan.append((key, _torch_dtype(info["dtype"]), list(info["shape"]), None))

    # Precompute offsets so the header can be finalized after a single pass.
    sizes = [_size(dtype, shape) for _, dtype, shape, _ in plan]
    offsets = []
    cursor = 0
    for n in sizes:
        offsets.append((cursor, cursor + n))
        cursor += n

    header_out = {
        name: {"dtype": _DTYPE_NAMES[dtype], "shape": shape, "data_offsets": list(offset)}
        for (name, dtype, shape, _), offset in zip(plan, offsets)
    }
    header_json = json.dumps(header_out).encode()

    n_quant = sum(1 for p in plan if p[3] is not None)
    start = time.time()
    print(f"Dequantizing {n_quant} int8 layers -> {dst.name}...")

    with open(dst, "wb") as f:
        f.write(len(header_json).to_bytes(8, "little"))
        header_start = f.tell()
        f.write(b"\x00" * len(header_json))
        data_start = f.tell()
        for i, (name, dtype, shape, base) in enumerate(plan):
            if base is None:
                tensor = st.get_tensor(name)
            else:
                spec = quantized[base]
                q = st.get_tensor(base + ".weight")
                scale = st.get_tensor(base + ".weight_scale")
                tensor = dequantize_int8(q, scale, spec["convrot"], spec["group_size"]).to(dtype)
            _write_tensor(f, tensor, data_start + offsets[i][0])
            if (i + 1) % 32 == 0 or i + 1 == len(plan):
                print(f"  {i + 1}/{len(plan)} tensors ({time.time() - start:.0f}s)")

        f.seek(header_start)
        f.write(header_json)

    print(f"Dequantization done: {time.time() - start:.0f}s")


def cache_dir() -> Path:
    override = os.environ.get("KIROSHI_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "kiroshi"
    return base / "dequant"


def cached_dequantized_file(path: str, dtype: torch.dtype = torch.bfloat16) -> str:
    """Path to a plain (dequantized) single-file checkpoint for ``path``.

    Unquantized Krea 2 checkpoints are returned as-is. Int8-quantized ones
    are dequantized into a cache file, invalidated when the source changes.
    Used by the LoRA path (LoRA fusion needs plain ``nn.Linear`` modules).
    """
    header = _header(path)
    if not is_quantized(header):
        return path

    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{Path(path).stem}-{hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]}"
    dst = directory / f"{name}.k2.safetensors"
    sidecar = directory / f"{name}.k2.json"

    try:
        state = json.loads(sidecar.read_text())
        if (
            state.get("size") == os.path.getsize(path)
            and state.get("mtime") == int(Path(path).stat().st_mtime_ns)
            and state.get("dtype") == str(dtype)
        ):
            return str(dst)
    except (OSError, json.JSONDecodeError):
        pass

    tmp = directory / f"{dst.name}.tmp"
    _dequantize_checkpoint(path, tmp, dtype)
    tmp.replace(dst)
    sidecar.write_text(
        json.dumps(
            {
                "size": os.path.getsize(path),
                "mtime": int(Path(path).stat().st_mtime_ns),
                "dtype": str(dtype),
            }
        )
    )
    return str(dst)
