"""
Real-ESRGAN x4 через ONNX Runtime (без PyTorch).
Модель: community ONNX на Hugging Face (imgdesignart/realesrgan-x4-onnx).
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_URL = (
    "https://huggingface.co/imgdesignart/realesrgan-x4-onnx/"
    "resolve/main/onnx/model_fp16.onnx"
)


def default_model_path() -> Path:
    base = Path(__file__).resolve().parent / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base / "model_fp16.onnx"


def ensure_model(path: Path | None = None, url: str = DEFAULT_MODEL_URL) -> Path:
    p = path or default_model_path()
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RealESRGAN-ONNX/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        p.write_bytes(r.read())
    return p


def _session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _pre_pad_reflect(img: np.ndarray, pre_pad: int) -> np.ndarray:
    """img: (1,3,H,W) float16/float32"""
    if pre_pad == 0:
        return img
    return np.pad(img, ((0, 0), (0, 0), (0, pre_pad), (0, pre_pad)), mode="reflect")


def _post_unpad(
    out: np.ndarray, pre_pad: int, mod_pad_h: int, mod_pad_w: int, scale: int
) -> np.ndarray:
    _, _, h, w = out.shape
    if mod_pad_h or mod_pad_w:
        out = out[:, :, : h - mod_pad_h * scale, : w - mod_pad_w * scale]
    if pre_pad != 0:
        _, _, h2, w2 = out.shape
        out = out[:, :, : h2 - pre_pad * scale, : w2 - pre_pad * scale]
    return out


def enhance_rgb(
    rgb: np.ndarray,
    *,
    model_path: Path | None = None,
    scale: int = 4,
    tile: int = 400,
    tile_pad: int = 10,
    pre_pad: int = 10,
) -> np.ndarray:
    """
    rgb: uint8 HxWx3 RGB
    returns: uint8 H'xW'x3 RGB (scale x4)
    """
    mp = ensure_model(model_path)
    sess = _session(mp)
    inp_name = sess.get_inputs()[0].name

    img = rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img, (2, 0, 1))
    batch = np.expand_dims(img_chw, 0)
    batch = _pre_pad_reflect(batch.astype(np.float16), pre_pad)

    _, _, h, w = batch.shape
    mod_pad_h = mod_pad_w = 0
    if scale == 2:
        ms = 2
    elif scale == 1:
        ms = 4
    else:
        ms = None
    if ms is not None:
        if h % ms != 0:
            mod_pad_h = ms - h % ms
        if w % ms != 0:
            mod_pad_w = ms - w % ms
        if mod_pad_h or mod_pad_w:
            batch = np.pad(
                batch,
                ((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)),
                mode="reflect",
            )

    _, _, height, width = batch.shape
    out_h, out_w = height * scale, width * scale
    output = np.zeros((1, 3, out_h, out_w), dtype=np.float32)

    if tile <= 0 or (height <= tile and width <= tile):
        out = sess.run(None, {inp_name: batch.astype(np.float16)})[0]
        out = out.astype(np.float32)
        out = _post_unpad(out, pre_pad, mod_pad_h, mod_pad_w, scale)
        out = np.clip(out.squeeze(0), 0.0, 1.0)
        out = np.transpose(out, (1, 2, 0))
        return (out * 255.0).round().astype(np.uint8)

    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile
            ofs_y = y * tile
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = batch[
                :,
                :,
                input_start_y_pad:input_end_y_pad,
                input_start_x_pad:input_end_x_pad,
            ].astype(np.float16)

            output_tile = sess.run(None, {inp_name: input_tile})[0].astype(np.float32)

            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            output[
                :,
                :,
                output_start_y:output_end_y,
                output_start_x:output_end_x,
            ] = output_tile[
                :,
                :,
                output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile,
            ]

    output = _post_unpad(output, pre_pad, mod_pad_h, mod_pad_w, scale)
    output = np.clip(output.squeeze(0), 0.0, 1.0)
    output = np.transpose(output, (1, 2, 0))
    return (output * 255.0).round().astype(np.uint8)
