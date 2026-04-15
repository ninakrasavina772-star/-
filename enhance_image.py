"""
Скачивание изображения по URL и улучшение:
- neural: Real-ESRGAN x4 (ONNX; первый запуск скачивает ~33 МБ модели);
- pillow: классическая резкость + LANCZOS (быстрее, слабее).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from realesrgan_onnx import enhance_rgb


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ImageEnhance/1.0)",
            "Accept": "image/webp,image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def enhance(
    img: Image.Image,
    *,
    upscale: int = 2,
    unsharp_radius: float = 1.2,
    unsharp_percent: int = 180,
    unsharp_threshold: int = 3,
    sharpness: float = 1.15,
    contrast: float = 1.05,
) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")

    w, h = img.size
    if upscale > 1:
        img = img.resize((w * upscale, h * upscale), Image.Resampling.LANCZOS)

    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=unsharp_radius,
            percent=unsharp_percent,
            threshold=unsharp_threshold,
        )
    )
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def process_pil(
    im: Image.Image,
    *,
    mode: str,
    model_path: Path | None = None,
    tile: int = 400,
    pillow_upscale: int = 2,
) -> Image.Image:
    im = im.convert("RGB")
    if mode == "pillow":
        return enhance(im, upscale=max(1, pillow_upscale))
    rgb = np.array(im)
    out_arr = enhance_rgb(rgb, model_path=model_path, tile=tile)
    return Image.fromarray(out_arr)


def process_url_to_image(
    url: str,
    *,
    mode: str,
    model_path: Path | None = None,
    tile: int = 400,
    pillow_upscale: int = 2,
) -> Image.Image:
    url = url.strip()
    if not url:
        raise ValueError("Пустая ссылка")
    fd, tmp = tempfile.mkstemp(suffix=".img", prefix="dl_")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        download(url, tmp_path)
        with Image.open(tmp_path) as im:
            return process_pil(
                im,
                mode=mode,
                model_path=model_path,
                tile=tile,
                pillow_upscale=pillow_upscale,
            )
    finally:
        tmp_path.unlink(missing_ok=True)


def image_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser(description="Улучшение фото по URL (локально)")
    p.add_argument("url", help="HTTPS-ссылка на изображение")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Путь для сохранения (по умолчанию: enhanced_<имя>)",
    )
    p.add_argument(
        "--mode",
        choices=("neural", "pillow"),
        default="neural",
        help="neural: Real-ESRGAN x4 (ONNX); pillow: резкость+LANCZOS (по умолчанию: neural)",
    )
    p.add_argument(
        "--model",
        type=Path,
        help="Путь к model_fp16.onnx (если не задан — models/model_fp16.onnx, при отсутствии — скачивание)",
    )
    p.add_argument(
        "--tile",
        type=int,
        default=400,
        help="Размер тайла для нейросети (меньше — меньше RAM; 0 — вся картинка целиком)",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Только для pillow: масштаб перед резкостью (1 = без)",
    )
    args = p.parse_args()

    url = args.url.strip()
    name = url.rstrip("/").split("/")[-1] or "image.webp"
    if not name.lower().endswith((".webp", ".png", ".jpg", ".jpeg")):
        name += ".webp"

    out = args.output
    if out is None:
        stem = Path(name).stem
        suf = Path(name).suffix or ".webp"
        out = Path.cwd() / f"enhanced_{stem}{suf}"

    try:
        improved = process_url_to_image(
            url,
            mode=args.mode,
            model_path=args.model,
            tile=args.tile,
            pillow_upscale=max(1, args.upscale),
        )
        improved.save(out, quality=92)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
