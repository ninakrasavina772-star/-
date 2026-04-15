"""
Редактор фото: ссылка → «Улучшение» → JPG.
Локально: streamlit run app.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st

from enhance_image import image_to_jpeg_bytes, process_url_to_image

SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_name(url: str) -> str:
    part = url.rstrip("/").split("/")[-1] or "photo"
    stem = Path(part).stem
    stem = SAFE.sub("_", stem)[:80] or "photo"
    return f"{stem}_улучшено.jpg"


def _is_streamlit_cloud() -> bool:
    e = os.environ
    if e.get("STREAMLIT_CLOUD") == "1" or e.get("STREAMLIT_SHARING_MODE"):
        return True
    try:
        cwd = str(Path.cwd())
        if cwd.startswith("/mount") or "/mount/src/" in cwd:
            return True
    except OSError:
        pass
    return False


st.set_page_config(page_title="Редактор фото", layout="centered")

st.title("Редактор фото")
st.caption(
    "Вставьте прямую ссылку на изображение (https://…), нажмите «Улучшение», скачайте JPG."
)

if _is_streamlit_cloud():
    st.info(
        "В **Streamlit Cloud** нейросеть может не поместиться в память. "
        "Если была ошибка «Oh no», в боковой панели выберите **«Быстро»** или обновите приложение после загрузки правок с GitHub."
    )

url = st.text_input(
    "Ссылка на фото",
    placeholder="https://example.com/image.webp",
    label_visibility="visible",
)

_default_mode_idx = 1 if _is_streamlit_cloud() else 0

with st.sidebar:
    st.header("Дополнительно")
    mode = st.radio(
        "Режим",
        ("neural", "pillow"),
        format_func=lambda x: "Нейросеть (лучше качество, дольше)"
        if x == "neural"
        else "Быстро (без тяжёлой модели)",
        index=_default_mode_idx,
    )
    jpg_q = st.slider("Качество JPG", 75, 100, 90)
    tile = st.number_input(
        "Тайл (нейросеть)",
        0,
        800,
        256 if _is_streamlit_cloud() else 400,
        50,
        help="В облаке лучше 200–256. Меньше — меньше памяти.",
    )
    p_up = st.number_input("Масштаб (только «Быстро»)", 1, 4, 2)

run = st.button("Улучшение", type="primary", use_container_width=True)

if run:
    if not (url or "").strip():
        st.warning("Вставьте ссылку на картинку.")
    else:
        try:
            with st.spinner(
                "Загрузка и улучшение… Первый запуск нейросети может скачать модель (~33 МБ)."
            ):
                img = process_url_to_image(
                    url.strip(),
                    mode=mode,
                    model_path=None,
                    tile=int(tile),
                    pillow_upscale=int(p_up),
                )
                jpg = image_to_jpeg_bytes(img, quality=int(jpg_q))
            st.success("Готово.")
            st.image(img, use_container_width=True)
            st.download_button(
                label="Скачать JPG",
                data=jpg,
                file_name=_safe_name(url.strip()),
                mime="image/jpeg",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Не получилось: {e}")
            st.exception(e)

st.divider()
st.caption(
    "Публикация: [Streamlit Community Cloud](https://streamlit.io/cloud), файл приложения `app.py`."
)
