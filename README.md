# Редактор фото

Любой человек с **ссылкой на приложение** может вставить URL картинки → **Улучшение** → скачать **JPG**.

Два режима: **нейросеть** (Real-ESRGAN) и **быстро** (резкость + масштаб).

## 1. Streamlit Community Cloud (проще всего)

1. Загрузите в **корень** репозитория GitHub: `app.py`, `enhance_image.py`, `realesrgan_onnx.py`, `requirements.txt`.
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → ваш репозиторий → **Main file:** `app.py` → **Deploy**.
3. Поделитесь ссылкой вида `https://….streamlit.app`.

**Важно:** первый деплой может идти **10–20 минут** (ставится `onnxruntime`). Если сборка не завершается или приложение падает с «Oh no», используйте вариант 2 или режим **«Быстро»** в боковой панели.

## 2. Docker (Render, Railway, Fly.io и т.п.)

Подходит, если нужно **больше памяти** или стабильнее нейросеть.

- Создайте сервис **Web Service** из этого репозитория, **Dockerfile** уже в корне.
- Порт **8501**, команда из Dockerfile уже задана.
- Выдайте пользователям **публичный URL** хостинга.

## Локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

Откройте `http://localhost:8501`.
