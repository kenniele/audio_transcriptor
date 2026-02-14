# Audio Parser — Транскрипция аудио

Скрипт для транскрипции аудио-файлов (m4a, mp3, wav и др.) с максимальной точностью.

Использует **faster-whisper** (OpenAI Whisper на CTranslate2) с моделью `large-v3` и **Silero VAD** для фильтрации тишины.

## Требования

- Python 3.10+
- **ffmpeg** (должен быть установлен в системе)

### Установка ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Установка

```bash
pip install -r requirements.txt
```

> При первом запуске модель (`large-v3`, ~3 ГБ) скачивается автоматически.

## Использование

```bash
# Базовая транскрипция (автоопределение языка)
python transcribe.py audio.m4a

# С сохранением в файл
python transcribe.py audio.m4a -o result.txt

# С таймстампами
python transcribe.py audio.m4a -t

# Указать язык явно (повышает точность)
python transcribe.py audio.m4a -l ru

# Использовать модель поменьше (быстрее, менее точно)
python transcribe.py audio.m4a -m medium
```

## Параметры

| Флаг | Описание |
|---|---|
| `-o, --output` | Сохранить результат в файл |
| `-m, --model` | Модель: `tiny`, `base`, `small`, `medium`, `large-v3` (по умолчанию) |
| `-l, --language` | Код языка (`ru`, `en`, `de` и т.д.) |
| `-d, --device` | Устройство: `auto`, `cpu`, `cuda` |
| `--compute-type` | Тип вычислений: `auto`, `float16`, `int8`, `float32` |
| `-t, --timestamps` | Добавить таймстампы к сегментам |
