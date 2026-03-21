#!/usr/bin/env python3
"""
Транскрипция аудио и видео файлов с максимальной точностью.
Опционально — диаризация (разделение по спикерам).

Используется faster-whisper (CTranslate2) с моделью large-v3 и Silero VAD.
Для диаризации — pyannote.audio (нужен Hugging Face токен).
Требуется установленный ffmpeg в системе.

Использование:
    python transcribe.py audio.m4a
    python transcribe.py video.mp4                          # видео → аудио → текст
    python transcribe.py audio.m4a -o result.txt
    python transcribe.py audio.m4a --model large-v3 --language ru
    python transcribe.py audio.m4a -s --hf-token YOUR_TOKEN  # с диаризацией
    python transcribe.py a.mp3 b.ogg c.mp4                   # batch-обработка
    python transcribe.py audio.m4a --format srt -t            # SRT-субтитры
"""

import argparse
import subprocess
import sys
import tempfile
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

# ── Константы ──

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".aac", ".wma", ".opus"}

VAD_PARAMETERS = dict(
    onset=0.5,
    offset=0.35,
    min_speech_duration_ms=250,
    max_speech_duration_s=float("inf"),
    min_silence_duration_ms=2000,
    speech_pad_ms=400,
)


# ── Утилиты ──

def format_timestamp(seconds: float) -> str:
    """Форматирует секунды в HH:MM:SS.mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def format_srt_timestamp(seconds: float) -> str:
    """Форматирует секунды в SRT-формат: HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}".replace(".", ",")


@lru_cache(maxsize=1)
def get_device() -> str:
    """Определяет доступное устройство (cuda или cpu). Результат кэшируется."""
    import ctranslate2
    try:
        ctranslate2.get_supported_compute_types("cuda")
        return "cuda"
    except ValueError:
        return "cpu"


def is_video(path: Path) -> bool:
    """Проверяет, является ли файл видео по расширению."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _run_ffmpeg(input_path: str, output_path: str, extra_args: list[str] | None = None) -> str:
    """Запускает ffmpeg для конвертации. Возвращает путь к выходному файлу."""
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path


def extract_audio_from_video(video_path: str) -> str:
    """Извлекает аудио из видеофайла в WAV 16kHz mono."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    print(f"Извлекаю аудио из видео...")
    return _run_ffmpeg(video_path, wav_path, extra_args=["-vn"])


def convert_to_wav(audio_path: str) -> str:
    """Конвертирует аудио в WAV 16kHz mono (нужно для pyannote)."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    print(f"Конвертирую в WAV для диаризации...")
    return _run_ffmpeg(audio_path, wav_path)


# ── Диаризация ──

def run_diarization(audio_path: str, hf_token: str, num_speakers: int | None = None):
    """
    Запускает диаризацию — определяет, кто говорит в каждый момент.
    Возвращает список (start, end, speaker), отсортированный по start.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print(
            "Ошибка: для диаризации нужен pyannote.audio.\n"
            "Установи: pip install pyannote.audio",
            file=sys.stderr,
        )
        sys.exit(1)

    wav_path = convert_to_wav(audio_path)

    print("Загрузка модели диаризации (pyannote)...")
    t0 = time.time()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    import torch
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    print(f"Модель диаризации загружена на {device_name} за {time.time() - t0:.1f} сек.")

    print("Определяю спикеров...")
    t0 = time.time()

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(wav_path, **kwargs)

    Path(wav_path).unlink(missing_ok=True)

    segments = []
    annotation = getattr(diarization, "speaker_diarization", diarization)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    speaker_set = {s[2] for s in segments}
    print(f"Найдено спикеров: {len(speaker_set)} — за {time.time() - t0:.1f} сек.\n")

    return segments


class SpeakerIndex:
    """Индекс для быстрого поиска спикера по временному интервалу (bisect)."""

    def __init__(self, diarization_segments: list[tuple[float, float, str]]):
        self._segments = diarization_segments
        self._starts = [s[0] for s in diarization_segments]
        self._ends = [s[1] for s in diarization_segments]

    def assign(self, seg_start: float, seg_end: float) -> str:
        """Определяет спикера по максимальному пересечению. O(log n + k)."""
        # Находим кандидатов: сегменты, чей start < seg_end И end > seg_start
        right_bound = bisect_left(self._starts, seg_end)
        left_bound = bisect_right(self._ends, seg_start)

        best_speaker = "?"
        best_overlap = 0.0

        for i in range(max(0, left_bound - 1), min(len(self._segments), right_bound + 1)):
            d_start, d_end, speaker = self._segments[i]
            overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        return best_speaker


# ── Форматирование вывода ──

def _format_line_txt(text: str, start: float, end: float, speaker_prefix: str, timestamps: bool) -> str:
    if timestamps:
        ts = f"[{format_timestamp(start)} --> {format_timestamp(end)}]"
        return f"{speaker_prefix}{ts}  {text}"
    if speaker_prefix:
        return f"{speaker_prefix}{text}"
    return text


def _format_segment_srt(index: int, text: str, start: float, end: float, speaker: str | None) -> str:
    ts_start = format_srt_timestamp(start)
    ts_end = format_srt_timestamp(end)
    prefix = f"[{speaker}] " if speaker else ""
    return f"{index}\n{ts_start} --> {ts_end}\n{prefix}{text}\n"


# ── Основная функция транскрипции ──

def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
    output_path: str | None = None,
    timestamps: bool = False,
    speakers: bool = False,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    fast: bool = False,
    output_format: str = "txt",
    _model: WhisperModel | None = None,
) -> tuple[str, str]:
    """
    Транскрибирует аудио/видео файл.

    fast=True — быстрый режим (beam=1, одна температура).
    fast=False — максимальная точность (beam=5, best_of=5, word_timestamps).
    _model — переиспользование загруженной модели (для batch).
    """
    audio = Path(audio_path)
    if not audio.exists():
        print(f"Ошибка: файл '{audio_path}' не найден.", file=sys.stderr)
        sys.exit(1)

    # ── Видео → аудио ──
    temp_audio = None
    if is_video(audio):
        temp_audio = extract_audio_from_video(str(audio))
        effective_audio = temp_audio
    else:
        effective_audio = str(audio)

    if speakers and not hf_token:
        print(
            "Ошибка: для диаризации нужен Hugging Face токен.\n"
            "Укажи через --hf-token YOUR_TOKEN\n\n"
            "Как получить:\n"
            "  1. Зарегистрируйся на https://huggingface.co\n"
            "  2. Прими условия: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. И тут тоже: https://huggingface.co/pyannote/segmentation-3.0\n"
            "  4. Создай токен: https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Определяем устройство ──
    if device == "auto":
        device = get_device()
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    # ── Параллельный запуск: диаризация + загрузка модели ──
    diarization_segments = None
    model = _model

    if speakers and model is None:
        # Запускаем параллельно: диаризацию и загрузку модели Whisper
        print("Параллельный запуск: диаризация + загрузка Whisper...\n")
        with ThreadPoolExecutor(max_workers=2) as executor:
            diar_future = executor.submit(run_diarization, effective_audio, hf_token, num_speakers)

            def _load_model():
                print(f"Загрузка модели '{model_size}' (device={device}, compute={compute_type})...")
                t0 = time.time()
                m = WhisperModel(model_size, device=device, compute_type=compute_type)
                print(f"Модель загружена за {time.time() - t0:.1f} сек.")
                return m

            model_future = executor.submit(_load_model)

            diarization_segments = diar_future.result()
            model = model_future.result()
    else:
        if speakers:
            diarization_segments = run_diarization(effective_audio, hf_token, num_speakers)

        if model is None:
            print(f"Загрузка модели '{model_size}' (device={device}, compute={compute_type})...")
            t0 = time.time()
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"Модель загружена за {time.time() - t0:.1f} сек.")

    print(f"Транскрибирую: {audio.name} ...")

    # ── Параметры транскрипции (DRY: общий базис) ──
    common_kwargs = dict(
        language=language,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=VAD_PARAMETERS,
    )

    if fast:
        print(">> Быстрый режим (скорость > точность)")
        common_kwargs.update(
            beam_size=1, best_of=1, patience=1.0,
            temperature=0.0, word_timestamps=False,
        )
    else:
        print(">> Режим максимальной точности")
        common_kwargs.update(
            beam_size=5, best_of=5, patience=2.0,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            word_timestamps=True,
        )

    segments_iter, info = model.transcribe(effective_audio, **common_kwargs)

    detected_lang = info.language
    lang_prob = info.language_probability
    duration = info.duration
    print(f"Язык: {detected_lang} (вероятность: {lang_prob:.1%})")
    print(f"Длительность: {duration:.1f} сек.\n")

    # ── Индекс спикеров (быстрый поиск) ──
    speaker_index = SpeakerIndex(diarization_segments) if diarization_segments else None

    # ── Файл вывода ──
    suffix = ".srt" if output_format == "srt" else ".txt"
    if output_path:
        out_file = Path(output_path)
    else:
        out_file = Path(audio.stem + "_transcript" + suffix)

    print(f"Результат: {out_file}\n")

    # ── Сборка результата с прогрессом (append-режим) ──
    full_text_parts: list[str] = []
    t_start = time.time()
    prev_speaker = None
    srt_index = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue

            full_text_parts.append(text)

            # ── Определение спикера ──
            current_speaker = None
            if speaker_index is not None:
                current_speaker = speaker_index.assign(segment.start, segment.end)

            # ── Форматирование и запись ──
            if output_format == "srt":
                srt_index += 1
                line = _format_segment_srt(srt_index, text, segment.start, segment.end, current_speaker)
            else:
                speaker_prefix = ""
                if current_speaker is not None and current_speaker != prev_speaker:
                    speaker_prefix = f"\n[{current_speaker}]\n" if prev_speaker is not None else f"[{current_speaker}]\n"
                    prev_speaker = current_speaker
                line = _format_line_txt(text, segment.start, segment.end, speaker_prefix, timestamps)

            f.write(line + "\n")
            f.flush()

            # ── Прогресс ──
            pct = min(segment.end / duration * 100, 100) if duration > 0 else 0
            elapsed = time.time() - t_start
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            print(f"\r[{pct:5.1f}%] {elapsed_str} — до {format_timestamp(segment.end)}", end="", flush=True)

    print()

    # ── Удаляем временный аудио из видео ──
    if temp_audio:
        Path(temp_audio).unlink(missing_ok=True)

    # ── Вывод ──
    total_time = time.time() - t_start
    full_text = " ".join(full_text_parts)

    print()
    print("=" * 60)
    print("РЕЗУЛЬТАТ ТРАНСКРИПЦИИ")
    print("=" * 60)
    result = out_file.read_text(encoding="utf-8")
    print(result)
    print("=" * 60)
    print(f"\nГотово за {total_time:.0f} сек.")
    print(f"Сохранено в: {out_file}")

    return full_text, str(out_file), model


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио/видео с максимальной точностью (faster-whisper + Silero VAD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python transcribe.py audio.m4a\n"
            "  python transcribe.py video.mp4                    # видео → текст\n"
            "  python transcribe.py a.mp3 b.ogg c.mp4            # batch\n"
            "  python transcribe.py audio.m4a -s --hf-token TOKEN # диаризация\n"
            "  python transcribe.py audio.m4a --format srt -t     # SRT-субтитры\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Путь к файлу (аудио: m4a/mp3/wav/ogg/flac, видео: mp4/mkv/avi/mov/webm)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Путь для сохранения транскрипции (только для одного файла)",
    )
    parser.add_argument(
        "-m", "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        help="Размер модели Whisper (по умолчанию: large-v3)",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Код языка (ru, en, de и т.д.). По умолчанию — автоопределение",
    )
    parser.add_argument(
        "-d", "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство для инференса (по умолчанию: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "int8", "float32"],
        help="Тип вычислений (auto → float16 для GPU, int8 для CPU)",
    )
    parser.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="Добавить таймстампы к каждому сегменту",
    )
    parser.add_argument(
        "-s", "--speakers",
        action="store_true",
        help="Включить диаризацию — разделение по спикерам (нужен --hf-token)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face токен для pyannote.audio (диаризация)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Кол-во спикеров (если известно). Улучшает точность диаризации",
    )
    parser.add_argument(
        "-f", "--fast",
        action="store_true",
        help="Быстрый режим: beam=1, одна температура (в 3-5x быстрее)",
    )
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "srt"],
        dest="output_format",
        help="Формат вывода: txt (текст) или srt (субтитры). По умолчанию: txt",
    )

    args = parser.parse_args()

    if args.output and len(args.input) > 1:
        print("Ошибка: -o/--output можно использовать только с одним файлом.", file=sys.stderr)
        sys.exit(1)

    # ── Batch-обработка: модель загружается один раз ──
    model = None
    for i, input_path in enumerate(args.input):
        if len(args.input) > 1:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(args.input)}] {input_path}")
            print(f"{'='*60}\n")

        _, _, model = transcribe(
            audio_path=input_path,
            model_size=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            output_path=args.output,
            timestamps=args.timestamps,
            speakers=args.speakers,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            fast=args.fast,
            output_format=args.output_format,
            _model=model,
        )


if __name__ == "__main__":
    main()
