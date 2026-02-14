#!/usr/bin/env python3
"""
Транскрипция аудио-файлов (m4a и других форматов) с максимальной точностью.
Опционально — диаризация (разделение по спикерам).

Используется faster-whisper (CTranslate2) с моделью large-v3 и Silero VAD.
Для диаризации — pyannote.audio (нужен Hugging Face токен).
Требуется установленный ffmpeg в системе.

Использование:
    python transcribe.py audio.m4a
    python transcribe.py audio.m4a -o result.txt
    python transcribe.py audio.m4a --model large-v3 --language ru
    python transcribe.py audio.m4a -s --hf-token YOUR_TOKEN   # с диаризацией
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды в HH:MM:SS.mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def get_device() -> str:
    """Определяет доступное устройство (cuda или cpu)."""
    import ctranslate2
    try:
        ctranslate2.get_supported_compute_types("cuda")
        return "cuda"
    except ValueError:
        return "cpu"


def convert_to_wav(audio_path: str) -> str:
    """Конвертирует аудио в WAV 16kHz mono (нужно для pyannote)."""
    wav_path = tempfile.mktemp(suffix=".wav")
    print(f"Конвертирую в WAV для диаризации...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            wav_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return wav_path


def run_diarization(audio_path: str, hf_token: str, num_speakers: int | None = None):
    """
    Запускает диаризацию — определяет, кто говорит в каждый момент.
    Возвращает список (start, end, speaker).
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

    # Конвертируем в WAV — pyannote плохо работает с m4a/mp3 напрямую
    wav_path = convert_to_wav(audio_path)

    print("Загрузка модели диаризации (pyannote)...")
    t0 = time.time()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    print(f"Модель диаризации загружена за {time.time() - t0:.1f} сек.")
    print("Определяю спикеров...")

    t0 = time.time()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(wav_path, **kwargs)

    # Удаляем временный wav
    Path(wav_path).unlink(missing_ok=True)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    speaker_set = {s[2] for s in segments}
    print(f"Найдено спикеров: {len(speaker_set)} — за {time.time() - t0:.1f} сек.\n")

    return segments


def assign_speaker(seg_start: float, seg_end: float, diarization_segments: list) -> str:
    """
    Определяет спикера для данного сегмента транскрипции
    по максимальному пересечению с диаризацией.
    """
    best_speaker = "?"
    best_overlap = 0.0

    for d_start, d_end, speaker in diarization_segments:
        # Пересечение интервалов
        overlap_start = max(seg_start, d_start)
        overlap_end = min(seg_end, d_end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


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
) -> str:
    """
    Транскрибирует аудио-файл с максимальной точностью.
    """
    audio = Path(audio_path)
    if not audio.exists():
        print(f"Ошибка: файл '{audio_path}' не найден.", file=sys.stderr)
        sys.exit(1)

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

    # ── Диаризация (если включена) ──
    diarization_segments = None
    if speakers:
        diarization_segments = run_diarization(str(audio), hf_token, num_speakers)

    # ── Определяем устройство ──
    if device == "auto":
        device = get_device()

    # ── Выбор compute_type ──
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"Загрузка модели '{model_size}' (device={device}, compute={compute_type})...")
    t0 = time.time()

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    print(f"Модель загружена за {time.time() - t0:.1f} сек.")
    print(f"Транскрибирую: {audio.name} ...")

    # ── Транскрипция с параметрами максимальной точности ──
    segments, info = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        best_of=5,
        patience=2.0,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=dict(
            onset=0.5,
            offset=0.35,
            min_speech_duration_ms=250,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=2000,
            speech_pad_ms=400,
        ),
        word_timestamps=True,
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    duration = info.duration
    print(f"Определён язык: {detected_lang} (вероятность: {lang_prob:.1%})")
    print(f"Длительность аудио: {duration:.1f} сек.\n")

    # ── Файл для инкрементальной записи ──
    out_file = Path(output_path) if output_path else Path(audio.stem + "_transcript.txt")
    print(f"Результат пишется в: {out_file} (обновляется по мере работы)\n")

    # ── Сборка результата с прогрессом ──
    lines: list[str] = []
    full_text_parts: list[str] = []
    t_start = time.time()
    prev_speaker = None

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        full_text_parts.append(text)

        # ── Определение спикера ──
        speaker_prefix = ""
        if diarization_segments is not None:
            speaker = assign_speaker(segment.start, segment.end, diarization_segments)
            # Добавляем метку спикера, когда он меняется
            if speaker != prev_speaker:
                speaker_prefix = f"\n[{speaker}]\n" if prev_speaker is not None else f"[{speaker}]\n"
                prev_speaker = speaker

        # ── Форматирование строки ──
        if timestamps:
            ts_start = format_timestamp(segment.start)
            ts_end = format_timestamp(segment.end)
            line = f"{speaker_prefix}[{ts_start} --> {ts_end}]  {text}"
        elif speaker_prefix:
            line = f"{speaker_prefix}{text}"
        else:
            line = text

        lines.append(line)

        # ── Прогресс ──
        pct = min(segment.end / duration * 100, 100) if duration > 0 else 0
        elapsed = time.time() - t_start
        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
        print(f"\r[{pct:5.1f}%] {elapsed_str} обработано до {format_timestamp(segment.end)}", end="", flush=True)

        # ── Записываем в файл по мере работы ──
        out_file.write_text("\n".join(lines), encoding="utf-8")

    print()  # перенос строки после прогресса

    result = "\n".join(lines)
    full_text = " ".join(full_text_parts)

    # ── Вывод ──
    print()
    print("=" * 60)
    print("РЕЗУЛЬТАТ ТРАНСКРИПЦИИ")
    print("=" * 60)
    print(result)
    print("=" * 60)
    print(f"\nГотово за {time.time() - t_start:.0f} сек.")
    print(f"Результат сохранён в: {out_file}")

    return full_text, str(out_file)


def main():
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио с максимальной точностью (faster-whisper + Silero VAD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        help="Путь к аудио-файлу (m4a, mp3, wav, ogg, flac и т.д.)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Путь для сохранения текста транскрипции в файл",
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
        help="Код языка аудио (например: ru, en, de). По умолчанию — автоопределение",
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
        help="Тип вычислений (по умолчанию: auto → float16 для GPU, int8 для CPU)",
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

    args = parser.parse_args()

    transcribe(
        audio_path=args.audio,
        model_size=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        output_path=args.output,
        timestamps=args.timestamps,
        speakers=args.speakers,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
    )


if __name__ == "__main__":
    main()
