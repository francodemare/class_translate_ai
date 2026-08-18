#!/usr/bin/env python3
"""
================================================================================
evals_whisper.py - Framework de Evaluación y Benchmarking de Whisper
Modelo Objetivo: whisper-large-v3-turbo
Comparativa Rigurosa: NVIDIA CUDA (Windows 11) vs. Apple Silicon MLX (macOS)
================================================================================

Este script evalúa de forma controlada y objetiva la calidad de transcripción
(WER, CER, Word Accuracy, Tasa de Alucinación) y la eficiencia del hardware
(RTF, Throughput, Consumo de Memoria Pico) del backend activo.

Uso Básico:
    # 1. Crear dataset de prueba de ejemplo (si no tienes uno)
    python evals_whisper.py --create-samples

    # 2. Ejecutar evaluación (5 corridas por muestra)
    python evals_whisper.py --dataset_dir eval_dataset --runs 5

    # 3. Comparar resultados de 2 máquinas (ej. Mac MLX vs Windows CUDA)
    python evals_whisper.py --compare results_mlx.json results_cuda.json
"""

import argparse
import datetime
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import numpy as np

# ==============================================================================
# Pipeline de Normalización de Texto (Standard Spanish Normalizer)
# ==============================================================================
standard_transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])


def strip_accents(text: str) -> str:
    """Elimina acentos/tildes para evitar discrepancias ortográficas secundarias."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(text: str, ignore_accents: bool = True) -> str:
    """Normaliza el texto para comparar de forma justa sin penalizar comas, mayúsculas o tildes."""
    if not text:
        return ""
    clean = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    norm = str(standard_transformation(clean)).strip()
    if ignore_accents:
        norm = strip_accents(norm)
    return norm


# ==============================================================================
# Detección e Inicialización del Backend de Hardware
# ==============================================================================
class WhisperEvaluatorBackend:
    def __init__(self, model_name: str = "whisper-large-v3-turbo"):
        self.model_name = model_name
        self.backend_type = "unknown"
        self.device_name = "unknown"
        self.os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        self.model_instance = None
        self._init_backend()

    def _init_backend(self):
        # 1. Intentar Apple Silicon MLX
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx.core as mx
                import mlx_whisper
                self.backend_type = "mlx"
                self.device_name = f"Apple Silicon {platform.processor() or 'M-Series'} (MLX Metal)"
                self.mlx_whisper = mlx_whisper
                self.mx = mx
                self.hf_repo = "mlx-community/whisper-large-v3-turbo"
                print(f"🚀 [Backend detectado] MLX Apple Silicon -> {self.device_name}")
                return
            except ImportError:
                print("⚠️ mlx_whisper no instalado en macOS, intentando otros backends...")

        # 2. Intentar faster-whisper (CTranslate2) con NVIDIA CUDA
        try:
            import faster_whisper
            import torch
            if torch.cuda.is_available():
                self.backend_type = "faster_whisper_cuda"
                gpu_name = torch.cuda.get_device_name(0)
                self.device_name = f"NVIDIA CUDA ({gpu_name})"
                self.torch = torch
                print(f"🚀 [Backend detectado] faster-whisper CUDA -> {self.device_name}")
                print(f"   Cargando modelo '{self.model_name}' con compute_type=float16...")
                self.model_instance = faster_whisper.WhisperModel(
                    "large-v3-turbo",
                    device="cuda",
                    compute_type="float16"
                )
                self._init_nvml()
                return
        except ImportError:
            pass

        # 3. Intentar PyTorch OpenAI Whisper con CUDA
        try:
            import torch
            import whisper
            self.torch = torch
            self.whisper = whisper
            if torch.cuda.is_available():
                self.backend_type = "openai_whisper_cuda"
                gpu_name = torch.cuda.get_device_name(0)
                self.device_name = f"NVIDIA CUDA ({gpu_name}) via PyTorch"
                print(f"🚀 [Backend detectado] OpenAI Whisper CUDA -> {self.device_name}")
                self.model_instance = whisper.load_model("turbo", device="cuda")
                self._init_nvml()
                return
            elif torch.backends.mps.is_available():
                self.backend_type = "openai_whisper_mps"
                self.device_name = "PyTorch MPS (Apple Silicon)"
                print(f"🚀 [Backend detectado] OpenAI Whisper MPS -> {self.device_name}")
                self.model_instance = whisper.load_model("turbo", device="mps")
                return
        except ImportError:
            pass

        # 4. Fallback a CPU
        try:
            import whisper
            self.backend_type = "openai_whisper_cpu"
            self.device_name = f"CPU ({platform.processor() or 'Standard CPU'})"
            print(f"⚠️ [Backend detectado] CPU Fallback -> {self.device_name}")
            self.model_instance = whisper.load_model("turbo", device="cpu")
        except Exception as e:
            raise RuntimeError(f"No se pudo inicializar ningún backend de Whisper: {e}")

    def _init_nvml(self):
        """Intenta inicializar NVML para lectura precisa de VRAM en Windows 11 sin sesgo de WDDM."""
        self.pynvml = None
        self.nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
            print("   ✅ NVML inicializado para medición exacta de VRAM dedicada.")
        except Exception:
            print("   ℹ️ pynvml no disponible. Usando torch.cuda.max_memory_allocated() como fallback.")

    def sync_device(self):
        """Fuerza la sincronización del acelerador para mediciones de tiempo exactas."""
        if "cuda" in self.backend_type and hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        elif self.backend_type == "mlx" and hasattr(self, "mx"):
            if hasattr(self.mx, "clear_cache"):
                self.mx.clear_cache()
            elif hasattr(self.mx, "metal") and hasattr(self.mx.metal, "clear_cache"):
                self.mx.metal.clear_cache()

    def reset_memory_tracker(self):
        """Reinicia los contadores de memoria pico antes de una corrida."""
        if self.backend_type == "mlx":
            if hasattr(self.mx, "reset_peak_memory"):
                self.mx.reset_peak_memory()
            elif hasattr(self.mx.metal, "reset_peak_memory"):
                self.mx.metal.reset_peak_memory()
        elif "cuda" in self.backend_type and hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def get_peak_memory_mb(self) -> float:
        """Retorna el pico máximo de memoria consumida en MB durante la inferencia."""
        if self.backend_type == "mlx":
            if hasattr(self.mx, "get_peak_memory"):
                return float(self.mx.get_peak_memory() / (1024 * 1024))
            elif hasattr(self.mx.metal, "get_peak_memory"):
                return float(self.mx.metal.get_peak_memory() / (1024 * 1024))
        elif "cuda" in self.backend_type:
            if self.pynvml and self.nvml_handle:
                try:
                    info = self.pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    return float(info.used / (1024 * 1024))
                except Exception:
                    pass
            if hasattr(self, "torch") and self.torch.cuda.is_available():
                return float(self.torch.cuda.max_memory_allocated() / (1024 * 1024))
        return 0.0

    def transcribe_audio_file(self, audio_path: str) -> str:
        """
        Ejecuta la transcripción con decodificación greedy determinista
        (temperature=0.0, beam_size=1, condition_on_previous_text=False).
        """
        if self.backend_type == "mlx":
            res = self.mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=self.hf_repo,
                language="es",
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            return str(res.get("text", "")).strip()

        elif self.backend_type == "faster_whisper_cuda":
            segments, _ = self.model_instance.transcribe(
                audio_path,
                language="es",
                temperature=0.0,
                beam_size=1,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )
            text_chunks = [s.text for s in segments]
            return " ".join(text_chunks).strip()

        else:
            # PyTorch OpenAI Whisper
            options = dict(
                language="es",
                temperature=0.0,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                fp16=(self.backend_type == "openai_whisper_cuda")
            )
            res = self.model_instance.transcribe(audio_path, **options)
            return str(res.get("text", "")).strip()


# ==============================================================================
# Funciones Utilitarias para Audio y Dataset
# ==============================================================================
def get_audio_duration_seconds(audio_path: str) -> float:
    """Obtiene la duración exacta del archivo de audio en segundos."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return float(info.duration)
    except Exception:
        pass

    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def create_sample_dataset(dataset_dir: str = "eval_dataset"):
    """Crea un conjunto de muestras sintéticas y de calibración (voz y silencio) para pruebas."""
    os.makedirs(dataset_dir, exist_ok=True)
    import soundfile as sf

    sr = 16000

    # 1. Muestra de silencio absoluto (para medir Tasa de Alucinación)
    silence_audio_path = os.path.join(dataset_dir, "sample_01_silence.wav")
    silence_txt_path = os.path.join(dataset_dir, "sample_01_silence.txt")
    silence_data = np.zeros(sr * 5, dtype=np.float32)  # 5 segundos de silencio
    sf.write(silence_audio_path, silence_data, sr)
    with open(silence_txt_path, "w", encoding="utf-8") as f:
        f.write("")  # Ground truth vacío

    # 2. Muestra con ruido blanco tenue
    noise_audio_path = os.path.join(dataset_dir, "sample_02_noise.wav")
    noise_txt_path = os.path.join(dataset_dir, "sample_02_noise.txt")
    noise_data = np.random.normal(0, 0.005, sr * 5).astype(np.float32)
    sf.write(noise_audio_path, noise_data, sr)
    with open(noise_txt_path, "w", encoding="utf-8") as f:
        f.write("")

    # 3. Muestra de calibración con voz si 'say' (en macOS) está disponible, o sintetizada
    speech_audio_path = os.path.join(dataset_dir, "sample_03_speech_es.wav")
    speech_txt_path = os.path.join(dataset_dir, "sample_03_speech_es.txt")
    reference_text = "Bienvenidos a la clase de inteligencia artificial y aprendizaje automatico."

    created_speech = False
    if platform.system() == "Darwin":
        try:
            aiff_temp = os.path.join(dataset_dir, "temp_say.aiff")
            subprocess.run(["say", "-v", "Jorge", "-o", aiff_temp, reference_text], check=True)
            cmd = ["ffmpeg", "-y", "-i", aiff_temp, "-ar", "16000", "-ac", "1", speech_audio_path]
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(aiff_temp):
                os.remove(aiff_temp)
            created_speech = True
        except Exception:
            created_speech = False

    if not created_speech:
        t = np.linspace(0, 4, sr * 4, endpoint=False)
        sine_data = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(speech_audio_path, sine_data, sr)

    with open(speech_txt_path, "w", encoding="utf-8") as f:
        f.write(reference_text if created_speech else "")

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    manifest_data = [
        {
            "audio_file": "sample_01_silence.wav",
            "reference_file": "sample_01_silence.txt",
            "type": "silence",
            "description": "5s silencio puro para verificar tasa de alucinación"
        },
        {
            "audio_file": "sample_02_noise.wav",
            "reference_file": "sample_02_noise.txt",
            "type": "noise",
            "description": "5s ruido blanco tenue"
        },
        {
            "audio_file": "sample_03_speech_es.wav",
            "reference_file": "sample_03_speech_es.txt",
            "type": "speech",
            "description": "Muestra de voz en español"
        }
    ]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Dataset de prueba creado exitosamente en: '{dataset_dir}/'")


def load_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """Carga los pares de audio y ground truth desde manifest.json o emparejando .wav y .txt."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"El directorio '{dataset_dir}' no existe.")

    manifest_file = dataset_path / "manifest.json"
    items = []

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_items = json.load(f)
        for item in manifest_items:
            audio_p = str(dataset_path / item["audio_file"])
            ref_p = str(dataset_path / item["reference_file"])
            ref_text = ""
            if os.path.exists(ref_p):
                with open(ref_p, "r", encoding="utf-8") as rf:
                    ref_text = rf.read().strip()
            
            items.append({
                "audio_path": audio_p,
                "reference_text": ref_text,
                "type": item.get("type", "speech"),
                "description": item.get("description", Path(audio_p).name)
            })
    else:
        # Auto-descubrir pares .wav/.mp3 y .txt
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        for audio_file in sorted(dataset_path.iterdir()):
            if audio_file.suffix.lower() in audio_extensions:
                txt_file = audio_file.with_suffix(".txt")
                ref_text = ""
                if txt_file.exists():
                    with open(txt_file, "r", encoding="utf-8") as rf:
                        ref_text = rf.read().strip()
                
                is_silence = ("silence" in audio_file.stem.lower() or ref_text == "")
                items.append({
                    "audio_path": str(audio_file),
                    "reference_text": ref_text,
                    "type": "silence" if is_silence else "speech",
                    "description": audio_file.name
                })

    return items


# ==============================================================================
# Motor de Evaluación y Métricas
# ==============================================================================
def run_evaluation(
    evaluator: WhisperEvaluatorBackend,
    dataset: List[Dict[str, Any]],
    runs_per_sample: int = 5
) -> Dict[str, Any]:
    """Ejecuta la suite de pruebas completa con warm-up, múltiples corridas y cálculo de métricas."""
    print("\n" + "=" * 75)
    print("INICIANDO SUITE DE EVALUACIÓN WHISPER")
    print(f"Backend: {evaluator.backend_type} | Dispositivo: {evaluator.device_name}")
    print(f"Muestras: {len(dataset)} | Corridas por muestra: {runs_per_sample}")
    print("=" * 75)

    # 1. Warm-up Run (1 corrida no registrada para inicializar shaders/kernels y eliminar overhead de arranque)
    if dataset:
        warmup_audio = dataset[0]["audio_path"]
        print(f"\n🔥 Ejecutando Warm-up Run sobre '{Path(warmup_audio).name}'...")
        evaluator.sync_device()
        evaluator.reset_memory_tracker()
        _ = evaluator.transcribe_audio_file(warmup_audio)
        evaluator.sync_device()
        print("   ✅ Warm-up completado. Iniciando registros de benchmark.\n")

    sample_results = []
    all_rtfs = []
    all_speedups = []
    all_latencies = []
    all_peak_memories = []

    total_ref_words = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0

    silence_sample_count = 0
    hallucinated_silence_count = 0

    for idx, item in enumerate(dataset, start=1):
        audio_path = item["audio_path"]
        raw_ref = item["reference_text"]
        sample_type = item["type"]
        sample_name = Path(audio_path).name

        duration_sec = get_audio_duration_seconds(audio_path)
        if duration_sec <= 0:
            print(f"⚠️ [{idx}/{len(dataset)}] No se pudo obtener duración de '{sample_name}', saltando...")
            continue

        print(f"[{idx}/{len(dataset)}] Evaluando '{sample_name}' ({duration_sec:.2f}s) - Tipo: {sample_type}")

        run_latencies = []
        run_memories = []
        hypotheses = []

        for r in range(runs_per_sample):
            evaluator.reset_memory_tracker()
            evaluator.sync_device()

            t0 = time.perf_counter()
            hyp_text = evaluator.transcribe_audio_file(audio_path)
            evaluator.sync_device()
            t1 = time.perf_counter()

            latency = t1 - t0
            peak_mem = evaluator.get_peak_memory_mb()

            run_latencies.append(latency)
            run_memories.append(peak_mem)
            hypotheses.append(hyp_text)

        mean_latency = statistics.mean(run_latencies)
        std_latency = statistics.stdev(run_latencies) if len(run_latencies) > 1 else 0.0
        mean_rtf = mean_latency / duration_sec
        mean_speedup = 1.0 / mean_rtf if mean_rtf > 0 else 0.0
        mean_peak_mem = statistics.mean(run_memories)

        all_latencies.extend(run_latencies)
        all_rtfs.append(mean_rtf)
        all_speedups.append(mean_speedup)
        if mean_peak_mem > 0:
            all_peak_memories.append(mean_peak_mem)

        # Usar la hipótesis más representativa (o la primera, siendo determinista greedy t=0.0)
        final_hyp_raw = hypotheses[0]
        norm_ref = normalize_text(raw_ref)
        norm_hyp = normalize_text(final_hyp_raw)

        # Cálculo de métricas de texto
        if sample_type == "silence" or norm_ref == "":
            silence_sample_count += 1
            has_hallucination = bool(norm_hyp)
            if has_hallucination:
                hallucinated_silence_count += 1
            sample_wer = 1.0 if has_hallucination else 0.0
            sample_acc = 0.0 if has_hallucination else 1.0
            sample_cer = 1.0 if has_hallucination else 0.0
            s, d, i, h, n = 0, 0, (len(norm_hyp.split()) if has_hallucination else 0), 0, 0
        else:
            word_out = jiwer.process_words(norm_ref, norm_hyp)
            s = word_out.substitutions
            d = word_out.deletions
            i = word_out.insertions
            h = word_out.hits
            n = s + d + h

            total_ref_words += n
            total_substitutions += s
            total_deletions += d
            total_insertions += i
            total_hits += h

            sample_wer = word_out.wer
            sample_acc = max(0.0, 1.0 - sample_wer)
            sample_cer = jiwer.cer(norm_ref, norm_hyp) if norm_ref else 0.0

        sample_summary = {
            "sample_name": sample_name,
            "sample_type": sample_type,
            "duration_sec": round(duration_sec, 2),
            "mean_latency_sec": round(mean_latency, 4),
            "std_latency_sec": round(std_latency, 4),
            "rtf": round(mean_rtf, 4),
            "speedup": round(mean_speedup, 2),
            "peak_memory_mb": round(mean_peak_mem, 1),
            "reference_text_raw": raw_ref,
            "hypothesis_text_raw": final_hyp_raw,
            "reference_normalized": norm_ref,
            "hypothesis_normalized": norm_hyp,
            "wer_percent": round(sample_wer * 100, 2),
            "word_accuracy_percent": round(sample_acc * 100, 2),
            "cer_percent": round(sample_cer * 100, 2),
            "word_counts": {"S": s, "D": d, "I": i, "H": h, "N": n}
        }
        sample_results.append(sample_summary)

        print(f"   ⏱️ Latencia: {mean_latency:.3f}s (±{std_latency:.3f}s) | RTF: {mean_rtf:.4f} ({mean_speedup:.1f}x)")
        print(f"   📊 Word Acc: {sample_summary['word_accuracy_percent']}% | WER: {sample_summary['wer_percent']}% | CER: {sample_summary['cer_percent']}%")
        print(f"   💾 Memoria Pico: {mean_peak_mem:.1f} MB\n")

    # Métricas Globales Agregadas
    global_wer = (
        (total_substitutions + total_deletions + total_insertions) / total_ref_words
        if total_ref_words > 0 else 0.0
    )
    global_word_accuracy = max(0.0, 1.0 - global_wer)
    global_hallucination_rate = (
        (hallucinated_silence_count / silence_sample_count) * 100
        if silence_sample_count > 0 else 0.0
    )

    global_mean_rtf = statistics.mean(all_rtfs) if all_rtfs else 0.0
    global_mean_speedup = statistics.mean(all_speedups) if all_speedups else 0.0
    global_peak_memory = max(all_peak_memories) if all_peak_memories else 0.0

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "backend": evaluator.backend_type,
        "device": evaluator.device_name,
        "os": evaluator.os_info,
        "model": evaluator.model_name,
        "parameters": {
            "temperature": 0.0,
            "beam_size": 1,
            "language": "es",
            "condition_on_previous_text": False
        },
        "runs_per_sample": runs_per_sample,
        "summary": {
            "word_accuracy_percent": round(global_word_accuracy * 100, 2),
            "wer_percent": round(global_wer * 100, 2),
            "hallucination_rate_percent": round(global_hallucination_rate, 2),
            "mean_rtf": round(global_mean_rtf, 4),
            "mean_speedup": round(global_mean_speedup, 2),
            "peak_memory_mb": round(global_peak_memory, 1),
            "total_speech_words_evaluated": total_ref_words,
            "breakdown": {
                "substitutions": total_substitutions,
                "deletions": total_deletions,
                "insertions": total_insertions,
                "hits": total_hits
            }
        },
        "samples": sample_results
    }

    return report


def print_summary_table(report: Dict[str, Any]):
    """Imprime una tabla formateada en consola con el resumen de la evaluación."""
    summary = report["summary"]
    print("\n" + "=" * 75)
    print("📊 RESUMEN EJECUTIVO DE RESULTADOS")
    print("=" * 75)
    print(f"• Entorno / OS:          {report['os']}")
    print(f"• Backend Whisper:       {report['backend']}")
    print(f"• Dispositivo:           {report['device']}")
    print(f"• Modelo:                {report['model']}")
    print("-" * 75)
    print(f"🏆 Word Accuracy:        {summary['word_accuracy_percent']}%  (1 - WER)")
    print(f"🎯 WER (Word Error Rate):{summary['wer_percent']}%")
    print(f"👻 Tasa de Alucinación:  {summary['hallucination_rate_percent']}% en tramos de silencio")
    print(f"⚡ RTF (Real-Time Factor):{summary['mean_rtf']}  (< 1.0 es más rápido que tiempo real)")
    print(f"🚀 Throughput (Speedup): {summary['mean_speedup']}x tiempo real")
    print(f"💾 Consumo Memoria Pico: {summary['peak_memory_mb']} MB")
    print(f"📝 Palabras Evaluadas:   {summary['total_speech_words_evaluated']} (S:{summary['breakdown']['substitutions']}, D:{summary['breakdown']['deletions']}, I:{summary['breakdown']['insertions']})")
    print("=" * 75)


def compare_two_results(file1_path: str, file2_path: str, output_md_path: Optional[str] = None):
    """Compara dos archivos JSON generados por evals_whisper.py e imprime una tabla comparativa."""
    with open(file1_path, "r", encoding="utf-8") as f1, open(file2_path, "r", encoding="utf-8") as f2:
        d1 = json.load(f1)
        d2 = json.load(f2)

    s1, s2 = d1["summary"], d2["summary"]

    lines = []
    lines.append("# Comparativa de Desempeño: Whisper-Large-v3-Turbo")
    lines.append(f"Fecha de reporte: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"| Métrica / Dimensión | **Entorno A ({d1['backend']})** | **Entorno B ({d2['backend']})** | Ganador / Diferencia |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Sistema Operativo** | {d1['os']} | {d2['os']} | - |")
    lines.append(f"| **Acelerador / GPU** | {d1['device']} | {d2['device']} | - |")
    lines.append(f"| **Word Accuracy (1 - WER)** | **{s1['word_accuracy_percent']}%** | **{s2['word_accuracy_percent']}%** | {'Entorno A' if s1['word_accuracy_percent'] > s2['word_accuracy_percent'] else 'Entorno B' if s2['word_accuracy_percent'] > s1['word_accuracy_percent'] else 'Empate'} |")
    lines.append(f"| **WER (Word Error Rate)** | {s1['wer_percent']}% | {s2['wer_percent']}% | {'Entorno A' if s1['wer_percent'] < s2['wer_percent'] else 'Entorno B' if s2['wer_percent'] < s1['wer_percent'] else 'Empate'} |")
    lines.append(f"| **Tasa de Alucinación** | {s1['hallucination_rate_percent']}% | {s2['hallucination_rate_percent']}% | {'Entorno A' if s1['hallucination_rate_percent'] < s2['hallucination_rate_percent'] else 'Entorno B' if s2['hallucination_rate_percent'] < s1['hallucination_rate_percent'] else 'Empate'} |")
    lines.append(f"| **RTF (Real-Time Factor)** | {s1['mean_rtf']} | {s2['mean_rtf']} | {'Entorno A' if s1['mean_rtf'] < s2['mean_rtf'] else 'Entorno B' if s2['mean_rtf'] < s1['mean_rtf'] else 'Empate'} |")
    lines.append(f"| **Throughput (Speedup)** | **{s1['mean_speedup']}x** | **{s2['mean_speedup']}x** | {'Entorno A' if s1['mean_speedup'] > s2['mean_speedup'] else 'Entorno B' if s2['mean_speedup'] > s1['mean_speedup'] else 'Empate'} |")
    lines.append(f"| **Memoria Pico** | {s1['peak_memory_mb']} MB | {s2['peak_memory_mb']} MB | {abs(s1['peak_memory_mb'] - s2['peak_memory_mb']):.1f} MB dif |")

    comparison_md = "\n".join(lines)
    print("\n" + comparison_md + "\n")

    if output_md_path:
        with open(output_md_path, "w", encoding="utf-8") as mf:
            mf.write(comparison_md)
        print(f"📄 Reporte comparativo guardado en: {output_md_path}")


# ==============================================================================
# CLI Principal
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Suite de Evaluación para whisper-large-v3-turbo (CUDA vs MLX)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="eval_dataset",
        help="Ruta al directorio del dataset (con manifest.json o pares .wav/.txt)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Número de ejecuciones por muestra para calcular promedios y desviación estándar (defecto: 5)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Ruta personalizada para el archivo JSON de salida"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Crea un conjunto de muestras sintéticas y de calibración en eval_dataset/"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RESULT_A.json", "RESULT_B.json"),
        help="Compara dos archivos de resultados JSON generados previamente"
    )

    args = parser.parse_args()

    # Modo Comparador
    if args.compare:
        out_md = f"comparison_{Path(args.compare[0]).stem}_vs_{Path(args.compare[1]).stem}.md"
        compare_two_results(args.compare[0], args.compare[1], out_md)
        return

    # Modo Crear Samples
    if args.create_samples:
        create_sample_dataset(args.dataset_dir)
        return

    # Si la carpeta del dataset no existe, sugerir o crear samples
    if not os.path.exists(args.dataset_dir):
        print(f"Directorio '{args.dataset_dir}' no encontrado. Creando muestras de calibración automáticas...")
        create_sample_dataset(args.dataset_dir)

    # 1. Cargar Dataset
    dataset = load_dataset(args.dataset_dir)
    if not dataset:
        print(f"❌ No se encontraron muestras de audio en '{args.dataset_dir}'.")
        sys.exit(1)

    # 2. Inicializar Evaluador de Hardware
    evaluator = WhisperEvaluatorBackend(model_name="whisper-large-v3-turbo")

    # 3. Ejecutar Evaluación
    report = run_evaluation(evaluator, dataset, runs_per_sample=args.runs)

    # 4. Mostrar Resultados
    print_summary_table(report)

    # 5. Guardar Reporte JSON
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.output_json or f"eval_results_{evaluator.backend_type}_{timestamp_str}.json"
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False)

    print(f"💾 Reporte detallado exportado exitosamente a: {out_json}")


if __name__ == "__main__":
    main()
