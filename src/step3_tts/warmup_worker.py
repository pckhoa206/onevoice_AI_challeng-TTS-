"""Warmup Worker for Step 3 TTS.
Thread-safe background initialization to eliminate cold-start JIT and NPU graph compilation latency.
"""
import time
import threading
from typing import Callable, Dict, Any


class WarmupWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._warmed_engines: Dict[str, bool] = {}

    def warmup_engine(self, engine_name: str, synth_fn: Callable[[str, str], Any], sample_text: str = "xin chào", lang: str = "vi") -> float:
        """Run a synchronized dummy synthesis to pre-allocate graph memory and bind ION/DMA pools.

        Returns elapsed warmup time in seconds.
        """
        with self._lock:
            if self._warmed_engines.get(engine_name, False):
                return 0.0

            print(f"[WarmupWorker] Warming up engine '{engine_name}' ({lang})...")
            t0 = time.perf_counter()
            try:
                # Perform dummy synthesis
                _ = synth_fn(sample_text, lang)
                self._warmed_engines[engine_name] = True
                elapsed = time.perf_counter() - t0
                print(f"[WarmupWorker] Engine '{engine_name}' warmed up in {elapsed:.3f}s")
                return elapsed
            except Exception as e:
                print(f"[WarmupWorker] Warning: Warmup failed for '{engine_name}': {e}")
                return 0.0

    def is_warmed(self, engine_name: str) -> bool:
        with self._lock:
            return self._warmed_engines.get(engine_name, False)
