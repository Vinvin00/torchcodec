# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Memory consumption benchmark for video decoders.

Each decoder runs in its own fresh subprocess, so one library's allocations
can't pollute another's numbers. In the subprocess we import the library,
build the decoder, record a baseline, and then decode the first N frames over
and over for ``--duration-seconds``, sampling resident set size (RSS) and
virtual memory size (VMS) in the background. We report the peak over the
baseline, and how much RSS grew between the end of the first iteration and the
end of the run (a rough leak indicator).

Quick check (a few seconds per decoder):
    python benchmark_memory.py --duration-seconds 1

Longer, more formal run:
    python benchmark_memory.py --duration-seconds 60

RSS growth includes one-time warm-up (allocator caches, thread pools). To tell
warm-up from a leak, run with two different durations: warm-up stays constant,
a leak scales with the duration. Note that VMS is very large on macOS, so look
at the "+base" column there.

Decoders are the ones from ``benchmark_decoders_library.decoder_registry``,
using the same ``name:opt=val+opt=val`` syntax as ``benchmark_decoders.py``.
Decoders whose library isn't installed are reported as unavailable.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

MiB = 1024 * 1024
DEFAULT_VIDEO = Path(__file__).parent / "../../test/resources/nasa_13013.mp4"
DEFAULT_DECODERS = "torchcodec_core,torchcodec_public,torchvision,opencv,decord"


class PeakSampler(threading.Thread):
    """Tracks the max RSS and VMS of this process, polling every `interval`."""

    def __init__(self, interval: float = 0.005):
        super().__init__(daemon=True)
        self._process = psutil.Process()
        self._interval = interval
        self._stop_event = threading.Event()
        self.peak_rss = self.peak_vms = 0

    def sample(self) -> None:
        info = self._process.memory_info()
        self.peak_rss = max(self.peak_rss, info.rss)
        self.peak_vms = max(self.peak_vms, info.vms)

    def run(self) -> None:
        while not self._stop_event.wait(self._interval):
            self.sample()

    def stop(self) -> None:
        self._stop_event.set()
        self.join()
        self.sample()


def run_worker(args) -> dict:
    # Imported here so that the parent process stays light, and so that a
    # missing library only fails this worker.
    from benchmark_decoders import parse_options_code
    from benchmark_decoders_library import decoder_registry

    name, _, options_code = args.decoder.partition(":")
    if name not in decoder_registry:
        raise ValueError(f"Unknown decoder: {name}")
    options = (
        parse_options_code(options_code)
        if options_code
        else decoder_registry[name].default_options
    )
    decoder = decoder_registry[name].kind(**options)

    process = psutil.Process()
    baseline = process.memory_info()
    sampler = PeakSampler()
    sampler.start()

    iterations = 0
    rss_after_first = None
    start = time.perf_counter()
    while iterations == 0 or time.perf_counter() - start < args.duration_seconds:
        frames = decoder.decode_first_n_frames(args.video, args.num_frames)
        del frames
        iterations += 1
        if iterations == 1:
            rss_after_first = process.memory_info().rss
    rss_end = process.memory_info().rss
    sampler.stop()

    return {
        "iterations": iterations,
        "baseline_rss": baseline.rss,
        "baseline_vms": baseline.vms,
        "peak_rss": sampler.peak_rss,
        "peak_vms": sampler.peak_vms,
        "rss_growth": rss_end - rss_after_first,
    }


def run_in_subprocess(decoder: str, args) -> dict:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--decoder",
        decoder,
        "--video",
        args.video,
        "--num-frames",
        str(args.num_frames),
        "--duration-seconds",
        str(args.duration_seconds),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        # Last line of the traceback, e.g. "ModuleNotFoundError: No module named 'decord'"
        lines = proc.stderr.strip().splitlines()
        return {"error": lines[-1] if lines else f"exit code {proc.returncode}"}
    # Libraries may print to stdout; our JSON is always the last line.
    return json.loads(proc.stdout.strip().splitlines()[-1])


def print_table(results: dict[str, dict]) -> None:
    header = (
        f"{'decoder':<40} {'iters':>6} {'peak RSS':>10} {'RSS +base':>10} "
        f"{'peak VMS':>10} {'VMS +base':>10} {'RSS growth':>11}"
    )
    print(header)
    print("-" * len(header))
    for decoder, r in results.items():
        if "error" in r:
            print(f"{decoder:<40} unavailable: {r['error']}")
            continue
        print(
            f"{decoder:<40} {r['iterations']:>6} "
            f"{r['peak_rss'] / MiB:>10.1f} "
            f"{(r['peak_rss'] - r['baseline_rss']) / MiB:>10.1f} "
            f"{r['peak_vms'] / MiB:>10.1f} "
            f"{(r['peak_vms'] - r['baseline_vms']) / MiB:>10.1f} "
            f"{r['rss_growth'] / MiB:>11.1f}"
        )
    print("\nAll values in MiB. '+base' is peak minus the value before decoding.")
    print("RSS growth is end-of-run RSS minus RSS after the first iteration.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--decoders",
        help="Comma-separated decoders, same syntax as benchmark_decoders.py.",
        default=DEFAULT_DECODERS,
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument(
        "--num-frames",
        help="Number of frames decoded from the start of the video per iteration.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--duration-seconds",
        help="How long to keep decoding, per decoder. Use ~1 for a quick check "
        "and 60+ for a formal run.",
        type=float,
        default=2.0,
    )
    # Internal: used by the parent to launch one decoder in a fresh process.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--decoder", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(run_worker(args)))
        return

    results = {}
    for decoder in args.decoders.split(","):
        print(f"Running {decoder} ...", file=sys.stderr, flush=True)
        results[decoder] = run_in_subprocess(decoder, args)
    print_table(results)


if __name__ == "__main__":
    main()
