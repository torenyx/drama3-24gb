# Diagnostic tools and native controls

The public user entry point is [`scripts/sample.py`](../scripts/sample.py). These tools retain the underlying experiments used in the [failure analysis](../docs/failures.md).

| Tool | Purpose |
|---|---|
| `benchmark_native_fish.py` | Published weights through native Fish Speech; measured control uses three GPUs |
| `asr_check.py` | Transcript coverage with a separately prepared Qwen3-ASR checkpoint |
| `diagnostic_a_opening.py` | Record the opening distribution around model switching |
| `diagnostic_a_sampling.py` | Controlled semantic sampling and bad-prefix replay |
| `profile_a_service.py` | Trace graph replay and dynamic residency across requests |

The diagnostic modules are imported by optional flags of `scripts/benchmark_comfy.py`; they are not separate user launchers. The production stalled-chunk retry is in [`runtime/stalled_chunk.py`](../runtime/stalled_chunk.py).

Use `python experiments/benchmark_native_fish.py --help` for the native control. Its environment is the pinned Fish Speech environment, separate from C. Reproducing the older Japanese records needs the original private references; the included LibriSpeech sample tests the same workflow with public inputs.
