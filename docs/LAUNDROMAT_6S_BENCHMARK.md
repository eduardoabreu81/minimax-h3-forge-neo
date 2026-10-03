# Laundromat: six-second Forge H3 benchmark

Date: 2026-10-02 (America/Sao_Paulo). Extension 0.1.2. This request ran through Forge's authenticated native txt2img Generate callback on the same dedicated NVIDIA A40 and standard FL2VA INT8 model set as the cinematic benchmark. No extension source changes were needed.

The requested scene is a washing machine bursting open in a laundromat, releasing a torrent with clothes, fish, a swimming dog and finally a seal. The submitted prompt adds a steady wide shot, natural sound and a cue for the seal to appear before the delivery cut. [Original prompt](prompts/laundromat-original.txt), [submitted prompt](prompts/laundromat-submitted.txt), [complete settings](LAUNDROMAT_SETTINGS.json).

## Configuration and measured results

| Item | Result |
| --- | --- |
| GPU | One NVIDIA A40, 48 GB advertised |
| Native generation | 448x672, 158 frames, 24 FPS, 6.583333 seconds |
| Exact delivery | 440x652, 144 frames, 24 FPS, 6 seconds |
| Steps / CFG / seed | 32 / 1 / 20261003 |
| Sampler / memory | Euler / Simple; Economical |
| Driver wall time | 903.2 seconds |
| Forge reported time | 902.3 seconds |
| Delivery processing and verification | 3.3 seconds, separate from generation |
| Torch active / reserved peaks | 23.94 / 24.03 GiB |
| Forge sampled system GPU peak | 24.6 GiB |
| Independent NVIDIA sampled maximum | 24951 MiB |
| Delivered codecs | H.264; stereo AAC, 32 kHz |
| Decoded audio RMS | -17.85 dBFS |
| Delivered file | 1,592,836 bytes |
| Delivered SHA256 | `c3fc8a5bcae067a1a19bd9b550d6545561de01d841319d688bfa551cb1ff7eaf` |
| Estimated request GPU cost | US$0.123, at captured US$0.49/hour |

H3 requires dimensions divisible by 32 and frames on the 17n+5 grid. Delivery center-crops 4 pixels from each horizontal edge and 10 pixels from each vertical edge, then removes the last 14 frames and trims audio to 6 seconds. It is not upscaled or time-stretched. H.264 CRF 16 and AAC 192 kbps are used for the final export. The native source is retained.

Both native and delivered streams passed complete FFmpeg decoding. FFprobe verified dimensions, FPS, frame counts and durations; the downloaded delivery matched its SHA256. Decodable nonzero audio does not establish sound semantics or synchronization. Sampled-frame observations, if performed, are separately recorded in [LAUNDROMAT_6S_RESULT.json](LAUNDROMAT_6S_RESULT.json).

## Environment and measurement limits

Immutable model repositories/revisions, complete weight hashes, software versions, driver/CUDA, memory policy and the Pod allocation are documented in [BENCHMARKS.md](BENCHMARKS.md) and [BENCHMARK_ENVIRONMENT.json](BENCHMARK_ENVIRONMENT.json). The same Python 3.12.3, Torch 2.8 CUDA 12.8 and pinned DiffSynth/Forge revisions are used here.

Driver wall time includes request setup, model loading, inference, decoding, native export and response. Forge's per-request Torch allocator peak counters and sampled system memory are distinct measurements; its HTML GB label represents binary GiB. Independent samples are in [LAUNDROMAT_GPU_SAMPLES.csv](LAUNDROMAT_GPU_SAMPLES.csv), with step changes in [LAUNDROMAT_STEP_MARKERS.json](LAUNDROMAT_STEP_MARKERS.json). Sampled totals are not instantaneous peak guarantees, and per-run host RAM is unmeasured.

This run keeps the cinematic test's 32 steps, CFG, memory policy, model set and GPU, while changing resolution, frames, prompt and seed. The timing comparison is observational; caches and per-stage times were not controlled. It does not isolate the individual effect of resolution or duration, and it does not establish support for other models or GPUs. Costs are estimates rather than invoices and exclude preparation, delivery processing and idle time.
