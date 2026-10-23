"""Run the upstream Comfy nodes directly, retaining codes, WAVs and stage timings."""
import argparse
import faulthandler
import importlib.util
import json
import logging
import os
import re
import subprocess
from pathlib import Path
import sys
import time

START = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument('--comfy', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--codec', required=True)
parser.add_argument('--assets', required=True)
parser.add_argument('--prompt-file', help='Override script text for a controlled prompt ablation')
parser.add_argument('--prompt-ablation', choices=['original', 'no-pauses', 'dialogue-only'], default='original',
                    help='Remove control tags after splitting, retaining original dialogue chunk boundaries')
parser.add_argument('--out', required=True)
parser.add_argument('--seeds', default='0,1,2')
parser.add_argument('--references', help='Reference ids; defaults to baseline_references in the asset manifest')
parser.add_argument('--split-layer', type=int, default=0)
parser.add_argument('--top-k', type=int, default=30)
parser.add_argument('--dynamic-vram', action='store_true',
                    help='Initialize the same allocator/compiler as normal ComfyUI startup')
parser.add_argument('--reuse-voices', action='store_true')
parser.add_argument('--encode-only', action='store_true')
parser.add_argument('--trace-sampling', action='store_true')
parser.add_argument('--trace-opening', action='store_true')
parser.add_argument('--codes-only', action='store_true',help='Keep TTS model resident; decode saved codes separately')
parser.add_argument('--resident-codec', action='store_true',help='Keep codec on last GPU without unloading/repatching TTS models')
parser.add_argument('--disable-comfy-compiler', action='store_true')
parser.add_argument('--codec-device',help='Place managed codec on this CUDA device, independently of TTS first device')
parser.add_argument('--external-codec-device', help='Decode in a persistent isolated process on this device; keep dynamic TTS untouched')
parser.add_argument('--keep-tts-resident', action='store_true', help='Reuse prepared TTS between dynamic placement refreshes when the codec is isolated')
parser.add_argument('--tts-refresh-every', type=int, default=0, help='Experimental dynamic placement refresh interval; 0 disables refresh')
parser.add_argument('--reset-dynamic-residency', action='store_true', help='Reset stale dynamic budget limits and reprioritize resident TTS pages before requests')
parser.add_argument('--semantic-top-p-crossing', action='store_true')
parser.add_argument('--semantic-top-k', type=int, help='Override semantic main/RAS top-k only, retaining acoustic sampling')
parser.add_argument('--semantic-top-p', type=float, help='Override semantic main/RAS top-p only, retaining acoustic sampling')
parser.add_argument('--semantic-from-chunk', type=int, default=0, help='First text chunk receiving semantic sampling overrides')
parser.add_argument('--trace-generation', action='store_true', help='Retain semantic logits on GPU and flush only after each complete take')
parser.add_argument('--retry-stalled', type=int, default=0, help='Retry a stalled chunk with unrestricted semantic candidates; reject it if retries also stall')
parser.add_argument('--profile-service', action='store_true', help='Profile each take and count actual CUDA graph replays for service slowdown diagnosis')
parser.add_argument('--disable-nvml-pressure', action='store_true', help='Disable AIMDO NVML pressure feedback for a controlled residency-budget comparison')
parser.add_argument('--resident-vbar-floor', action='store_true', help='Keep the prepared TTS dynamic-weight range resident when it fits on the assigned GPUs')
parser.add_argument('--allocator-owner-cleanup', action='store_true', help='Destroy each AIMDO allocation graph under its owning CUDA and accounting device context')
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, args.comfy)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
assets = Path(args.assets)
manifest = json.loads((assets / 'references.json').read_text())
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / 'runtime'))
sys.path.insert(0, str(project / 'experiments'))
faulthandler.dump_traceback_later(60, repeat=True)

from comfy.cli_args import args as comfy_args
if args.disable_nvml_pressure:
    comfy_args.disable_nvml_pressure = True
if args.disable_comfy_compiler:
    comfy_args.disable_comfy_compiler = True
if args.dynamic_vram:
    import comfy_aimdo.control as aimdo
    aimdo.init(simple_vram_headroom=None, nvml_pressure=not comfy_args.disable_nvml_pressure)

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import folder_paths
import comfy.model_management as mm
if args.dynamic_vram:
    import comfy.memory_management
    import comfy.model_patcher
    if not aimdo.init_devices((d.index, int(comfy_args.vram_headroom * 2**30))
                              for d in mm.get_all_torch_devices()):
        raise RuntimeError('ComfyUI dynamic allocator initialization failed')
    aimdo.set_log_info()
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    print('DynamicVRAM initialized as in ComfyUI main.py', flush=True)
if args.allocator_owner_cleanup:
    import comfy_aimdo.malloc_graph as allocation_graph
    destroy_graph = allocation_graph.MallocGraph.__del__

    def destroy_on_owner(graph):
        if getattr(graph, '_handle', None):
            device = graph._stream.device
            with torch.cuda.device(device):
                # This exported query also selects AIMDO's thread-local accounting context.
                aimdo.lib.get_total_vram_usage(aimdo.get_devctx(device.index))
                destroy_graph(graph)

    allocation_graph.MallocGraph.__del__ = destroy_on_owner

pack = project / 'upstream/comfy-drama3'
spec = importlib.util.spec_from_file_location('drama3_pack', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
faulthandler.cancel_dump_traceback_later()
nodes = sys.modules['drama3_pack.nodes']
if args.profile_service:
    from profile_a_service import install
    install(sys.modules['drama3_pack.drama3.generate'], out)
tail_trace = None
if args.trace_generation or args.semantic_top_k is not None or args.semantic_top_p is not None or args.retry_stalled:
    from stalled_chunk import install as install_tail
    tail_trace = install_tail(sys.modules['drama3_pack.drama3.generate'], out,
                              args.trace_generation, args.semantic_top_k, args.semantic_top_p,
                              args.semantic_from_chunk, args.retry_stalled)
trace_sampling = None
trace_opening = None
if args.trace_opening:
    from diagnostic_a_opening import install as install_opening
    trace_opening = install_opening(sys.modules['drama3_pack.drama3.generate'],out)
if (args.trace_sampling or args.semantic_top_p_crossing) and os.environ.get('DRAMA3_FORCE_BAD_PREFIX') != '1':
    from diagnostic_a_sampling import install
    trace_sampling = install(sys.modules['drama3_pack.drama3.generate'], out, args.semantic_top_p_crossing)
folder_paths.add_model_folder_path('diffusion_models', str(Path(args.model).parent))
folder_paths.add_model_folder_path('vae', str(Path(args.codec).parent))
torch.set_num_threads(16)
torch.set_grad_enabled(False)
metrics = {
    'backend': 'ComfyUI-Drama3', 'model': args.model,
    'comfy_root': args.comfy,
    'comfy_version': __import__('comfyui_version').__version__,
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'references': args.references.split(',') if args.references else manifest['baseline_references'],
    'reference_cache': args.reuse_voices,
    'dynamic_vram': args.dynamic_vram,
    'disable_comfy_compiler': args.disable_comfy_compiler,
    'disable_nvml_pressure': args.disable_nvml_pressure,
    'allocator_owner_cleanup': args.allocator_owner_cleanup,
    'sampling': {'temperature': 1.0, 'top_p': 0.9, 'top_k': args.top_k, 'max_frames_per_chunk': 600},
    'semantic_sampling_override': {'top_k': args.semantic_top_k, 'top_p': args.semantic_top_p,
                                   'from_chunk': args.semantic_from_chunk},
    'stalled_chunk_retries': args.retry_stalled,
    'prompt_ablation': args.prompt_ablation,
    'split_long_text': True, 'speakers': 'one take per line',
    'reference_processing': 'upstream tidy_clip: mono, trim quiet edges, peak normalization',
    'output_processing': 'upstream shorten_long_pauses=True',
    'imports_seconds': time.perf_counter() - START,
    'process_started_epoch': time.time() - (time.perf_counter() - START),
    'stages': {}, 'takes': [],
}


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def stage(name, operation):
    sync()
    start = time.perf_counter()
    value = operation()
    sync()
    metrics['stages'][name] = time.perf_counter() - start
    print('STAGE', name, metrics['stages'][name], flush=True)
    return value


def persist():
    metrics['updated_epoch'] = time.time()
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))


with torch.inference_mode():
    codec = stage('load_codec', lambda: nodes.Drama3CodecLoader.execute(Path(args.codec).name)[0])
    if args.codec_device:
        codec.patcher.load_device = torch.device(args.codec_device)
        metrics['codec_placement'] = f'managed {args.codec_device}'
    if args.resident_codec:
        codec_device = torch.device(f'cuda:{torch.cuda.device_count()-1}')
        codec.model.to(codec_device)
        def resident_decode(codec, codes, gap=0.0):
            takes = [take for take in codes['codes'].split(codes['takes'],-1) if take.shape[-1]]
            if not takes:
                return torch.zeros((1,nodes.FRAME_LENGTH))
            with mm.cuda_device_context(codec_device):
                audio = [codec.model.decode(take[None].to(codec_device),torch.float32).float().cpu() for take in takes]
            silence = torch.zeros((1,round(gap*nodes.SAMPLE_RATE)))
            return torch.cat([piece for take in audio for piece in (silence,take)][1:],-1)
        nodes.decode = resident_decode
        metrics['codec_placement'] = f'unmanaged resident {codec_device}; TTS remains loaded'
    manifest = json.loads((assets / 'references.json').read_text())
    by_id = {r['id']: r for r in manifest['references']}
    voices = {}
    cache_dir = Path(os.environ.get('DRAMA3_COMFY_VOICE_CACHE', out.parent / 'voice-cache-comfy'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for slot, ref_id in enumerate(metrics['references']):
        ref = by_id[ref_id]
        transcript = '<|speaker:0|>' + ref['transcript']
        cached = cache_dir / f'{ref_id}.safetensors'
        if args.reuse_voices:
            with safe_open(cached, framework='pt') as f:
                assert f.metadata()['transcript'] == transcript
                voice = {'codes': f.get_tensor('codes').long(), 'transcript': transcript}
        else:
            waveform, rate = sf.read(assets / ref['audio'], dtype='float32', always_2d=True)
            audio = {'waveform': torch.from_numpy(waveform.T.copy())[None], 'sample_rate': rate}
            voice = stage(f'encode_{ref_id}', lambda: nodes.Drama3EncodeVoice.execute(codec, audio, transcript)[0])
            save_file({'codes': voice['codes'].short().contiguous()}, cached,
                      metadata={'transcript': transcript, 'source': ref['audio']})
        # Both references explicitly carry speaker 0, overriding the voice1 slot's default.
        voices[f'voice{slot}'] = voice
    if args.encode_only:
        metrics['elapsed_process_seconds'] = time.perf_counter() - START
        persist()
        print('VOICE CACHE READY', cache_dir, flush=True)
        raise SystemExit(0)
    model = stage('load_model', lambda: nodes.Drama3Loader.execute(
        Path(args.model).name, experts='in VRAM', second_gpu='cuda:1', split_layer=args.split_layer)[0])
    persist()
    prompt_path = Path(args.prompt_file) if args.prompt_file else assets / 'prompt.txt'
    if not args.prompt_file and not prompt_path.is_file():
        prompt_path = assets / 'makima_ja.txt'
    text = prompt_path.read_text()
    (out / 'prompt.txt').write_text(text)
    original_split_text = nodes.split_text

    def split_for_ablation(script, *positional, **keywords):
        chunks = original_split_text(script, *positional, **keywords)
        if args.prompt_ablation == 'no-pauses':
            chunks = [chunk.replace('[pause]', '') for chunk in chunks]
        elif args.prompt_ablation == 'dialogue-only':
            chunks = [re.sub(r'\[[^\[\]]*\]', '', chunk) for chunk in chunks]
        (out / 'input-chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
        return chunks

    nodes.split_text = split_for_ablation
    decoder = None
    if args.external_codec_device:
        decoder_log = open(out / 'external-codec.log', 'w')
        decoder = subprocess.Popen(
            [sys.executable, '-u', str(project / 'scripts/isolated_comfy_codec.py'),
             '--comfy', args.comfy, '--codec', args.codec, '--device', args.external_codec_device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=decoder_log,
            text=True, bufsize=1)
        ready_line = decoder.stdout.readline()
        if not ready_line or json.loads(ready_line).get('type') != 'ready':
            raise RuntimeError('Isolated codec failed to start; see external-codec.log')
        metrics['codec_placement'] = f'isolated process {args.external_codec_device}'
        (out / 'raw-decode').mkdir(exist_ok=True)

        def external_decode(codec, codes, gap=0.0):
            raw_path = out / 'raw-decode' / f'seed-{seed}.wav'
            decoder.stdin.write(json.dumps({'codes': str(out / f'seed-{seed}.codes.safetensors'),
                                            'wav': str(raw_path), 'gap': gap}) + '\n')
            decoder.stdin.flush()
            response_line = decoder.stdout.readline()
            if not response_line:
                raise RuntimeError('Isolated codec stopped; see external-codec.log')
            take['external_codec'] = json.loads(response_line)
            waveform, rate = sf.read(raw_path, dtype='float32', always_2d=True)
            assert rate == nodes.SAMPLE_RATE
            return torch.from_numpy(waveform.T.copy())

        nodes.decode = external_decode
    original_generate = nodes.generate
    if args.keep_tts_resident:
        if not args.external_codec_device:
            raise ValueError('--keep-tts-resident requires an isolated codec')
        original_load_models = mm.load_models_gpu
        tts_patchers = set(model.patchers())
        tts_prepared = False
        tts_requests = 0

        def load_tts_once(models, *positional, **keywords):
            global tts_prepared, tts_requests
            if set(models) == tts_patchers:
                refresh = args.tts_refresh_every and tts_requests % args.tts_refresh_every == 0
                tts_requests += 1
                if tts_prepared and not refresh:
                    return
                result = original_load_models(models, *positional, **keywords)
                if args.resident_vbar_floor:
                    metrics['resident_vbar_floor'] = []
                    for patcher in models:
                        vbar = patcher._vbar_get()
                        vbar.set_watermark_limit(vbar.offset)
                        vbar.prioritize()
                        metrics['resident_vbar_floor'].append({'device': vbar.device, 'bytes': vbar.offset})
                tts_prepared = True
                return result
            return original_load_models(models, *positional, **keywords)

        mm.load_models_gpu = load_tts_once
        metrics['keep_tts_resident'] = True
        metrics['tts_refresh_every'] = args.tts_refresh_every
    for seed in map(int, args.seeds.split(',')):
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        take = {'seed': seed, 'first_frame_seconds': None, 'chunks': []}
        sync()
        request_start = time.perf_counter()
        if args.reset_dynamic_residency:
            import comfy_aimdo.model_vbar as model_vbar
            model_vbar.vbars_reset_watermark_limits()
            take['dynamic_residency'] = []
            for patcher in model.patchers():
                vbar = patcher._vbar_get()
                if vbar is not None:
                    before = {'loaded': vbar.loaded_size(), 'watermark': vbar.get_watermark()}
                    vbar.prioritize()
                    take['dynamic_residency'].append({'before': before,
                        'after': {'loaded': vbar.loaded_size(), 'watermark': vbar.get_watermark()}})

        def instrumented_generate(model, prompt, max_frames, temperature, top_k, top_p, seed, on_frame=None, **kwargs):
            if tail_trace:
                tail_trace(seed - len(take['chunks']), len(take['chunks']))
            if trace_opening:
                trace_opening[0]()
            if os.environ.get('DRAMA3_FORCE_BAD_PREFIX') == '1':
                save_file({'prompt': prompt.cpu().contiguous()}, out / 'forced-prompt.safetensors')
                from diagnostic_a_sampling import force_prefix_logits
                force_prefix_logits(model, prompt, out)
                raise SystemExit(0)
            chunk_start = time.perf_counter()
            chunk = {'prompt_rows': len(prompt), 'first_frame_seconds': None}

            def observed_frame(step):
                now = time.perf_counter()
                if chunk['first_frame_seconds'] is None:
                    chunk['first_frame_seconds'] = now - chunk_start
                if take['first_frame_seconds'] is None:
                    take['first_frame_seconds'] = now - request_start
                if on_frame:
                    on_frame(step)

            frames = original_generate(model, prompt, max_frames, temperature, top_k, top_p,
                                       seed, observed_frame, **kwargs)
            if trace_opening:
                trace_opening[1](seed,len(take['chunks']),frames,prompt)
            chunk.update(seconds=time.perf_counter() - chunk_start, frames=len(frames))
            take['chunks'].append(chunk)
            print('CHUNK', json.dumps(chunk), flush=True)
            return frames

        nodes.generate = instrumented_generate
        codes = nodes.Drama3Generate.execute(model, text, voices, seed, 1.0, 0.9, args.top_k, 600)[0]
        sync()
        take['generation_seconds'] = time.perf_counter() - request_start
        if trace_sampling:
            trace_sampling(seed)
        save_file({'codes': codes['codes'].short().contiguous()}, out / f'seed-{seed}.codes.safetensors',
                  metadata={'lines': json.dumps(codes['lines']), 'takes': json.dumps(codes['takes'])})
        if args.codes_only:
            take['request_seconds'] = time.perf_counter() - request_start
            metrics['takes'].append(take)
            persist()
            print('TAKE CODES',json.dumps(take),flush=True)
            continue
        decode_start = time.perf_counter()
        audio = nodes.Drama3Decode.execute(codec, codes, shorten_long_pauses=True)[0]
        sync()
        take['decode_seconds'] = time.perf_counter() - decode_start
        take['request_seconds'] = time.perf_counter() - request_start
        # This native node returns a complete waveform: first frame is not first playable audio.
        take['first_playable_audio_seconds'] = take['request_seconds']
        samples = audio['waveform'][0, 0].numpy()
        sf.write(out / f'seed-{seed}.wav', samples, audio['sample_rate'], subtype='PCM_16')
        take['audio_seconds'] = len(samples) / audio['sample_rate']
        take['rtfx'] = take['audio_seconds'] / take['request_seconds']
        take['peak_dbfs'] = float(20 * np.log10(max(float(np.abs(samples).max()), 1e-12)))
        take['rms_dbfs'] = float(20 * np.log10(max(float(np.sqrt(np.mean(samples**2))), 1e-12)))
        take['gpu_peak_allocated_gib'] = [torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())]
        take['gpu_peak_reserved_gib'] = [torch.cuda.max_memory_reserved(i)/2**30 for i in range(torch.cuda.device_count())]
        metrics['takes'].append(take)
        metrics['elapsed_process_seconds'] = time.perf_counter() - START
        persist()
        print('TAKE', json.dumps(take), flush=True)
    if decoder is not None:
        decoder.stdin.close()
        decoder.wait(timeout=30)
        decoder_log.close()
