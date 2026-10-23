"""Observe identical repeated requests without changing generation or placement."""
import cProfile
import json
import pstats
import time
from collections import Counter

import torch
import comfy.model_prefetch as prefetch
import comfy_aimdo.model_vbar as vbar_module
import comfy_aimdo.control as control


def install(module, out):
    take = module.take
    replay = torch.cuda.CUDAGraph.replay
    counts = Counter()
    sequence = 0
    fault = vbar_module.ModelVBAR.fault
    pin = prefetch.pin_modules

    def observed_fault(vbar, alloc, size):
        result = fault(vbar, alloc, size)
        counts[f'fault_{vbar.device}_{"oom" if result is None else "ok"}'] += 1
        return result

    def observed_pin(modules, device, dtype=None):
        result = pin(modules, device, dtype)
        counts[f'pin_{device}_{result[1]}'] += 1
        return result

    vbar_module.ModelVBAR.fault = observed_fault
    prefetch.pin_modules = observed_pin

    def observed_replay(graph):
        counts['replay_calls'] += 1
        return replay(graph)

    torch.cuda.CUDAGraph.replay = observed_replay

    def observed_take(drama3, *args, **kwargs):
        nonlocal sequence
        sequence += 1
        counts.clear()
        layers, devices, *_ = drama3.parts()
        diagnostic = {'sequence': sequence,
                      'compile': [module.comfy.model_prefetch.malloc_graph_enabled(d) for d in set(devices)],
                      'sync_free': all(l.mlp.experts.sync_free(d) for l, d in zip(layers, devices))}
        profile = cProfile.Profile()
        started = time.perf_counter()
        profile.enable()
        try:
            return take(drama3, *args, **kwargs)
        finally:
            profile.disable()
            diagnostic.update(seconds=time.perf_counter() - started, counts=dict(counts))
            diagnostic['residency'] = []
            for patcher in drama3.patchers():
                vbar = patcher._vbar_get()
                statuses = vbar.get_residency()
                diagnostic['residency'].append({'device': vbar.device, 'pages': len(statuses),
                    'resident': sum(bool(s & 1) for s in statuses),
                    'pinned': sum(bool(s & 2) for s in statuses), 'watermark': vbar.get_watermark(),
                    'offset': vbar.offset,
                    'aimdo_recorded': control.lib.get_total_vram_usage(vbar._devctx),
                    'cuda_free': torch.cuda.mem_get_info(vbar.device)[0]})
            profile.dump_stats(str(out / f'profile-{sequence:02d}.pstats'))
            with (out / f'profile-{sequence:02d}.txt').open('w') as stream:
                pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(45)
            with (out / 'profile-service.jsonl').open('a') as stream:
                stream.write(json.dumps(diagnostic) + '\n')
            print('SERVICE PROFILE', json.dumps(diagnostic), flush=True)

    module.take = observed_take
