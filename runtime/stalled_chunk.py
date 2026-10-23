"""Observe complete semantic trajectories without host synchronization inside sampling."""
import json
import logging
from pathlib import Path
import torch
from safetensors.torch import save_file


def install(module, directory, trace, semantic_k, semantic_p, from_chunk=0, retry_stalled=0):
    original_sample, original_take = module.sample_top_k_top_p, module.take
    state = {'buffer': None, 'call': 0, 'seed': 0, 'chunk': 0, 'attempt': 0, 'retrying': False}
    events = []

    def sample(logits, temperature, top_k, top_p, generator):
        semantic = logits.shape[-1] == 4097
        if semantic and state['chunk'] >= from_chunk:
            top_k = top_k if semantic_k is None else semantic_k
            top_p = top_p if semantic_p is None else semantic_p
        if semantic and state['retrying']:
            top_k, top_p = 0, 1.0
        chosen = original_sample(logits, temperature, top_k, top_p, generator)
        if semantic and trace:
            index = state['call']
            state['buffer'][index].copy_(logits[0])
            state['picked'][index].copy_(chosen[0])
            state['call'] += 1
        return chosen

    def observed_take(drama3, prompt, max_frames, *args, **kwargs):
        if trace:
            device = drama3.parts()[1][-1]
            state['buffer'] = torch.empty((2 * max_frames, 4097), device=device, dtype=torch.float32)
            state['picked'] = torch.empty((2 * max_frames,), device=device, dtype=torch.long)
            state['call'] = 0
        frames, stalled = original_take(drama3, prompt, max_frames, *args, **kwargs)
        if trace:
            n = state['call']
            name = f"tail-seed-{state['seed']}-chunk-{state['chunk']}-attempt-{state['attempt']}"
            save_file({'logits': state['buffer'][:n].cpu().contiguous(),
                       'picked': state['picked'][:n].cpu().contiguous(), 'frames': frames.cpu()},
                      Path(directory, name + '.safetensors'),
                      metadata={'stalled': json.dumps(stalled), 'prompt_rows': str(len(prompt))})
            state['buffer'] = None
            state['picked'] = None
        state['attempt'] += 1
        return frames, stalled

    def take(drama3, prompt, max_frames, *args, **kwargs):
        for retry in range(retry_stalled + 1):
            state['retrying'] = retry > 0
            frames, stalled = observed_take(drama3, prompt, max_frames, *args, **kwargs)
            if retry_stalled:
                events.append({'seed': state['seed'], 'chunk': state['chunk'], 'retry': retry,
                               'frames': len(frames), 'stalled': stalled,
                               'semantic_fallback': state['retrying']})
                Path(directory, 'generation-status.json').write_text(json.dumps(events, indent=2))
            if not stalled:
                state['retrying'] = False
                return frames, stalled
            if retry < retry_stalled:
                logging.warning('Drama 3: incomplete chunk %s stalled; retrying the same text with semantic k=0,p=1.', state['chunk'])
        state['retrying'] = False
        if retry_stalled and stalled:
            raise RuntimeError(f"Chunk {state['chunk']} stalled after {retry_stalled + 1} attempts; incomplete output rejected")
        return frames, stalled

    module.sample_top_k_top_p = sample
    module.take = take

    def start(seed, chunk):
        state.update(seed=seed, chunk=chunk, attempt=0, retrying=False)
    return start
