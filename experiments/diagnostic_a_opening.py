"""Observe the first two semantic draws without replacing sampling arithmetic."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

def install(module, directory):
    original = module.sample_top_k_top_p
    captured = []
    def sample(*args, **kwargs):
        chosen = original(*args, **kwargs)
        if args[0].shape[-1] == 4097 and len(captured)<2:
            captured.append((args[0].detach().float().cpu(), chosen.detach().cpu(),
                {'temperature': args[1], 'top_k': args[2], 'top_p':args[3]}))
        return chosen
    module.sample_top_k_top_p = sample
    def start():
        captured.clear()
    def finish(seed, index, frames, prompt):
        tensors = {'prompt':prompt.cpu(), 'frames':frames.cpu()}
        for i,(logits,picked,_) in enumerate(captured):
            tensors[f'logits-{i}'] = logits
            tensors[f'draw-{i}'] = picked
        save_file(tensors,Path(directory,f'opening-seed-{seed}-chunk-{index}.safetensors'),
            metadata={'sampling':json.dumps([r[2] for r in captured])})
    return start, finish
