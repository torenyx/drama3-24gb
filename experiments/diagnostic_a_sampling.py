"""Controlled semantic-only nucleus ablation; retain acoustic policy and RNG draws."""
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def force_prefix_logits(drama3, prompt, directory):
    """Teacher-force recorded bad frames through the backbone, without sampling/graphs."""
    import comfy.model_management
    from drama3_pack.drama3.generate import embed_rows
    backbone = drama3.model.model
    dtype = drama3.model.dtype
    layers, devices, norm, audio_head, tower = drama3.parts()
    first, last = devices[0], devices[-1]
    with safe_open(os.environ['DRAMA3_BAD_PREFIX_CODES'], framework='pt') as f:
        count = json.loads(f.metadata()['takes'])[0]
        codes = f.get_tensor('codes')[:, :count].long().to(last)
    frames = torch.cat((codes[:1] + drama3.config.semantic_start_token_id, codes)).T
    rows = prompt.to(last)
    caches = backbone.init_caches(len(rows) + len(frames) + 1, layers, devices, dtype)
    def forward(rows):
        text = embed_rows(backbone.embed_tokens, rows[:, 0].to(first), dtype)
        x = tower.embed_backbone(rows, text)[None]
        return norm(backbone.run_layers(x, caches, layers, devices))
    logits = []
    with drama3.session(), comfy.model_management.cuda_device_context(last):
        hidden = forward(rows)
        logits.append(audio_head(hidden[0, -1:]).float().cpu())
        for frame in frames:
            hidden = forward(frame[None])
            logits.append(audio_head(hidden[0, -1:]).float().cpu())
    save_file({'semantic_logits': torch.cat(logits)}, Path(directory, 'forced-prefix-logits.safetensors'))
    print('FORCED PREFIX', len(frames), 'frames', len(logits), 'logit rows', flush=True)


def install(module, directory, crossing):
    original = module.sample_top_k_top_p
    rows = []

    def sample(logits, temperature, top_k, top_p, generator):
        semantic = logits.shape[-1] == 4097
        if not semantic:
            return original(logits, temperature, top_k, top_p, generator)
        baseline_pick = None if crossing else original(logits, temperature, top_k, top_p, generator)
        values, ids = torch.sort(logits.float() / max(temperature, module.MIN_TEMPERATURE), descending=True, dim=-1)
        probability = values.softmax(-1)
        cumulative = probability.cumsum(-1)
        dropped = cumulative > top_p
        if crossing and semantic:
            dropped[..., 1:] = dropped[..., :-1].clone()
        dropped |= torch.arange(values.shape[-1], device=values.device) >= (top_k or values.shape[-1])
        dropped[..., 0] = False
        probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
        if baseline_pick is None:
            noise = torch.empty_like(probabilities).exponential_(generator=generator)
            picked = ids.gather(-1, (probabilities / noise).argmax(-1, keepdim=True)).squeeze(-1)
        else:
            picked = baseline_pick
        if semantic:
            rank = int((ids[0] == 0).nonzero()[0])
            rows.append({'call': len(rows), 'top_k': top_k, 'top_p': top_p,
                         'picked': int(picked[0]), 'eos_rank': rank + 1,
                         'eos_probability': float(probability[0, rank]),
                         'eos_retained': not bool(dropped[0, rank]),
                         'top1': int(ids[0, 0]), 'top1_probability': float(probability[0, 0])})
        return picked

    module.sample_top_k_top_p = sample

    def persist(seed):
        Path(directory, f'seed-{seed}.semantic-trace.json').write_text(json.dumps(rows, indent=2))
        rows.clear()

    return persist
