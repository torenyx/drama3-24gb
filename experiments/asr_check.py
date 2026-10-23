"""Transcribe test WAVs and retain literal text plus a normalized character audit."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

p=argparse.ArgumentParser()
p.add_argument('--comfy', required=True)
p.add_argument('--model', required=True)
p.add_argument('--language', default='Japanese')
p.add_argument('--runs', required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--groups', default='a-formal,a-cu130,b-single')
p.add_argument('--out', default='asr-check.json')
a=p.parse_args()
sys.argv=[sys.argv[0]]
sys.path.insert(0,a.comfy)
import numpy as np
import soundfile as sf
import torch
import folder_paths
torch.set_num_threads(16)
torch.set_grad_enabled(False)
pack=Path(__file__).resolve().parents[1]/'upstream/comfy-drama3'
spec=importlib.util.spec_from_file_location('drama3_pack',pack/'__init__.py',submodule_search_locations=[str(pack)])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
nodes=sys.modules['drama3_pack.nodes']
folder_paths.add_model_folder_path('asr',str(Path(a.model).parent))
model=nodes.Qwen3ASRLoader.execute(Path(a.model).name)[0]

def normalize(text):
    text=unicodedata.normalize('NFKC',text)
    return ''.join(c for c in text if not unicodedata.category(c).startswith(('P','Z','C')))

def distance(x,y):
    row=list(range(len(y)+1))
    for i,c in enumerate(x,1):
        new=[i]
        for j,d in enumerate(y,1):
            new.append(min(new[-1]+1,row[j]+1,row[j-1]+(c!=d)))
        row=new
    return row[-1]

expected=re.sub(r'<\|speaker:\d+\|>|\[[^\]]*\]','',Path(a.prompt).read_text())
normal=normalize(expected)
result={'asr_model':a.model,'language':a.language,'expected':expected,'normalized_expected':normal,
        'note':'CER is an ASR transcript diagnostic; spelling differences need review. No voice-similarity score.','samples':[]}
root=Path(a.runs)
with torch.inference_mode():
    for group in a.groups.split(','):
        for path in sorted((root/group).glob('*.wav')):
            samples,rate=sf.read(path,dtype='float32',always_2d=True)
            audio={'waveform':torch.from_numpy(samples.T.copy())[None],'sample_rate':rate}
            start=time.perf_counter()
            text=nodes.Qwen3ASRTranscribe.execute(model,audio,language=a.language,context='',max_new_tokens=4096)[0]
            heard=normalize(text)
            edits=[{'kind':tag,'expected':normal[i:j],'heard':heard[k:l]}
                   for tag,i,j,k,l in difflib.SequenceMatcher(None,normal,heard,autojunk=False).get_opcodes() if tag!='equal']
            record={'file':str(path.relative_to(root)),'transcript':text,'normalized':heard,
                    'cer':distance(normal,heard)/len(normal),'edits':edits,'asr_seconds':time.perf_counter()-start}
            result['samples'].append(record)
            (root/a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print(json.dumps(record,ensure_ascii=False),flush=True)
