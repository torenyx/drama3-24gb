"""Measure real HTTP PCM arrival for one resident Drama3 service."""
import argparse
import json
from pathlib import Path
import time
import wave

import requests

p=argparse.ArgumentParser()
p.add_argument('--assets',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--url',default='http://127.0.0.1:18080')
p.add_argument('--takes',type=int,default=4)
p.add_argument('--top-k',type=int)
p.add_argument('--top-p',type=float,default=.9)
p.add_argument('--temperature',type=float,default=1.)
p.add_argument('--references',help='Comma-separated reference ids; otherwise baseline references')
p.add_argument('--prompt-file',type=Path)
p.add_argument('--placement',default='dual-stage')
p.add_argument('--wait-ready',type=float,default=600)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.assets/'references.json').read_text())
by_id={r['id']:r for r in manifest['references']}
reference_ids=a.references.split(',') if a.references else manifest['baseline_references']
references=[{'audio_path':str(a.assets/by_id[i]['audio']),
             'text':by_id[i]['transcript']} for i in reference_ids]
prompt_path = a.prompt_file or a.assets/'prompt.txt'
if a.prompt_file is None and not prompt_path.is_file():
    prompt_path = a.assets/'makima_ja.txt'
body={'input':prompt_path.read_text(),'references':references,
      'temperature':a.temperature,'top_p':a.top_p,'max_new_tokens':1200,
      'stream':True,'response_format':'pcm'}
if a.top_k is not None:
    body['top_k']=a.top_k
(a.out/'request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2))
metrics={'backend':'SGLang-Omni INT4','placement':a.placement,'references':reference_ids,
         'sampling':{'temperature':a.temperature,'top_p':a.top_p,'top_k':a.top_k or 'upstream default (uncapped main token)'},
         'seed_note':'This upstream speech API refuses seed; trials are repeated requests, not matched-seed A/B.',
         'process_started_epoch':time.time(),'takes':[]}
session=requests.Session()
session.trust_env=False
ready_start=time.perf_counter()
while True:
    try:
        health=session.get(a.url+'/health',timeout=3)
        if health.status_code==200:
            break
    except requests.RequestException:
        pass
    if time.perf_counter()-ready_start>a.wait_ready:
        raise TimeoutError('SGLang service did not become healthy')
    time.sleep(3)
metrics['wait_ready_seconds']=time.perf_counter()-ready_start
print('Service ready after client wait',metrics['wait_ready_seconds'],flush=True)
for trial in range(a.takes):
    start=time.perf_counter()
    take={'trial':trial,'cold_request':trial==0,'started_epoch':time.time()}
    chunks=[]
    try:
        with session.post(a.url+'/v1/audio/speech',json=body,stream=True,timeout=(10,600)) as response:
            take['status']=response.status_code
            take['headers']=dict(response.headers)
            if response.status_code >= 400:
                take['response_error'] = response.text
            response.raise_for_status()
            rate=int(response.headers['X-Sample-Rate'])
            assert response.headers['X-Channels']=='1' and response.headers['X-Bit-Depth']=='16'
            for chunk in response.iter_content(chunk_size=None):
                if not chunk:
                    continue
                if not chunks:
                    take['ttfa_seconds']=time.perf_counter()-start
                chunks.append(chunk)
        raw=b''.join(chunks)
        assert raw and len(raw)%2==0,'Expected nonempty mono PCM16'
        path=a.out/f'trial-{trial}.wav'
        with wave.open(str(path),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(raw)
        take.update(request_seconds=time.perf_counter()-start,audio_seconds=len(raw)/2/rate,
                    sample_rate=rate,pcm_chunks=len(chunks),pcm_bytes=len(raw))
        take['rtfx']=take['audio_seconds']/take['request_seconds']
    except Exception as exc:
        take['error']=repr(exc)
        take['request_seconds']=time.perf_counter()-start
        if chunks:
            (a.out/f'trial-{trial}-partial.pcm').write_bytes(b''.join(chunks))
    metrics['takes'].append(take)
    metrics['updated_epoch']=time.time()
    (a.out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print(json.dumps(take,ensure_ascii=False),flush=True)
    if 'error' in take:
        raise RuntimeError(take['error'])
