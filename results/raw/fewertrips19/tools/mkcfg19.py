#!/usr/bin/env python3
"""mkcfg19.py BASE.json TAG [K=V ...] [+ARG ...] [-ARG] [@--flag=V] -> ~/strata-fewertrips19/cfg/TAG.json : BASE + env additions (K=- removes) + model_name suffix (ON the mini).  (mkcfg18 with the FewerTrips19 paths)"""
import json, os, sys
base, tag = sys.argv[1], sys.argv[2]
c = json.load(open(os.path.expanduser(base))); e = c['env']
for kv in sys.argv[3:]:
    if kv.startswith('@'):
        f, v = kv[1:].split('=', 1); i = c['args'].index(f); c['args'][i + 1] = v; continue
    if kv.startswith('+'): c['args'].append(kv[1:]); continue
    if kv.startswith('-') and '=' not in kv: c['args'].remove(kv[1:]); continue
    k, v = kv.split('=', 1)
    if v == '-': e.pop(k, None)
    else: e[k] = os.path.expanduser(v)
c['model_name'] = c['model_name'].split("-d15")[0] + '-ft19-' + tag
out = os.path.expanduser('~/strata-fewertrips19/cfg/%s.json' % tag)
json.dump(c, open(out, 'w'), indent=1); print(out)
