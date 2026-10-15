#!/usr/bin/env python3
"""mkcfg14.py BASE.json TAG [K=V ...] [+ARG ...] -> ~/strata-decode15/cfg/TAG.json : BASE + env additions (K=- removes) + model_name suffix (ON the mini).
   An argument beginning with '@--flag=V' sets that flag's value.  An argument beginning with '+' is appended to the engine args ('+--kv-grow'); '-ARG' removes one occurrence (flag only)."""
import json, os, sys
base, tag = sys.argv[1], sys.argv[2]
c = json.load(open(os.path.expanduser(base))); e = c['env']
for kv in sys.argv[3:]:
    if kv.startswith('@'):   # @--flag=value: replace the value that follows --flag in the engine args
        f, v = kv[1:].split('=', 1); i = c['args'].index(f); c['args'][i + 1] = v; continue
    if kv.startswith('+'): c['args'].append(kv[1:]); continue
    if kv.startswith('-') and '=' not in kv: c['args'].remove(kv[1:]); continue
    k, v = kv.split('=', 1)
    if v == '-': e.pop(k, None)
    else: e[k] = os.path.expanduser(v)
c['model_name'] = c['model_name'].split("-d13")[0] + '-d15-' + tag
out = os.path.expanduser('~/strata-decode15/cfg/%s.json' % tag)
json.dump(c, open(out, 'w'), indent=1); print(out)
