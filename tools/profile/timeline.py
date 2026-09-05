import json,collections,re,sys
d=json.load(open(sys.argv[1])); ev=d["traceEvents"]
pname={}
for e in ev:
    if e.get("ph")=="M" and e["name"]=="process_name": pname[e["pid"]]=e["args"]["name"]
stacks=collections.defaultdict(list); spans=[]
for e in sorted([e for e in ev if e.get("ph") in ("B","E")], key=lambda e:(e["ts"], 0 if e["ph"]=="E" else 1)):
    k=(e["pid"],e["tid"])
    if e["ph"]=="B": stacks[k].append(e)
    elif stacks[k]:
        b=stacks[k].pop(); spans.append((b["pid"],b.get("name",""),b["ts"],e["ts"]-b["ts"]))
spans=[s for s in spans if s[1] not in ("null descriptor","VPU_DEBUG_TRACE_EVENT")]
t0=min(s[2] for s in spans); t1=max(s[2]+s[3] for s in spans); print("window %.3f ms, %d spans" % ((t1-t0)/1000, len(spans)))
def optype(n):
    n=re.sub(r"_optimized_bundle_\d+","",n); n=re.sub(r"_bundle_\d+","",n); n=re.sub(r"/op_\d+.*$","",n); n=re.sub(r"_\d+$","",n); return n
agg={}
lay=sys.argv[2] if len(sys.argv)>2 else "l1_"
for pid,name,ts,dur in spans:
    o=optype(name)
    if not o.startswith(lay): continue
    eng=pname.get(pid,str(pid))
    a=agg.setdefault((o,eng),[ts,ts+dur,0,0]); a[0]=min(a[0],ts); a[1]=max(a[1],ts+dur); a[2]+=1; a[3]+=dur
for (o,eng),(s,e,c,busy) in sorted(agg.items(), key=lambda kv: kv[1][0]):
    if e-s < 3 and c < 10: continue
    print("  %-40s %-6s %7.3f %7.3f %4d busy %6.3f" % (o[:40], eng.replace(" (accel","").rstrip(")"), (s-t0)/1000, (e-t0)/1000, c, busy/1000))
