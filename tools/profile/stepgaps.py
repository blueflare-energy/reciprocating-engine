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
        b=stacks[k].pop(); spans.append((pname.get(b["pid"],str(b["pid"])),b.get("name",""),b["ts"],e["ts"]-b["ts"]))
spans=[s for s in spans if s[1] not in ("null descriptor","VPU_DEBUG_TRACE_EVENT") and not s[0].startswith("Host")]
spans.sort(key=lambda s:s[2])
def cls(eng): return re.sub(r"\d+.*","",eng.split(" ")[0])
def op(n):
    n=re.sub(r"_optimized_bundle_\d+","",n); n=re.sub(r"_bundle_\d+","",n); n=re.sub(r"/op_\d+.*$","",n); n=re.sub(r"_\d+$","",n); return n
# launch boundaries: starts of l0_norm1 groups separated by > 50 us
starts=[s[2] for s in spans if s[1].startswith("l0_norm1")]
bounds=[starts[0]]
for t in starts[1:]:
    if t > bounds[-1]+50: bounds.append(t)
print("launches by l0_norm1:", len(bounds))
bounds.append(spans[-1][2]+1)
def union(iv):
    iv=sorted(iv); tot=0; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    return tot+ce-cs
for li in range(len(bounds)-1):
    L=[s for s in spans if bounds[li] <= s[2] < bounds[li+1]]
    if not L: continue
    t0=L[0][2]; t1=max(s[2]+s[3] for s in L); wall=t1-t0
    alliv=[(ts,ts+dur) for _,_,ts,dur in L]
    per=collections.defaultdict(list)
    for eng,name,ts,dur in L: per[cls(eng)].append((ts,ts+dur))
    print("launch %d: wall %7.1f us, %5d spans, any-busy %5.1f%%, %s" % (li, wall, len(L), 100*union(alliv)/wall, " ".join("%s %4.1f%%" % (k, 100*union(v)/wall) for k,v in sorted(per.items()))))
L=[s for s in spans if bounds[-2] <= s[2] < bounds[-1]]
t0=L[0][2]
lay=collections.defaultdict(list)
for eng,name,ts,dur in L:
    m=re.match(r"l(\d+)_", name)
    if m: lay[int(m.group(1))].append((ts,ts+dur))
print("last launch per layer (wall / union busy us):", " ".join("%d:%.0f/%.0f" % (l, max(b for a,b in iv)-min(a for a,b in iv), union(iv)) for l,iv in sorted(lay.items())[:6]))
# collapsed node sequence of layer 5 in the last launch
agg={}
for eng,name,ts,dur in L:
    if not name.startswith("l5_"): continue
    o=(op(name), cls(eng)); a=agg.setdefault(o,[ts,ts+dur,0]); a[0]=min(a[0],ts); a[1]=max(a[1],ts+dur); a[2]+=1
prev_end=None
for (o,eng),(s,e,c) in sorted(agg.items(), key=lambda kv: kv[1][0]):
    gap = s-prev_end if prev_end is not None else 0
    print("  %8.1f .. %8.1f (%5.1f) gap %5.1f  %-5s x%-3d %s" % (s-t0, e-t0, e-s, gap, eng, c, o[:48])); prev_end=max(prev_end or 0, e)
