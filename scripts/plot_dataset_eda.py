"""Describe the committed demand splits without changing the experiment."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenarios/yubei/central_park"
OUT = ROOT / "docs/figures"

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((SCENE / "demand_manifest.json").read_text())
    tunnels = json.loads((SCENE / "tunnels.json").read_text())["tunnel_edges"]
    edges = {}
    for edge in ET.parse(SCENE / "central_park.net.xml").getroot().findall("edge"):
        lane = edge.find("lane")
        if lane is not None and edge.get("function") != "internal":
            shape = np.array([[float(v) for v in pair.split(",")] for pair in lane.get("shape").split()])
            lengths = np.linalg.norm(np.diff(shape, axis=0), axis=1)
            cumulative = np.r_[0., np.cumsum(lengths)]
            mid = np.array([np.interp(cumulative[-1]/2, cumulative, shape[:, k]) for k in range(2)])
            edges[edge.get("id")] = (shape, mid)
    data, checks = {}, {}
    for split, files in manifest["split"].items():
        rows = []
        for filename in files:
            root = ET.parse(SCENE / filename).getroot()
            assert len(root.findall("trip")) == 20
            assert len(root.findall("person")) == 50
            for person in root.findall("person"):
                ride = person.find("ride")
                a, b = edges[ride.get("from")][1], edges[ride.get("to")][1]
                rows.append([float(person.get("depart")), *a, *b, float(np.linalg.norm(a-b))])
            checks[filename] = hashlib.sha256((SCENE/filename).read_bytes()).hexdigest()
        data[split] = np.array(rows)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    colors = {"train":"#176B87", "val":"#CA7C26", "test":"#754AB0"}
    labels = {"train":"Training (700 requests)", "val":"Validation (150)", "test":"Test (150)"}
    fig, axes = plt.subplots(1, 2, figsize=(10,4.5), constrained_layout=True)
    for ax, column, title in zip(axes, (1,3), ("Pickup edge midpoints", "Drop-off edge midpoints")):
        ax.add_collection(LineCollection([s/1000 for s,m in edges.values()],colors="#D0D5DA",linewidths=.55))
        ax.add_collection(LineCollection([edges[t][0]/1000 for t in tunnels],colors="#B52E31",linewidths=3,label="Tunnel trigger edges"))
        for split in data:
            a=data[split]; ax.scatter(a[:,column]/1000,a[:,column+1]/1000,s=11,alpha=.4,color=colors[split],label=labels[split])
        ax.autoscale(); ax.set_aspect("equal"); ax.set(title=title,xlabel="Local SUMO x (km)",ylabel="Local SUMO y (km)")
    axes[1].legend(fontsize=7,loc="upper left")
    fig.savefig(OUT/"dataset_spatial_eda.png",dpi=230); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    summary={}
    for split,a in data.items():
        for ax,col,scale in ((axes[0],0,1),(axes[1],5,1000)):
            x=np.sort(a[:,col]/scale); ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",label=labels[split],color=colors[split])
        summary[split]={"variants":len(manifest["split"][split]),"requests":len(a),"arrival_mean_s":float(a[:,0].mean()),"arrival_median_s":float(np.median(a[:,0])),"arrival_min_s":float(a[:,0].min()),"arrival_max_s":float(a[:,0].max()),"od_median_m":float(np.median(a[:,5])),"od_p10_m":float(np.quantile(a[:,5],.1)),"od_p90_m":float(np.quantile(a[:,5],.9))}
    axes[0].set(xlabel="Request arrival time (s)",ylabel="Cumulative proportion",title="Request arrival times by split",xlim=(0,600))
    axes[1].set(xlabel="Straight-line OD proxy (km)",ylabel="Cumulative proportion",title="Origin–destination separation by split")
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/"dataset_split_eda.png",dpi=230); plt.close(fig)
    (OUT/"dataset_eda_summary.json").write_text(json.dumps({"definition":"Euclidean distance between lane-0 polyline midpoints; not routed distance or observed travel time", "non_internal_edges":len(edges),"trigger_edges":len(tunnels),"splits":summary,"input_sha256":checks},indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
