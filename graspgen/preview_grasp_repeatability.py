"""Static per-reference repeatability results; display best tested condition explicitly."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(path):
    report=json.loads(path.read_text())
    grouped={}
    for c in report['conditions']:grouped.setdefault(c['reference'],[]).append(c)
    den=np.array([5,8,4])
    kinds=['nominal','object_offset','base_offset']
    def value(c):return np.array([c['successes'][k] for k in kinds])
    best=[max(rows,key=lambda c:(c['passed'],min(value(c)/den),sum(value(c)))) for rows in grouped.values()]
    data=np.array([value(c)/den for c in best])
    fig,ax=plt.subplots(figsize=(9,max(3.5,.29*len(best)+2)))
    ax.imshow(data,aspect='auto',vmin=0,vmax=1,cmap='RdYlGn')
    for i,c in enumerate(best):
        for j in range(3):ax.text(j,i,f'{value(c)[j]}/{den[j]}',ha='center',va='center',fontsize=9)
    radius=1000*np.abs(report['criteria']['object_offsets_m']).max()
    ax.set_xticks(range(3),['Nominal repeats (need >=4/5)',f'Object +/-{radius:g}mm (need 8/8)','Base +/-5mm (need >=3/4)'])
    labels=[f"{c['reference']} | {c['region']}/{c['depth']} | {'PASS' if c['passed'] else 'FAIL'}" for c in best]
    ax.set_yticks(range(len(best)),labels,fontsize=8)
    ax.set_title(f"{report['object']}: {report['retained_reference_count']}/{report['tested_reference_count']} references retained\nBest tested condition per reference; other conditions remain in the JSON report",fontsize=12,pad=15)
    fig.text(.5,.01,'Finite direct-IK simulation diagnostics. Local conditions only; no expert approach trajectories.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.03,1,1]);target=path.with_name('review.png')
    fig.savefig(target,dpi=160);plt.close(fig);print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('review',type=Path)
    main(parser.parse_args().review)
