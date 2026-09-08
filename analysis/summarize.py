"""Summarize scalar snapshots using explicit environment-step windows."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault('MPLCONFIGDIR','/tmp/srb_curve_analysis_DfK1ku/mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = Path('/tmp/srb_curve_analysis_DfK1ku')
OUT = Path('/root/space_robotics_bench_l/logs/_analysis/20260907_latest')
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((DATA/'manifest.json').read_text())
arrays={name:dict(np.load(DATA/f'{meta["algo"]}.npz')) for name,meta in manifest.items()}
COLORS={'SAC':'#0072B2','PPO':'#D55E00','ExO-PPO':'#009E73','FPO':'#CC79A7'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.2,'figure.dpi':130})
metrics=[('rollout/ep_rew_mean','Episode return (raw)',1),
         ('rollout/ep_len_mean','Episode length (seconds; step = 0.04 s)',.04),
         ('rollout/metrics/command_lin_error','Planar velocity error (m/s)',1),
         ('rollout/metrics/command_ang_error','Yaw velocity error (rad/s)',1),
         ('rollout/metrics/tracking_success','Instantaneous tracking success (%)',100),
         ('rollout/metrics/undesired_contact','Undesired contact prevalence (%)',100)]

def bins(data,width=500_000):
    # Each point is a mean of logged scalars, not a mean across random seeds.
    idx=np.maximum(0,np.ceil(data[:,0]/width).astype(int)-1)
    keys=np.unique(idx)
    count=np.bincount(idx)
    x=np.bincount(idx,weights=data[:,0])/np.maximum(1,count)
    y=np.bincount(idx,weights=data[:,2])/np.maximum(1,count)
    return x[keys],y[keys],count[keys]

def window(data,start,end):
    chosen=data[(data[:,0]>start)&(data[:,0]<=end)]
    if len(chosen)==0:return None
    return dict(n=len(chosen),first_step=int(chosen[0,0]),last_step=int(chosen[-1,0]),
                mean=float(chosen[:,2].mean()),median=float(np.median(chosen[:,2])),
                p10=float(np.quantile(chosen[:,2],.1)),p90=float(np.quantile(chosen[:,2],.9)))

summary={}
for name,series in arrays.items():
    end=float(series['rollout/metrics/command_lin_error'][-1,0])
    checkpoints={}
    for boundary in (5e6,10e6,15e6,20e6,22e6,24e6,26e6,28e6,30e6,34e6,40e6,60e6,100e6):
        if boundary>end:continue
        checkpoints[str(int(boundary))]={tag:window(data,boundary-2e6,boundary)
                                        for tag,data in series.items()}
    tail={tag:window(data,end-2e6,end) for tag,data in series.items()}
    reward=series['rollout/ep_rew_mean']
    bx,by,n=bins(reward,2_000_000)
    best=int(np.argmax(by))
    summary[name]=dict(end_step=int(end),elapsed_hours=manifest[name]['elapsed_hours'],
                      measured_steps_per_second=end/(manifest[name]['elapsed_hours']*3600),
                      tail_2m=tail,checkpoints=checkpoints,
                      peak_2m_bin=dict(mean_return=float(by[best]),mean_step=float(bx[best]),n=int(n[best])))
    print(name,'end',end,'hours',manifest[name]['elapsed_hours'],'peak2M',summary[name]['peak_2m_bin'])
    for tag,_,_ in metrics:
        print(' ',tag,round(tail[tag]['mean'],6),'at32-34M',round(checkpoints['34000000'][tag]['mean'],6))

for limit,suffix in ((100.01e6,'full'),(34e6,'common_34m')):
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(tag,label,scale) in zip(axes.flat,metrics):
        for name,series in arrays.items():
            data=series[tag];data=data[data[:,0]<=limit]
            x,y,n=bins(data)
            ax.plot(x/1e6,y*scale,label=name,color=COLORS[name],linewidth=1.8)
        ax.set(xlabel='Environment transitions (millions)',ylabel=label,xlim=(0,limit/1e6))
        if 'command_' in tag: ax.axhline(.1,color='#777777',linestyle=':',linewidth=1)
    axes[0,0].legend(loc='upper right',frameon=False)
    fig.suptitle('Latest SRB H1 / Moon runs: SAC, PPO, ExO-PPO, FPO\n'
                 '0.5M-transition bins; one run per method; dashed error thresholds = 0.1',fontsize=14)
    fig.savefig(OUT/f'curves_{suffix}.png',dpi=180)
    fig.savefig(OUT/f'curves_{suffix}.pdf')
    plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
panels=[('PPO',('train/std',),'PPO Gaussian std (log scale)',True),
        ('ExO-PPO',('train/std',),'ExO conditional Gaussian std (log scale)',True),
        ('FPO',('train/clip_fraction','train/approx_kl'),'FPO surrogate clip fraction / KL',False),
        ('PPO',('train/clip_fraction','train/approx_kl'),'PPO clip fraction / approximate KL',False),
        ('ExO-PPO',('train/ratio','train/approx_kl'),'ExO importance ratio / KL estimator',False),
        ('SAC',('train/ent_coef',),'SAC entropy coefficient (log scale)',True)]
for ax,(name,tags,title,log) in zip(axes.flat,panels):
    for i,tag in enumerate(tags):
        x,y,n=bins(arrays[name][tag]);ax.plot(x/1e6,y,label=tag.removeprefix('train/'),
                                          color=COLORS[name] if i==0 else '#444444',linewidth=1.6)
    if log:ax.set_yscale('log')
    ax.set(title=title,xlabel='Environment transitions (millions)')
    ax.legend(frameon=False,fontsize=9)
fig.suptitle('Optimization diagnostics (definitions differ between algorithms)',fontsize=14)
fig.savefig(OUT/'optimization.png',dpi=180)
plt.close(fig)

with (OUT/'binned_curves.csv').open('w',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['algorithm','tag','mean_environment_step','mean_value','num_logged_scalars'])
    for name,series in arrays.items():
        for tag,data in series.items():
            x,y,n=bins(data)
            writer.writerows((name,tag,float(xx),float(yy),int(nn)) for xx,yy,nn in zip(x,y,n))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'sources.json').write_text(json.dumps(manifest,indent=2))
print('OUTPUT',OUT)
