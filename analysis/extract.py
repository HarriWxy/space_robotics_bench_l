"""Read a fixed-size snapshot of local TensorBoard TFRecords without TensorFlow."""
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import struct
import time

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event

ROOT = Path('/root/space_robotics_bench_l/logs/locomotion_velocity_tracking_c')
OUT = Path('/tmp/srb_curve_analysis_DfK1ku')
RUNS = {'SAC': ('sb3_sac','20260906T191305'), 'PPO': ('sb3_ppo','20260906T191254'),
        'ExO-PPO': ('exoppo','20260906T191107'), 'FPO': ('fpo','20260906T185310')}
manifest = {}
for name, (algo, run) in RUNS.items():
    started = time.monotonic()
    series = defaultdict(lambda: array('d'))
    files = []
    for path in sorted((ROOT/algo/run).rglob('*tfevents*')):
        limit = path.stat().st_size
        incomplete = False
        records = 0
        with path.open('rb', buffering=4*1024*1024) as stream:
            while stream.tell() < limit:
                header = stream.read(12)
                if len(header) < 12:
                    incomplete = True
                    break
                length = struct.unpack('<Q', header[:8])[0]
                if stream.tell() + length + 4 > limit:
                    incomplete = True
                    break
                data = stream.read(length)
                crc = stream.read(4)
                if len(data) != length or len(crc) != 4:
                    incomplete = True
                    break
                event = Event.FromString(data)
                records += 1
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        series[value.tag].extend((event.step,event.wall_time,value.simple_value))
        files.append(dict(path=str(path),snapshot_bytes=limit,records=records,incomplete_tail=incomplete))
    arrays = {tag:np.asarray(values).reshape(-1,3) for tag, values in series.items()}
    np.savez_compressed(OUT/f'{algo}.npz',**arrays)
    summary = {}
    for tag, data in arrays.items():
        summary[tag] = dict(n=len(data),first_step=int(data[0,0]),last_step=int(data[-1,0]),
                            first=float(data[0,2]),last=float(data[-1,2]),
                            min=float(np.nanmin(data[:,2])),max=float(np.nanmax(data[:,2])),
                            nonfinite=int(np.sum(~np.isfinite(data[:,2]))),
                            decreasing_steps=int(np.sum(np.diff(data[:,0])<0)))
    all_times = [data[:,1] for data in arrays.values()]
    first_time = min(float(x[0]) for x in all_times)
    last_time = max(float(x[-1]) for x in all_times)
    manifest[name] = dict(algo=algo,run=str(ROOT/algo/run),files=files,tags=summary,
                          first_event_utc=datetime.fromtimestamp(first_time,timezone.utc).isoformat(),
                          last_event_utc=datetime.fromtimestamp(last_time,timezone.utc).isoformat(),
                          elapsed_hours=(last_time-first_time)/3600)
    print(name, 'seconds',round(time.monotonic()-started,1),'tags',len(arrays),
          'max_step',max(int(x[-1,0]) for x in arrays.values()),
          'hours',round((last_time-first_time)/3600,2),flush=True)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
