"""
FINAL proven-stable scenario for 08_visualization.ipynb.
Start: 28 m/s (101 km/h), brake to 64 km/h, chicane at 60-65 km/h.
"""
import sys
sys.path.insert(0, 'lib')
import numpy as np
from bicycle_model import Full2DModel

def ramp(t, t0, t1, v0, v1):
    if t <= t0: return v0
    if t >= t1: return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

def chicane_inputs(t):
    RAMP    = 0.5
    MAX_DEG = 8.0

    if t < 0.5:
        return 0.0, 0.0, 0.0                               # coast

    elif t < 2.0:
        return 0.0, 0.0, 0.25                              # braking

    elif t < 2.0 + RAMP:
        d = ramp(t, 2.0, 2.5, 0.0, np.radians(MAX_DEG))
        return float(d), 0.0, 0.0                          # ramp steer left

    elif t < 5.5:
        return np.radians(MAX_DEG), 0.0, 0.0               # hold left

    elif t < 5.5 + RAMP:
        d = ramp(t, 5.5, 6.0, np.radians(MAX_DEG), 0.0)
        return float(d), 0.0, 0.0                          # ramp back to 0

    elif t < 9.0:
        return 0.0, 0.12, 0.0                              # stabilisation gap

    elif t < 9.0 + RAMP:
        d = ramp(t, 9.0, 9.5, 0.0, -np.radians(MAX_DEG))
        return float(d), 0.0, 0.0                          # ramp steer right

    elif t < 12.5:
        return -np.radians(MAX_DEG), 0.0, 0.0              # hold right

    elif t < 12.5 + RAMP:
        d = ramp(t, 12.5, 13.0, -np.radians(MAX_DEG), 0.0)
        return float(d), 0.0, 0.0                          # ramp back to 0

    else:
        return 0.0, 1.0, 0.0                               # full throttle exit

DT      = 0.005
T_TOTAL = 22.0
N_STEPS = int(T_TOTAL / DT)   # 4400

car = Full2DModel(Vx=28.0)

print_times = [i*0.5 for i in range(int(T_TOTAL/0.5)+1)]
print(f"{'t':>6}  {'Vx km/h':>8}  {'Vy m/s':>7}  {'psi_dot':>8}  {'af_deg':>7}  {'ar_deg':>7}  {'delta':>7}  gear")
for _ in range(N_STEPS):
    d, thr, brk = chicane_inputs(car.t)
    t = car.t
    if any(abs(t - pt) < 0.003 for pt in print_times):
        h = car.history
        if len(h['Vx']) > 0:
            Vx = h['Vx'][-1]
            Vy = h['Vy'][-1]
            pd = h['psi_dot'][-1]
            af = h['alpha_f'][-1]
            ar = h['alpha_r'][-1]
            g  = h['gear'][-1]
            print(f"{t:6.2f}  {Vx*3.6:8.2f}  {Vy:7.3f}  {pd:8.4f}  {af:7.2f}  {ar:7.2f}  {np.degrees(d):7.3f}  {g}")
    car.step(d, throttle=thr, brake=brk, dt=DT)

h = car.history
print()
print(f"Start  : {h['Vx'][0]*3.6:.1f} km/h  gear {h['gear'][0]}")
print(f"End    : {h['Vx'][-1]*3.6:.1f} km/h  gear {h['gear'][-1]}")
print(f"Peak lat G  : {max(abs(a) for a in h['ay']) / 9.81:.2f} g")
print(f"Peak decel G: {abs(min(h['ax'])) / 9.81:.2f} g")
print(f"Peak psi_dot: {max(abs(p) for p in h['psi_dot']):.3f} rad/s")
print(f"Max |Vy|    : {max(abs(v) for v in h['Vy']):.3f} m/s")
print(f"Max |af|    : {max(abs(a) for a in h['alpha_f']):.2f} deg")
print(f"Max |ar|    : {max(abs(a) for a in h['alpha_r']):.2f} deg")
print(f"N_STEPS/frames: {N_STEPS}/{N_STEPS//11}")
