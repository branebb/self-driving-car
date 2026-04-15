# Self-Driving Car — `newCar` Physics Model

This document describes the physics model implemented in `vehicle/vehicle.py` (`newCar` class).

---

## Simulation Clock

All per-frame rates are hardcoded for a fixed timestep:

$$dt = \frac{1}{f_s}$$

where $f_s$ is the simulation frequency (frames per second). Every call to `update_position(dt)` advances the state by one tick.

---

## Steering Accumulator

The steering input is a continuous accumulator $s \in [-180°, 180°]$ driven by a signed force $u_s \in [-1, 1]$:

$$s_{t+1} = \text{clip}\!\left(s_t + u_s \cdot r_{\text{press}} - \text{sgn}(s_t) \cdot r_{\text{return}},\ -180°,\ 180°\right)$$

where $r_{\text{press}}$ is the press rate and $r_{\text{return}}$ is the self-centering return rate per frame. Both are fractions of the full range normalised to $f_s$.

The physical front-wheel steering angle $\delta$ is a linear reduction of $s$:

$$\delta = \frac{s}{k_s}$$

so that the maximum wheel deflection is $\pm\,\frac{180°}{k_s}$.

---

## Heading — Kinematic Bicycle Model

The car's heading $\theta$ (radians, $0$ = facing right) evolves according to the kinematic bicycle model:

$$\dot{\theta} = \frac{v}{L}\tan\delta$$

Discretised:

$$\theta_{t+1} = \theta_t + \frac{v_t}{L}\tan(\delta_t)\,dt$$

where $L$ is the wheelbase and $v$ is the longitudinal speed. There is no lateral grip limit; the model assumes perfect tracking of the geometric path.

---

## Pedal Dynamics

Throttle $\alpha \in [0, 1]$ and brake $\beta \in [0, 1]$ each ramp with a press rate and a passive release rate per frame:

$$\alpha_{t+1} = \text{clip}\!\left(\alpha_t + u_\alpha \cdot r_{\text{press}} - r_{\text{rel}},\ 0,\ 1\right)$$

$$\beta_{t+1} = \text{clip}\!\left(\beta_t + u_\beta \cdot r_{\text{press}} - r_{\text{rel}},\ 0,\ 1\right)$$

When braking is active ($\beta > 0$), the throttle command $u_\alpha$ is forced to zero.

---

## Drivetrain — RPM from Velocity

Given a gear with ratio $G_g$ and a final-drive ratio $G_f$, the engine RPM $N$ is derived from wheel speed:

$$\omega_{\text{wheel}} = \frac{v}{r_w}$$

$$\omega_{\text{engine}} = \omega_{\text{wheel}} \cdot G_g \cdot G_f$$

$$N = \omega_{\text{engine}} \cdot \frac{60}{2\pi}$$

$N$ is clamped to $[N_{\text{idle}},\, N_{\text{redline}}]$.

### Auto-shift

| Condition | Action |
|-----------|--------|
| $N \geq N_{\text{up}}$ and gear $< 8$ | Upshift |
| $N \leq N_{\text{down}}$ and gear $> 1$ | Downshift |

---

## Torque Curve

Engine torque $T$ is modelled as a Gaussian peak over a baseline:

$$T(N) = T_a + T_b \exp\!\left(-T_c\,(N - T_d)^2\right)$$

| Parameter | Role |
|-----------|------|
| $T_a$ | Constant baseline torque |
| $T_b$ | Peak torque above baseline |
| $T_c$ | Curve width (sharpness of peak) |
| $T_d$ | RPM at peak torque |

---

## Engine Force

Drive force at the contact patch:

$$F_{\text{engine}} = \frac{T(N) \cdot |G_g| \cdot G_f}{r_w} \cdot \alpha$$

---

## Aerodynamics

Both drag and downforce scale with the square of speed.

**Drag:**

$$F_{\text{drag}} = K_D\,v^2, \qquad K_D = \tfrac{1}{2}\,C_d\,A\,\rho$$

**Downforce:**

$$F_{\text{down}} = K_L\,v^2, \qquad K_L = \tfrac{1}{2}\,C_l\,A\,\rho$$

where $C_d$ and $C_l$ are the drag and lift coefficients, $A$ is the reference area, and $\rho$ is air density.

---

## Normal Force and Dynamic Braking

The aerodynamic downforce adds to the static weight, increasing the effective normal force:

$$F_N = mg + F_{\text{down}}$$

Brake force scales with this dynamic normal force:

$$F_{\text{brake}} = \beta \cdot F_N \cdot \mu_{\text{brake}}$$

This means braking authority grows with speed — at high speed, downforce significantly amplifies the available stopping force.

---

## Engine Braking

When the throttle is below a low threshold, the drivetrain applies a resistive torque proportional to how close the engine is to redline:

$$F_{\text{eb}} = C_{\text{eb}} \cdot \frac{N}{N_{\text{redline}}}$$

This is set to zero when $\alpha$ is above the threshold.

---

## Rolling Resistance

$$F_{\text{roll}} = \mu_r \cdot m \cdot g$$

---

## Longitudinal Dynamics

**While rolling** ($v > v_{\text{min}}$):

$$F_{\text{net}} = F_{\text{engine}} - \left(F_{\text{drag}} + F_{\text{roll}} + F_{\text{brake}} + F_{\text{eb}}\right)$$

**Static hold** ($v \leq v_{\text{min}}$): if drive force cannot overcome static resistance, engine force is set to zero to prevent creep.

Acceleration and velocity update:

$$a = \frac{F_{\text{net}}}{m}$$

$$v_{t+1} = \max\!\left(0,\; v_t + a\,dt\right)$$

---

## Position Integration

$$x_{t+1} = x_t + v_t \cos(\theta_t)\,dt$$

$$y_{t+1} = y_t + v_t \sin(\theta_t)\,dt$$

---

## Upgrade History

### v1 — Initial Model
- Kinematic bicycle heading model
- 8-speed gearbox with Gaussian torque curve and auto-shift
- Aerodynamic drag ($F_{\text{drag}} = K_D v^2$)
- Fixed maximum brake force constant (independent of speed)

### v2 — Upgrade A: Downforce + Dynamic Braking
- Added aerodynamic downforce: $F_{\text{down}} = K_L v^2$
- Replaced fixed brake force with speed-dependent formula: $F_{\text{brake}} = \beta \cdot (mg + F_{\text{down}}) \cdot \mu_{\text{brake}}$
- At race speeds, braking force more than doubles relative to v1, consistent with real F1 deceleration figures
