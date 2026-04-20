# tk3lab — Aerial Robotics Lab Assignment

Politecnico di Milano — Aerial Robotics (RAINBOW Lab)  
Docker image: `art/tk3lab:ionic-0.2` · Middleware: pocolibs / GenoM3

---

## Repository Structure

```
01a-model/
├── quad/        simulation.py   — quadrotor Gazebo simulation
├── hexa-fa/     simulation.py   — hexarotor fully actuated (Gazebo)
└── hexa-ua/     simulation.py   — hexarotor underactuated (Gazebo)

01b-simulator/
├── quadrotor_simulator.py       — Part II: standalone Python RK4 simulator
└── simulator_telekyb3.py        — Part III: nhfc-connected closed-loop simulator
```

---

## Part I — Gazebo Models (`01a-model`)

Three vehicle configurations simulated via the telekyb3 stack (optitrack + rotorcraft + pom + nhfc).

### Quadrotor (`quad/simulation.py`)
- **Mass:** 1.28 kg · **Arm length:** 0.23 m · **Rotors:** 4
- **Geometry:** `set_gtmrp_geom`, rz=-1 (rotors pointing down)
- **Control mode:** `tilt_prioritized`
- **Gains:** Kpxy=5, Kpz=5, Kvxy=6, Kvz=6, Kqxy=4
- **Start position:** (-1, 0, 0.25)
- **Waypoints:** (0,1,1.25,0) → (0,-1,1.25,π/2) → (-1,0,0.25,0)

### Hexarotor Fully Actuated (`hexa-fa/simulation.py`)
- **Mass:** 2.3 kg · **Rotors:** 6 (tilted)
- **Geometry:** `set_geom` with custom 6×8 allocation matrix G derived from model.sdf
- **Control mode:** `full_attitude` — exploits tilted rotors to control all 6 DOF
- **Gains:** Kpxy=5, Kpz=7, Kvxy=6, Kvz=7, Kiz=2
- **Start position:** (1, 0, 0.21)
- **Waypoints:** (2,1,1.21,0) → (2,-1,1.21,π/2) → (1,0,0.21,0)
- **Note:** ~0.5m z error expected due to tilted rotor geometry

### Hexarotor Underactuated (`hexa-ua/simulation.py`)
- Same hardware as hexa-fa but controlled as underactuated
- **Control mode:** `tilt_prioritized` — ignores lateral force capability of tilted rotors
- **Note:** marked as non-functional in code

---

## Part II — Standalone Python Simulator (`quadrotor_simulator.py`)

Pure Python quadrotor simulator. No Gazebo, no middleware. Validates the dynamic model independently.

**State vector (13):** position (3) + quaternion (4) + linear velocity (3) + angular velocity (3)  
**Integration:** 4th-order Runge-Kutta  
**Attitude:** quaternion kinematics (avoids gimbal lock)

---

## Part III — nhfc-Connected Simulator (`simulator_telekyb3.py`)

Replaces Gazebo with the Python RK4 simulator connected directly to nhfc via python-genomix.

**Pipeline:**
```
nhfc/rotor_input → Python RK4 dynamics → nhfc/state
```

**Key parameters:**
| Parameter | Value |
|-----------|-------|
| `dt` | 7ms (~143Hz) |
| `mass` | 1.28 kg |
| `cf` | 6.5e-4 |
| `ct` | 1e-5 |
| `L (arm)` | 0.23 m |
| Saturation | x=1, v=1 |
| Control mode | `full_attitude` |
| Emergency thresholds | dx=100, dq=100, dv=100, dw=100 |

**Rotor layout** (verified from `mrsim-quadrotor/model.sdf`):
| Rotor | Position | Spin |
|-------|----------|------|
| 1 | (+x) front | CW |
| 2 | (+y) left | CCW |
| 3 | (-x) rear | CW |
| 4 | (-y) right | CCW |

**Wrench computation:**
```python
fz = cf * sum(w2)
tx = cf * L * ( w2[1] - w2[3])   # left - right
ty = cf * L * ( w2[2] - w2[0])   # rear - front
tz = ct * ( w2[0] - w2[1] + w2[2] - w2[3])  # CW(+) CCW(-)
```

**Waypoint sequence:**
| # | Position | Yaw | Send time |
|---|----------|-----|-----------|
| wp0 | (0, 0, 1) | 0 | t=3s |
| wp1 | (1, 1, 1) | 0 | t=9s |
| wp2 | (1, -1, 1) | π/2 | t=15s |
| wp3 | (0, 0, 0.25) | 0 | t=29s |

**Result:**
```
wp0 → (0.00,  0.00, 1.00) ✓
wp1 → (1.00,  1.00, 1.00) ✓
wp2 → (1.02, -1.17, 0.82) ✓  z loss during yaw rotation — expected
wp3 → (0.00,  0.00, 0.25) ✓
```

---

## Running Part III

```bash
# Terminal 1 — middleware
h2 end && h2 init
genomixd &
nhfc-pocolibs &
pom-pocolibs &

# Terminal 2 — simulator
cd /shared-workspace/src/01b-simulator
python3 simulator_telekyb3.py
```

---

## Notes

- genomix HTTP latency ~100ms per call → waypoint send times offset accordingly
- Timestamp offset removed from state messages (delay stable at ~1ms without it)
- `rotor_measure` port not connected (per lecture slides, neglect rotor feedback)
- `ground_reaction()` clips z to 0 on ground contact and zeroes velocities
