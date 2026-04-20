# Copyright (c) 2026 IRISA/CNRS-INRIA
# All rights reserved.
# Template filled by: Oguzhan Enes Isik

# ============================================================
# Part III — Python Simulator connected to telekyb3/nhfc
#
# Pipeline:
#   nhfc --rotor_input--> [this script] --state--> nhfc
#
# No Gazebo. No optitrack. No pom. No rotorcraft.
# We write state directly to nhfc's state input port.
# ============================================================

import genomix  # type: ignore
import math
import numpy as np  # type: ignore
import os
import sys
import select
import time

# ─────────────────────────────────────────────────────────────
# ROBOT PARAMETERS — quadrotor (from mrsim-quadrotor/model.sdf)
# ─────────────────────────────────────────────────────────────
mass = 1.28
grav = 9.81
Jxx, Jyy, Jzz = 0.015, 0.015, 0.007
J = np.diag([Jxx, Jyy, Jzz])
Ji = np.diag([1/Jxx, 1/Jyy, 1/Jzz])
cf = 6.5e-4
ct = 1e-5
L = 0.23

# ─────────────────────────────────────────────────────────────
# DYNAMICS
# ─────────────────────────────────────────────────────────────

def quat_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qw*qz),    2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),    1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),    2*(qy*qz+qw*qx),    1-2*(qx**2+qy**2)]
    ])

def quat_mult(q, p):
    a1,b1,c1,d1 = q
    a2,b2,c2,d2 = p
    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ])

# Rotor layout:
#     2       1
#       \   /
#         X
#       /   \
#     3       4

def compute_wrench(w2):
    # w2: rotor speeds squared [rad^2/s^2]
    fz = cf * np.sum(w2)               # total vertical thrust — all rotors push up
    tx = cf * L * (-w2[3] + w2[1])    # roll torque — rotors 2 and 4 oppose each other
    ty = cf * L * ( w2[2] - w2[0])    # pitch torque — rotors 1 and 3 oppose each other
    tz = ct * (w2[0] - w2[1] + w2[2] - w2[3])  # yaw torque — CW/CCW reaction torques
    return np.array([0.0, 0.0, fz, tx, ty, tz])

def f_dynamics(x, wrench):
    q      = x[3:7]    # quaternion [qw, qx, qy, qz]
    v      = x[7:10]   # linear velocity in world frame
    w      = x[10:13]  # angular velocity in body frame
    f_body = wrench[0:3]
    tau    = wrench[3:6]
    R      = quat_to_R(q)

    # position derivative: pdot = v
    p_dot = v

    # quaternion kinematics: qdot = 0.5 * q ⊗ [0, ω]
    q_dot = 0.5 * quat_mult(q, np.array([0.0, w[0], w[1], w[2]]))

    # Newton's 2nd law in world frame: vdot = (1/m)(F_gravity + R*f_body)
    v_dot = (1/mass) * (np.array([0., 0., -mass*grav]) + R @ f_body)

    # Euler's rotation equation in body frame: wdot = J^-1 * (tau - w x Jw)
    w_dot = Ji @ (-np.cross(w, J @ w) + tau)

    return np.concatenate([p_dot, q_dot, v_dot, w_dot])

def rk4_step(x, wrench, dt):
    # 4th order Runge-Kutta integration
    k1 = f_dynamics(x,              wrench)
    k2 = f_dynamics(x + dt/2 * k1, wrench)
    k3 = f_dynamics(x + dt/2 * k2, wrench)
    k4 = f_dynamics(x + dt   * k3, wrench)
    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def normalize_quat(x):
    # prevent quaternion drift due to numerical integration errors
    x = x.copy()
    x[3:7] /= np.linalg.norm(x[3:7])
    return x

def ground_reaction(x):
    # simple ground contact: prevent drone from going below z=0
    x = x.copy()
    if x[2] < 0.0:
        x[2]    = 0.0 
        x[7:13] = 0.0  # zero velocities on ground contact
    return x

# ─────────────────────────────────────────────────────────────
# NHFC INTERFACE
# ─────────────────────────────────────────────────────────────

def setup_state_port():
    # Create a new port 'my_state' with the same type as nhfc's state input
    # Then redirect nhfc to read its state from our port instead of pom
    port = nhfc.state('my_state')
    nhfc.connect_port({'local': 'state', 'remote': 'my_state'})
    return port

def state_to_nhfc(state_port, x, xdot):
    # Build the full state message nhfc expects and publish it

    def _now():  
        t = math.modf(time.clock_gettime(time.CLOCK_REALTIME))
        sec=int(t[1])
        nsec=int(t[0]*1e9)
        if nsec >= 1e9:
            sec += 1
            nsec -= int(1e9)
        return sec, nsec

    # Small covariances — we provide perfect state (no sensor noise)
    ps = 1e-3; qs = 1e-3; vs = 1e-3; ws = 3e-3; as_ = 2e-2

    pos_cov     = [ps**2, 0, ps**2, 0, 0, ps**2]
    vel_cov     = [vs**2, 0, vs**2, 0, 0, vs**2]
    avel_cov    = [ws**2, 0, ws**2, 0, 0, ws**2]
    acc_cov     = [as_**2, 0, as_**2, 0, 0, as_**2]
    aacc_cov    = [0]*6   # no jerk model
    att_pos_cov = [0]*12  # no cross-covariance between position and attitude

    qw, qx, qy, qz = x[3], x[4], x[5], x[6]
    att_cov = [
        qs**2*(1-qw*qw), qs**2*(-qw*qx),
        qs**2*(1-qx*qx), qs**2*(qw*qy),
        qs**2*(-qx*qy),  qs**2*(1-qy*qy),
        qs**2*(-qw*qz),  qs**2*(-qx*qz),
        qs**2*(-qy*qz),  qs**2*(1-qz*qz),
    ]

    sec, nsec = _now()
    data = {"state": {
        "ts":          {"sec": sec, "nsec": nsec},
        "intrinsic":   False,  # world frame
        "pos":         {"x": x[0],      "y": x[1],      "z": x[2]},
        "att":         {"qw": x[3],     "qx": x[4],     "qy": x[5],     "qz": x[6]},
        "vel":         {"vx": x[7],     "vy": x[8],     "vz": x[9]},
        "avel":        {"wx": x[10],    "wy": x[11],    "wz": x[12]},
        "acc":         {"ax": xdot[7],  "ay": xdot[8],  "az": xdot[9]},    # f̃_{1:3} linear acceleration
        "aacc":        {"awx": xdot[10],"awy": xdot[11],"awz": xdot[12]},  # f̃_{4:6} angular acceleration
        "pos_cov":     {"cov": pos_cov},
        "att_cov":     {"cov": att_cov},
        "att_pos_cov": {"cov": att_pos_cov},
        "vel_cov":     {"cov": vel_cov},
        "avel_cov":    {"cov": avel_cov},
        "acc_cov":     {"cov": acc_cov},
        "aacc_cov":    {"cov": aacc_cov},
    }}
    state_port(data)

def read_rotor_speeds(n=4):
    # Read nhfc's commanded rotor speeds from output port 'rotor_input'
    speeds = np.zeros(n)
    try:
        data = nhfc.rotor_input()["rotor_input"]
        for i in range(n):
            s = data["desired"][i]
            if s:
                speeds[i] = s
    except Exception:
        pass  # return zeros if nhfc not ready yet
    return speeds

# ─────────────────────────────────────────────────────────────
# CONNECT TO GENOMIX
# ─────────────────────────────────────────────────────────────
g = genomix.connect()
g.rpath('/opt/openrobots/lib/genom/pocolibs/plugins')

nhfc = g.load('nhfc')

LOG_DIR = os.path.join(os.environ['TK3LAB_WS'], 'logs', '01b-simulator')
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# NHFC SETUP
# ─────────────────────────────────────────────────────────────
nhfc.set_gtmrp_geom({
    'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0,
    'armlen': 0.23, 'mass': 1.28,
    'rx': 0, 'ry': 0, 'rz': -1,
    'cf': 6.5e-4, 'ct': 1e-5
})
# Large emergency thresholds — prevent nhfc from entering emergency during simulation
nhfc.set_emerg({'emerg': {'descent': 0.1, 'dx': 100.0, 'dq': 100.0, 'dv': 100.0, 'dw': 100.0}})
nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}}) 
nhfc.set_servo_gain({'gain': {
    'Kpxy': 5, 'Kpz': 8, 'Kqxy': 4, 'Kqz': 0.1,
    'Kvxy': 6, 'Kvz': 8, 'Kwxy': 1, 'Kwz': 0.1,
    'Kixy': 0, 'Kiz': 0
}})
nhfc.set_control_mode({'att_mode': '::nhfc::full_attitude'})

# Redirect nhfc state input to our simulator port
state_port = setup_state_port()

# ─────────────────────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────────────────────
try: nhfc.stop()
except: pass
time.sleep(0.5)

nhfc.log(os.path.join(LOG_DIR, 'nhfc.log'))

# Initial state: on ground, level, at rest
x    = np.zeros(13)
x[3] = 1.0  # qw=1 → identity quaternion, no rotation

wrench0 = compute_wrench(np.zeros(4))
xdot0   = f_dynamics(x, wrench0)

# Send initial state and arm nhfc
state_to_nhfc(state_port, x, xdot0)
time.sleep(1.0)
nhfc.set_current_position()
time.sleep(1.0)
nhfc.servo(ack=True)
time.sleep(1.0)

# Keep sending state while waiting for Enter
# This prevents nhfc from seeing "obsolete state" and entering emergency
print("\nPress Enter to start simulation...")
while True:
    state_to_nhfc(state_port, x, xdot0)
    time.sleep(0.005)  # 200Hz — enough to keep nhfc happy
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        break

# ─────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────
dt = 7e-3   
tf = 37.00

N     = math.ceil(tf / dt)
tt    = np.linspace(0.0, tf, N)

x_log = np.zeros((N, 13))
u_log = np.zeros((N, 4))
t_log = np.zeros(N)

set_wp0 = True 
was_on_ground = False
set_wp1 = True
set_wp2 = True
set_wp3 = True

print("Simulation running...")

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
for i, ts in enumerate(tt):

    t_start = time.clock_gettime_ns(time.CLOCK_REALTIME) * 1e-6  # ms

    # ── Send waypoints at scheduled times ────────────────────
    if set_wp0 and ts >= 3.0:
        nhfc.set_position(0, 0, 1, 0)
        set_wp0 = False
        print(f"t={ts:.1f}s → wp0 takeoff (0,0,1,0)")
    elif set_wp1 and ts >= 9.0:
        nhfc.set_position(1, 1, 1, 0)
        set_wp1 = False
        print(f"t={ts:.1f}s → wp1 (1,1,1,0)")
    elif set_wp2 and ts >= 15.0:
        nhfc.set_position(1, -1, 1, math.pi/2)
        set_wp2 = False
        print(f"t={ts:.1f}s → wp2 (1,-1,1,π/2)")
    elif set_wp3 and ts >= 29.0:
        nhfc.set_position(0, 0, 0.25, 0)
        set_wp3 = False
        print(f"t={ts:.1f}s → wp3 (0,0,0.25,0)")

    # ── Read rotor commands from nhfc ─────────────────────────
    rotor_speeds = read_rotor_speeds(n=4)
    w2           = np.square(rotor_speeds) 

    # ── Integrate dynamics one step forward ───────────────────
    wrench = compute_wrench(w2)
    x      = rk4_step(x, wrench, dt)
    x      = normalize_quat(x)

    x      = ground_reaction(x)

    on_ground = x[2] <= 0.01
    if on_ground and not was_on_ground:
        try:
            nhfc.servo(ack=True)
        except:
            pass
    was_on_ground = on_ground


    # ── Compute derivatives at new state (for acc, aacc) ──────
    xdot = f_dynamics(x, wrench)


    # ── Log data ──────────────────────────────────────────────
    t_log[i]    = ts
    x_log[i, :] = x
    u_log[i, :] = np.square(rotor_speeds) * cf  # thrust per rotor [N]

    if i % 1000 == 0:
        print(f"t={ts:.1f}s  pos=({x[0]:.2f},{x[1]:.2f},{x[2]:.2f})")
        
    state_to_nhfc(state_port, x, xdot)
    # ── Real-time pacing ──────────────────────────────────────
    t_end     = time.clock_gettime_ns(time.CLOCK_REALTIME) * 1e-6
    elapsed   = t_end - t_start
    remaining = dt * 1000 - elapsed
    if remaining > 0:
        time.sleep(remaining * 1e-3)
    else:
        print(f"Warning: overrun {elapsed:.2f}ms > {dt*1000:.1f}ms")

    # ── Send updated state to nhfc ────────────────────────────
    # This closes the control loop: nhfc gets fresh state every dt
    

# ─────────────────────────────────────────────────────────────
# STOP
# ─────────────────────────────────────────────────────────────
nhfc.stop()
try: nhfc.log_stop()
except: pass

# ─────────────────────────────────────────────────────────────
# SAVE LOGS
# ─────────────────────────────────────────────────────────────
print("\nSaving logs...")

state_path = os.path.join(LOG_DIR, 'simulator_state.log')
with open(state_path, 'w') as f:
    f.write('# ts px py pz qw qx qy qz vx vy vz wx wy wz\n')
    for i in range(N):
        row = [t_log[i]] + list(x_log[i])
        f.write(' '.join(f'{v:.8f}' for v in row) + '\n')

input_path = os.path.join(LOG_DIR, 'simulator_input.log')
with open(input_path, 'w') as f:
    f.write('# ts f1 f2 f3 f4\n')
    for i in range(N):
        row = [t_log[i]] + list(u_log[i])
        f.write(' '.join(f'{v:.8f}' for v in row) + '\n')

print(f"State log:  {state_path}")
print(f"Input log:  {input_path}")
print("=== Done ===")