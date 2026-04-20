"""
Aerial Robotics Lab — Modeling Part II
Standalone Quadrotor Simulator (no telekyb3, no GenoM3)

What this script does:
    Replicates what happens INSIDE Gazebo:
    1. Receive rotor speed commands (input u)
    2. Compute wrench from rotor speeds using allocation matrix
    3. Integrate dynamic model with RK4 (dt = 1ms)
    4. Normalize quaternion
    5. Apply ground reaction
    6. Repeat -> produces new state x at each time step

Two tests as required by PDF:
    - Test 1: lifting force = weight force (mg)  -> drone hovers, z stays constant
    - Test 2: lifting force = weight force + 1N  -> drone slowly climbs

No python-genomix, no GenoM3 components.
Only numpy, matplotlib, scipy (for euler angles).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT PARAMETERS — from mrsim-quadrotor/model.sdf and nhfc set_gtmrp_geom
# ─────────────────────────────────────────────────────────────────────────────

mass = 1.28          # total mass [kg]
grav = 9.81          # gravitational acceleration [m/s²]
Jxx  = 0.015         # inertia around x-axis [kg·m²] — from SDF ixx
Jyy  = 0.015         # inertia around y-axis [kg·m²] — from SDF iyy
Jzz  = 0.007         # inertia around z-axis [kg·m²] — from SDF izz
J    = np.diag([Jxx, Jyy, Jzz])        # 3x3 inertia matrix (diagonal assumption)
Ji   = np.diag([1/Jxx, 1/Jyy, 1/Jzz])  # J^{-1} — trivial since J is diagonal

cf   = 6.5e-4        # thrust coefficient [N/(rad/s)²] — from SDF plugin
ct   = 1e-5          # drag coefficient   [Nm/(rad/s)²] — from SDF plugin
L    = 0.23          # arm length [m] — from set_gtmrp_geom armlen

# ─────────────────────────────────────────────────────────────────────────────
# TIME PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

dt    = 1e-3         # sampling time Δt = 1ms (required by PDF)
T     = 5.0          # total simulation time [s]
steps = int(T / dt)  # number of steps = 5000

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL STATE x0 — shape (13,)
# ─────────────────────────────────────────────────────────────────────────────
# State vector x :
#   x[0:3]   = position p = [px, py, pz]           in world frame W
#   x[3:7]   = quaternion q = [qw, qx, qy, qz]     scalar first! represents orientation of B w.r.t. W
#   x[7:10]  = linear velocity v = [vx, vy, vz]    in world frame W
#   x[10:13] = angular velocity ω = [wx, wy, wz]   in body frame B
#
# Initial condition: drone sitting on ground, level, at rest
#   p = [0, 0, 0]
#   q = [1, 0, 0, 0]  → identity quaternion = no rotation = drone is perfectly level
#   v = [0, 0, 0]     → at rest
#   ω = [0, 0, 0]     → no rotation

x0      = np.zeros(13)
x0[3]   = 1.0        # qw = 1 → identity quaternion

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Quaternion → Rotation Matrix
# ─────────────────────────────────────────────────────────────────────────────
def quat_to_R(q):
    """
    Convert quaternion q = [qw, qx, qy, qz] to 3x3 rotation matrix R.

    Why we need this:
        The rotor thrust force f is expressed in body frame B.
        Newton's law is expressed in world frame W.
        To compute linear acceleration in W, we need: a = (1/m)(R @ f_body - mg*zW)
        R rotates vectors from body frame to world frame.

    The standard quaternion-to-rotation formula (from quaternion algebra):
        R[0,0] = 1 - 2(qy² + qz²)   etc.
    """
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),    2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),      2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
    ])
    return R

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Hamilton Product
# ─────────────────────────────────────────────────────────────────────────────
def quat_mult(q, p):
    """
    Compute Hamilton product q ⊗ p (PDF slides 25-26).

    Why we need this:
        Quaternion kinematics :
            dq/dt = (1/2) * q ⊗ ω_pure
        where ω_pure = [0, wx, wy, wz] is angular velocity as a pure quaternion.
        This is how we propagate orientation over time — instead of using
        rotation matrices (9 numbers), we use quaternions (4 numbers).

    Formula (PDF slide 26):
        q = (a1, b1, c1, d1)
        p = (a2, b2, c2, d2)
        w = a1a2 - b1b2 - c1c2 - d1d2
        x = a1b2 + b1a2 + c1d2 - d1c2
        y = a1c2 - b1d2 + c1a2 + d1b2
        z = a1d2 + b1c2 - c1b2 + d1a2
    """
    a1, b1, c1, d1 = q
    a2, b2, c2, d2 = p
    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ])

# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION MATRIX: rotor speeds² → wrench
# ─────────────────────────────────────────────────────────────────────────────
def compute_wrench(w2):
    """
    Compute body-frame wrench [fx, fy, fz, tx, ty, tz] from rotor speeds squared.

    Why we need this:
        The dynamic model takes forces and torques as input (wrench).
        But the controller (nhfc) outputs rotor velocity commands.
        So we need to convert: rotor ω² → wrench.
        This is exactly what mrsim-gazebo plugin does inside Gazebo.

    Quadrotor geometry (from SDF):
        Rotor 1: (+L,  0,  0), spin CW  → drag torque in -z
        Rotor 2: ( 0, +L,  0), spin CCW → drag torque in +z
        Rotor 3: (-L,  0,  0), spin CW  → drag torque in -z
        Rotor 4: ( 0, -L,  0), spin CCW → drag torque in +z

    All rotors point upward (+z body), so:
        fz = cf * (w1² + w2² + w3² + w4²)    — total upward thrust
        fx = fy = 0                          — no lateral force (under-actuated)
        tx = cf*L * (-w2² + w4²)             — roll torque from rotor imbalance
        ty = cf*L * ( w1² - w3²)             — pitch torque from rotor imbalance
        tz = ct  * (-w1² + w2² - w3² + w4²)  — yaw torque from drag (CW vs CCW)

    These match exactly the commented allocation matrix in mrsim-quadrotor/model.sdf.

    Args:
        w2: np.array shape (4,) — rotor speeds squared [rad²/s²]
    Returns:
        wrench: np.array shape (6,) — [fx, fy, fz, tx, ty, tz] in body frame [N, Nm]
    """
    fz = cf * (w2[0] + w2[1] + w2[2] + w2[3])
    tx = cf * L * (-w2[1] + w2[3])
    ty = cf * L * ( w2[0] - w2[2])
    tz = ct      * (-w2[0] + w2[1] - w2[2] + w2[3])
    return np.array([0.0, 0.0, fz, tx, ty, tz])

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC MODEL: f(x, u) — continuous-time equations of motion
# ─────────────────────────────────────────────────────────────────────────────
def f_dynamics(x, wrench):
    """
    Compute state derivative xdot = f(x, u) — PDF slides 22-24.

    This is the continuous-time dynamic model of the quadrotor.
    It takes the current state x and the wrench u, and returns how
    fast each state variable is changing right now.

    This is what Gazebo's dynamic engine evaluates at each time step.
    We replicate it here in pure Python.

    Equations of motion (Newton-Euler formalism, PDF slide 22):
        [m·p̈ ]   =  -[  mg·zW         ] + G·w(u)
        [J·ω̇ ]      -[ω × J·ω         ]

    where G = [[R, 0], [0, I]] rotates the body-frame force into world frame.

    This expands to (PDF slide 23, diagonal J assumption):
        ṗ  = v                                  (1. position kinematics)
        q̇  = (1/2) q ⊗ [0, ω]                   (2. orientation kinematics)
        v̇  = (1/m)(-mg·ẑW + R·f_body)           (3. linear acceleration)
        ω̇  = J⁻¹(-ω × J·ω + τ_body)             (4. angular acceleration)

    Args:
        x:      state vector (13,)
        wrench: [fx, fy, fz, tx, ty, tz] in body frame (6,)
    Returns:
        xdot:   state derivative (13,)
    """
    # unpack state
    # p  = x[0:3]   # position (not needed directly in equations)
    q  = x[3:7]    # quaternion [qw, qx, qy, qz]
    v  = x[7:10]   # linear velocity in world frame
    w  = x[10:13]  # angular velocity in body frame

    # unpack wrench
    f_body = wrench[0:3]   # force in body frame [N]
    tau    = wrench[3:6]   # torque in body frame [Nm]

    # rotation matrix: body → world
    R = quat_to_R(q)

    # ── 1. Position kinematics: dp/dt = v ────────────────────────────────────
    # Velocity IS the time derivative of position. Simple.
    p_dot = v

    # ── 2. Orientation kinematics: dq/dt = (1/2) q ⊗ ω_pure ─────────────────
    # Angular velocity ω is in body frame.
    # To propagate quaternion, we form a pure quaternion [0, wx, wy, wz]
    # and use Hamilton product. This is the quaternion kinematics equation.
    w_pure = np.array([0.0, w[0], w[1], w[2]])
    q_dot  = 0.5 * quat_mult(q, w_pure)

    # ── 3. Linear acceleration: dv/dt = (1/m)(-mg·zW + R·f_body) ────────────
    # Gravity pulls down in world frame: [0, 0, -mg]
    # Rotor thrust is generated in body frame (+z body direction)
    # R rotates it to world frame so we can add it to gravity
    gravity = np.array([0.0, 0.0, -mass * grav])
    v_dot   = (1.0 / mass) * (gravity + R @ f_body)

    # ── 4. Angular acceleration: dω/dt = J⁻¹(-ω × Jω + τ) ──────────────────
    # -ω × Jω is the gyroscopic term: a spinning body resists changes
    #   to its rotation axis (like a gyroscope). It appears because
    #   we write the equations in the rotating body frame.
    # τ is the net torque from rotors (roll/pitch/yaw moments)
    Jw    = J @ w
    w_dot = Ji @ (-np.cross(w, Jw) + tau)

    return np.concatenate([p_dot, q_dot, v_dot, w_dot])

# ─────────────────────────────────────────────────────────────────────────────
# RK4 INTEGRATION STEP
# ─────────────────────────────────────────────────────────────────────────────
def rk4_step(x, wrench, dt):
    """
    Advance state by one time step using 4th-order Runge-Kutta (PDF slide 30).

    Why RK4 instead of Euler?
        Euler: x_{k+1} = x_k + dt * f(x_k, u_k)
        → uses slope only at t_k → accumulates error quickly

        RK4: evaluates f() at 4 different points within [t_k, t_{k+1}]
        and takes a weighted average → much more accurate for same dt.

    The 4 intermediate slopes:
        k1 = f(x_k,           u)  slope at start
        k2 = f(x_k + dt/2*k1, u)  slope at midpoint using k1
        k3 = f(x_k + dt/2*k2, u)  slope at midpoint using k2 (better midpoint)
        k4 = f(x_k + dt*k3,   u)  slope at end using k3

    Weighted average (k2 and k3 count double — midpoints are more representative):
        x_{k+1} = x_k + (dt/6)(k1 + 2k2 + 2k3 + k4)

    The input wrench is kept constant over dt (zero-order holder assumption).

    Args:
        x:      current state (13,)
        wrench: current wrench (6,) — constant over this step
        dt:     time step [s]
    Returns:
        x_new:  state at next time step (13,)
    """
    k1 = f_dynamics(x,              wrench)  # slope at t_k
    k2 = f_dynamics(x + dt/2 * k1, wrench)
    k3 = f_dynamics(x + dt/2 * k2, wrench)
    k4 = f_dynamics(x + dt   * k3, wrench)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def normalize_quat(x):
    """
    Normalize the quaternion part of state x (PDF slides 31-32).

    Why needed:
        A valid rotation quaternion must satisfy ||q|| = 1.
        Numerical integration (RK4, Euler, etc.) does NOT preserve this.
        Each step introduces tiny errors → norm drifts from 1 → invalid rotation.
        Solution: after every integration step, divide q by its norm.

    Only touches x[3:7], leaves position, velocity, angular velocity untouched.
    """
    x      = x.copy()
    q      = x[3:7]
    x[3:7] = q / np.linalg.norm(q)
    return x

# ─────────────────────────────────────────────────────────────────────────────
# GROUND REACTION
# ─────────────────────────────────────────────────────────────────────────────
def ground_reaction(x, z_ground=0.0):
    """
    Apply ground reaction when drone hits the ground (PDF slide 33).

    Why needed:
        The dynamic model is a "floating body" — it has no floor.
        Without this, the drone would fall through z=0 indefinitely.

    Rule (from PDF):
        If z < z_ground:
            → set z = z_ground  (keep drone at ground level)
            → set all velocities and accelerations to 0
               (ground absorbs kinetic energy instantly)

    This is a simplified contact model — Gazebo uses more complex
    collision physics, but for our purposes this is sufficient.
    """
    x = x.copy()
    if x[2] < z_ground: #13 elements in x , goes as px py pz qw qx qy qz vx vy vz wx wy wz
        x[2]     = z_ground   # z = ground level
        x[7:13]  = 0.0        # zero all velocities: v and ω
    return x

# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION → EULER ANGLES (for plotting)
# ─────────────────────────────────────────────────────────────────────────────
def quat_to_euler(q): # we didnt have roll pitch yaw before ! 
    # quaternion is a more compact representation of orientation, but for plotting we want roll/pitch/yaw angles in degrees. 
    """
    Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw] in radians.
    Uses extrinsic XYZ convention (same as pom-genom3 log convention).
    """
    qw, qx, qy, qz = q
    roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = np.arcsin (np.clip(2*(qw*qy - qz*qx), -1, 1))
    yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return roll, pitch, yaw

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def simulate(rotor_w2, label):
    """
    Run the full simulation for T seconds with constant rotor input.

    This replicates the Gazebo simulation loop:
        for each time step k:
            a. get input u_k (here: constant rotor speeds)
            b. compute wrench from rotor speeds
            c. integrate dynamic model with RK4
            d. normalize quaternion
            e. apply ground reaction

    Args:
        rotor_w2: np.array (4,) — constant rotor speeds squared [rad²/s²]
        label:    string for saving plots
    Returns:
        history: np.array (steps+1, 13) — full state trajectory
    """
    print(f"\n=== Simulating: {label} ===")
    print(f"  Rotor w² = {rotor_w2[0]:.1f} rad²/s²")
    print(f"  Expected fz = {cf * 4 * rotor_w2[0]:.4f} N  (mg = {mass*grav:.4f} N)")

    x       = x0.copy()
    history = np.zeros((steps + 1, 13))
    history[0] = x

    # Pre-compute wrench (constant input → same wrench every step)
    wrench = compute_wrench(rotor_w2)

    for k in range(steps):
        # Step a+b: wrench already computed (constant input)

        # Step c: RK4 integration — advance state by dt
        x = rk4_step(x, wrench, dt)

        # Step d: normalize quaternion — keep ||q|| = 1
        x = normalize_quat(x)

        # Step e: ground reaction — prevent drone from falling through floor
        x = ground_reaction(x)

        history[k + 1] = x

    print(f"  Final z = {history[-1, 2]:.4f} m")
    return history

# ─────────────────────────────────────────────────────────────────────────────
# DEFINE INPUTS: hover and climb
# ─────────────────────────────────────────────────────────────────────────────

# Test 1: Hover — lifting force exactly equals weight
#   fz = cf * 4 * w²  =  mg
#   → w² = mg / (4 * cf)
#   Expected result: drone stays at z=0, all states constant
hover_w2_val = (mass * grav) / (4 * cf)
w2_hover     = np.array([hover_w2_val] * 4)

# Test 2: Climb — lifting force 1N above weight
#   fz = mg + 1  →  w² = (mg + 1) / (4 * cf)
#   Expected result: drone slowly accelerates upward
climb_w2_val = (mass * grav + 1.0) / (4 * cf)
w2_climb     = np.array([climb_w2_val] * 4)

# ─────────────────────────────────────────────────────────────────────────────
# RUN SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────
hist_hover = simulate(w2_hover, 'hover')
hist_climb = simulate(w2_climb, 'climb')

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING — same layout as Part I (PDF requirement)
# Only measured state — no desired setpoints (PDF slide 35)
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs('/shared-workspace/logs/01b-simulator', exist_ok=True)

t = np.linspace(0, T, steps + 1)   # time axis starting from 0

XYZ = ['red', 'green', 'blue']


def plot_state(hist, wrench, label):
    """
    Generate state plots matching Part I layout.
    5 subplots: position, attitude, linear velocity, angular velocity, acceleration.
    Only measured (solid lines) — no desired setpoints.
    """

    # Pre-compute Euler angles from quaternion history
    rolls, pitches, yaws = [], [], []
    for i in range(len(hist)):
        r, p, y = quat_to_euler(hist[i, 3:7])
        rolls.append(r); pitches.append(p); yaws.append(y)
    rolls   = np.degrees(np.array(rolls))
    pitches = np.degrees(np.array(pitches))
    yaws    = np.degrees(np.array(yaws))

    # Pre-compute accelerations: a = v_dot from f_dynamics
    # We evaluate f() at each stored state to get the acceleration
    accs = np.zeros((len(hist), 3))
    for i in range(len(hist)):
        xdot      = f_dynamics(hist[i], wrench)
        accs[i]   = xdot[7:10]   # linear acceleration part of xdot

    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    fig.suptitle(f'Quadrotor Standalone Simulator — {label}', fontsize=14)

    # 1. Position [m]
    ax = axes[0]
    for col, c, lbl in zip([0,1,2], XYZ, ['x','y','z']):
        ax.plot(t, hist[:, col], color=c, label=lbl)
    ax.set_ylabel('Position [m]')
    ax.set_title('Position')
    ax.legend(); ax.grid(True)

    # 2. Attitude [deg]
    ax = axes[1]
    ax.plot(t, rolls,   color='red',   label='roll')
    ax.plot(t, pitches, color='green', label='pitch')
    ax.plot(t, yaws,    color='blue',  label='yaw')
    ax.set_ylabel('Attitude [deg]')
    ax.set_title('Attitude (Euler extrinsic XYZ)')
    ax.legend(); ax.grid(True)

    # 3. Linear velocity [m/s]
    ax = axes[2]
    for col, c, lbl in zip([7,8,9], XYZ, ['vx','vy','vz']):
        ax.plot(t, hist[:, col], color=c, label=lbl)
    ax.set_ylabel('Linear Velocity [m/s]')
    ax.set_title('Linear Velocity')
    ax.legend(); ax.grid(True)

    # 4. Angular velocity [deg/s]
    ax = axes[3]
    for col, c, lbl in zip([10,11,12], XYZ, ['wx','wy','wz']):
        ax.plot(t, np.degrees(hist[:, col]), color=c, label=lbl)
    ax.set_ylabel('Angular Velocity [deg/s]')
    ax.set_title('Angular Velocity')
    ax.legend(); ax.grid(True)

    # 5. Linear acceleration [m/s²]
    ax = axes[4]
    for col, c, lbl in zip([0,1,2], XYZ, ['ax','ay','az']):
        ax.plot(t, accs[:, col], color=c, label=lbl)
    ax.set_ylabel('Linear Acceleration [m/s²]')
    ax.set_title('Linear Acceleration')
    ax.set_xlabel('Time [s]')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    out = f'/shared-workspace/logs/01b-simulator/{label}_state.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.close()


# Compute wrenches for plotting
wrench_hover = compute_wrench(w2_hover)
wrench_climb = compute_wrench(w2_climb)

plot_state(hist_hover, wrench_hover, 'hover')
plot_state(hist_climb, wrench_climb, 'climb')

print('\n=== All done! ===')
print('Plots saved to /shared-workspace/logs/01b-simulator/')
