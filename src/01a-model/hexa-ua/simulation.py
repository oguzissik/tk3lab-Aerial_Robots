import genomix # type: ignore
import os, time, math

g = genomix.connect()
g.rpath('/opt/openrobots/lib/genom/pocolibs/plugins')

optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom       = g.load('pom')
nhfc      = g.load('nhfc')

LOG_DIR = os.path.join(os.environ['TK3LAB_WS'], 'logs', '01a-model', 'hexa-ua')

# Allocation matrix computed from model.sdf rotor poses
# G is 6x8 (6 wrench components, 8 rotor slots — last 2 columns zero)

#DOES NOT WORK

G = [
-0.00031826,  0.00013418, -0.00013417,  0.00031826,  0.00045243, -0.00045243,  0.0, 0.0,
 0.00033868, -0.00044496, -0.00044496,  0.00033869,  0.00010628,  0.00010628,  0.0, 0.0,
 0.00087431,  0.00087431,  0.00087431,  0.00087431,  0.00087431,  0.00087431,  0.0, 0.0,
 6.11e-06,    0.00029783,  0.00029783,  6.11e-06,   -0.00030394, -0.00030394,  0.0, 0.0,
-0.00034746, -0.00017903,  0.00017903,  0.00034746,  0.00016845, -0.00016845,  0.0, 0.0,
 0.0001153,  -0.0001153,   0.0001153,  -0.00011531,  0.00011529, -0.00011529,  0.0, 0.0,
]

# Inertia matrix from model.sdf (row-major, 3x3)
J = [0.0115, 0, 0,
     0, 0.0114, 0,
     0, 0, 0.0194]


def setup():
    optitrack.connect({
        'host': 'localhost', 'host_port': '1509',
        'mcast': '', 'mcast_port': '0'
    })

    rotorcraft.connect({'serial': '/tmp/pty-hr6', 'baud': 0})

    rotorcraft.set_sensor_rate({'rate': {
        'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
    }})

    rotorcraft.set_imu_filter({
        'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
    })

    rotorcraft.connect_port({
        'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
    })

    # Use set_geom with full allocation matrix instead of set_gtmrp_geom
    # because tilthex has non-uniform rotor tilts
    nhfc.set_geom({
        'mass': 2.3,
        'G': G,
        'J': J
    })

    nhfc.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
    }})

    nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})

    nhfc.set_servo_gain({'gain': {
        'Kpxy': 5,  'Kpz': 7, 
        'Kqxy': 4,  'Kqz': 0.1,
        'Kvxy': 6,  'Kvz': 7,
        'Kwxy': 1,  'Kwz': 0.1,
        'Kixy': 0,  'Kiz': 2
    }})

    # Under-actuated: tilt_prioritized (ignores lateral force capability)
    nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

    nhfc.connect_port({
        'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
    })
    nhfc.connect_port({
        'local': 'state', 'remote': 'pom/frame/robot'
    })

    pom.set_prediction_model('::pom::constant_acceleration')
    pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
    pom.set_history_length({'history_length': 0.25})
    pom.set_mag_field({'magdir': {
        'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
    }})

    pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
    pom.add_measurement('imu')
    pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
    pom.add_measurement('mag')
    pom.connect_port({'local': 'measure/mocap', 'remote': 'optitrack/bodies/HR_6'})
    pom.add_measurement('mocap')


def start():
    pom.log_state(os.path.join(LOG_DIR, 'pom.log'))
    pom.log_measurements('/tmp/pom-measurements-hexa-ua.log')
    optitrack.set_logfile(os.path.join(LOG_DIR, 'optitrack.log'))
    rotorcraft.log(os.path.join(LOG_DIR, 'rotorcraft.log'))
    rotorcraft.start()
    rotorcraft.servo(ack=True)
    nhfc.log(os.path.join(LOG_DIR, 'nhfc.log'))

    # pom gets real mocal value 
    print("  Waiting for pom state to stabilize...")
    time.sleep(5)
    nhfc.set_current_position()
    print("  Current position set.")


def stop():
    rotorcraft.stop()
    rotorcraft.log_stop()
    nhfc.stop()
    nhfc.log_stop()
    pom.log_stop()
    optitrack.unset_logfile()


def simulation():
    print("=== Hexa-UA Simulation Starting from 1, 0, 0.21 ===")
    #pro takes it as  0 0 0 and adding the positions given on pdf on top respectively

    print("[1/6] Running setup...")
    setup()

    print("[2/6] Starting drone...")
    start()

    print("[3/6] wp0: Hovering at start, waiting 5s...")
    time.sleep(5)

    print("[4/6] wp1: Moving to (2, 1, 1.21, yaw=0)...")
    print("there is around 0.5 error in z expected")
    nhfc.set_position(2, 1, 1.21, 0)
    time.sleep(20)

    print("[5/6] wp2: Moving to (2, -1, 1.21, yaw=pi/2)...")
    print("there is around 0.5 error in z expected")
    nhfc.set_position(2, -1, 1.21, math.pi / 2)
    time.sleep(10)

    print("[6/6] wp3: Returning to (1, 0, 0.21, yaw=0)...")
    nhfc.set_position(1, 0, 0.21, 0)
    time.sleep(5)

    print("=== Simulation complete, stopping ===")
    stop()
    print("=== Done ===")