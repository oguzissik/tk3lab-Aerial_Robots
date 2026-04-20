import genomix # type: ignore
import os
import time
import math

g = genomix.connect()
g.rpath('/opt/openrobots/lib/genom/pocolibs/plugins')

optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom       = g.load('pom')
nhfc      = g.load('nhfc')

# Log directory for quadrotor
LOG_DIR = os.path.join(os.environ['TK3LAB_WS'], 'logs', '01a-model', 'quad')


def setup():

    # Connect to simulated optitrack on localhost:1509
    optitrack.connect({
        'host': 'localhost', 'host_port': '1509',
        'mcast': '', 'mcast_port': '0'
    })

    # /tmp/pty-qr4 is the virtual serial port for the quadrotor in Gazebo
    rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})

    # IMU at 1kHz, motor speeds at 20Hz, magnetometer disabled in simulation
    rotorcraft.set_sensor_rate({'rate': {
        'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
    }})

    # Low-pass filter: 20Hz for gyroscope, 5Hz for accelerometer
    rotorcraft.set_imu_filter({
        'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
    })

    # Connect nhfc rotor commands to rotorcraft
    rotorcraft.connect_port({
        'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
    })

    # Quadrotor geometry: 4 rotors, 23cm arms, 1.28kg, rotors pointing down
    nhfc.set_gtmrp_geom({
        'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0,
        'armlen': 0.23, 'mass': 1.28,
        'rx': 0, 'ry': 0, 'rz': -1,
        'cf': 6.5e-4, 'ct': 1e-5
    })

    # Emergency descent parameters
    nhfc.set_emerg({'emerg': {
        'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
    }})

    # PID gains
    nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
    nhfc.set_servo_gain({'gain': {
        'Kpxy': 5,  'Kpz': 5,
        'Kqxy': 4,  'Kqz': 0.1,
        'Kvxy': 6,  'Kvz': 6,
        'Kwxy': 1,  'Kwz': 0.1,
        'Kixy': 0,  'Kiz': 0
    }})

    # Under-actuated: tilt prioritized mode
    nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

    # Connect measured rotor speeds from rotorcraft to nhfc
    nhfc.connect_port({
        'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
    })

    # Connect estimated state from pom to nhfc
    nhfc.connect_port({
        'local': 'state', 'remote': 'pom/frame/robot'
    })

    # UKF prediction model: constant acceleration
    pom.set_prediction_model('::pom::constant_acceleration')
    pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})

    # Accept sensor data up to 250ms old
    pom.set_history_length({'history_length': 0.25})

    # Magnetic field configuration
    pom.set_mag_field({'magdir': {
        'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
    }})

    # IMU and magnetometer from rotorcraft to pom
    pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
    pom.add_measurement('imu')
    pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
    pom.add_measurement('mag')

    # QR_4 is the quadrotor model name in Gazebo optitrack plugin
    pom.connect_port({
        'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'
    })
    pom.add_measurement('mocap')


def start():
    pom.log_state(os.path.join(LOG_DIR, 'pom.log'))
    pom.log_measurements('/tmp/pom-measurements.log')
    optitrack.set_logfile(os.path.join(LOG_DIR, 'optitrack.log'))
    rotorcraft.log(os.path.join(LOG_DIR, 'rotorcraft.log'))
    rotorcraft.start()
    rotorcraft.servo(ack=True)
    nhfc.log(os.path.join(LOG_DIR, 'nhfc.log'))
    nhfc.set_current_position() 


def stop():
    rotorcraft.stop()
    rotorcraft.log_stop()
    nhfc.stop()
    nhfc.log_stop()
    pom.log_stop()
    optitrack.unset_logfile()


def simulation():
    print("=== QUAD Simulation Starting from -1, 0, 0.25 ===")
    #pro takes it as  0 0 0 and adding the positions given on pdf on top respectively

    print("[1/6] Running setup...")
    setup()

    print("[2/6] Starting drone...")
    start()

    print("[3/6] wp0: Hovering at start, waiting 5s...")
    time.sleep(5)

    print("[4/6] wp1: Moving to (0, 1, 1.25, yaw=0)...")
    print("no error expected in z unlike hexa")
    nhfc.set_position(0, 1, 1.25, 0)
    time.sleep(20)

    print("[5/6] wp2: Moving to (0, -1, 1.25, yaw=pi/2)...")
    nhfc.set_position(0, -1, 1.25, math.pi / 2)
    time.sleep(10)

    print("[6/6] wp3: Returning to (-1, 0, 0.25, yaw=0)...")
    nhfc.set_position(-1, 0, 0.25, 0)
    time.sleep(5)

    print("=== Simulation complete, stopping ===")
    stop()
    print("=== Done ===")