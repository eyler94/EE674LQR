"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
from chap6.pid_control import pid_control#, pi_control, pd_control_with_rate
from message_types.msg_state import msg_state
from tools.tools import Euler2Quaternion, Quaternion2Euler
from control import matlab


class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pid_control( #pd_control_with_rate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        Ts=ts_control,
                        limit=np.radians(45))
        self.course_from_roll = pid_control( #pi_control(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.sideslip_from_rudder = pid_control( #pi_control(
                        kp=AP.sideslip_kp,
                        ki=AP.sideslip_ki,
                        Ts=ts_control,
                        limit=np.radians(45))
        self.yaw_damper = matlab.tf([0.5, 0.],[1.0, ],ts_control)
                        #
                        # num=np.array([[AP.yaw_damper_kp, 0]]),
                        # den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
                        # Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = pid_control( #pd_control_with_rate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = pid_control( #pi_control(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = pid_control( #pi_control(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.5,
                        throttle_flag=True)
        self.commanded_state = msg_state()

    def update(self, cmd, state):

        # lateral autopilot

        phi_c = self.course_from_roll.update(cmd.course_command,state.chi,reset_flag=True)  #cmd.course_command
        # delta_a = -8.13462186e-09  # Trim state
        delta_a = self.roll_from_aileron.update_with_rate(phi_c, state.phi, state.p) # Controller based on chi command#
        # delta_r = -1.21428507e-08
        delta_r = self.sideslip_from_rudder.update(0,state.beta)

        # longitudinal autopilot
        h_c = cmd.altitude_command
        theta_c = np.pi/16
        theta_c = self.altitude_from_pitch.update(h_c, state.h)
        # delta_e = -1.24785989e-01
        delta_e = self.pitch_from_elevator.update_with_rate(theta_c, state.theta, state.q)
        # delta_t =  3.14346798e-01 # Trim state
        delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)

        # construct output and commanded states
        delta = np.array([[delta_e], [delta_t], [delta_a], [delta_r]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
