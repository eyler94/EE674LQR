"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from message_types.msg_state import msg_state
from chap6LQR.lqr_control import lqr_control
import chap6LQR.mat as mat

class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.lat = lqr_control(mat.A_lat, mat.B_lat, mat.C_lat, mat.Klat, mat.xlat_eq, mat.ylat_eq, mat.ulat_eq, mat.limitlat, mat.Kilat, ts_control)
        self.lon = lqr_control(mat.A_lon, mat.B_lon, mat.C_lon, mat.Klon, mat.xlon_eq, mat.ylon_eq, mat.ulon_eq, mat.limitlon, mat.Kilon, ts_control, throttle_flag = True)

        # self.roll_from_aileron = pid_control( #pd_control_with_rate(
        #                 kp=AP.roll_kp,
        #                 kd=AP.roll_kd,
        #                 Ts=ts_control,
        #                 limit=np.radians(45))
        # self.course_from_roll = pid_control( #pi_control(
        #                 kp=AP.course_kp,
        #                 ki=AP.course_ki,
        #                 Ts=ts_control,
        #                 limit=np.radians(30))
        # self.sideslip_from_rudder = pid_control( #pi_control(
        #                 kp=AP.sideslip_kp,
        #                 ki=AP.sideslip_ki,
        #                 Ts=ts_control,
        #                 limit=np.radians(45))
        # self.yaw_damper = matlab.tf([0.5, 0.],[1.0, ],ts_control)
        #                 #
        #                 # num=np.array([[AP.yaw_damper_kp, 0]]),
        #                 # den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
        #                 # Ts=ts_control)
        #
        # # instantiate longitudinal controllers
        # self.pitch_from_elevator = pid_control( #pd_control_with_rate(
        #                 kp=AP.pitch_kp,
        #                 kd=AP.pitch_kd,
        #                 limit=np.radians(45))
        # self.altitude_from_pitch = pid_control( #pi_control(
        #                 kp=AP.altitude_kp,
        #                 ki=AP.altitude_ki,
        #                 Ts=ts_control,
        #                 limit=np.radians(30))
        # self.airspeed_from_throttle = pid_control( #pi_control(
        #                 kp=AP.airspeed_throttle_kp,
        #                 ki=AP.airspeed_throttle_ki,
        #                 Ts=ts_control,
        #                 limit=1.5,
        #                 throttle_flag=True)
        self.commanded_state = msg_state()

    def update(self, cmd, state):

        # lateral autopilot
        r_lat = np.array([[0,cmd.course_command]]).T
        y_lat = np.array([[state.beta,state.psi]]).T
        x_lat = np.array([[state.beta,state.p,state.r,state.phi,state.psi]]).T
        u_lat = self.lat.update(r_lat,y_lat,x_lat,"lat",wrap_flag=True)

        phi_c = 0.
        # delta_a = -8.13462186e-09  # Trim state
        delta_a = u_lat.item(0)
        # delta_r = -1.21428507e-08
        delta_r = u_lat.item(1)

        # longitudinal autopilot
        r_lon = np.array([[cmd.airspeed_command, cmd.altitude_command]]).T
        y_lon = np.array([[state.Va, state.h]]).T
        x_lon = np.array([[state.Va, state.alpha, state.q, state.theta, state.h]]).T
        u_lon = self.lon.update(r_lon,y_lon,x_lon,"lon")


        h_c = cmd.altitude_command
        theta_c = 0.05001388909468854
        # delta_e = mat.de_eq#-1.24785989e-01
        delta_e = u_lon.item(0)
        # delta_t =  mat.dt_eq#3.14346798e-01 # Trim state
        delta_t = u_lon.item(1)

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
