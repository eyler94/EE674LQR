"""
mavDynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:
        2/16/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state
from message_types.msg_sensors import msg_sensors

import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.tools import Quaternion2Rotation, Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self._forces = np.array([[0.], [0.], [0.]])
        # initialize the sensors message
        self.sensors = msg_sensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

        ######################### Old init code
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                                [MAV.pe0],  # (1)
                                [MAV.pd0],  # (2)
                                [MAV.u0],  # (3)
                                [MAV.v0],  # (4)
                                [MAV.w0],  # (5)
                                [MAV.e0],  # (6)
                                [MAV.e1],  # (7)
                                [MAV.e2],  # (8)
                                [MAV.e3],  # (9)
                                [MAV.p0],  # (10)
                                [MAV.q0],  # (11)
                                [MAV.r0]])  # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.msg_true_state = msg_state()


    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
                    Integrate the differential equations defining dynamics, update sensors
                    delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
                    wind is the wind vector in inertial coordinates
                    Ts is the time step between function calls.
                '''
        # get forces and moments acting on rigid body
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2. * k1, forces_moments)
        k3 = self._derivatives(self._state + time_step / 2. * k2, forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        # print("\ne0:", e0,\
        #       "\ne1:", e1,\
        #       "\ne2:", e2,\
        #       "\ne3:", e3,\
        #       )
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6][0] = self._state.item(6) / normE
        self._state[7][0] = self._state.item(7) / normE
        self._state[8][0] = self._state.item(8) / normE
        self._state[9][0] = self._state.item(9) / normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    def update_sensors(self):
        "Return value of sensors on MAV: gyros, accels, static_pressure, dynamic_pressure, GPS"
        ### Gyro sensors
        p = self.msg_true_state.p
        q = self.msg_true_state.q
        r = self.msg_true_state.r
        B_gyro_x=0.
        B_gyro_y=0.
        B_gyro_z=0.
        eta_gyro_x = np.random.randn()*SENSOR.gyro_sigma
        eta_gyro_y = np.random.randn()*SENSOR.gyro_sigma
        eta_gyro_z = np.random.randn()*SENSOR.gyro_sigma
        self.sensors.gyro_x = p + B_gyro_x + eta_gyro_x
        self.sensors.gyro_y = q + B_gyro_y + eta_gyro_y
        self.sensors.gyro_z = r + B_gyro_z + eta_gyro_z

        ### Accel sensors
        fx = self._forces.item(0)
        fy = self._forces.item(1)
        fz = self._forces.item(2)
        mass = MAV.mass
        phi = self.msg_true_state.phi
        theta = self.msg_true_state.theta
        psi = self.msg_true_state.psi
        eta_accel_x = np.random.randn()*SENSOR.accel_sigma
        eta_accel_y = np.random.randn()*SENSOR.accel_sigma
        eta_accel_z = np.random.randn()*SENSOR.accel_sigma
        self.sensors.accel_x = fx/mass + MAV.gravity*np.sin(theta) + eta_accel_x
        self.sensors.accel_y = fy/mass - MAV.gravity*np.cos(theta)*np.sin(phi) + eta_accel_y
        self.sensors.accel_z = fz/mass - MAV.gravity*np.cos(theta)*np.cos(phi) + eta_accel_z

        ### Pressure sensors
        B_abs = 0.
        B_diff = 0.
        eta_abs = np.random.randn()*SENSOR.static_pres_sigma
        eta_diff = np.random.randn()*SENSOR.diff_pres_sigma
        self.sensors.static_pressure = MAV.rho*MAV.gravity*self.msg_true_state.h + B_abs + eta_abs
        # print("SP:",self.sensors.static_pressure)
        self.sensors.diff_pressure = MAV.rho*self.msg_true_state.Va**2./2. + B_diff + eta_diff
        # print("DP:", self.sensors.diff_pressure)

        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = np.exp(-SENSOR.gps_beta*SENSOR.ts_gps)*self._gps_eta_n+np.random.randn()*SENSOR.gps_n_sigma
            self._gps_eta_e = np.exp(-SENSOR.gps_beta*SENSOR.ts_gps)*self._gps_eta_e+np.random.randn()*SENSOR.gps_e_sigma
            self._gps_eta_h = np.exp(-SENSOR.gps_beta*SENSOR.ts_gps)*self._gps_eta_h+np.random.randn()*SENSOR.gps_h_sigma
            self.sensors.gps_n = self.msg_true_state.pn + self._gps_eta_n
            self.sensors.gps_e = self.msg_true_state.pe + self._gps_eta_e
            self.sensors.gps_h = self.msg_true_state.h + self._gps_eta_h
            Vn = self.msg_true_state.Va*np.cos(self.msg_true_state.psi)+self.msg_true_state.wn
            Ve = self.msg_true_state.Va * np.sin(self.msg_true_state.psi) + self.msg_true_state.we
            self.sensors.gps_Vg = np.sqrt(Vn**2 + Ve**2) + np.random.randn()*SENSOR.gps_Vg_sigma
            self.sensors.gps_course = np.arctan2(Ve,Vn) + np.random.randn()*SENSOR.gps_course_sigma
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
                for the dynamics xdot = f(x, u), returns f(x, u)
                """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        pn_dot = (e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2) * u + 2 * (e1 * e2 - e3 * e0) * v + 2 * (e1 * e3 + e2 * e0) * w
        pe_dot = 2 * (e1 * e2 + e3 * e0) * u + (e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2) * v + 2 * (e2 * e3 - e1 * e0) * w
        pd_dot = 2 * (e1 * e3 - e2 * e0) * u + 2 * (e2 * e3 + e1 * e0) * v + (e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2) * w

        # position dynamics
        mass = MAV.mass
        u_dot = (r * v - q * w) + fx / mass
        v_dot = (p * w - r * u) + fy / mass
        w_dot = (q * u - p * v) + fz / mass
        self.ud = u_dot
        self.vd = v_dot
        self.wd = w_dot

        # rotational kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # rotational dynamics
        p_dot = MAV.gamma1 * p * q - MAV.gamma2 * q * r + MAV.gamma3 * l + MAV.gamma4 * n
        q_dot = MAV.gamma5 * p * r - MAV.gamma6 * (p ** 2 - r ** 2) + m / MAV.Jy
        r_dot = MAV.gamma7 * p * q - MAV.gamma1 * q * r + MAV.gamma4 * l + MAV.gamma8 * n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # Split wind into components
        self._ur = self._state.item(3) - wind.item(0)  # u - uw
        self._vr = self._state.item(4) - wind.item(1)  # v - vw
        self._wr = self._state.item(5) - wind.item(2)  # w - ww
        # compute airspeed
        self._Va = np.sqrt(self._ur ** 2 + self._vr ** 2 + self._wr ** 2)
        # compute angle of attack
        self._alpha = np.arctan(self._wr / self._ur)
        # compute sideslip angle
        self._beta = np.arcsin(self._vr / self._Va)

    def _forces_moments(self, delta):
        """
                return the forces on the UAV based on the state, wind, and control surfaces
                :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
                :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
                """
        # assert delta.shape == (4,1)
        de = delta[0, 0]
        dt = delta[1, 0]
        da = delta[2, 0]
        dr = delta[3, 0]

        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        Fg = MAV.mass * MAV.gravity * np.array([[2 * (e1 * e3 - e2 * e0)],
                                                [2 * (e2 * e3 + e1 * e0)],
                                                [e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2],
                                                ])

        M_e = 25
        sig = lambda a: (1 + np.exp(-M_e * (a - MAV.alpha0)) + np.exp(M_e * (a + MAV.alpha0))) / (
                    (1 + np.exp(-M_e * (a - MAV.alpha0))) * (1 + np.exp(M_e * (a + MAV.alpha0))))
        cla = lambda a: (1 - sig(a)) * (MAV.C_L_0 + MAV.C_L_alpha * a) + sig(a) * (
                    2 * np.sign(a) * np.sin(a) ** 2 * np.cos(a))
        cda = lambda a: MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * a) ** 2 / (np.pi * MAV.e * MAV.AR)
        cxa = lambda a: -(cda(a)) * np.cos(a) + (cla(a)) * np.sin(a)

        cxq = lambda a: -MAV.C_D_q * np.cos(a) + MAV.C_L_q * np.sin(a)

        cxde = lambda a: -MAV.C_D_delta_e * np.cos(a) + MAV.C_L_delta_e * np.sin(a)

        cza = lambda a: -(cda(a)) * np.sin(a) - (cla(a)) * np.cos(a)

        czq = lambda a: -MAV.C_D_q * np.sin(a) - MAV.C_L_q * np.cos(a)

        czde = lambda a: -MAV.C_D_delta_e * np.sin(a) - MAV.C_L_delta_e * np.cos(a)

        c = MAV.c / (2 * self._Va)
        b = MAV.b / (2 * self._Va)

        Fa = 0.5 * MAV.rho * self._Va ** 2 * MAV.S_wing * np.array([ \
            [1, 0, 0], [0, 1, 0], [0, 0, 1]]).dot(
            np.array([[cxa(self._alpha) + cxq(self._alpha) * c * q + cxde(self._alpha) * de],
                      [
                          MAV.C_Y_0 + MAV.C_Y_beta * self._beta + MAV.C_Y_p * b * p + MAV.C_Y_r * b * r + MAV.C_Y_delta_a * da + MAV.C_Y_delta_r * dr],
                      [cza(self._alpha) + czq(self._alpha) * c * q + czde(self._alpha) * de],
                      ]))

        F = Fg + Fa
        #
        # print("Fa:",Fa)

        Fp = 0.5 * MAV.rho * MAV.S_prop * MAV.C_prop * ((MAV.k_motor * dt) ** 2 - self._Va ** 2)

        # print("FP:", Fp)

        fx = F.item(0) \
             + Fp \
            # + 0.5*MAV.rho*self._Va**2*MAV.S_wing*(\
        #     +cxa(self._alpha)\
        #     + cxq(self._alpha)*c*q\
        #     + cxde(self._alpha)*de
        #     )

        fy = F.item(1)
        fz = F.item(2)

        #  Moment time!!!
        Ma = 0.5 * MAV.rho * self._Va ** 2 * MAV.S_wing * np.array([ \
            [MAV.b * (
                        MAV.C_ell_0 + MAV.C_ell_beta * self._beta + MAV.C_ell_p * b * p + MAV.C_ell_r * b * r + MAV.C_ell_delta_a * da + MAV.C_ell_delta_r * dr)],
            [MAV.c * (MAV.C_m_0 + (MAV.C_m_alpha * self._alpha) + (MAV.C_m_q * c * q) + (MAV.C_m_delta_e * de))],
            [MAV.b * (MAV.C_n_0 + (MAV.C_n_beta * self._beta) + (MAV.C_n_p * b * p) + (MAV.C_n_r * b * r) + (
                        MAV.C_n_delta_a * da) + (MAV.C_n_delta_r * dr))]
        ])
        # print("\nMa:", Ma)
        Mp = np.array([[-MAV.kTp * (MAV.kOmega * dt) ** 2],
                       [0.],
                       [0.]
                       ])

        M = Mp + Ma

        Mx = M.item(0)
        My = M.item(1)
        Mz = M.item(2)

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # def _motor_thrust_torque(self, Va, delta_t):
    #     return T_p, Q_p

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = np.sqrt(
            (self._state.item(3)) ** 2 + (self._state.item(4)) ** 2 + (self._state.item(5)) ** 2)
        Vg = np.array([self._state.item(3), self._state.item(4), self._state.item(5)])
        Vg_M = np.linalg.norm(Vg)
        Vg_h = np.array([self._state.item(3), self._state.item(4), 0.])
        Vg_h_M = np.linalg.norm(Vg_h)
        Va_h = np.array([self._ur, self._vr, 0.])
        Va_h_M = np.linalg.norm(Va_h)
        self.msg_true_state.gamma = np.arccos(Vg.dot(Vg_h) / (Vg_M * Vg_h_M))
        num = Vg_h.dot(Va_h)
        den = (Vg_h_M * Va_h_M)
        frac = np.round(num / den, 8)
        self.msg_true_state.chi = psi + self._beta + np.arccos(frac)
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
