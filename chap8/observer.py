"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
# from tools.tools import Euler2Rotation

from message_types.msg_state import msg_state

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.5)
        self.lpf_accel_y = alpha_filter(alpha=0.5)
        self.lpf_accel_z = alpha_filter(alpha=0.5)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9,y0=-MAV.rho*MAV.gravity*MAV.pd0)
        self.lpf_diff = alpha_filter(alpha=0.5,y0=MAV.rho*MAV.Va0**2./2.)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements):
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        gyro_x = measurements.gyro_x
        gyro_y = measurements.gyro_y
        gyro_z = measurements.gyro_z
        self.estimated_state.p = self.lpf_gyro_x.update(gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(gyro_z) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        static_pressure = measurements.static_pressure
        diff_pressure = measurements.diff_pressure
        self.estimated_state.h = self.lpf_static.update(static_pressure)/(MAV.rho*MAV.gravity)
        self.estimated_state.Va = np.sqrt(2.0*self.lpf_diff.update(diff_pressure)/MAV.rho)

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha*self.y + (1-self.alpha)*u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        self.Q = np.eye(2)*1.*10**(-9.)
        self.Q_gyro = SENSOR.gyro_sigma**2.*np.eye(3)
        self.R_accel = SENSOR.accel_sigma**2.*np.eye(3)
        self.N = 9  # number of prediction step per sample
        self.xhat = np.array([[0.],[0.05]]) # initial state: phi, theta
        self.P = np.eye(2)*0.1#(np.pi/10.)**2.# Max distance away, squared
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        p = state.p
        q = state.q
        r = state.r
        phi = x.item(0)
        theta = x.item(1)
        _f = np.array([[p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)],\
                        [q*np.cos(phi) - r*np.sin(phi)]])
        return _f

    def h(self, x, state):
        # measurement model y
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        g = MAV.gravity
        phi = x.item(0)
        theta = x.item(1)
        _h = np.array([[q*Va*np.sin(theta) + g*np.sin(theta)],\
                       [r*Va*np.cos(theta) - p*Va*np.sin(theta) - g*np.cos(theta)*np.sin(phi)],\
                       [-q*Va*np.cos(theta) - g*np.cos(theta)*np.cos(phi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat = self.xhat + (self.Ts)*self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state) # missing brackets on the columns-> np.array([[q * np.cos(phi) * np.tan(theta) - r * np.sin(phi) * np.tan(theta), (q*np.sin(phi)-r*np.cos(phi))/np.cos(theta)**2.], [-q * np.sin(phi) - r * np.cos(theta), 0]])
            # compute G matrix for gyro noise
            p = state.p
            q = state.q
            r = state.r
            phi = self.xhat.item(0)
            theta = self.xhat.item(1)
            G = np.array([[1.,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
                          [0.,np.cos(phi),-np.sin(phi)]])
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(2) + A*self.Ts + A@A*(self.Ts**2.)/2.
            G_d = G*self.Ts
            Q_d = self.Q*self.Ts**2.
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + G_d@self.Q_gyro@G_d.T + Q_d

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T
        C = jacobian(self.h, self.xhat, state)
        L = self.P@C.T@np.linalg.inv(self.R_accel+C@self.P@C.T)
        self.P = (np.eye(2)-L@C)@self.P@(np.eye(2)-L@C).T + L@self.R_accel@L.T
        self.xhat = self.xhat + L@(y-h)

        # for i in range(0, 3):
        #     if np.abs(y[i]-h[i,0]) < threshold:
        #         Ci =
        #         L =
        #         self.P = (np.eye()-L@Ci)@self.P@(np.eye()-L@Ci).T + L@self.R_accel@L.T
        #         self.xhat = self.xhat +

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        self.Q = np.eye(7)*1.*10**-2
        self.R_gps = np.eye(4)
        self.R_gps[0,0] = SENSOR.gps_n_sigma**2
        self.R_gps[1,1] = SENSOR.gps_e_sigma**2
        self.R_gps[2,2] = SENSOR.gps_Vg_sigma**2
        self.R_gps[3,3] = SENSOR.gps_course_sigma**2
        self.R_psuedo = np.eye(2)
        self.R_psuedo[0,0] = 0.01
        self.R_psuedo[1,1] = 0.01
        self.N = 4  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([[0.],[0.],[25.],[0.],[0.],[0.],[0.]])
        self.P = np.eye(7)*0.1
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        # state.pn = measurement.gps_n#self.xhat.item(0)
        # state.pe = measurement.gps_e#self.xhat.item(1)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        # state.chi = measurement.gps_course#self.xhat.item(3)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        phi = state.phi
        theta = state.theta
        Va = state.Va
        q = state.q
        r = state.r
        psid = q*np.sin(phi)/np.cos(theta)+r*np.cos(phi)/np.cos(theta)
        _f = np.array([[Vg*np.cos(chi)],\
                       [Vg*np.sin(chi)],\
                       [((Va*np.cos(psi)+wn)*(-Va*psid*np.sin(psi))+(Va*np.sin(psi)+we)*(Va*psid*np.cos(psi)))/Vg],\
                       [MAV.gravity/Vg*np.tan(phi)*np.cos(chi-psi)],\
                       [0],\
                       [0],\
                       [psid]])
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2)
        chi = x.item(3)
        _h = np.array([[pn],\
                       [pe],\
                       [Vg],\
                       [chi]])
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangle pseudo measurement
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = state.Va
        _h = np.array([[Va*np.cos(psi)+wn-Vg*np.cos(chi)],\
                       [Va*np.sin(psi)+we-Vg*np.sin(chi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat = self.xhat + (self.Ts)*self.f(self.xhat,state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(7) + A*self.Ts + A @ A * (self.Ts**2)/2.
            Q_d = self.Q*self.Ts**2
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + Q_d

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([[0, 0]]).T
        L = self.P @ C.T @ np.linalg.inv(self.R_psuedo + C @ self.P @ C.T)
        self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + L @ self.R_psuedo @ L.T
        # self.xhat = self.xhat + L@(y-C@self.xhat)
        self.xhat = self.xhat + L @ (y - h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([[measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course]]).T
            L = self.P @ C.T @ np.linalg.inv(self.R_gps + C @ self.P @ C.T)
            self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + L @ self.R_gps @ L.T
            # print("\ny:",y[3,0])
            # print("h:",h[3,0])
            y[3,0] = self.wrap(y[3,0],h[3,0])
            # print("ywrap:",y[3,0])
            self.xhat = self.xhat + L @ (y - h)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

    def wrap(self, chi_c, chi):
        while chi_c-chi >= np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

def jacobian(fun, x, state):
    # compute jacobian of fun with respect to x
    f = fun(x, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J