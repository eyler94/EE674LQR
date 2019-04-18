"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
import numpy as np
import control.matlab as ctl

class wind_simulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[0., 0., 0.]]).T
        # self.steady_state = np.array([[3., 1., 0.]]).T

        #   Dryden gust model parameters (pg 56 UAV book)


        # HACK:  Setting Va to a constant value is a hack.  We set a nominal airspeed for the gust model.
        # Could pass current Va into the gust function and recalculate A and B matrices.
        Va = 17

        # self._A = np.array([[1-Ts*c, -Ts*d],[Ts, 1]])
        # self._B = np.array([[Ts],[0]])
        # self._C = np.array([[a, b]])

        suv= 1.06
        sw= 0.7
        luv = 200
        lw = 50

        hu_n = np.array([suv*np.sqrt(2*Va/luv)])
        hu_d = np.array([1, Va/luv])

        hv_n = suv*np.sqrt(3*Va/luv)*np.array([1, Va/(np.sqrt(3)*luv)])
        hv_d = np.array([1, 2*Va/luv, (Va/luv)**2])

        hw_n = sw*np.sqrt(3*Va/lw)*np.array([1, Va/(np.sqrt(3)*lw)])
        hw_d = np.array([1, 2*Va/lw, (Va/lw)**2])

        self.hu = ctl.ss(ctl.tf(hu_n, hu_d, Ts))
        self.hv = ctl.ss(ctl.tf(hv_n, hv_d, Ts))
        self.hw = ctl.ss(ctl.tf(hw_n, hw_d, Ts))
        self.huA = np.asarray(self.hu.A)
        self.huB = np.asarray(self.hu.B)
        self.huC = np.asarray(self.hu.C)

        self.hvA = np.asarray(self.hv.A)
        self.hvB = np.asarray(self.hv.B)
        self.hvC = np.asarray(self.hv.C)

        self.hwA = np.asarray(self.hw.A)
        self.hwB = np.asarray(self.hw.B)
        self.hwC = np.asarray(self.hw.C)

        self._gust_state_u = np.array([0.])
        self._gust_state_v = np.array([[0.], [0.]])
        self._gust_state_w = np.array([[0.], [0.]])
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        return np.concatenate((self._steady_state, self._gust()))

    def _gust(self):
        # calculate wind gust using Dryden model.  Gust is defined in the body frame
        wn_u = np.random.randn()  # zero mean unit variance Gaussian (white noise)
        wn_v = np.random.randn()  # zero mean unit variance Gaussian (white noise)
        wn_w = np.random.randn()  # zero mean unit variance Gaussian (white noise)
        # propagate Dryden model (Euler method): x[k+1] = x[k] + Ts*( A x[k] + B w[k] )
        self._gust_state_u = self._gust_state_u + self._Ts * (self.huA @ self._gust_state_u + self.huB * wn_u)
        self._gust_state_v = self._gust_state_v + self._Ts * (self.hvA @ self._gust_state_v + self.hvB * wn_v)
        self._gust_state_w = self._gust_state_w + self._Ts * (self.hwA @ self._gust_state_w + self.hwB * wn_w)
        # output the current gust: y[k] = C x[k]
        gust_vect = np.array([self.huC @ self._gust_state_u,
                             self.hvC @ self._gust_state_v,
                             self.hwC @ self._gust_state_w
                             ])
        return gust_vect.reshape((3,1))

