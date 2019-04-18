import numpy as np
from math import sin, cos, atan, atan2
import sys
sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot
from parameters import aerosonde_parameters as MAV

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(50)  # approach angle for large distance from straight-line path
        self.k_path = 0.02  # proportional gain for straight-line path following
        self.k_orbit = 2.5  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        q = path.line_direction
        r = path.line_origin

        # Airspeed and Course
        chi_q = np.arctan2(q.item(1),q.item(0))
        chi_q = self._wrap(chi_q,state.chi)
        cc = np.cos(chi_q)
        sc = np.sin(chi_q)
        Rpi = np.array([[cc, sc, 0.],\
                        [-sc, cc, 0.],\
                        [0., 0., 1.]])
        p = np.array([[state.pn, state.pe, -state.h]]).T
        ep = Rpi@(p-r)
        epy = ep.item(1)
        self.autopilot_commands.airspeed_command = 25.0
        self.autopilot_commands.course_command = chi_q - self.chi_inf*2./np.pi*np.arctan(self.k_path*epy)

        # Altitude and phi feedforward
        k = np.array([[0.,0.,1.]])
        n = (np.cross(q.T,k)/np.linalg.norm(np.cross(q.T,k))).T
        ei_p = p-r
        s = ei_p-(ei_p.T@n)*n
        s = s.reshape(3,1)
        rd = r.item(2)
        sn = s.item(0)
        se = s.item(1)
        qn = q.item(0)
        qe = q.item(1)
        qd = q.item(2)
        self.autopilot_commands.altitude_command = -rd - np.sqrt(sn**2.+se**2.)*qd/np.sqrt(qn**2.+qe**2.)
        self.autopilot_commands.phi_feedforward = 0.

    def _follow_orbit(self, path, state):
        p = np.array([[state.pn, state.pe, -state.h]]).T
        g = MAV.gravity
        Vg = state.Vg
        psi = state.psi
        chi = state.chi

        c = path.orbit_center
        rho = path.orbit_radius
        dir = path.orbit_direction
        d = np.sqrt((p.item(1)-c.item(1))**2+(p.item(0)-c.item(0))**2)
        varphi = np.arctan2(p.item(1)-c.item(1), p.item(0)-c.item(0))
        # varphi = self._wrap(varphi,state.phi)
        chi_c = varphi + dir*(np.pi/2+np.arctan(self.k_orbit*(d-rho)/rho))
        self.autopilot_commands.airspeed_command = 25.0
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = -c.item(2)
        self.autopilot_commands.phi_feedforward = dir*np.arctan(Vg**2./(g*rho*np.cos(chi-psi)))

    def _wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

