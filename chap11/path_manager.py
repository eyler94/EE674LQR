import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path

class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        # dubins path parameters
        self.dubins_path = dubins_parameters()


    def update(self, waypoints, radius, state):
            # this flag is set for one time step to signal a redraw in the viewer
            # if self.path.flag_path_changed == True:
            #     self.path.flag_path_changed = False
            if waypoints.num_waypoints == 0:
                waypoints.flag_manager_requests_waypoints = True
            else:
                if waypoints.type == 'straight_line':
                    self.line_manager(waypoints, state)
                elif waypoints.type == 'fillet':
                    self.fillet_manager(waypoints, radius, state)
                elif waypoints.type == 'dubins':
                    self.dubins_manager(waypoints, radius, state)
                else:
                    print('Error in Path Manager: Undefined waypoint type.')
            return self.path

    def line_manager(self, waypoints, state):
        # print("waypoints",waypoints.ned)#,waypoints.course)
        P = np.array([[state.pn,state.pe,-state.h]]).T
        wi_1 = np.array([[waypoints.ned[0][self.ptr_previous],waypoints.ned[1][self.ptr_previous],waypoints.ned[2][self.ptr_previous]]]).T
        wi = np.array([[waypoints.ned[0][self.ptr_current],waypoints.ned[1][self.ptr_current],waypoints.ned[2][self.ptr_current]]]).T
        wip1 = np.array([[waypoints.ned[0][self.ptr_next],waypoints.ned[1][self.ptr_next],waypoints.ned[2][self.ptr_next]]]).T
        ri_1 = wi_1
        qi_1 = (wi-wi_1)/np.linalg.norm(wi-wi_1)
        qi = (wip1-wi)/np.linalg.norm(wip1-wi)

        ### Increment path
        self.halfspace_r = wi
        self.halfspace_n = (qi_1 + qi) / np.linalg.norm(qi_1 + qi)
        if self.inHalfSpace(P):
            self.increment_pointers()

        ### Return
        self.path.airspeed = waypoints.airspeed.item(self.ptr_previous)
        self.path.line_origin = ri_1
        self.path.line_direction = qi_1

    def fillet_manager(self, waypoints, radius, state):
        # print("waypoints",waypoints.ned)#,waypoints.course)
        P = np.array([[state.pn,state.pe,-state.h]]).T
        R = radius
        wi_1 = np.array([[waypoints.ned[0][self.ptr_previous],waypoints.ned[1][self.ptr_previous],waypoints.ned[2][self.ptr_previous]]]).T
        wi = np.array([[waypoints.ned[0][self.ptr_current],waypoints.ned[1][self.ptr_current],waypoints.ned[2][self.ptr_current]]]).T
        wip1 = np.array([[waypoints.ned[0][self.ptr_next],waypoints.ned[1][self.ptr_next],waypoints.ned[2][self.ptr_next]]]).T
        qi_1 = (wi-wi_1)/np.linalg.norm(wi-wi_1)
        qi = (wip1-wi)/np.linalg.norm(wip1-wi)
        varrho = np.arccos(float(-qi_1.T@qi))
        if self.manager_state == 1:
            self.path.flag = 'line'
            self.path.line_origin = wi_1
            self.path.line_direction = qi_1
            self.halfspace_r = wi - R/np.tan(varrho/2.)*qi_1
            self.halfspace_n = qi_1
            if self.inHalfSpace(P):
                self.manager_state = 2
        elif self.manager_state == 2:
            self.path.flag = 'orbit'
            self.path.orbit_center = wi - R/np.sin(varrho/2.)*(qi_1-qi)/np.linalg.norm(qi_1-qi)
            # self.path.orbit_center.item(2) = -100
            self.path.orbit_radius = R
            self.path.orbit_direction = np.sign((qi_1.item(0)*qi.item(1))-(qi_1.item(1)*qi.item(0)))
            print("orbit center:", self.path.orbit_center)
            self.halfspace_r = wi + R/np.tan(varrho/2.)*qi
            self.halfspace_n = qi
            if self.inHalfSpace(P):
                self.increment_pointers()
                self.manager_state = 1

        ### Return
        self.path.airspeed = waypoints.airspeed.item(self.ptr_previous)

    def dubins_manager(self, waypoints, radius, state):
        P = np.array([[state.pn,state.pe,-state.h]]).T
        R = radius
        wi_1 = np.array([[waypoints.ned[0][self.ptr_previous],waypoints.ned[1][self.ptr_previous],waypoints.ned[2][self.ptr_previous]]]).T
        chii_1 = waypoints.course[0][self.ptr_previous]
        wi = np.array([[waypoints.ned[0][self.ptr_current],waypoints.ned[1][self.ptr_current],waypoints.ned[2][self.ptr_current]]]).T
        chii = waypoints.course[0][self.ptr_current]
        # wip1 = np.array([[waypoints.ned[0][self.ptr_next],waypoints.ned[1][self.ptr_next],waypoints.ned[2][self.ptr_next]]]).T
        # chip1 = waypoints.course[self.ptr_next]
        self.dubins_path.update(wi_1,chii_1,wi,chii,R)
        if self.manager_state == 1:
            self.path.flag = 'orbit'
            self.path.orbit_center = self.dubins_path.center_s
            self.path.orbit_radius = self.dubins_path.radius
            self.path.orbit_direction = self.dubins_path.dir_s
            self.halfspace_r = self.dubins_path.r1 #z1
            self.halfspace_n = -self.dubins_path.n1 #-q1
            if self.inHalfSpace(P):
                self.manager_state = 2
        elif self.manager_state == 2:
            self.halfspace_r = self.dubins_path.r1 #z1
            self.halfspace_n = self.dubins_path.n1 #q1
            if self.inHalfSpace(P):
                self.manager_state = 3
        elif self.manager_state == 3:
            self.path.flag = 'line'
            self.path.line_origin = self.dubins_path.r1
            self.path.line_direction = self.dubins_path.n1
            self.halfspace_r = self.dubins_path.r2 #z2
            self.halfspace_n = self.dubins_path.n1 #q1
            if self.inHalfSpace(P):
                self.manager_state = 4
        elif self.manager_state == 4:
            self.path.flag = 'orbit'
            self.path.orbit_center = self.dubins_path.center_e
            self.path.orbit_radius = self.dubins_path.radius
            self.path.orbit_direction = self.dubins_path.dir_e
            self.halfspace_r = self.dubins_path.r3 #z3
            self.halfspace_n = -self.dubins_path.n3 #-q3
            if self.inHalfSpace(P):
                self.manager_state = 5
        else: #state 5
            if self.inHalfSpace(P):
                self.manager_state = 1
                self.increment_pointers()
                self.dubins_path.update(wi_1,chii_1,wi,chii,R)

    def initialize_pointers(self):
        print("initialize pointers")

    def increment_pointers(self):
        print("Increment pointers")
        self.ptr_previous+=1
        self.ptr_current+=1
        self.ptr_next+=1

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

