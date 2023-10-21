"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np

# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation, Euler2Quaternion


class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                                [MAV.east0],  # (1)
                                [MAV.down0],  # (2)
                                [MAV.u0],  # (3)
                                [MAV.v0],  # (4)
                                [MAV.w0],  # (5)
                                [MAV.e0],  # (6)
                                [MAV.e1],  # (7)
                                [MAV.e2],  # (8)
                                [MAV.e3],  # (9)
                                [MAV.p0],  # (10)
                                [MAV.q0],  # (11)
                                [MAV.r0],  # (12)
                                [0],  # (13)
                                [0],  # (14)
                                ])
        # initialize true_state message
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state[0:13], forces_moments)
        k2 = self._derivatives(self._state[0:13] + time_step / 2. * k1, forces_moments)
        k3 = self._derivatives(self._state[0:13] + time_step / 2. * k2, forces_moments)
        k4 = self._derivatives(self._state[0:13] + time_step * k3, forces_moments)
        self._state[0:13] += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6][0] = self._state.item(6) / normE
        self._state[7][0] = self._state.item(7) / normE
        self._state[8][0] = self._state.item(8) / normE
        self._state[9][0] = self._state.item(9) / normE

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        ##### TODO: Update position derivatives with quaternion formula #####

        # Extract the States
        ned = np.array([state.item(0), state.item(1), state.item(2)])
        uvw = np.array([state.item(3), state.item(4), state.item(5)])

        phi, theta, psi = Quaternion2Euler(state[6:10])
        quaternionAngles = np.array([state.item(6), state.item(7), state.item(8), state.item(9)])

        pqr = np.array([state.item(10), state.item(11), state.item(12)])

        # Extract Forces/Moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        mx = forces_moments.item(3)
        my = forces_moments.item(4)
        mz = forces_moments.item(5)

        forces = np.array([fx, fy, fz])
        moments = np.array([mx, my, mz])


        # Position Kinematics
        rotationMatrix = np.array(
            [
                [np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                 np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],

                [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                 np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],

                [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]])

        position_derivative = np.matmul(rotationMatrix, uvw)
        n_dot = position_derivative[0]
        e_dot = position_derivative[1]
        d_dot = position_derivative[2]

        # Position Dynamics
        uvw_dot = np.array(
            [pqr[2] * uvw[1] - pqr[1] * uvw[2],
             pqr[0] * uvw[2] - pqr[2] * uvw[0],
             pqr[1] * uvw[0] - pqr[0] * uvw[1]]) + ((1 / MAV.mass) * forces)
        u_dot = uvw_dot[0]
        v_dot = uvw_dot[1]
        w_dot = uvw_dot[2]

        # rotational kinematics
        quaternionRotMat = 0.5 * np.array([
            [0, -pqr[0], -pqr[1], -pqr[2]],
            [pqr[0], 0, pqr[2], -pqr[1]],
            [pqr[1], -pqr[2], 0, pqr[0]],
            [pqr[2], pqr[1], -pqr[0], 0]
        ])

        quat_dot = np.matmul(quaternionRotMat, quaternionAngles)
        e0_dot = quat_dot.item(0)
        e1_dot = quat_dot.item(1)
        e2_dot = quat_dot.item(2)
        e3_dot = quat_dot.item(3)

        # rotatonal dynamics
        gammaMatA = np.array([MAV.gamma1 * pqr[0] * pqr[1] - MAV.gamma2 * pqr[1] * pqr[2],
                              MAV.gamma5 * pqr[0] * pqr[2] - MAV.gamma6 * (pqr[0] ** 2 - pqr[2] ** 2),
                              MAV.gamma7 * pqr[0] * pqr[2] - MAV.gamma1 * pqr[1] * pqr[2]])
        gammaMatB = np.array([MAV.gamma3 * moments[0] + MAV.gamma4 * moments[2],
                              moments[1] / MAV.Jy,
                              MAV.gamma4 * moments[0] + MAV.gamma8 * moments[2]])
        pqr_dot = gammaMatA + gammaMatB
        p_dot = pqr_dot[0]
        q_dot = pqr_dot[1]
        r_dot = pqr_dot[2]

        # collect the derivative of the states
        # x_dot = np.array([[north_dot, east_dot,... ]]).T
        x_dot = np.array([[n_dot, e_dot, d_dot, u_dot, v_dot, w_dot, e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
