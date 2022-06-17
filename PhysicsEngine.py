from numba import jit
import numpy as np

@jit(nopython=True)
def pqr_to_ang_vel(p, q, r, theta, phi):

    phi_dot = (np.cos(phi) / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*p\
                + 0\
                + (np.sin(theta) / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*r          # phi_dot

    theta_dot = (np.sin(phi)*np.sin(theta) / (np.cos(phi)*np.cos(theta)**2 + np.cos(theta)*np.sin(theta)**2))*p\
                + 1*q\
                - (np.sin(phi) / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*r            # theta_dot

    psi_dot = (-np.sin(theta) / (np.cos(phi)*np.cos(theta)**2 + np.cos(theta)*np.sin(theta)**2))*p\
                + 0\
                + (1 / (np.sin(theta)**2 + np.cos(phi)*np.cos(theta)))*r                      # psi_dot

    return phi_dot, theta_dot, psi_dot


@jit(nopython=True)
def get_sderivative(F, M1, M2, M3, S, m, I_xx, I_yy, I_zz, g):

    phi     = S[3]
    theta   = S[4]
    psi     = S[5]
    p       = S[9]
    q       = S[10]
    r       = S[11]
    s_dot   = np.zeros(12)
    
    s_dot[:3] = S[6:9]                                                                    # x_dot, y_dot, z_dot

    phi_dot, theta_dot, psi_dot = pqr_to_ang_vel(p, q, r, theta, phi)
    s_dot[3] = phi_dot

    s_dot[4] = theta_dot

    s_dot[5] = psi_dot

    s_dot[6] = (np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi))*F/m          # x_dot_dot

    s_dot[7] = (np.sin(psi)*np.sin(theta) - np.cos(theta)*np.sin(phi)*np.cos(psi))*F/m          # y_dot_dot

    s_dot[8] = -g + (np.cos(phi)*np.cos(theta))*F/m                                         # z_dot_dot

    s_dot[9] = (1/I_xx) * (M1 - (-I_yy*q*r + I_zz*r*q))                                       # p_dot

    s_dot[10] = (1/I_yy) * (M2 - (I_xx*p*r - I_zz*r*p))                                       # q_dot

    s_dot[11] = (1/I_zz) * (M3 - (-I_xx*p*q + I_yy*q*p))                                      # r_dot

    return s_dot



@jit(nopython = True)
def update(F, M1, M2, M3, curr, t, delta_t, m, I_xx, I_yy, I_zz, g):

    t += delta_t

    k1 = delta_t*get_sderivative(F, M1, M2, M3, curr, m, I_xx, I_yy, I_zz, g)

    k2 = delta_t*get_sderivative(F, M1, M2, M3, curr + k1/4, m, I_xx, I_yy, I_zz, g)

    k3 = delta_t*get_sderivative(F, M1, M2, M3, curr + (3/32)*k1 + (9/32)*k2, m, I_xx, I_yy, I_zz, g)

    k4 = delta_t*get_sderivative(F, M1, M2, M3, curr + (1932/2197)*k1 + (-7200/2197)*k2 + (7296/2197)*k3, m, I_xx, I_yy, I_zz, g)

    k5 = delta_t*get_sderivative(F, M1, M2, M3, curr + (439/216)*k1 + (-8)*k2 + (3680/513)*k3 + (-845/4104)*k4, m, I_xx, I_yy, I_zz, g)

    # curr += (16/135)*k1 + 0*k2 + (6656/12825)*k3 + (28561/56430)*k4 + (-9/50)*k5
    curr += (25/216)*k1 + 0*k2 + (1408/2565)*k3 + (2197/4104)*k4 + (-1/5)*k5

    return curr, t





'''
                (1)            (2)                      x
                     \       /                          |   
                        (O)                             |           
                     /       \                 y________|
                (3)            (4)
'''
'''
curr = [x, y, z, phi, theta, psi,
        x_dot, y_dot, z_dot, p, q, r]
'''

class EnvPhysicsEngine:

    def __init__(self):
        self.t = 0.0            # s
        self.m = 1.236 + 0.25          # drone + payload
        self.I_xx = 0.0113      # kg m^2
        self.I_yy = 0.0133      # kg m^2
        self.I_zz = 0.0187      # kg m^2
        self.g = 9.81           # m s^-2
        self.l = 0.16           # m
        self.f_MF = 0.24        # if battery voltage > 11.5 
        self.curr = np.zeros(12)
        self.delta_t = 0.01     # time step in s

        # useful notations
        self.J = [self.I_xx, 0, 0,
                    0, self.I_yy, 0,
                    0, 0, self.I_zz]
        self.J = np.reshape(self.J, (3,3))

        # Attitude Controller gains (Geometric)
        '''phi'''
        kr_phi = 5              #5
        kw_phi = 0.5            #0.5

        '''theta'''
        kr_theta = 5            #5
        kw_theta = 0.5          #0.5

        '''psi'''
        kr_psi = 5.0            #5
        kw_psi = 0.6            #0.6

        kr = np.array([kr_phi,  0,   0,
                            0,  kr_theta,   0,
                            0,  0,  kr_psi ])
        self.kr = np.reshape(kr, (3,3))

        kw = np.array([kw_phi,  0,   0,
                            0,  kw_theta,   0,
                            0,  0,  kw_psi ])
        self.kw = np.reshape(kw, (3,3))

        # Position Controller Gains (PD)

        self.k_x = 10           #10.0
        self.k_y = 10           #10.0

        self.k_xd = 5           #5.0
        self.k_yd = 5           #5.0

        # Altitude Controller gains (PD)
        self.kpz = 25.0                #25.0
        self.kdz = 11.0                 #11.0
      

    def get_currentState(self):
        return self.curr


    def get_time(self):
        return self.t


    def get_derivative(self, F, M1, M2, M3):

        
        s_dot = get_sderivative(F, M1, M2, M3, self.curr, self.m,
                                self.I_xx, self.I_yy, self.I_zz, self.g)

        return s_dot


    def stepSimulation(self, x_d, y_d, z_d, phi_d, theta_d, psi_d):

        F =  self.get_Force(z_d)
        M1, M2, M3 = self.get_Moment(phi_d, theta_d, psi_d, x_d, y_d)


       
        # print('forces:-------',F, M1, M2, M3)
        self.curr, self.t = update(F, M1, M2, M3, self.curr, self.t, self.delta_t,
                                   self.m, self.I_xx, self.I_yy, self.I_zz, self.g)
        pass


    def get_Force(self, z_d):

        z = self.curr[2]
        z_dot = self.curr[8]

        # Altitude Controller (PD)
        z_d_dot = 0.0

        F = (self.m)*self.g + self.kpz*(z_d - z) + self.kdz*(z_d_dot - z_dot)
        # F = 0.0
        return F


    def get_Moment(self, phi_d, theta_d, psi_d, x_d, y_d):

        x, y = self.curr[:2]
        x_dot, y_dot = self.curr[6:8]
        phi, theta, psi = self.curr[3:6]
        p, q, r = self.curr[9:]
        y_d_dot = 0
        x_d_dot = 0
    

        # Position Controller (PD)
        phi_h = -1/self.g * (self.k_yd*(y_d_dot - y_dot) + self.k_y*(y_d - y))
        theta_h = 1/self.g * (self.k_xd*(x_d_dot - x_dot) + self.k_x*(x_d - x))
        

        # Attitude Controller (Geometric)
        
        phi_d = phi_d + phi_h
        theta_d = theta_d + theta_h

        '''constraining drone's rpy to +-45 degree'''
        if abs(np.rad2deg(phi_d)) > 45.0:
            if phi_d > 0.0:
                phi_d = np.deg2rad(45)
            else:
                phi_d = np.deg2rad(-45)

        if abs(np.rad2deg(theta_d)) > 45.0:
            if theta_d > 0.0:
                theta_d = np.deg2rad(45)
            else:
                theta_d = np.deg2rad(-45)


        R = np.reshape(self.rotationalmat(phi, theta, psi), (3,3))
        Rd = np.reshape(self.rotationalmat(phi_d, theta_d, psi_d), (3,3))
        er = np.dot(np.transpose(Rd), R) - np.dot(np.transpose(R), Rd)
        er = 0.5*self.veemap(er)
        omega = self.pqr_to_ang_vel(p, q, r, theta, phi)
        e_omega = omega

        M = -np.dot(self.kr, er) - np.dot(self.kw,e_omega) + np.cross(omega, np.dot(self.J, omega))
        

        return M


    def veemap(self, R):

        R = np.reshape(R, (9,))
        c = -R[1]
        b = R[2]
        a = -R[5]

        return np.array([a, b, c])


    def rotationalmat(self, phi, theta, psi):

        # arguments are in radian

        R = [np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta), - np.cos(phi) * np.sin(psi) , np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
            np.sin(psi)*np.cos(theta) + np.cos(psi) * np.sin(phi) * np.sin(theta), np.cos(phi)* np.cos(psi), np.sin(psi) * np.sin(theta) - np.cos(theta) * np.sin(phi) * np.cos(psi), 
            -np.sin(theta) * np.cos(phi) , np.sin(phi) , np.cos(phi) * np.cos(theta) ] 
 
        return R


    def reset(self, x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot):

        self.t = 0.0
        self.curr[0] = x 
        self.curr[1] = y 
        self.curr[2] = z 
        self.curr[3] = phi 
        self.curr[4] = theta
        self.curr[5] = psi
        self.curr[6] = x_dot
        self.curr[7] = y_dot
        self.curr[8] = z_dot
        p,q,r = self.ang_vel_to_pqr(phi_dot, theta_dot, psi_dot, phi, theta)
        self.curr[9] = p
        self.curr[10] = q
        self.curr[11] = r

    
    def ang_vel_to_pqr(self, phi_dot, theta_dot, psi_dot, phi, theta):
        
        p = np.cos(theta)*phi_dot - np.sin(theta)*np.cos(theta)*psi_dot
        q = theta_dot + np.sin(phi)*psi_dot
        r = np.sin(theta)*phi_dot + np.cos(theta)*np.cos(phi)*psi_dot
        
        return p, q, r


    def pqr_to_ang_vel(self, p, q, r, theta, phi):


        phi_dot = np.cos(theta)*p + np.sin(theta)*r
        theta_dot  = np.tan(phi)*np.sin(theta) * p + q  - np.cos(theta) * np.tan(phi) * r 
        psi_dot = -(np.sin(theta)/ np.cos(phi)) * p  + (np.cos(theta) /np.cos(phi) ) * r

        return phi_dot, theta_dot, psi_dot

    
