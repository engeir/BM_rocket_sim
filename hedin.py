import os
import multiprocessing as mp
import ast
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from main import FindFiles
import config as cf


class BrownianAmbient(FindFiles):
    """Class containing information about the ambient gas.

    Make N equally spaced particles and trace their trajectory through the detector.

    Arguments:
        FindFiles {class} -- class object holding methods for obtaining data
    """

    def __init__(self, altitudes):
        """Initialize the class.

        Files are obtained from .mat files (detector design) and .dat files (DSMC parameters).

        Arguments:
            altitudes {int} -- the altitude that should be used
        """
        # Find the detector design.
        mat_file = self.load_mat(the_file='filer/hedin.mat')
        # Make a polygon of the design.
        self.all_polygons = self.make_polygon(mat_file, version='hedin')
        file_dict = self.make_files(
            [altitudes], the_file='filer/flow_Hedin_paper.DAT')
        # Find the correct .dat file.
        file = file_dict[altitudes]
        # Get the important parameters from the .dat file.
        x, y, u, v, w = file[:, 0], file[:, 1], file[:, 4], file[:, 5], file[:, 6]
        T, N = file[:, 10], file[:, 2]
        self.x_lim, self.y_lim = (min(x), max(x)), (min(y), max(y))
        # A 2D interpolator used in the cubic version scipy's griddata interpolator.
        # This creates a 2D function that can be called for each of the parameters.
        self.u = sc.interpolate.CloughTocher2DInterpolator((x, y), u)
        self.v = sc.interpolate.CloughTocher2DInterpolator((x, y), v)
        self.w = sc.interpolate.CloughTocher2DInterpolator((x, y), w)
        self.T = sc.interpolate.CloughTocher2DInterpolator((x, y), T)
        self.N = sc.interpolate.CloughTocher2DInterpolator((x, y), N)
        # Create a log folder and log files to keep the data that is made.
        os.makedirs(cf.LOG_PATH, exist_ok=True)
        with open(cf.LOG_PATH + cf.DATE + 'pos.txt', 'w') as log:
            log.write('')
        with open(cf.LOG_PATH + cf.DATE + 'vel.txt', 'w') as log:
            log.write('')
        self.pool_method()

    @staticmethod
    def log_results(results):
        with open(cf.LOG_PATH + cf.DATE + 'pos.txt', 'a') as log:
            log.write(f'{results[0]}\n')
        with open(cf.LOG_PATH + cf.DATE + 'vel.txt', 'a') as log:
            log.write(f'{results[1]}\n')

    # Reference to pool_method $\label{lst:line_pool_method}$
    def pool_method(self):
        """Simulate particles in parallel, using the multiprocessing.Pool() class.

        The returned values from collide() which is called, are put in log_results().
        This way, the returned value from collide() is the input of log_result().
        """
        pool = mp.Pool()
        for y in np.linspace(-cf.Y_DOMAIN, cf.Y_DOMAIN, cf.N):
            pool.apply_async(self.collide, args=(y,), callback=self.log_results)
        pool.close()
        pool.join()

    def collide(self, y):
        """Create a particle as a BrownianParticle() and simulate the trajectory.

        Arguments:
            y {float} -- the y position where the partile starts

        Returns:
            list -- two lists containing lists with position, velocity and mean free path
        """
        p = BrownianParticle(self.x_lim[0] + 0.07, y)
        # Check the time when the particle is created, and trace it for no more than 't' seconds.
        t0 = time.clock()
        t = cf.TIMEOUTERROR
        mfp = None  # Mean free path
        while 1:
            if time.clock() - t0 > t:
                # If the particle is still being simulated when the time is up,
                # the mean free path at the position it was stopped is saved
                # and the loop which keeps movement() running is terminated.
                print(f'RunTime warning: particle stopped after {t} seconds.')
                mfp = p.mfp
                break
            p.movement(self.gas_velocity, self.it_crashed,
                       self.all_polygons, self.x_lim, self.y_lim)
            if not p.alive:
                if p.detected:
                    mfp = 'detected'
                break
        return ([p.trace_x, p.trace_y], [mfp, p.trace_u, p.trace_v])

    def gas_velocity(self, x, y):
        """Gas velocity at a point x,y is returned, along with the standard
        deviation set equal to the thermal velocity and the number density.

        Arguments:
            x {float} -- x position
            y {float} -- y position

        Returns:
            tuple, float, float -- tuple containing velocity in three dimensions, standard deviation and number density
        """
        u = self.u(x, y)
        v = self.v(x, y)
        w = self.w(x, y)

        temp = self.T(x, y)
        N = self.N(x, y)
        # Standard deviation: $\tn{sd}=\sqrt{\frac{k_BT}{m_g}}$
        sd = np.sqrt(cf.K_B * temp / cf.GAS_MASS)
        return (u, v, w), sd, N


class BrownianParticle:
    """Make a particle that move through a gas due to
    binary collisions between spherical collision partners.
    """
    def __init__(self, x, y):
        """Initialize particle with position, velocity,
        collision time step, mean free path value and logging lists.

        Arguments:
            x {float} -- x position
            y {float} -- y position
        """
        self.alive = True
        self.detected = False
        self.x = x
        self.y = y
        self.z = 0
        self.u = cf.U_VELOCITY  # m/s
        self.v = 0  # m/s
        self.w = 0  # m/s
        self.dt = 1e-7
        self.mfp = None
        self.trace_x = []
        self.trace_y = []
        self.trace_u = []
        self.trace_v = []

    # Reference to project_v_gas. $\label{lst:scaling}$
    @staticmethod
    def project_v_gas(vel, sd, scaling=True):
        """Scale gas particle velocity based on point of contact between collision partners.

        Arguments:
            vel {tuple} -- 3D tuple with velocities
            sd {float} -- standard deviation/thermal velocity

        Keyword Arguments:
            scaling {bool} -- if scaling should be used (True) or not (False) (default: {True})

        Returns:
            floats -- three floats for the new 3D velocity
        """
        theta = np.random.uniform(0, np.pi / 2)
        phi = 2 * np.pi * np.random.uniform(0, 1)
        if scaling:
            u = np.random.normal(0, sd) * np.cos(theta)
            v = np.random.normal(0, sd) * np.sin(theta) * np.cos(phi)
            w = np.random.normal(0, sd) * np.sin(theta) * np.sin(phi)
        else:
            u = np.random.normal(0, sd)
            v = np.random.normal(0, sd)
            w = np.random.normal(0, sd)
        return u + vel[0], v + vel[1], w + vel[2]

    @staticmethod
    def project_v_rel(rel_v):
        """Scale relative gas particle velocity according to the direction of the dust particle.

        Arguments:
            rel_v {float} -- the magnitude of the relative velocity between gas particle and dust particle

        Returns:
            floats -- three floats for the 3D relative velocity
        """
        theta = np.arccos(np.random.uniform(-1, 1))  # pylint: disable=assignment-from-no-return
        phi = 2 * np.pi * np.random.uniform(0, 1)
        u_rel = rel_v * np.cos(theta)
        v_rel = rel_v * np.sin(theta) * np.cos(phi)
        w_rel = rel_v * np.sin(theta) * np.sin(phi)
        return u_rel, v_rel, w_rel

    def mean_velocity(self, u, v, w):
        """Calculate the velocity of the colission partners center of mass.

        Arguments:
            u {float} -- x velocity
            v {float} -- y velocity
            w {float} -- z velocity

        Returns:
            floats -- three floats for the 3D center of mass velocity
        """
        u_m = (cf.GAS_MASS * u + cf.DUST_MASS * self.u) / (cf.GAS_MASS + cf.DUST_MASS)
        v_m = (cf.GAS_MASS * v + cf.DUST_MASS * self.v) / (cf.GAS_MASS + cf.DUST_MASS)
        w_m = (cf.GAS_MASS * w + cf.DUST_MASS * self.w) / (cf.GAS_MASS + cf.DUST_MASS)
        return u_m, v_m, w_m

    # Reference to do_n_collisions $\label{lst:line_do_n_collisions}$
    def do_n_collisions(self, velocity, sd, it_crashed, all_polygons, x_lim, y_lim):
        """Do an arbitrary number of collisions given that the time
        between collisions is smaller than the simulation time step.

        Arguments:
            velocity {tuple} -- 3D velocity
            sd {float} -- standard deviation/thermal velocity
            it_crashed {method} -- method that check if the particle have crashed with anything on the domain
            all_polygons {list} -- containing all shapes in the detector design
            x_lim {tuple} -- the start and end of the x domain
            y_lim {tuple} -- the start and end of the y domain
        """
        # Radius of how far the particle had moved with the simulation time step.
        # During the simulations between simulation steps, the particle should
        # not move farther than that before the collision time is recalculated.
        R = np.sqrt((self.u * cf.DT)**2 + (self.v * cf.DT)**2 + (self.w * cf.DT)**2)
        r = 0
        for _ in range(int(cf.DT / self.dt)):
            # Solves $\cref{eq:vector_incident_gas}$ (True) or $\cref{eq:vector_big_v_g}$ (False).
            vel_u, vel_v, vel_w = self.project_v_gas(velocity, sd, scaling=True)
            u_rel = vel_u - self.u
            v_rel = vel_v - self.v
            w_rel = vel_w - self.w
            relative_velocity = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
            u_rel, v_rel, w_rel = self.project_v_rel(relative_velocity)
            u_m, v_m, w_m = self.mean_velocity(vel_u, vel_v, vel_w)
            self.u = u_m - cf.GAS_MASS / (cf.GAS_MASS + cf.DUST_MASS) * u_rel
            self.v = v_m - cf.GAS_MASS / (cf.GAS_MASS + cf.DUST_MASS) * v_rel
            self.w = w_m - cf.GAS_MASS / (cf.GAS_MASS + cf.DUST_MASS) * w_rel
            self.x += self.u * self.dt
            self.y += self.v * self.dt
            self.z += self.w * self.dt
            # Add the new movement to the allowed radius.
            r += np.sqrt((self.u * self.dt)**2 + (self.v * self.dt)**2 + (self.w * self.dt)**2)
            # Check if the particle is outside the plotting domain,
            bound_x = x_lim[1] < self.x or self.x < x_lim[0]
            bound_y = y_lim[1] < self.y or self.y < y_lim[0]
            # and if the particle has crashed with the detector itself.
            crash, detect = it_crashed(self.x, self.y, all_polygons)
            if crash or bound_x or bound_y:
                if detect:
                    self.detected = True
                self.alive = False
                break
            # As an extra check that the particle do not move too far away
            # from the the point where the collision time was calculated.
            if r > R:
                break

    def movement(self, function, it_crashed, all_polygons, x_lim, y_lim):
        """Move the particle in a direction based on the velocity.

        The state of the ambient gas is checked and a collision time is found.
        If this time is greater than the simulation time, DT, then no collisions
        should occur and the velocity remains unchanged. If the time is smaller
        than DT, do_n_collisions() is called.

        Arguments:
            function {method} -- method that return the ambient gas parameters at the point self.x, self.y
            it_crashed {method} -- to check if the particle have crashed
            all_polygons {list} -- containing the detector design
            x_lim {tuple} -- start and end of the x domain
            y_lim {tuple} -- start and end of the y domain
        """
        # Get parameters from gas_velocity() in BrownianAmbient().
        velocity, sd, N = function(self.x, self.y)
        v_d = np.sqrt((velocity[0] - self.u)**2 + (velocity[1] - self.v)**2 + (velocity[2] - self.w)**2)
        # Mean thermal velocity: $ \gf{v}_{gth}=\sqrt{\frac{8k_B T}{\pi m}} $
        v_gth = sd * np.sqrt(8 / np.pi)
        a = 2 * np.abs(v_d) / (np.sqrt(np.pi) * v_gth)
        integral = (np.sqrt(np.pi) * (2 * a**2 + 1) * (sc.special.erf(a) + 1) +
                    2 * np.exp(- a**2) * a - 2 * np.sqrt(np.pi) * sc.special.erfc(a)) / 4
        v_collision = v_gth / (2 * a) * integral
        tau, self.mfp = self.collision_time(v_collision, N)
        self.dt = np.abs(tau)
        if self.dt <= cf.DT:
            self.do_n_collisions(velocity, sd, it_crashed,
                                 all_polygons, x_lim, y_lim)
        else:
            self.x += self.u * cf.DT
            self.y += self.v * cf.DT
            self.z += self.w * cf.DT

        bound_x = x_lim[1] < self.x or self.x < x_lim[0]
        bound_y = y_lim[1] < self.y or self.y < y_lim[0]
        if self.alive:
            crash, detect = it_crashed(self.x, self.y, all_polygons)
            if crash or bound_x or bound_y:
                if detect:
                    self.detected = True
                self.alive = False

        self.trace_x.append(self.x)
        self.trace_y.append(self.y)
        self.trace_u.append(self.u)
        self.trace_v.append(self.v)

    @staticmethod
    def collision_time(v_c, N):
        """Calculate the collision time a an arbitrary point.

        Arguments:
            v_c {float} -- collision velocity used to find the collision time
            N {float} -- number density

        Returns:
            floats -- floats for collision time and mean free path
        """
        # Find a random number $ r=\ln(r_1), r_1\in(0,1] $
        random_float = np.log2(1 - np.random.ranf())
        mfp = (np.pi * (cf.DUST_RADIUS + cf.GAS_RADIUS)**2 * N)**(-1)
        t_mean = mfp / v_c
        return - random_float * t_mean, mfp


if __name__ == '__main__':
    ensemble = BrownianAmbient(85)
