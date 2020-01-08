"""Script that makes plot of the historical data obtained through all runs.
"""
import os
import glob
import ast
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import config as cf
from main import FindFiles


class ProbeDesign(FindFiles):
    """Find the design of the Hedin detector design.

    Arguments:
        FindFiles {class} -- class object holding methods for obtaining data
    """
    def __init__(self):
        self.mat_file = self.load_mat(the_file='filer/hedin.mat')
        self.header, self.header_dict = self.make_header()
        self.file_dict = self.make_files([55], the_file='filer/flow_Hedin_paper.DAT')
        all_polygons = self.make_polygon(self.mat_file, version='hedin')
        self.patch = self.make_patches(all_polygons, 'k', False)
        files = self.file_dict[55]
        x, y = files[:, 0], files[:, 1]
        xi = np.arange(min(x), max(x), 0.0001)
        yi = np.arange(min(y), max(y), 0.0001)
        self.x_lim, self.y_lim = (min(x), max(x)), (min(y), max(y))
        self.xi, self.yi = np.meshgrid(xi, yi)

    def draw(self):
        """Draw the detector probe of the Hedin detector.
        """
        _, ax = plt.subplots(figsize=(7, 5))
        ax.add_collection(self.patch)
        plt.xlim(self.x_lim[0] + .06, self.x_lim[1])
        plt.ylim(self.y_lim[0], self.y_lim[1])


class Finder:
    """Find log files and make plots from all or a selection of them.
    """
    def __init__(self):
        """Set the path to the log files folder, and check if it exist.

        Also gives default values for saving, which files to use and which version.
        """
        self.path = 'log_files'
        if not os.path.isdir(self.path):
            print(f'No folder in directory {self.path}')
            exit()
        self.save = 'n'
        self.the_files = ''
        self.version = 'traj_phase'

    def choose_version(self):
        """Choose which version to use.

        All plots, plot of trajectory and phase or just plot of velocity distribution.
        """
        string = 'Choose to plot all types (1), only trajectory and phase space (2) or only velocity distribution (3).'
        while True:
            try:
                version = int(input(f'{string}\t'))
            except Exception:
                print('It must be one of the numbers 1, 2, or 3.')
            else:
                if any([x == version for x in range(1, 4)]):
                    break
        if version == 1:
            self.version = 'all'
        elif version == 3:
            self.version = 'v_distribution'

    def saver(self):
        os.makedirs(cf.SAVE_PATH, exist_ok=True)
        self.save = input('Do you want to save the figure? (y or n)\t').lower()

    def choose_file(self, latest=False):
        """Choose which log files to use for plotting.

        The complete file name as a string is not needed, just enough so the
        given string can be found somwhere in one of the files' name.

        Keyword Arguments:
            latest {bool} -- if True, the latest simulation log files are used (default: {False})
        """
        if latest:
            for f in sorted(glob.glob(os.path.join(self.path, '*.txt')), reverse=True):
                self.the_files = [f.replace(self.path + '/', '')[:-8]]
                break
        else:
            # All available files are printed.
            for f in sorted(glob.glob(os.path.join(self.path, '*.txt'))):
                print(f.replace(self.path + '/', ''))
            txt = 'Print all files (a) or a selection by listing unique parts of the file names.\t'
            # Input where the files are given as a list.
            self.the_files = list(input(txt).lower().split(', '))

    def print_info(self):
        if self.version == 'all':
            print(f'Making plots of trajectory, phase space and velocity distribution.')
        elif self.version == 'traj_phase':
            print(f'Making plots of trajectory and phase space.')
        elif self.version == 'v_distribution':
            print(f'Making plots of velocity distribution.')
        if self.save == 'y':
            print(f'Saves figures to {cf.SAVE_PATH} as {self.the_files[0]}...')
        if self.the_files[0] == 'a':
            print('Using all files.')
        else:
            print('Using...')

    def convert_files(self):
        """Given the specifications on what files to use,
        files are fetched and converted to lists using ast package.

        Returns:
            list of None -- returns the appropriate lists according to the specified version
        """
        # N counts the total amount of particles that are used
        # C counts the total amount of particles that hit the detector
        N = 0
        C = 0
        log_pos = []
        log_vel = []
        vel_dist = []
        mean_mfp = []
        # Find files, sort them and loop through.
        for file in sorted(glob.glob(os.path.join(self.path, '*.txt'))):
            # Check if the file should be used.
            if any(x in file for x in self.the_files) or self.the_files[0] == 'a':
                if self.the_files[0] != 'a':
                    # If all files are not used, the chosen file is printed
                    # to confirm to the user that correct files are used.
                    print(file.replace(self.path + '/', ''))
                if file.find('pos') != -1 and self.version != 'v_distribution':
                    with open(file, 'r') as log_position:
                        the_log_pos = log_position.readlines()
                    the_log_pos = [x.strip() for x in the_log_pos]
                    for p in the_log_pos:
                        p = ast.literal_eval(p)
                        log_pos.append(p)
                if file.find('vel') != -1:
                    with open(file, 'r') as log_velocity:
                        the_log_vel = log_velocity.readlines()
                    the_log_vel = [x.strip() for x in the_log_vel]
                    for p in the_log_vel:
                        p = ast.literal_eval(p)
                        # Check if the particle was detected, crashed or stopped in the ambient gas.
                        # If None, the particle crashed, if float, this is the mean free path and
                        # the particle was stopped in the gas. Finally, if str, this is 'detected'.
                        # and hence the particle was detected.
                        if p[2] is None or isinstance(p[2], (float, str)):
                            if isinstance(p[2], float):
                                vel_dist.append(p[0][-1])
                                mean_mfp.append(p[2])
                            elif isinstance(p[2], str):
                                C += 1
                            if self.version != 'v_distribution':
                                log_vel.append([p[2], p[0], p[1]])
                        else:
                            if isinstance(p[0], float):
                                vel_dist.append(p[1][-1])
                                mean_mfp.append(p[0])
                            elif isinstance(p[0], str):
                                C += 1
                            if self.version != 'v_distribution':
                                log_vel.append(p)
                        N += 1
        if N == 0:
            print('Found no such files. Exiting...')
            exit()

        txt = 'Percentage of particles that hit the detector:'
        print(f'{txt} {round(C * 100/N, 2)}')
        if self.version == 'all':
            return log_pos, log_vel, vel_dist, mean_mfp
        elif self.version == 'traj_phase':
            return log_pos, log_vel, None, mean_mfp
        return None, None, vel_dist, mean_mfp

    @staticmethod
    def print_mfp(mean_mfp):
        # Round the mean value to an appropriate value
        try:
            mean_mfp = np.mean(mean_mfp)
            mean_mfp = round(mean_mfp, -(int(np.floor(np.log10(abs(mean_mfp)))) - 3))
            print(f'Average mfp = {mean_mfp} m')
        except Exception:
            mean_mfp = 'They all crashed.'
            print(mean_mfp)

    def draw_traj(self, log_pos):
        """Draw the trajectory of the particles found in the files.

        Arguments:
            log_pos {list} -- all positions listed in two lists (x,y) in the log_pos list
        """
        if log_pos is not None:
            for p in log_pos:
                plt.plot(p[0], p[1], 'purple', mec='w', linewidth=0.7)
            plt.plot(- np.ones(10) * .07, np.linspace(-.04, .04, 10), 'k--', linewidth=0.4)
            plt.plot(- np.ones(10) * .1, np.linspace(-.04, .04, 10), 'k--', linewidth=0.4)
            for i, y in zip(range(cf.N), np.linspace(cf.Y_DOMAIN, - cf.Y_DOMAIN, cf.N)):
                plt.annotate(f'{i + 1}', (-.18, y), ha='right', va='center', size='x-small')
            plt.xlabel('$x$ position [m]')
            plt.ylabel('$y$ position [m]')
            if self.save == 'y':
                plt.savefig(f'{cf.SAVE_PATH}{self.the_files[0]}_trajs_V_g.pdf',
                            bbox_inches='tight', format='pdf', dpi=600)

    def draw_phase_space(self, log_pos, log_vel, xlim):
        """Make a phase space plot of the particles found in the files.

        Arguments:
            log_pos {list} -- list containing two lists for x,y position
            log_vel {list} -- list containing two lists for x,y velocities
            xlim {tuple} -- tuple with the start and end of the simulated area in x direction
        """
        if log_pos is not None and log_vel is not None:
            plt.figure(figsize=(7, 5))
            plt.grid(which='both', alpha=.3)
            l1, l2, l3 = False, False, False
            color, style = [], []
            for i, p in enumerate(log_vel):
                if isinstance(p[0], float):
                    # Particles that was stopped in the gas
                    if l1 is False:
                        plt.plot(log_pos[i][0], np.array(p[1]) / 1000, 'r', alpha=0.3, label='Particles that did not crash', linewidth=1)
                        l1 = True
                        color.append('r')
                        style.append('-')
                    else:
                        plt.plot(log_pos[i][0], np.array(p[1]) / 1000, 'r', alpha=0.3, linewidth=1)
                    plt.plot(log_pos[i][0][-1], p[1][-1] / 1000, 'ro', markersize=3.0)
                elif p[0] is None:
                    # Particles that crashed with the walls
                    if l2 is False:
                        plt.plot(log_pos[i][0], np.array(p[1]) / 1000, 'k--', alpha=0.3, label='Particles that crashed', linewidth=1)
                        l2 = True
                        color.append('k')
                        style.append('--')
                    else:
                        plt.plot(log_pos[i][0], np.array(p[1]) / 1000, 'k--', alpha=0.3, linewidth=1)
                    plt.plot(log_pos[i][0][-1], p[1][-1] / 1000, 'ko', markersize=3.0)
                elif isinstance(p[0], str):
                    # Particles that was detected
                    if l3 is False:
                        plt.plot(log_pos[i][0], np.array(p[1]) / 1000, 'b:', alpha=0.3, label='Particles that was detected', linewidth=1)
                        l3 = True
                        color.append('b')
                        style.append(':')
                    else:
                        plt.plot(log_pos[i][0], np.array(p[1]) / 1000, 'b:', alpha=0.3, linewidth=1)
                    plt.plot(log_pos[i][0][-1], p[1][-1] / 1000, 'bo', markersize=3.0)
            plt.xlabel('$x$ position [m]')
            plt.ylabel('$u/u_0$ relative velocity (along $x$)')
            plt.xlim([xlim[0] + .06, xlim[1]])
            # New legend is made to prevent semitransparent lines being
            # presented in the legend. They may be harder to see.
            leg = plt.legend(loc='best')
            for i, l in enumerate(leg.get_lines()):
                l.set_alpha(1)
                l.set_linestyle(style[i])
                l.set_color(color[i])
            if self.save == 'y':
                plt.savefig(f'{cf.SAVE_PATH}{self.the_files[0]}_phase_space_V_g.pdf',
                            bbox_inches='tight', format='pdf', dpi=600)

    @staticmethod
    def draw_vel_distribution(vel_dist):
        """Plot velocity distribution of stopped particles (velocity at last point).

        Arguments:
            vel_dist {list} -- list containing the last saved velocity of stopped particles
        """
        if vel_dist is not None:
            plt.figure()
            plt.title('Velocity distribution: last velocity')
            plt.hist(vel_dist, 50)
            plt.xlabel('Velocity distribution [m/s]')
            plt.ylabel('Instances')


if __name__ == '__main__':
    find = Finder()
    c = ProbeDesign()
    find.choose_version()
    find.saver()
    find.choose_file(latest=False)
    os.system('cls' if os.name == 'nt' else 'clear')
    find.print_info()
    pos, vel, vel_d, m_mfp = find.convert_files()
    find.print_mfp(m_mfp)
    c.draw()
    find.draw_traj(pos)
    find.draw_phase_space(pos, vel, c.x_lim)
    find.draw_vel_distribution(vel_d)
    plt.show()
