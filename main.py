import os
from copy import copy

import matplotlib.collections as col
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sc
import scipy.io

colormap = 'gist_stern'  # , terrain, 'PuOr'


class FindFiles:
    """Class that implement methods for handling data that other classes can inherit from.
    """

    @staticmethod
    def load_mat(the_file=None):
        """Load the .mat files that holds the geometry of the detector design.

        Keyword Arguments:
            the_file {str} -- string object that specifies where to look for .mat files (default: {None})

        Returns:
            scipy.io.loadmat -- the .mat file containing the design description
        """
        if the_file is None:
            loaded_mat = scipy.io.loadmat(
                'filer/GeomSPID_correct.mat')
        else:
            loaded_mat = scipy.io.loadmat(the_file)
        return loaded_mat

    @staticmethod
    def make_header():
        """Make a header which hold the available parameters from the .dat files which
        holds values obtained from the DSMC model.

        Also includes explanation to each parameter.

        Returns:
            dict -- a dict with the parameter name and explanation
        """
        header = []
        for line in open('filer/flow_SPID_55.DAT', 'r'):
            my_list = line.split()[2].split()[0].split(',')
            for items in range(len(my_list)):
                get_values = list(
                    [val for val in my_list[items] if val.isalnum()])
                header.append("".join(get_values))
            break
        explanations = [
            'x position',
            'y position',
            'Number density',
            'Density',
            'Velocity along x',
            'Velocity along y',
            'Velocity along z',
            'Translational temperature',
            'Rotational temperature',
            'Vibrational temperature',
            'Temperature',
            'Mach number',
            'Molecules per cell',
            'Mean collision time',
            'Mean free path',
            'SOF: (mean collisional separation) / (mean free path)',
            'FSP: (in-plane) speed of the flow',
            'The in-plane flow angle relative to x axis',
            'Scalar pressure (nkT)'
        ]
        header_dict = dict(zip(header, explanations))
        return header, header_dict

    @staticmethod
    def make_polygon(files, version='SPID'):
        """Make a polygon based on the detector design.

        Arguments:
            files {scipy.io.loadmat} -- the returned object from load_mat

        Keyword Arguments:
            version {str} -- describing which polygon to make.
                             Different method for different designs (default: {'SPID'})

        Returns:
            list -- list object containing the complete design as np.arrays
        """
        all_polygons = []
        if version == 'SPID':
            SPID_MP = files['GeomSPID_MP_correct']
            SPID_side = files['GeomSPID_side_correct'].T
            SPID_side[:, [0, 1]] = SPID_side[:, [1, 0]]
            SPID_side[:, 0] *= -1
            start = 0
            all_polygons.append(SPID_side)
            all_polygons.append(np.c_[SPID_side[:, 0], -1 * SPID_side[:, 1]])
            for index, value in enumerate(SPID_MP[0, :]):
                # Check if 'value' is np.nan.
                if value != value:
                    hold = SPID_MP[:, start:index].T
                    hold[:, [0, 1]] = hold[:, [1, 0]]
                    hold[:, 0] *= -1
                    all_polygons.append(hold)
                    all_polygons.append(np.c_[hold[:, 0], -1 * hold[:, 1]])
                    start = index + 1
        elif version == 'hedin':
            hedin = files['hedin']
            all_polygons.append(hedin)
        return all_polygons

    @staticmethod
    def make_patches(all_p, color='w', make_mirror=True):
        """Make polygons as patches that are to be drawn in the scatter plot to represent the instrument.

        Arguments:
            all_p {list} -- a list containing all the polygons

        Keyword Arguments:
            color {str} -- the face color of the patch object (default: {'w'})
            make_mirror {bool} -- if True, a mirror image about x=0 is made (default: {True})

        Returns:
            PatchCollection -- a collection of Polygon objects
        """
        patches = []
        for polygon in all_p:
            if polygon.shape[1] != 2:
                polygon = polygon.T
            patches.append(pat.Polygon(polygon, True))
            if make_mirror:
                patches.append(pat.Polygon(
                    np.c_[polygon[:, 0], -1 * polygon[:, 1]], True))
        p = col.PatchCollection(patches, alpha=1)
        p.set_facecolor(color)

        return p

    # Reference to make_masks $\label{lst:make_masks}$
    @staticmethod
    def make_masks(xi, yi, all_polygons, version='SPID'):
        """Make masks for the interpolation style that represents the instrument,
        an which removes values at these 'mask' points.

        Arguments:
            xi {np.array} -- the x-coordinates returned from the meshgrid method
            yi {np.array} -- the y-coordinates returned from the meshgrid method
            all_polygons {list} -- the list returned from make_polygon

        Keyword Arguments:
            version {str} -- decide what design is used.
                             Different designs have different masks (default: {'SPID'})

        Returns:
            list -- a list containing the different masks
        """
        if version == 'SPID':
            # Mask for the rightmost instrument body.
            mask1 = (xi > 0.05) & (yi < 0.04) & (yi > - 0.04)
            # Top SPID side
            mask2 = (yi < (all_polygons[0][0, 1] - all_polygons[0][1, 1]) /
                    (all_polygons[0][0, 0] - all_polygons[0][1, 0]) *
                    (xi - all_polygons[0][1, 0]) + all_polygons[0][1, 1]) & \
                    (yi > all_polygons[0][1, 1]) & \
                    (xi < all_polygons[0][2, 0]) & \
                    (yi < all_polygons[0][3, 1])
            mask3 = (yi > (all_polygons[1][1, 1] - all_polygons[1][0, 1]) /
                    (all_polygons[1][1, 0] - all_polygons[1][0, 0]) *
                    (xi - all_polygons[1][0, 0]) + all_polygons[1][0, 1]) & \
                    (yi < all_polygons[1][1, 1]) & \
                    (xi < all_polygons[1][2, 0]) & \
                    (yi > all_polygons[1][3, 1])
            mask4 = (yi < (all_polygons[2][1, 1] - all_polygons[2][0, 1]) /
                    (all_polygons[2][1, 0] - all_polygons[2][0, 0]) *
                    (xi - all_polygons[2][0, 0]) + all_polygons[2][0, 1]) & \
                    (yi > all_polygons[2][2, 1]) & \
                    (xi < all_polygons[2][2, 0])
            mask5 = (yi > (all_polygons[3][0, 1] - all_polygons[3][1, 1]) /
                    (all_polygons[3][0, 0] - all_polygons[3][1, 0]) *
                    (xi - all_polygons[3][1, 0]) + all_polygons[3][1, 1]) & \
                    (yi < all_polygons[3][2, 1]) & \
                    (xi < all_polygons[3][2, 0])

            mask_list = [mask1, mask2, mask3, mask4, mask5]
            for i in range(len(all_polygons)):
                if i > 3:
                    if i % 2:
                        mask = (yi > (all_polygons[i][2, 1] - all_polygons[i][1, 1]) /
                                (all_polygons[i][2, 0] - all_polygons[i][1, 0]) *
                                (xi - all_polygons[i][1, 0]) + all_polygons[i][1, 1]) & \
                            (yi < (all_polygons[i][3, 1] - all_polygons[i][0, 1]) /
                             (all_polygons[i][3, 0] - all_polygons[i][0, 0]) *
                             (xi - all_polygons[i][0, 0]) + all_polygons[i][0, 1]) & \
                            (xi < all_polygons[i][3, 0]) & \
                            (xi > all_polygons[i][0, 0])
                    else:
                        mask = (yi < (all_polygons[i][2, 1] - all_polygons[i][1, 1]) /
                                (all_polygons[i][2, 0] - all_polygons[i][1, 0]) *
                                (xi - all_polygons[i][1, 0]) + all_polygons[i][1, 1]) & \
                            (yi > (all_polygons[i][3, 1] - all_polygons[i][0, 1]) /
                             (all_polygons[i][3, 0] - all_polygons[i][0, 0]) *
                             (xi - all_polygons[i][0, 0]) + all_polygons[i][0, 1]) & \
                            (xi < all_polygons[i][3, 0]) & \
                            (xi > all_polygons[i][0, 0])
                    mask_list.append(mask)
        elif version == 'hedin':
            mask1 = (xi > all_polygons[0][0, 0]) & (xi < all_polygons[0][0, 4]) & \
                (yi < all_polygons[0][1, 1]) & (yi > all_polygons[0][1, 0])
            mask2 = (xi > all_polygons[0][0, 2]) & (xi < all_polygons[0][0, 4]) & \
                (yi > all_polygons[0][1, 2]) & (yi < all_polygons[0][1, 4])
            mask3 = (xi > all_polygons[0][0, 11]) & (xi < all_polygons[0][0, 4]) & \
                (yi > all_polygons[0][1, 10]) & (yi < all_polygons[0][1, 11])
            mask4 = (xi > all_polygons[0][0, 8]) & (xi < all_polygons[0][0, 4]) & \
                (yi > all_polygons[0][1, 8]) & (yi < all_polygons[0][1, 9])
            mask5 = (xi > all_polygons[0][0, 6]) & (xi < all_polygons[0][0, 4]) & \
                (yi > all_polygons[0][1, 6]) & (yi < all_polygons[0][1, 7])
            mask_list = [mask1, mask2, mask3, mask4, mask5]

        return mask_list

    @staticmethod
    def it_crashed(xi, yi, all_polygons):
        """Works only for the Hedin design.

        Method for checking if the particle have crashed with the probe or if it is outside of the plotting area.

        Arguments:
            xi {float} -- the x position of the particle
            yi {float} -- the y position of the particle
            all_polygons {list} -- list containing all elements and the corners of the elements as tuples

        Returns:
            bool -- True if the particle has crashed, False if it has not.
            bool -- True if the particle hit the detector, False if it did not.
        """
        # Upper side
        mask1 = (xi > all_polygons[0][0, 0]) & (xi < all_polygons[0][0, 4]) & \
            (yi < all_polygons[0][1, 1]) & (yi > all_polygons[0][1, 0])
        if mask1:
            return True, False
        # Upper back end
        mask2 = (xi > all_polygons[0][0, 2]) & (xi < all_polygons[0][0, 4]) & \
            (yi > all_polygons[0][1, 2]) & (yi < all_polygons[0][1, 4])
        if mask2:
            return True, False
        # Detection plate
        mask3 = (xi > all_polygons[0][0, 11]) & (xi < all_polygons[0][0, 4]) & \
            (yi > all_polygons[0][1, 10]) & (yi < all_polygons[0][1, 11])
        if mask3:
            return True, True
        # Lower side
        mask4 = (xi > all_polygons[0][0, 8]) & (xi < all_polygons[0][0, 4]) & \
            (yi > all_polygons[0][1, 8]) & (yi < all_polygons[0][1, 9])
        if mask4:
            return True, False
        # Lower back end
        mask5 = (xi > all_polygons[0][0, 6]) & (xi < all_polygons[0][0, 4]) & \
            (yi > all_polygons[0][1, 6]) & (yi < all_polygons[0][1, 7])
        if mask5:
            return True, False
        return False, False

    @staticmethod
    def make_files(altitudes, the_file=None):
        """Fetch the .dat files you need to be able to make all plots,
        and put them in a files dictionary based on altitudes.

        Arguments:
            altitudes {list} -- a list containing the altitudes that the user specified

        Keyword Arguments:
            the_file {str} -- specifying where to look for the .dat file (default: {None})
        """
        unique_altitudes = list(set(altitudes))
        file_list = []
        for i, alt in enumerate(unique_altitudes):
            print(f'Looking for files' + '.' * (i + 1), end='\r')
            if the_file is None:
                f = np.genfromtxt('filer/flow_SPID_' + str(alt) + '.DAT',
                                  skip_header=1, comments='Z')
            else:
                f = np.genfromtxt(the_file,
                                  skip_header=1, comments='Z')
            # Make the mirror image to get the full xy-plane. y=0 is removed.
            test = np.c_[f[:-1, 0], -1 * f[:-1, 1], f[:-1, 2:5], -1 * f[:-1, 5], f[:-1, 6:]]
            file_list.append(np.r_[f, test])
        print('')
        return dict(zip(unique_altitudes, file_list))

    def interpolator(self, file, all_polygons, mask, index=None, version='SPID'):
        """Make interpolated data for position and velocity.

        Arguments:
            file {dat file} -- a .dat file from one of the altitudes
            all_polygons {list} -- a list containing numpy arrays of polygons
            mask {list} -- a list containing statements saying which positions should have nan

        Keyword Arguments:
            index {None or int} -- give the index of the parameter you want (default: {None})
            version {str} -- specify what design is used (default: {'SPID'})

        Returns:
            numpy arrays and list -- the 2D numpy arrays for x, y, u and v, in addition to a list of masks
        """
        if index is None:
            xi = np.arange(min(file[:, 0]), max(file[:, 0]), 0.0001)
            yi = np.arange(min(file[:, 1]), max(file[:, 1]), 0.0001)
            xi, yi = np.meshgrid(xi, yi)
            # Set mask, i.e. the instrument.
            ui = sc.interpolate.griddata(
                (file[:, 0], file[:, 1]), file[:, 4], (xi, yi), method='cubic')
            vi = sc.interpolate.griddata(
                (file[:, 0], file[:, 1]), file[:, 5], (xi, yi), method='cubic')
            if mask is None:
                mask = self.make_masks(xi, yi, all_polygons, version=version)
            for the_mask in mask:
                ui[the_mask] = np.nan
                vi[the_mask] = np.nan

            return xi, yi, ui, vi, mask

        xi = np.arange(min(file[:, 0]), max(file[:, 0]), 0.0001)
        yi = np.arange(min(file[:, 1]), max(file[:, 1]), 0.0001)
        xi, yi = np.meshgrid(xi, yi)
        # Set mask, i.e. the instrument.
        zi = sc.interpolate.griddata(
            (file[:, 0], file[:, 1]), file[:, index], (xi, yi), method='cubic')
        if mask is None:
            mask = self.make_masks(xi, yi, all_polygons, version=version)
        for the_mask in mask:
            zi[the_mask] = 178  # np.nan

        return xi, yi, zi, mask


class Plotter(FindFiles):
    """Make a plotter object with methods for plotting in different styles.
    """

    def __init__(self):
        """Initialize the plotter object.
        """
        self.files = self.load_mat(the_file='filer/hedin.mat')
        self.all_polygons = self.make_polygon(self.files, version='hedin')

        self.header, self.header_dict = self.make_header()
        self.mask = None
        self.file_dict = None

    def makes_files(self, altitudes):
        """Fetch the files you need to be able to make all plots, and put them in a files dictionary.

        Arguments:
            altitudes {list} -- a list containing the altitudes that the user specified
        """
        self.file_dict = self.make_files(
            altitudes, the_file='filer/flow_Hedin_paper.DAT')

    def scatter_plots(self, n_plots, altitudes, indices, save=False):
        """Make one or more scatter plots from the specified data.

        Arguments:
            n_plots {int} -- the amount of plots the user wanted
            altitudes {list} -- a list of all altitudes specified by the user
            indices {list} -- a list of the indices of the parameters according to how they are stored in 'header'

        Keyword Arguments:
            save {bool} -- if True, the plot will be saved to the current directory (default: {False})
        """
        if self.file_dict is None:
            print('No files found.')
            exit()
        p = self.make_patches(self.all_polygons, 'k')
        pp = []
        for i in range(n_plots):
            pp.append(copy(p))
            _, ax = plt.subplots(figsize=(7, 5))
            print("Making scatter plots" + "." * (i + 1), end='\r')
            file = self.file_dict[altitudes[i]]
            if self.header[indices[i]] == 'ND':
                plt.scatter(file[:, 0], file[:, 1], c=file[:, indices[i]] / 1.2e20, s=0.2, cmap=colormap)
                plt.xlabel('$x$ position [m]')
                plt.ylabel('$y$ position [m]')
                plt.colorbar().ax.set_ylabel('$n/n_0$ normalized number density')
            else:
                plt.scatter(file[:, 0], file[:, 1], c=file[:, indices[i]], s=0.2, cmap=colormap)
                plt.colorbar()
            plt.title(
                f'Scatter plot view.\n{self.header_dict[self.header[indices[i]]]} at {altitudes[i]} km altitude')
            ax.add_collection(pp[i])
            plt.tight_layout()
            if save:
                plt.savefig(f'number_density_hedin_SMALL.pdf',
                            bbox_inches='tight', format='pdf', dpi=600)

    def interpolated_plots(self, n_plots, altitudes, indices, save=False):
        """Make one or more plots with interpolated data from the specified data.

        Arguments:
            n_plots {int} -- the amount of plots the user wanted
            altitudes {list} -- a list of all altitudes specified by the user
            indices {list} -- a list of the indices of the parameters according to how they are stored in 'header'

        Keyword Arguments:
            save {bool} -- if True, the plot will be saved to the current directory (default: {False})
        """
        if self.file_dict is None:
            print('No files found.')
            exit()
        for i in range(n_plots):
            plt.figure(figsize=(8.5, 5))
            print("Making interpolated plots" + "." * (i + 1), end='\r')
            file = self.file_dict[altitudes[i]]
            xi, yi, zi, self.mask = self.interpolator(file, self.all_polygons, self.mask, indices[i], version='hedin')
            if self.header[indices[i]] == 'ND':
                # plt.contourf(xi, yi, zi / 1.2e20, 200, cmap=colormap)
                plt.imshow(zi / 1.2e20, cmap=colormap, origin='lower', extent=[
                           xi.min(), xi.max(), yi.min(), yi.max()], aspect=.8)
                plt.xlabel('$x$ position [m]')
                plt.ylabel('$y$ position [m]')
                plt.colorbar().ax.set_ylabel('Normalized number density $n/n_0$')
            elif self.header[indices[i]] == 'TOV':
                plt.imshow(zi, cmap=colormap, origin='lower', extent=[
                           xi.min(), xi.max(), yi.min(), yi.max()], aspect=.8)
                plt.xlabel('$x$ position [m]')
                plt.ylabel('$y$ position [m]')
                plt.colorbar().ax.set_ylabel('Temperature $T$ [K]')
            else:
                plt.contour(xi, yi, zi, 200, cmap=colormap)
                plt.colorbar()
                plt.title(
                    f'Interpolated view.\n{self.header_dict[self.header[indices[i]]]} at {altitudes[i]} km altitude')
            plt.xlim(min(file[:, 0]) + .06, max(file[:, 0]))
            plt.ylim(min(file[:, 1]), max(file[:, 1]))
            plt.tight_layout()
            if save:
                plt.savefig(f'number_density_hedin.pdf',
                            bbox_inches='tight', format='pdf', dpi=600)

    def stream_lines(self, altitudes, save=False):
        """Make up to three plots showing the stream lines for the three different heights.

        Arguments:
            altitudes {list} -- a list of all altitudes specified by the user

        Keyword Arguments:
            save {bool} -- if True, the plot will be saved to the current directory (default: {False})
        """
        if self.file_dict is None:
            print('No files found.')
            exit()
        for i, altitude in enumerate(altitudes):
            plt.figure()
            print("Making stream line plots" + "." * (i + 1), end='\r')
            file = self.file_dict[altitude]
            xi, yi, ui, vi, self.mask = self.interpolator(file, self.all_polygons, self.mask)
            plt.streamplot(xi, yi, ui, vi)
            plt.contourf(xi, yi, ui, 100)
            plt.title(
                f'Stream line view.\n At {altitude} km altitude')
            plt.colorbar()
            plt.tight_layout()
            if save:
                plt.savefig('test_with_griddata_stream_line_60_V.eps')


class Chooser(FindFiles):
    """Make an object for finding correct files and to choose what to plot.
    """

    def __init__(self):
        """Initialize the object with all needed attributes.
        """
        self.header, self.header_dict = self.make_header()
        self.show = Plotter()
        self.which_plot, self.n_plots, self.altitudes, self.indices = self.data_set()
        self.show.makes_files(self.altitudes)

    def reset(self):
        """Possible to run again after doing a run of the code.

        Returns:
            bool -- True if the code should run again, False if not
        """
        try:
            again = str(
                input(f'Press yes(y) if you want to go again with more plotting.\t').upper())
        except Exception:
            pass
        if again in ['YES', 'Y']:
            self.which_plot, self.n_plots, self.altitudes, self.indices = self.data_set()
            self.show.makes_files(self.altitudes)
            return True
        return False

    # pylint: disable=too-many-branches
    def data_set(self):
        """Method for choosing how many plots, and what to plot in each plot.
        """
        # Choose what type of plot you want to make.
        while True:
            try:
                welcome = 'Choose scatter plot, interpolated plot, a comparison of the two or stream line plot.'
                the_choice = '("s=scatter", "i=inter", "b=both", "sl=streamline")'
                which_plot = str(input(f'{welcome} {the_choice}\t'))
            except Exception:
                pass
            else:
                if which_plot in ['scatter', 'inter', 'both', 'streamline', 's', 'i', 'b', 'sl']:
                    break
        # Give the amount of plotted variables.
        while True:
            try:
                if which_plot in ['sl', 'streamline']:
                    n_plots = int(
                        input(f'How many plots do you want to look at? (Max. 3)\t'))
                else:
                    n_plots = int(
                        input(f'How many plots do you want to look at? (Max. 9)\t'))
            except Exception:
                pass
            else:
                if which_plot in ['sl', 'streamline']:
                    if any([n_plots == item + 1 for item in range(3)]):
                        break
                else:
                    if any([n_plots == item + 1 for item in range(9)]):
                        break
        # Define what altitudes you want.
        while True:
            try:
                st = 'Give the altitudes you want to look at; 55, 60 or 85 km.'
                altitudes = list(map(int, input(
                    f'{st} [{n_plots} altitude(s).]\t').split(', ')))
            except Exception:
                print('It must be separated with a comma and whitespace, e.g. "55, 55".')
            else:
                if len(altitudes) == n_plots and set(altitudes).issubset(set([55, 60, 85])):
                    break
        if which_plot == 'streamline' or which_plot == 'sl':
            return which_plot, n_plots, altitudes, None
        # Define what parameters you want.
        while True:
            print(self.header)
            parameters = list(
                input(f'Give the parameters you want to look at. [{n_plots} parameter(s).]\t').upper().split(', '))
            if len(parameters) == n_plots and set(parameters).issubset(set(self.header)):
                indices = [self.header.index(item) for item in parameters]
                break
            else:
                print('It must be separated with a comma and whitespace, e.g. "ANG, V".')
        return which_plot, n_plots, altitudes, indices

    def plotter(self):
        """Method for making a plotter object that is able to plot the data that has been decided.
        """
        # Flush the terminal for a better overview.
        os.system('cls' if os.name == 'nt' else 'clear')
        try:
            # Have no indices if we are in stream line mode.
            parameters = [self.header[index] for index in self.indices]
        except Exception:
            pass

        # Adding save=True as a last argument to the self.show...(..., save=True) will save figures. $\label{lst:save_figures}$
        if self.which_plot == 'b' or self.which_plot == 'both':
            showing = f'Showing {self.n_plots} scatter and interpolated plot(s):'
            summary = f'Height(s):\t {self.altitudes} km\n Parameter(s):\t {parameters}'
            print(f'{showing}\n {summary}')
            self.show.interpolated_plots(
                self.n_plots, self.altitudes, self.indices)
            print('')
            self.show.scatter_plots(self.n_plots, self.altitudes, self.indices)
        elif self.which_plot == 'inter' or self.which_plot == 'i':
            st1 = f'Showing {self.n_plots} interpolated plot(s):\n'
            st2 = f'Height(s):\t {self.altitudes} km\n Parameter(s):\t {parameters}'
            print(f'{st1} {st2}')
            self.show.interpolated_plots(
                self.n_plots, self.altitudes, self.indices)
        elif self.which_plot == 'streamline' or self.which_plot == 'sl':
            st1 = f'Showing {self.n_plots} stream line plot(s):\n'
            st2 = f'Height(s):\t {self.altitudes} km'
            print(f'{st1} {st2}')
            self.show.stream_lines(self.altitudes)
        else:
            st1 = f'Showing {self.n_plots} scatter plot(s):\n'
            st2 = f'Height(s):\t {self.altitudes} km\n Parameter(s):\t {parameters}'
            print(f'{st1} {st2}')
            self.show.scatter_plots(self.n_plots, self.altitudes, self.indices)
        plt.show()


if __name__ == '__main__':
    task = Chooser()
    more = True
    while more:
        task.plotter()
        more = task.reset()
    os.system('cls' if os.name == 'nt' else 'clear')
