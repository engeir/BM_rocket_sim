"""Script that creates a .mat file where the points describing the
design of the probe used in Hedin et al. (2007) is saved.
"""

import numpy as np
import scipy.io

a = np.array([[-0.1001, -0.1001, -0.002, -0.002, 0, 0, -0.002, -0.002, -0.1001, -0.1001, -0.04, -0.04],
              [0.04, 0.043, 0.043, 0.08, 0.08, -0.08, -0.08, -0.043, -0.043, -0.04, -0.04, 0.04]], dtype=np.object)
scipy.io.savemat('../filer_Henriette/hedin.mat', mdict={'hedin': a})
