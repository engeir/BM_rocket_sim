"""Constant values used throughout the hedin.py script.
"""
from datetime import datetime
import numpy as np
import scipy.constants as scc

# Physical constants
K_B = scc.k  # Boltzmanns constant
AMU = scc.physical_constants["atomic mass constant"][0]  # Atomic mass in kg

# Dust particle
DUST_RADIUS = 3.0e-9  # m
DUST_DENSITY = 3000  # 3 g/cm^3 = 3000 kg/m^3
VOLUME = 4 * np.pi * DUST_RADIUS**3 / 3
DUST_MASS = DUST_DENSITY * VOLUME
U_VELOCITY = 1000  # m/s

# Ambient gas
GAS_RADIUS = .185e-9  # m
GAS_MASS = 29 * AMU  # 29 amu

# Name of log files as dates
LOG_PATH = 'log_files/'
SAVE_PATH = 'figures/'
date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
DATE = date + '_r' + str(DUST_RADIUS).replace('.', '_') + '_'

# === SIMULATION PARAMETERS ===
DT = 1e-8
TIMEOUTERROR = 18.0  # How long particles are allowed to live until termination
Y_DOMAIN = 0.04  # How wide spread the dust particles are
N = 21  # Number of test particles
