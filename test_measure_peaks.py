import numpy as np
import matplotlib.pyplot as plt
from measure_peaks import *


# Analysis Parameters
#
standards = ["Asp", "Ser", "Glu", "Gly", "His", "NH3", "Arg", "Thr", "Ala", "Pro", "Cys", "Tyr", "Val", "Met", "Lys", "Ile", "Leu", "Phe"]
standards_exclude = ["NH3"]  # standards to be excluded in the final computations
standards_directory = 'standards'
standards_filename = []  # [] means all files available in standards_directory
samples_directory = 'samples'
samples_filename = []  # [] means all files available in samples_directory
# time window (in sec.):
timestep_min = 920.0  # or 980 ?
timestep_max = 2000.0
# signal smoothing:
smoothing_window = 5
smoothing_polyorder = 1
# baseline correction:
baseline_order = 10  # or 5 ?
baseline_threshold=1.0e-2
# automatic peak detection:
peak_delta = 1.0e-3
peak_min = 0.002
# compute_peak_beginning_end:
be_order = 5
# match_peaks_standard_sample_lap_greedy:
match_threshold = 20.0
# General plotting options
plot = False
savefig = False  # save the main graph as PDF


peaks_area_standards, timestamps_standard, signal_standard, \
    peaks_idx_standard = crunch_standards(standards,
                                          standards_filename=standards_filename,
                                          standards_directory=standards_directory,
                                          plot=plot)



peaks_area, matching, timestamps, \
    signal, molar_fraction = crunch_samples(standards,
                                            standards_exclude,
                                            timestamps_standard=timestamps_standard,
                                            signal_standard=signal_standard,
                                            peaks_idx_standard=peaks_idx_standard,
                                            peaks_area_standards=peaks_area_standards,
                                            samples_filename=samples_filename,
                                            samples_directory=samples_directory,
                                            plot=plot,
                                            savefig=savefig)


# Expected:
# data/34-24s.txt
# | Protein | F (stand) | Area  |   Conc   | % Molar |
# |--------------------------------------------------|
# | Asp     |  580.731  | 0.104 |   60.603 |    2.54 |
# | Ser     |  207.491  | 1.009 |  209.428 |    8.79 |
# | Glu     |  600.555  | 0.022 |   12.996 |    0.55 |
# | Gly     |  214.749  | 5.407 | 1161.041 |   48.75 |
# | His     |  187.126  | 0.000 |    0.000 |    0.00 |
# | Arg     |  256.790  | 0.121 |   30.955 |    1.30 |
# | Thr     |  170.654  | 0.284 |   48.471 |    2.04 |
# | Ala     |  205.298  | 2.645 |  542.936 |   22.80 |
# | Pro     |  194.895  | 0.075 |   14.572 |    0.61 |
# | Cys     |  220.858  | 0.000 |    0.000 |    0.00 |
# | Tyr     |  190.023  | 0.681 |  129.419 |    5.43 |
# | Val     |  211.090  | 0.389 |   82.121 |    3.45 |
# | Met     |  197.067  | 0.000 |    0.000 |    0.00 |
# | Lys     |  130.333  | 0.169 |   21.986 |    0.92 |
# | Ile     |  220.659  | 0.130 |   28.690 |    1.20 |
# | Leu     |  204.115  | 0.097 |   19.808 |    0.83 |
# | Phe     |  211.256  | 0.088 |   18.626 |    0.78 |
# |--------------------------------------------------|
