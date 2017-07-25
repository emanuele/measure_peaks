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
peak_lookahead = 3
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
                                          timestep_min=timestep_min,
                                          timestep_max=timestep_max,
                                          smoothing_window=smoothing_window,
                                          smoothing_polyorder=smoothing_polyorder,
                                          baseline_order=baseline_order,
                                          baseline_threshold=baseline_threshold,
                                          peak_lookahead=peak_lookahead,
                                          peak_delta=peak_delta,
                                          peak_min=peak_min,
                                          be_order=be_order,
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
                                            timestep_min=timestep_min,
                                            timestep_max=timestep_max,
                                            smoothing_window=smoothing_window,
                                            smoothing_polyorder=smoothing_polyorder,
                                            baseline_order=baseline_order,
                                            baseline_threshold=baseline_threshold,
                                            peak_lookahead=peak_lookahead,
                                            peak_delta=peak_delta,
                                            peak_min=peak_min,
                                            be_order=be_order,
                                            plot=plot,
                                            savefig=savefig)
