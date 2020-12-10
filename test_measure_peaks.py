import numpy as np
import matplotlib.pyplot as plt
from measure_peaks import *


# Analysis Parameters
#
standards = ["Asp", "Ser", "Glu", "Gly", "His", "NH3", "Arg", "Thr", "Ala", "Pro", "Cys", "Tyr", "Val", "Met", "Lys", "Ile", "Leu", "Phe"]
standards_exclude = ["NH3"]  # standards to be excluded in the final computations
standards_directory = '20201210/standards'
standards_filename = []  # [] means all files available in standards_directory
samples_directory = '20201210/samples'
samples_filename = []  # [] means all files available in samples_directory
# time window (in sec.):
timestep_min = 880.0 # 920.0  # or 980 ?
timestep_max = 2000.0
# signal smoothing:
smoothing_window = 7 # 5
smoothing_polyorder = 1
# baseline correction:
baseline_order = 5 # 10  # or 5 ?
baseline_threshold = 1.0e-2
# automatic peak detection:
peak_lookahead = 3
peak_delta = 1.0e-3
peak_min = 0.002  # minimum inetnsity below which the signal is considered as potential peak
# compute_peak_beginning_end:
be_order = 5
# match_peaks_standard_sample_lap_greedy:
match_threshold = 40.0 # 20.0  # time distance between standard and sample peaks in order to be considered as potential match
# General plotting options
plot = False  # set True if you want ALL figures step by step
savefig = True  # save the main graph as PDF
merge = True  # set False to avoid merging two sets of measurements at different scales
if merge:
    filename_smaller = '20201210/samples/sample_small_scale.txt'
    filename_larger = '20201210/samples/sample_large_scale.txt'
    aminoacids_to_merge = ['Ala', 'Gly']


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



samples_filename, peaks_areas, matchings, timestamps, \
    signals, molar_fractions = crunch_samples(standards,
                                              standards_exclude,
                                              timestamps_standard=timestamps_standard,
                                              signal_standard=signal_standard,
                                              peaks_idx_standard=peaks_idx_standard,
                                              peaks_area_standards=peaks_area_standards,
                                              samples_filename=samples_filename,
                                              samples_directory=samples_directory,
                                              match_threshold=match_threshold,
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


if merge:
    molar_fraction = merge_two_results_and_compute_molar_fraction(filename_smaller,
                                                                  filename_larger,
                                                                  samples_filename,
                                                                  peaks_areas,
                                                                  matchings, standards,
                                                                  standards_exclude,
                                                                  peaks_area_standards,
                                                                  aminoacids_to_merge)
