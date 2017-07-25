import numpy as np
import matplotlib.pyplot as plt
# from pandas import read_csv
import time
from scipy.signal import savgol_filter, detrend, find_peaks_cwt, argrelextrema
from scipy.interpolate import interp1d
from scipy.integrate import trapz, simps
from peakdetect import peakdetect
# from DSPFP import DSPFP
# from linear_assignment_numpy import LinearAssignment
from glob import glob
from os.path import join

plt.interactive(True)


def load(filename, plot=False):
    """Load and parse CSV-like high-performance liquid chromatography (HLPC)
    measurements file.
    """
    print("Loading and parsing %s" % filename)
    timestamp_format = "%d/%m/%Y %H:%M:%S"
    timestamps = []
    signal = []
    with open(filename) as f:
        assert(f.readline().strip() == 'Date, HPLC flow chart')
        for line in f:
            t, s = line.split(',')
            timestamps.append(time.mktime(time.strptime(t, timestamp_format)))
            signal.append(float(s))

    timestamps = np.array(timestamps)
    signal = np.array(signal)
    if plot:
        plt.figure()
        plt.plot(timestamps, signal)
        plt.xlabel('Time (sec.)')
        plt.title('%s : raw data' % filename)
        # plt.figure()
        # plt.plot(timestamps, label='timestamps')
        # plt.title('%s : sequence of timestamps' % filename)
        # plt.legend()
        # plt.xlabel('Time (sec.)')

    print("Read %s timestamps and %s measures" % (timestamps.size,
                                                  signal.size))
    return timestamps, signal


def timestamps_signal_fix(timestamps, signal, plot=True):
    """Fix timestamps/signal, because the timestamp resolution of the
    instrument is 'seconds' while there are multiple measurements per
    second, leading to multiple entries with the same timestamp.
    """
    print("Fixing and clipping time information.")
    timestamps = timestamps - timestamps[0]
    # Issue: there are multiple measurements at the same timestamp.
    # Solution 1: resample timestamps using linspace. This is WRONG in
    # some cases
    # timestamps = timestamps - timestamps[0]
    # timestamps_fixed = np.linspace(timestamps[0], timestamps[-1],
    #                                len(timestamps), endpoint=True)
    # Solution 2: average all measurements with the same timestamp:
    timestamps_fixed = np.unique(timestamps)
    signal_fixed = np.zeros(len(timestamps_fixed))
    for i, t in enumerate(np.unique(timestamps)):
        signal_fixed[i] = signal[np.where(timestamps == t)[0]].mean()

    if plot:
        plt.figure()
        plt.plot(timestamps_fixed, signal_fixed)
        plt.xlabel('Time (sec.)')
        plt.title('Fixed data')

    return timestamps_fixed, signal_fixed


def timestamps_signal_clip(timestamps, signal, timestep_min=980.0,
                           timestep_max=2000.0, plot=True):
    """Clip the interval of the signal to a given time window.
    """
    time_window_idx = np.logical_and(timestamps >= timestep_min,
                                     timestamps <= timestep_max)
    timestamps_clipped = timestamps[time_window_idx]
    signal_clipped = signal[time_window_idx]
    if plot:
        plt.figure()
        plt.plot(timestamps_clipped, signal_clipped)
        plt.xlabel('Time (sec.)')
        plt.title('Clipped data')

    return timestamps_clipped, signal_clipped


def signal_smoothing(timestamps, signal, window_length=5, polyorder=1,
                     plot=True):
    """Smooth signal to remove high-frequency noise.
    """
    print("Smoothing")
    signal_smoothed = savgol_filter(signal,
                                    window_length=window_length,
                                    polyorder=polyorder)
    if plot:
        plt.figure()
        plt.plot(timestamps, signal_smoothed)
        plt.xlabel('Time (sec.)')
        plt.title('Smoothed data')

    return signal_smoothed


def baseline_correction(timestamps, signal, order=5,
                        baseline_threshold=1.0e-2, plot=True):
    """Detrend the signal with a polyline going through some of the lowest
    points in the signal.
    """
    print("Baseline correction.")
    break_points = argrelextrema(signal, np.less, order=order)[0]
    # Remove break_points too far from their linear fit:
    linear_fit = np.poly1d(np.polyfit(timestamps[break_points],
                                      signal[break_points], 1))
    losses = signal[break_points] - linear_fit(timestamps[break_points])
    # print(losses)
    break_points = break_points[np.abs(losses) < baseline_threshold]
    # Add the first and last measured points as break-points for detrending:
    break_points = np.concatenate([[0], break_points, [len(signal) - 1]])
    if plot:
        plt.figure()
        plt.plot(timestamps, signal)
        plt.plot(timestamps[break_points], signal[break_points], 'g*')
        plt.xlabel('Time (sec.)')
        # linear_fit before removing utliers in breakpoints
        # plt.plot(timestamps[break_points],
        #          linear_fit(timestamps[break_points]), 'm*')
        plt.title('Break points for detrending')

    # Detrending:
    # linear detrend:
    # signal_detrended = detrend(signal, type='linear', bp=break_points)
    # polyline detrend:
    trend = interp1d(timestamps[break_points], signal[break_points])
    signal_detrended = signal - trend(timestamps)
    if plot:
        plt.figure()
        plt.plot(timestamps, signal_detrended)
        plt.xlabel('Time (sec.)')
        plt.title("Detrended data")

    return signal_detrended


def load_and_prepare(filename, timestep_min=920, timestep_max=2000,
                     window_length=5, polyorder=1, order=10, baseline_threshold=1.0e-2,
                     plot=False):
    """Convenience function to load and preprocess HLPC data.
    """
    timestamps, signal = load(filename, plot=plot)
    timestamps_fixed, signal_fixed = timestamps_signal_fix(timestamps,
                                                           signal, plot=plot)
    timestamps_clipped, signal_clipped = timestamps_signal_clip(timestamps_fixed,
                                                                signal_fixed, timestep_min=timestep_min,
                                                                timestep_max=timestep_max, plot=plot)
    signal_smoothed = signal_smoothing(timestamps_clipped,
                                       signal_clipped,
                                       window_length=window_length,
                                       polyorder=polyorder, plot=plot)
    signal_detrended = baseline_correction(timestamps_clipped,
                                           signal_smoothed,
                                           order=order,
                                           baseline_threshold=baseline_threshold,
                                           plot=plot)
    return timestamps_clipped, signal_detrended, signal


def automatic_peak_detection(timestamps, signal, lookahead=3,
                             delta=1.0e-3, peak_min=0.002,
                             expected_peaks=None, plot=True,
                             offset=0.0):
    """Authomatic peak detection and post-detection filtering.
    """
    print("Automatic peak detection.")
    # there are a number of options
    # See: https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
    # or https://github.com/MonsieurV/py-findpeaks
    # This does not work well:
    # peaks_idx = np.array(find_peaks_cwt(signal, widths=peak_widths),
    #                      dtype=np.int)
    # This one works very well:
    peaks_idx = np.array(peakdetect(signal, lookahead=lookahead,
                                    delta=delta)[0])[:, 0].astype(np.int)
    print("Peaks detected: %s" % len(peaks_idx))
    if plot:
        if offset == 0.0:
            plt.figure()
            plt.xlabel('Time (sec.)')

        plt.plot(timestamps, signal + offset)
        plt.plot(timestamps[peaks_idx], signal[peaks_idx] + offset,
                 'w*', label='rejected peaks')

    print("Peak filtering")
    peaks_idx_filtered = peaks_idx[signal[peaks_idx] >= peak_min]
    print("Peaks retained after filtering: %s" % len(peaks_idx_filtered))
    for i, pif in enumerate(peaks_idx_filtered):
        print("Peak %s, time=%0.2f min" % (i+1, timestamps[pif]/60.0))

    if expected_peaks is not None:
        assert(len(peaks_idx_filtered) == expected_peaks)

    if plot:
        plt.plot(timestamps[peaks_idx_filtered],
                 signal[peaks_idx_filtered] + offset, 'r*',
                 label='confirmed peaks')
        plt.title("Detected peaks")
        plt.legend()

    return peaks_idx_filtered


def plot_peaks_area(timestamps, signal, peaks_idx, beginnings, ends,
                    offset=0.0):
    """Plot signal, peaks and their areas.
    """
    plt.figure()
    plt.plot(timestamps, signal + offset)
    plt.plot(timestamps[peaks_idx], signal[peaks_idx] + offset, 'r*')
    plt.xlabel('Time (sec.)')
    for i in range(len(beginnings)):
        plt.fill_between(timestamps[beginnings[i]:ends[i]+1],
                         signal[beginnings[i]:ends[i]+1] + offset, 0)
        plt.title("Peaks areas")

    return


def compute_peak_beginning_end(timestamps, signal, peaks_idx, order=5,
                               plot=True, offset=0.0):
    """Compute beginning and end of a peak.
    """
    print("Detecting beginnings and ends of peaks")
    beginnings = np.zeros(len(peaks_idx), dtype=np.int)
    ends = np.zeros(len(peaks_idx), dtype=np.int)
    # add 0 and last idx
    tmp = np.concatenate([[0], peaks_idx, [len(signal) - 1]])
    for i in range(len(tmp) - 2):
        try:  # sometimes argrelextrema does not find any result...
            beginnings[i] = tmp[i] + argrelextrema(signal[tmp[i]:tmp[i+1]],
                                                   np.less, order=order)[0][-1]  # better!
        except IndexError:
            try:
                beginnings[i] = tmp[i] + np.argmin(signal[tmp[i]:tmp[i+1]])  # basic
            except ValueError:
                beginnings[i] = tmp[i]  # if everything else fail...

        try:  # sometimes argrelextrema does not find any result...
            ends[i] = tmp[i+1] + argrelextrema(signal[tmp[i+1]:tmp[i+2]], np.less, order=order)[0][0]  # better!
        except IndexError:
            try:
                ends[i] = tmp[i+1] + np.argmin(signal[tmp[i+1]:tmp[i+2]])  # basic
            except ValueError:
                ends[i] = tmp[i+1]  # if everything else fail...

    if plot:
        plot_peaks_area(timestamps, signal, peaks_idx, beginnings,
                        ends, offset=0.0)

    return beginnings, ends


def compute_peaks_area(timestamps, signal, beginnings, ends):
    """Compute area of peaks. The area is in the units of timestamps X
    signal.
    """
    print("Measuring area of peaks.")
    peaks_area = np.zeros(len(beginnings))
    for i in range(len(beginnings)):
        # Several ways to compute a peak's area:
        # dx = (timestamps[-1] - timestamps[0]) / len(timestamps)
        # peaks_area[i] = signal[beginnings[i]:ends[i]].sum() * dx  # basic
        # peaks_area[i] = trapz(signal[beginnings[i]:ends[i]], dx=dx)  # better
        peaks_area[i] = trapz(signal[beginnings[i]:ends[i]+1],
                              timestamps[beginnings[i]:ends[i]+1])  # even better

    for i in range(len(beginnings)):
        print("Peak %s, area: %0.2f \t percentage: %0.2f" % (i+1, peaks_area[i], peaks_area[i] / peaks_area.sum() * 100.0))

    return peaks_area


def glob_filenames(directory,
                   filenames=None):
    """mix filenames with directory and retrieve (glob) filenames if not
    available.
    """
    mix = False
    if (filenames is None) or (filenames == []):
        filenames = sorted(glob(join(directory, '*.txt')))
        mix = True
    elif type(filenames) == str:
        filenames = [filenames]
    else:
        raise Exception

    if len(filenames) == 0:
        print("NO STANDARDS IN DIRECTORY %s" % directory)
        raise Exception
    else:
        if not mix:
            filenames = [join(directory, fn) for fn in filenames]

    return filenames


def crunch_standards(standards,
                     standards_filename=None,
                     standards_directory='standards', plot=True):
    """Load, prepare, find peaks, measure area and compute means of standards.
    """
    print("Crunching standards...")
    standards_filename = glob_filenames(directory=standards_directory,
                                        filenames=standards_filename)
    print("Standards' files: %s" % standards_filename)
    peaks_area_standards = np.zeros((len(standards_filename),
                                     len(standards)), dtype=np.float)
    for i, filename_standard in enumerate(standards_filename):
        timestamps_standard, signal_standard, \
            raw_standard = load_and_prepare(filename_standard,
                                            plot=plot)
        peaks_idx_standard = automatic_peak_detection(timestamps_standard,
                                                      signal_standard,
                                                      expected_peaks=len(standards),
                                                      plot=plot)
        beginnings_standard, \
            ends_standard = compute_peak_beginning_end(timestamps_standard,
                                                       signal_standard,
                                                       peaks_idx_standard,
                                                       plot=plot)
        peaks_area_standard = compute_peaks_area(timestamps_standard,
                                                 signal_standard,
                                                 beginnings_standard,
                                                 ends_standard)
        peaks_area_standards[i] = peaks_area_standard

        print("")

    F = 100.0 / peaks_area_standards
    print("Standards: F ([pmol/ul]/Area)")
    print("| Protein |  F (mean) |  F (std) |")
    print("|--------------------------------|")
    for i, st in enumerate(standards):
        print("| %s     | %8.3f  | %7.3f  |" % (st, F.mean(0)[i], F.std(0)[i]))

    print("|--------------------------------|")
    print("")

    return peaks_area_standards, timestamps_standard, \
        signal_standard, peaks_idx_standard


def plot_matching_peaks(timestamps, signal, peaks_idx,
                        timestamps_standard, signal_standard,
                        peaks_idx_standard, matching, standards):
    """Plot sample (above) and standard (below) signals with
    markers on peaks and straight lines connecting matching peaks.
    """
    plt.figure()
    plt.plot(timestamps_standard, signal_standard, label='standard')
    plt.plot(timestamps_standard[peaks_idx_standard],
             signal_standard[peaks_idx_standard], 'r*')
    for i in range(len(peaks_idx_standard)):
        plt.text(timestamps_standard[peaks_idx_standard][i],
                 -signal_standard.max() * 0.1, standards[i], ha='center')

    offset = signal_standard.max() * 1.1
    plt.plot(timestamps, signal + offset, label='sample')
    plt.plot(timestamps[peaks_idx], signal[peaks_idx] + offset, 'r*')
    for k in matching.keys():
        plt.plot([timestamps_standard[peaks_idx_standard[k]],
                  timestamps[peaks_idx[matching[k]]]],
                 [signal_standard[peaks_idx_standard[k]],
                  signal[peaks_idx[matching[k]]] + offset], 'k-')

    plt.xlabel('Time (sec.)')
    plt.legend()
    plt.title("Matching standard peaks (below) to sample peaks (above)")
    plt.legend(loc='upper left')
    return


def match_peaks_standard_sample_nn(peaks_idx, peaks_idx_standard,
                                   signal, signal_standard, standards,
                                   match_threshold=20.0, plot=True):
    """Standard-to-sample peak matching based on nearest neighbors, with
    some additions.

    Note: this function is work in progress and may not guarantee unique
    matching, at the moment.
    """
    distance_matrix = np.abs(np.subtract.outer(timestamps[peaks_idx],
                                               timestamps_standard[peaks_idx_standard]))
    match = distance_matrix.argmin(0)
    matching = {}
    for sample_idx in np.unique(match):
        standard_idxs = np.where(match == sample_idx)[0]
        # standard's peak with minimum distance from sample peak:
        standard_idx = standard_idxs[distance_matrix[sample_idx,
                                                     standard_idxs].argmin()]
        print("Sample peak %s: candidates = %s , winner: %s , at distance %s" % (sample_idx, standard_idxs, standard_idx, distance_matrix[sample_idx, standard_idx]))
        if distance_matrix[sample_idx, standard_idx] <= match_threshold:
            matching[standard_idx] = sample_idx

    if plot:
        plot_matching_peaks(timestamps, signal, peaks_idx,
                            timestamps_standard, signal_standard,
                            peaks_idx_standard, matching, standards)

    return matching


def greedy_assignment(X):
    """A simple greedy algorithm for the assignment problem.
    Note: the X matrix is a benefit function, not a cost function!

    Returns a partial permutation matrix.
    """
    XX = np.nan_to_num(X.copy())
    min = XX.min() - 1.0
    P = np.zeros(X.shape)
    while (XX > min).any():
        row, col = np.unravel_index(XX.argmax(), XX.shape)
        P[row, col] = 1.0
        XX[row, :] = min
        XX[:, col] = min

    return P


def match_peaks_standard_sample_lap_greedy(timestamps,
                                           timestamps_standard,
                                           peaks_idx,
                                           peaks_idx_standard, signal,
                                           signal_standard, standards,
                                           match_threshold=20.0,
                                           plot=True):
    """Standard-to-sample peaks matching based on greedy LAP, with
    threshold. This algorithms works well in practice. The threshold is
    the maximum allowed timestamp distance between corresponding peaks.

    Returns a dictionary where the key is the index of
    peaks_idx_standard and the value is the index of the matching
    peaks_idx. Only matching peaks are in the dictionary.
    """
    print("Matching standard peaks with sample peaks:")
    distance_matrix = np.abs(np.subtract.outer(timestamps[peaks_idx],
                                               timestamps_standard[peaks_idx_standard]))
    P = greedy_assignment(-distance_matrix)  # '-' is used to transform cost in benefit
    match = P.argmax(0)
    distances = distance_matrix[match, range(len(peaks_idx_standard))]
    tmp = (distances <= match_threshold)
    peaks_sample_match = np.sort(match[tmp])  # sort() ensures temporal order
    peaks_standard_match = np.arange(len(peaks_idx_standard),
                                     dtype=np.int)[tmp]
    matching = dict(zip(peaks_standard_match, peaks_sample_match))
    for peak_idx_standard in matching.keys():
        print("Standard peak %s matches sample peak %s (distance: %s)" % (peak_idx_standard+1, matching[peak_idx_standard]+1, distance_matrix[matching[peak_idx_standard], peak_idx_standard]))

    if plot:
        plot_matching_peaks(timestamps, signal, peaks_idx,
                            timestamps_standard, signal_standard, peaks_idx_standard,
                            matching, standards)

    return matching


def compute_molar_fraction(standards, standards_exclude, peaks_area,
                           peaks_area_standards, matching, filename):
    F = 100.0 / peaks_area_standards
    print("")
    print(filename)
    peaks_area_full = np.zeros(len(standards), dtype=np.float)
    for psidx in matching.keys():
        peaks_area_full[psidx] = peaks_area[matching[psidx]]

    Conc = F.mean(0) * peaks_area_full
    tmp = np.ones(len(standards), dtype=np.bool)
    tmp[[standards.index(se) for se in standards_exclude]] = False  # mask to exclude proteins in standards_exclude
    Molar_percent_full = -np.ones(len(standards), dtype=np.float)
    Molar_percent_full[tmp] = Conc[tmp] / Conc[tmp].sum() * 100.0
    print("| Protein | F (stand) | Area  |   Conc   | % Molar |")
    print("|--------------------------------------------------|")
    for i, st in enumerate(standards):
        if st in standards_exclude:
            continue
        else:
            print("| %s     | %8.3f  | %.3f | %8.3f | %7.2f |" % (st, F.mean(0)[i], peaks_area_full[i], Conc[i], Molar_percent_full[i]))

    print("|--------------------------------------------------|")
    print("")
    return Molar_percent_full


def plot_timelines(timestamps_standard, timestamps, plot=True):
    """Plot comparison of two timelines.
    """
    if plot:
        plt.figure()
        plt.plot(timestamps_standard, label='standard')
        plt.plot(timestamps, label='sample')
        plt.title('Timestamps')
        plt.xlabel('Time (sec.)')


def crunch_samples(standards,
                   standards_exclude,
                   timestamps_standard,
                   signal_standard,
                   peaks_idx_standard,
                   peaks_area_standards,
                   samples_filename=None,
                   samples_directory='samples',
                   match_threshold=20,
                   peak_min=0.002,
                   plot=True,
                   savefig=True):
    """Crunch all samples.
    """
    print("Crunching samples...")
    samples_filename = glob_filenames(directory=samples_directory,
                                      filenames=samples_filename)
    print("Standards' files: %s" % samples_filename)
    for i, sample_filename in enumerate(samples_filename):
        timestamps_sample, signal_sample, \
            raw_sample = load_and_prepare(sample_filename,
                                          plot=plot)
        peaks_idx_sample = automatic_peak_detection(timestamps_sample,
                                                    signal_sample,
                                                    peak_min=peak_min,
                                                    plot=plot)
        matching = match_peaks_standard_sample_lap_greedy(timestamps_sample,
                                                          timestamps_standard,
                                                          peaks_idx_sample,
                                                          peaks_idx_standard,
                                                          signal_sample,
                                                          signal_standard,
                                                          standards,
                                                          match_threshold=match_threshold,
                                                          plot=True)
        beginnings_sample, \
            ends_sample = compute_peak_beginning_end(timestamps_sample,
                                                     signal_sample,
                                                     peaks_idx_sample,
                                                     plot=plot)
        peaks_area_sample = compute_peaks_area(timestamps_sample,
                                                 signal_sample,
                                                 beginnings_sample,
                                                 ends_sample)
        molar_fraction = compute_molar_fraction(standards,
                                                standards_exclude,
                                                peaks_area_sample,
                                                peaks_area_standards,
                                                matching,
                                                sample_filename)
        if savefig:
            tmp = sample_filename[:-4] + '.pdf'
            print("Saving %s" % (sample_filename[:-4] + '.pdf'))
            plt.savefig(tmp)

        print("")
        print("")

    print('Done.')
    return peaks_area_sample, matching, timestamps_sample, signal_sample, molar_fraction
