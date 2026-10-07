import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from mne.stats import permutation_t_test, permutation_cluster_test, bonferroni_correction, fdr_correction
import scipy
from mne.time_frequency import tfr_morlet, tfr_multitaper

''' --------------- Import Data ---------------'''

epochs = mne.read_epochs('ERP_epochs.fif', preload=True)
epochs.info
epochs.plot()

''' -------------- Remove EOG channel ---------'''
epochs = epochs.pick_types(eeg=True)
epochs.info


''' ------------- Compute Permutation Statistics -------'''
data = epochs['visual/left'].get_data()
times = epochs['visual/left'].times

temporal_mask = np.logical_and(0.1 <= times, times <= 0.2)
data = np.mean(data[:, :, temporal_mask], axis=2)

n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=None)

significant_sensors = p_values <= 0.05
significant_sensors_names = np.array(epochs.ch_names)[significant_sensors]

print("Number of significant sensors : %d" % len(significant_sensors_names))
print("Sensor names : %s" % significant_sensors_names)

evoked = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                         epochs.info, tmin=0.)

# Extract mask and indices of active sensors in the layout
stats_picks = mne.pick_channels(evoked.ch_names, significant_sensors_names)
spatial_mask = p_values[:, np.newaxis] <= 0.05

evoked.plot_topomap(ch_type='eeg',
                    times=[0],
                    scalings=1,
                    cmap='Reds',
                    vlim=(0, 3),#np.max),
                    units='-log10(p)',
                    cbar_fmt='-%0.2f',
                    mask=spatial_mask,
                    size=3,
                    show_names=lambda x: x[4:] + ' ' * 10,
                    time_unit='s')


plt.plot(p_values)
plt.hlines(0.05,0,60, 'k', linestyle='dashed')


# ----------------------------------------------------------------------

'''------------- Fake subjects' data '----------------------'''
data_evoked = []
for _ in range(50):
    r = np.random.randint(low=0, high=len(epochs), size=20)
    data_evoked.append(epochs._data[r, :, :].mean(axis=0))
data_evoked = np.array(data_evoked)

''' Temporal Mask '''
temporal_mask = np.logical_and(0.1 <= times, times <= 0.18)
data_evoked = np.mean(data_evoked[:, :, temporal_mask], axis=2)

''' Calculate Permutation Statistics '''
n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data_evoked, n_permutations, n_jobs=None)



significant_sensors = p_values <= 0.05
significant_sensors_names = np.array(epochs.ch_names)[significant_sensors]
print("Number of significant sensors : %d" % sum(significant_sensors))
print("Sensors names : %s" % significant_sensors_names)
reject_b, pval_b = bonferroni_correction(p_values, alpha=0.05)
reject_fdr, pval_fdr = fdr_correction(p_values, alpha=0.05, method='indep')


evoked = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                         epochs.info, tmin=0.)

''' Extract mask and indices of active sensors in the layout '''
stats_picks = mne.pick_channels(evoked.ch_names, significant_sensors_names)


fig, (ax1, ax2, ax3) = plt.subplots(1,3)


''' No correction topo '''
spatial_mask = p_values[:, np.newaxis] <= 0.05
evoked.plot_topomap(ch_type='eeg',
                    times=[0],
                    scalings=1,
                    time_format=None,
                    cmap='Reds',
                    vlim=(0, 6),
                    units='-log10(p)',
                    cbar_fmt='-%0.1f',
                    mask=spatial_mask,
                    size=3,
                    show_names=lambda x: x[4:] + ' ' * 10,
                    time_unit='s',
                    axes=[ax1, ax4])

''' Bonferroni correction topo '''
spatial_mask = pval_b[:, np.newaxis] <= 0.05
evoked.plot_topomap(ch_type='eeg',
                    times=[0],
                    scalings=1,
                    time_format=None,
                    cmap='Reds',
                    vlim=(0, 6),
                    units='-log10(p)',
                    cbar_fmt='-%0.1f',
                    mask=spatial_mask,
                    size=3,
                    show_names=lambda x: x[4:] + ' ' * 10,
                    time_unit='s',
                    axes=[ax2, ax4])

''' FDR correction topo '''
spatial_mask = pval_fdr[:, np.newaxis] <= 0.05
evoked.plot_topomap(ch_type='eeg',
                    times=[0],
                    scalings=1,
                    time_format=None,
                    cmap='Reds',
                    vlim=(0, 6),
                    units='-log10(p)',
                    cbar_fmt='-%0.1f',
                    mask=spatial_mask,
                    size=3,
                    show_names=lambda x: x[4:] + ' ' * 10,
                    time_unit='s',
                    axes=[ax3, ax4])

#%% --------------------------------------------------------------------------

data_vis = epochs['visual'].get_data()
data_aud = epochs['audial'].get_data()

'''------------- Fake subjects' data '----------------------'''
data_evoked_diff = []
for _ in range(50):
    r_v = np.random.randint(low=0, high=len(data_vis), size=20)
    r_a = np.random.randint(low=0, high=len(data_aud), size=20)
    diff = data_vis[r_v, :, :].mean(axis=0) - data_aud[r_a, :, :].mean(axis=0)
    data_evoked_diff.append(diff)
data_evoked_diff = np.array(data_evoked_diff)

''' Temporal Mask '''
temporal_mask = np.logical_and(0.1 <= times, times <= 0.18)
data_evoked_diff = np.mean(data_evoked_diff[:, :, temporal_mask], axis=2)

''' Calculate Permutation Statistics '''
n_permutations = 50000
T0, p_values, H0 = permutation_t_test(data_evoked_diff, n_permutations, n_jobs=None)

significant_sensors = p_values <= 0.05
significant_sensors_names = np.array(epochs.ch_names)[significant_sensors]
print("Number of significant sensors : %d" % sum(significant_sensors))
print("Sensors names : %s" % significant_sensors_names)


evoked_diff = mne.EvokedArray(-np.log10(p_values)[:, np.newaxis],
                         epochs.info, tmin=0.)

reject_b, pval_b = bonferroni_correction(p_values, alpha=0.05)
spatial_mask = pval_b[:, np.newaxis] <= 0.05

evoked_diff.plot_topomap(ch_type='eeg',
                         times=[0],
                         scalings=1,
                         time_format=None,
                         cmap='Reds',
                         vlim=(0, 6),
                         units='-log10(p)',
                         cbar_fmt='-%0.1f',
                         mask=spatial_mask,
                         size=3,
                         show_names=lambda x: x[4:] + ' ' * 10,
                         time_unit='s')

'''' Plot evoked significant sensors '''
data_diff_amp = data_evoked_diff.mean(axis=0)
evoked_diff_amp = mne.EvokedArray(data_diff_amp[:, np.newaxis], epochs.info, tmin=0.)

evoked_diff_amp.plot_topomap(ch_type='eeg',
                         times=[0],
                         cmap='jet',
                         vlim=(-3, 3),
                         mask=spatial_mask,
                         size=3,
                         show_names=lambda x: x[4:] + ' ' * 10)

fig, (ax1, ax2) = plt.subplots(1,2)
evoked_diff.plot_topomap(ch_type='eeg',
                         times=[0],
                         scalings=1,
                         time_format=None,
                         cmap='Reds',
                         vlim=(0, 6),
                         units='-log10(p)',
                         cbar_fmt='-%0.1f',
                         mask=spatial_mask,
                         size=3,
                         show_names=lambda x: x[4:] + ' ' * 10,
                         time_unit='s',
                         axes=[ax1, ax3])
evoked_diff_amp.plot_topomap(ch_type='eeg',
                         times=[0],
                         cmap='jet',
                         vlim=(-3, 3),
                         mask=spatial_mask,
                         size=3,
                         show_names=lambda x: x[4:] + ' ' * 10,
                         axes=[ax2, ax3])

#%% ------------------------------------------------------------------

'''' permutation cluster test over time axis'''

data_vis_s = []
data_aud_s = []
for _ in range(50):
    r_v = np.random.randint(low=0, high=len(data_vis), size=20)
    r_a = np.random.randint(low=0, high=len(data_aud), size=20)
    data_vis_s.append(data_vis[r_v, :, :].mean(axis=0))
    data_aud_s.append(data_aud[r_a, :, :].mean(axis=0))
data_vis_s = np.array(data_vis_s)
data_aud_s = np.array(data_aud_s)


n=50
data_vis_ch = data_vis_s[:, n, :]
data_aud_ch = data_aud_s[:, n, :]

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([data_vis_ch, data_aud_ch], 
                                                                 n_permutations=10000,
                                                                 threshold=6.0, 
                                                                 tail=0,
                                                                 out_type='mask')

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.plot(times, data_vis_ch.mean(axis=0) - data_aud_ch.mean(axis=0),
        label="ERF Contrast (Vis - Aud)")
ax.set_ylabel("EEG, mV")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = ax2.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)

hf = plt.plot(times, T_obs, 'g')
ax2.legend((h, ), ('cluster p-value < 0.05', ))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("f-values")

#%% -------------------------------------------------------------------------
''' permutation cluster test over time and topo axes'''
pval = 0.05   # arbitrary
dfn = 2 - 1   # degrees of freedom numerator
dfd = 60 - 2  # degrees of freedom denominator
threshold = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([data_vis_s, data_aud_s], 
                                                                 n_permutations=10000,
                                                                 threshold=threshold, 
                                                                 tail=0,
                                                                 out_type='mask')

cl = 10
plt.figure(figsize=(20, 10))
#plt.pcolormesh(significant_points * t_obs.T, cmap='Reds')
plt.pcolormesh(T_obs*clusters[cl], cmap='Reds')
plt.colorbar().set_label('F-value', fontsize=14)
plt.yticks(ticks=np.linspace(0,59,60), labels=epochs.ch_names)
plt.xticks(ticks=np.linspace(1,61,31), labels=np.round(np.linspace(-0.2,0.4,31), 2))
plt.xlabel('Time', fontsize=14)
plt.ylabel('Channels', fontsize=14)
plt.tight_layout()

epochs.plot_sensors(show_names=True)


'''================ Time - Frequency Decomposition ======================'''

epochs = mne.read_epochs('TF-epo.fif', preload=True)
epochs.plot()
epochs.info

freqs = np.linspace(1, 30, num=30)
freqs = np.logspace(*np.log10([1, 30]), num=29)
n_cycles = freqs / 2.

power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=3, n_jobs=1)
power, itc = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=2, return_itc=True)


power.plot_topo(baseline=(-0.4, -0.0), mode='logratio', tmin=-0.3, tmax=0.7, cmap='jet') #  vmin=-0.001, vmax=0.001,
itc.plot_topo(mode='mean', tmin=-0.4, tmax=0.7, vmin=0, vmax=0.5, cmap='Reds') # baseline=(-0.7, -0.2)

