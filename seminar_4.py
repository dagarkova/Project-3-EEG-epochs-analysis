import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet, tfr_multitaper
import mplcyberpunk
plt.style.use('default') # cyberpunk

' Import data'
epochs = mne.read_epochs('TF-epo.fif', preload=True)
epochs.plot()
epochs.info

'time-frequency decomposition: wavelets and multitaper'
freqs = np.linspace(1, 25, num=25)
freqs = np.logspace(*np.log10([1, 30]), num=29)
n_cycles = freqs / 2.

power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)


power_vis, itc_vis = tfr_morlet(epochs['visual'], freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)
power_aud, itc_aud = tfr_morlet(epochs['auditory'], freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)

epochs.plot_sensors(show_names=True)


kwargs = dict(baseline=(-0.3, -0.1), mode='logratio', tmin=-0.3, tmax=0.7, cmap='jet')

'''
mode‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
Perform baseline correction by
subtracting the mean of baseline values (‘mean’) (default)
dividing by the mean of baseline values (‘ratio’)
dividing by the mean of baseline values and taking the log (‘logratio’)
subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
subtracting the mean of baseline values and dividing by the standard deviation of baseline values (‘zscore’)
dividing by the mean of baseline values, taking the log, and dividing by the standard deviation of log baseline values (‘zlogratio’)
'''

'''Power'''
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
power_vis.plot('EEG 058', **kwargs, axes=ax1) #  vmin=-0.001, vmax=0.001,
power_aud.plot('EEG 058', **kwargs, axes=ax2) #  vmin=-0.001, vmax=0.001,
ax1.set_title('Visual')
ax2.set_title('Audial')

power_vis_df = power_vis.to_data_frame()
power_aud_df = power_aud.to_data_frame()

chan = 58 # 58 26
band_power_vis = pd.pivot_table(power_vis_df.query('freq >= 7 and freq <= 13'), values=f'EEG 0{chan}', columns='time')
band_power_aud = pd.pivot_table(power_aud_df.query('freq >= 7 and freq <= 13'), values=f'EEG 0{chan}', columns='time')
band_power_vis = np.log(band_power_vis.T / band_power_vis.T[-0.3:-0.1].mean())
band_power_aud = np.log(band_power_aud.T / band_power_aud.T[-0.3:-0.1].mean())

plt.figure()
plt.plot(band_power_vis[-0.3:0.7], label='Visual')
plt.plot(band_power_aud[-0.3:0.7], label='Audial')
plt.xlabel('Time, s')
plt.ylabel('EEG reactivity, dB')
plt.title('Band: 3 - 7 Hz')
plt.legend()


def tf_reac(data_vis, data_aud, chan, ax, low_freq=5, high_freq=10, plot=True):
    band_power_vis = pd.pivot_table(data_vis.query('freq >= @low_freq and freq <= @high_freq'), values=f'EEG 0{chan}', columns='time')
    band_power_aud = pd.pivot_table(data_aud.query('freq >= @low_freq and freq <= @high_freq'), values=f'EEG 0{chan}', columns='time')
    band_power_vis = np.log(band_power_vis.T / band_power_vis.T[-0.3:-0.1].mean())
    band_power_aud = np.log(band_power_aud.T / band_power_aud.T[-0.3:-0.1].mean())
    if plot == True:
        ax.plot(band_power_vis[-0.3:0.7], label='Visual')
        ax.plot(band_power_aud[-0.3:0.7], label='Audial')
        ax.set_xlabel('Time, s')
        ax.set_ylabel('EEG reactivity, dB')
        ax.set_title(f'Band: {low_freq} - {high_freq} Hz')
        ax.legend()
        ax.hlines(0, -0.3, 0.7, 'k')

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5), sharey=True)
tf_reac(power_vis_df, power_aud_df, 58, ax1, low_freq=1, high_freq=5, plot=True)
tf_reac(power_vis_df, power_aud_df, 58, ax2, low_freq=5, high_freq=10, plot=True)
tf_reac(power_vis_df, power_aud_df, 58, ax3, low_freq=10, high_freq=15, plot=True)
fig.tight_layout()


'''ITC'''
itc_vis.plot('EEG 058', mode='mean', tmin=-0.4, tmax=0.8, cmap='Reds') # baseline=(-0.7, -0.2)


itc_vis_df = itc_vis.to_data_frame()
itc_aud_df = itc_aud.to_data_frame()

chan = 58 # 58 26
band_itc_vis = pd.pivot_table(itc_vis_df.query('freq >= 3 and freq <= 7'), values=f'EEG 0{chan}', columns='time')
band_itc_aud = pd.pivot_table(itc_aud_df.query('freq >= 3 and freq <= 7'), values=f'EEG 0{chan}', columns='time')
band_itc_vis = band_itc_vis.T
band_itc_aud = band_itc_aud.T

plt.figure()
plt.plot(band_itc_vis[-0.3:0.7], label='Visual')
plt.plot(band_itc_aud[-0.3:0.7], label='Audial')
plt.xlabel('Time, s')
plt.ylabel('EEG reactivity, dB')
plt.title('Band: 3 - 7 Hz')
plt.legend()



'''substract evoked'''
epochs_ica_vis_se = epochs['visual'].subtract_evoked()
epochs_ica_aud_se = epochs['auditory'].subtract_evoked()

power_vis_se, itc_vis_se = tfr_morlet(epochs_ica_vis_se, freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)
power_aud_se, itc_aud_se = tfr_morlet(epochs_ica_aud_se, freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
power_vis_se.plot('EEG 058', baseline=(-0.3, -0.1), mode='logratio', tmin=-0.3, tmax=0.7, cmap='jet', axes=ax1) #  vmin=-0.001, vmax=0.001,
power_aud_se.plot('EEG 058', baseline=(-0.3, -0.1), mode='logratio', tmin=-0.3, tmax=0.7, cmap='jet', axes=ax2) #  vmin=-0.001, vmax=0.001,
ax1.set_title('Visual')
ax2.set_title('Audial')


power_aud_se.plot_topo(baseline=(-0.3, -0.1), tmin=-0.3, tmax=0.7, mode='logratio', title='Average power', cmap='jet')
power_vis_se.plot_topo(baseline=(-0.3, -0.1), tmin=-0.3, tmax=0.7, mode='logratio', title='Average power', cmap='jet')


itc_vis_se.plot_topo(tmin=-0.3, tmax=0.7, mode='mean', title='Average ITC', vmin=0, vmax=1.0, cmap='Reds')


''' Multitaper'''
power_vis, itc_vis = tfr_multitaper(epochs['visual'], freqs=freqs, n_cycles=n_cycles, time_bandwidth=2, return_itc=True)
power_aud, itc_aud = tfr_multitaper(epochs['auditory'], freqs=freqs, n_cycles=n_cycles, time_bandwidth=2, return_itc=True)


'''substract evoked'''
power_vis_se, itc_vis_se = tfr_multitaper(epochs_ica_vis_se, freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)
power_aud_se, itc_aud_se = tfr_multitaper(epochs_ica_aud_se, freqs=freqs, n_cycles=n_cycles, use_fft=False, return_itc=True, decim=3, n_jobs=1)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
power_vis_se.plot('EEG 058', baseline=(-0.3, -0.1), mode='logratio', tmin=-0.3, tmax=0.7, cmap='jet', axes=ax1) #  vmin=-0.001, vmax=0.001,
power_aud_se.plot('EEG 058', baseline=(-0.3, -0.1), mode='logratio', tmin=-0.3, tmax=0.7, cmap='jet', axes=ax2) #  vmin=-0.001, vmax=0.001,
ax1.set_title('Visual')
ax2.set_title('Audial')


#%% ========= CONNECTIVITY ==================
import mne
import pandas as pd
import numpy as np

epochs = mne.read_epochs('TF-epo.fif', preload=True)
epochs.plot()

freqs = np.linspace(1, 20, num=20)
n_cycles = freqs / 2.
kwargs = dict(freqs=freqs, n_cycles=n_cycles, 
              average=False, output='complex', 
              use_fft=False, return_itc=False, 
              decim=3, n_jobs=1)

power_vis_comp = tfr_morlet(epochs['visual'], **kwargs)
power_aud_comp = tfr_morlet(epochs['auditory'], **kwargs)

power_vis_comp_df = power_vis_comp.to_data_frame()
power_aud_comp_df = power_vis_comp.to_data_frame()

power_vis_comp_st1 = power_vis_comp_df.query('freq == 5 and epoch == 1')['EEG 058'].reset_index(drop=True)
power_vis_comp_st3 = power_vis_comp_df.query('freq == 5 and epoch == 3')['EEG 058'].reset_index(drop=True)
power_vis_comp_st5 = power_vis_comp_df.query('freq == 5 and epoch == 5')['EEG 058'].reset_index(drop=True)

power_aud_comp_st1 = power_vis_comp_df.query('freq == 5 and epoch == 1')['EEG 058'].reset_index(drop=True)
power_aud_comp_st3 = power_vis_comp_df.query('freq == 5 and epoch == 3')['EEG 058'].reset_index(drop=True)
power_aud_comp_st5 = power_vis_comp_df.query('freq == 5 and epoch == 5')['EEG 058'].reset_index(drop=True)

st_real = list(map(lambda x: x.real, power_vis_comp_st1))
st_imag = list(map(lambda x: x.imag, power_vis_comp_st1))
plt.plot(st_real, st_imag)


amp =   lambda x: np.sqrt(x.real**2 + x.imag**2)
phase = lambda x: np.arctan(x.imag / x.real)

amp_st1 = list(map(amp, power_aud_comp_st1))
amp_st3 = list(map(amp, power_aud_comp_st3))
amp_st5 = list(map(amp, power_aud_comp_st5))

phase_st1 = list(map(phase, power_aud_comp_st1))
phase_st3 = list(map(phase, power_aud_comp_st3))
phase_st5 = list(map(phase, power_aud_comp_st5))

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(amp_st1)
ax1.plot(amp_st3)
ax1.plot(amp_st5)
ax2.plot(phase_st1)
ax2.plot(phase_st3)
ax2.plot(phase_st5)



from mne_connectivity import spectral_connectivity_epochs, seed_target_indices


picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
picks_ch_names = [epochs['visual'].ch_names[i] for i in picks]
seed_ch = 'EEG 058'
seed = picks_ch_names.index(seed_ch)
targets = np.arange(len(picks))
indices = seed_target_indices(seed, targets)

epochs_vis_eeg = epochs['visual'].pick_types(eeg=True)
con = spectral_connectivity_epochs(epochs_vis_eeg, 
                                   sfreq=100, 
                                   method='plv', # ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased']
                                   mode='cwt_morlet',
                                   fmin=(4,8,13),
                                   fmax=(8,13,25),
                                   tmin=0.05,
                                   tmax=0.55,
                                   faverage=True,
                                   cwt_freqs=np.linspace(3, 25, num=23),
                                   cwt_n_cycles=3,
                                   indices=indices)

'''
methodstr | list of str
Connectivity measure(s) to compute. These can be ['coh', 'cohy', 'imcoh', 'cacoh', 'mic', 'mim', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased', 'gc', 'gc_tr']. These are:

‘coh’ : Coherence

‘cohy’ : Coherency

‘imcoh’ : Imaginary part of Coherency

‘cacoh’ : Canonical Coherency (CaCoh)

‘mic’ : Maximised Imaginary part of Coherency (MIC)

‘mim’ : Multivariate Interaction Measure (MIM)

‘plv’ : Phase-Locking Value (PLV)

‘ciplv’ : Corrected Imaginary PLV (ciPLV)

‘ppc’ : Pairwise Phase Consistency (PPC)

‘pli’ : Phase Lag Index (PLI)

‘pli2_unbiased’ : Unbiased estimator of squared PLI

‘dpli’ : Directed PLI (DPLI)

‘wpli’ : Weighted PLI (WPLI)

‘wpli2_debiased’ : Debiased estimator of squared WPLI

‘gc’ : State-space Granger Causality (GC)

‘gc_tr’ : State-space GC on time-reversed signals


mode
Spectrum estimation mode can be either: ‘multitaper’, ‘fourier’, or ‘cwt_morlet’.
'''

con_data = con.get_data()

plt.pcolor(con_data[:, 1, :])
plt.colorbar()

plt.plot(con_data.mean(axis=2))
plt.xticks(ticks=np.linspace(0,58,59), labels=picks_ch_names, rotation=90)
plt.legend(['Theta', 'Alpha', 'Beta'])
