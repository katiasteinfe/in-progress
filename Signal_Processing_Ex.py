#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:20:54 2024

@author: ks7689
"""


import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mne
import random
import scipy
#%%
#for Windows
#path=r'E:\BrainTrace - ongoing\BrainTrace\BrainTrace_DATA\75Hz_2r\BrainTrace_Analysis\Analysis_Aug24\Selected_sbjs_psds_2'

#for MAC
module_folder = '/Volumes/T7/BrainTrace - ongoing/BrainTrace/BrainTrace_Scripts/75Hz_2r/BrainTrace_Analysis/Analysis_Aug24/Selected_sbjs_psds_2'

methods=['fft']
conditions=['8epo']
sizes=['500','501','503','1250','125025','1251','80']

#%%
# Add the folder containing the module to sys.path
#for WINDOWS
#module_folder = r'E:\BrainTrace - ongoing\BrainTrace\BrainTrace_Scripts\75Hz_2r\BrainTrace_Analysis\Analysis_Aug24'

#for MAC
module_folder = '/Volumes/T7/BrainTrace - ongoing/BrainTrace/BrainTrace_Scripts/75Hz_2r/BrainTrace_Analysis/Analysis_Aug24'
sys.path.append(module_folder)

# Import my functions from my module
from psd_noise_defs import calculatePSD_evoked,  noise_std_eval_channelwise, read_d, prepare_ROI, prepare_antneuro_124

mne.set_log_level('Warning')

#%%

#FOR Windows
#module_folder = r'E:\BrainTrace - ongoing\BrainTrace\BrainTrace_Scripts\75Hz_2r\BrainTrace_Analysis\Analysis_Aug24'
#sys.path.append(module_folder)

#For MAC
module_folder = '/Volumes/T7/BrainTrace - ongoing/BrainTrace/BrainTrace_Scripts/75Hz_2r/BrainTrace_Analysis/Analysis_Aug24/Sbjs_Level_Analysis_Phase_Shuffle'
sys.path.append(module_folder)

# Import my functions from my module
from phase_shuffle_pipeline import  phase_shuffle_evoked_or_psd
# , generate_null_distribution,
# extract_power_at_frequencies, sum_psds_at_frequencies, 
# compare_psds_to_null_zscores, compare_psds_to_null_pvalue, 
#  generate_null_distribution_condition_diff, 
# compare_real_vs_null_condition_diff)#%%

#%%
# Read previously stored ROI and select data 
d=read_d()
ROIs, ROI=prepare_ROI(0,0,d)
ROI=[x for x in ROI.keys()]
ants,ant=prepare_antneuro_124(0,0, d)
ROI_idx=[]
for i in ROI:
    for nj,j in enumerate(ant):
        if i==j:
            ROI_idx.append(nj)
#%%
# Read example evoked (1 subject, 1 condition)
#for MAC
evoked_path='/Volumes/T7/BrainTrace - ongoing/BrainTrace/BrainTrace_DATA/75Hz_2r/BrainTrace_Preprocessed/Preprocessed/All_chs/Preprocessed_8epo/S1823_500-8-epo-ave.fif'

#for WINDOWS
#evoked_path=r'E:\BrainTrace - ongoing\BrainTrace\BrainTrace_DATA\75Hz_2r\BrainTrace_Preprocessed\Preprocessed\All_chs\Preprocessed_8epo\S1823_500-8-epo-ave.fif'

evokedData=mne.read_evokeds( evoked_path, condition=0).pick_channels(ROI).crop(tmin=0, tmax=10)
#%%
# Plot evoked accross ROI
evokedData=evokedData.apply_baseline(baseline=(None, None))
evokedData.plot()
#%%
avgtrialsData=evokedData.get_data()[:,0:10240]
#%%
# Sample signal
Fs = 1024 # Sampling frequency (Hz)
T = 1/Fs   # Sampling period (seconds)
n = avgtrialsData.shape[1] # Length of the signal

#%%
# Perform FFT on the data for each channel
fft_results = np.fft.fft(avgtrialsData,axis=1) 
original_freq_data = fft_results  # Save the raw FFT coefficients

# Get corresponding frequencies
freqs = np.fft.fftfreq(n, d=T)

# Take the phase
phase_shuffled = [np.angle(fft_results[ch]) for ch in range(35)]
print(np.shape(phase_shuffled[0]))

# Shuffle the phase
[np.random.shuffle(phase_shuffled[ch]) for ch in range(35)]
print(np.shape(phase_shuffled[0]))
print({
    ROI[ch]: sum([1 if x == y else 0 for x, y in zip(np.angle(original_freq_data[ch]), phase_shuffled[ch])])
    for ch in range(35)
})

# Create new frequency-domain data with original magnitude and shuffled phase
new_freq_data = [np.abs(fft_results[ch]) * np.exp(1j * phase_shuffled[ch]) for ch in range(35)]
print(np.shape(new_freq_data))

# Perform inverse FFT to get the shuffled time-domain signal
shuffled_data_all_v2=np.fft.ifft(new_freq_data, axis=1)
evoked_data_new_v2 = [np.real(shuffled_data_all_v2[ch]) for ch in range(35)]
print(np.shape(shuffled_data_all_v2))
#%%
# Alternatively, shuffle the phase by adding a cnst to each freq
original_magnitudes = np.abs(fft_results)  # Save magnitudes
original_phases = np.angle(fft_results)  # Save phases

# Generate shuffled phases
random_phases = 2 * np.pi * np.random.rand(35, 10240)
shuffled_ffts = original_magnitudes * np.exp(1j * (original_phases + random_phases))

# Inverse FFT to get the shuffled time-domain signals
shuffled_data_all = np.fft.ifft(shuffled_ffts, axis=1)
evoked_data_new = np.real(shuffled_data_all)  # Take real part of inverse FFT

# Check the shapes to confirm they match expectations
print("Shape of original FFT results:", fft_results.shape)
print("Shape of shuffled FFT results:", shuffled_ffts.shape)
print("Shape of shuffled time-domain data:", shuffled_data_all .shape)


#%%
labels={'o':'Original data', 'n':'Phase-shuffled data'}

times=np.arange(0,np.round(n/Fs),T)
# Plot new and original time series
ch=random.randint(0,35)
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(times,avgtrialsData[ch] , color='blue', label=labels['o'])
plt.plot(times,evoked_data_new[ch], color='red', label=labels['n'])
plt.title('v1')
plt.ylabel('EEG sinal (mv)')
plt.xlabel('Time (ms)')
plt.legend()


plt.subplot(2,1,2)
plt.plot(times,avgtrialsData[ch] , color='blue', label=labels['o'])
plt.plot(times,evoked_data_new_v2[ch], color='red', label=labels['n'])
plt.title('v2')
plt.ylabel('EEG sinal (mv)')
plt.xlabel('Time (ms)')
plt.legend()

plt.suptitle(f'EEG sinal for {ROI[ch]}')
#%%
# Plot the new and original freq series
plt.figure(figsize=(10, 8))
plt.plot(original_freq_data[ch], color='blue', label=labels['o'])
plt.plot(new_freq_data[ch], color='red', label=labels['n'], alpha=0.5)
plt.title(f'FFT magnitudes for {ROI[ch]}')
plt.ylabel('FFT magnitudes')
plt.xlabel('Freqs (Hz)')
plt.legend()
#Why does it look like that?
#%%
# Take the single-sided spectrum

fft_magnitudes={'original':[],'new':[]}

for fft_result,name in zip([original_freq_data,new_freq_data],['original','new']):
    fft_magnitude = [np.abs(fft_result[ch])[:n//2] * 2/n for ch in range(len(fft_result))] # Magnitude of the FFT
    fft_magnitudes[name].append(fft_magnitude) # Take only the first half and scale
    
#%%    
# Generate frequency axis (for positive frequencies)
frequencies = np.fft.fftfreq(n, T)[:n//2]

#%%
plt.figure(figsize=(10, 8))
plt.plot(frequencies,fft_magnitudes['original'][0][0] , color='blue', label='Original data')
plt.plot(frequencies,fft_magnitudes['new'][0][0], color='red', alpha=0.5, label='Phase-shuffled data')

plt.title(f'FFT for {ROI[ch]}')
plt.ylabel('PSD (DB)')
plt.xlabel('Freqs (Hz)')
plt.legend()

#%%
# What happens if we shuffle in the time domain instead?
ch=np.random.randint(0,35)
avgtrialsData_timeshuffle=avgtrialsData[ch].copy()
# Shuffle time points
for _ in range(1000):
    np.random.shuffle(avgtrialsData_timeshuffle)
fft_timeshuffle = np.fft.fft(avgtrialsData_timeshuffle)
magnitude_timeshuffle = np.abs(fft_timeshuffle)[:n // 2] * 2 / n
phase_timeshuffle= np.angle(avgtrialsData_timeshuffle)

#%%
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(frequencies,fft_magnitudes['original'][0][ch] , color='blue', label='Original data')
plt.plot(frequencies,fft_magnitudes['new'][0][ch], color='red', alpha=0.5, label='Phase-shuffled data')
plt.ylabel('Amplitude - normalized')
plt.xlabel('Freqs (Hz)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(frequencies,fft_magnitudes['original'][0][ch] , color='blue', alpha=0.5, label='Original data')
plt.plot(frequencies, magnitude_timeshuffle, color='green', alpha=0.5, label='Time-shuffled data')
plt.ylabel('Amplitude - normalized')
plt.xlabel('Freqs (Hz)')
plt.legend()

plt.suptitle(f'FFT for {ROI[ch]}')
#%%
fft_psd_db={'original':[],'new':[]}
fft_psd_db['original'] = [10 * np.log10((np.abs(original_freq_data[ch])[:n // 2] ** 2) / n) for ch in range(35)]
fft_psd_db['new'] = [10 * np.log10((np.abs(new_freq_data[ch])[:n // 2] ** 2) / n) for ch in range(35)]
fft_psd_timeshuffle=[10 * np.log10((np.abs(fft_timeshuffle)[:n // 2] ** 2) / n) for ch in range(35)]
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(frequencies,fft_psd_db['original'][ch] , color='blue', label='Original data')
plt.plot(frequencies,fft_psd_db['new'][ch], color='red', alpha=0.5, label='Phase-shuffled data')
plt.ylabel('PSD (db)')
plt.xlabel('Freqs (Hz)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(frequencies,fft_psd_db['original'][ch] , color='blue', alpha=0.5, label='Original data')
plt.plot(frequencies, magnitude_timeshuffle, color='green', alpha=0.5, label='Time-shuffled data')
plt.ylabel('PSD (db)')
plt.xlabel('Freqs (Hz)')

plt.legend()

plt.suptitle(f'PSD for {ROI[ch]}')
#%%
#1/f distribution
# Remove 0Hz from data before taking log
plt.figure(figsize=(10, 8))
plt.plot(frequencies[1:200],fft_magnitudes['original'][0][ch][1:200] , color='blue', label=labels['o'])
plt.plot(frequencies[1:200],fft_magnitudes['new'][0][ch][1:200], color='red', alpha=0.5, label=labels['n'])
plt.vlines(0, ymin=0, ymax=fft_magnitudes['new'][0][ch][1:200].max(), linestyles='--', color='black')
plt.title(f'FFT for {ROI[ch]}')
plt.ylabel('PSD (DB)')
plt.xlabel('Freqs (Hz)')
plt.legend()

#%%
#Illustrate the power-law relationship, y varies as a power of x (y= kx^-alpha) -> log-log makes it linear y=-alpha *x + k
# Take the logarithm of frequency and power spectral density
log_frequencies = np.log(frequencies[1:])
log_psd = { name: [np.log(fft_magnitudes[name][0][ch][1:]) for ch in range(35)] for name in fft_magnitudes.keys()} 

# Fit a linear model to the log-log plot
regression_res = { name: [ scipy.stats.linregress(log_frequencies, log_psd[name][ch]) for ch in range(35)] for name in fft_magnitudes.keys()}
#slope, intercept, r_value, p_value, std_err 

#%%
# Plot the data and the fitted line
plt.figure(figsize=(10, 8))
plt.plot(log_frequencies, log_psd['original'][ch], label=labels['o'], color='blue')
plt.plot(log_frequencies, log_psd['new'][ch], label=labels['o'], color='red', alpha=0.5)
plt.plot(log_frequencies, regression_res['new'][ch][0] * log_frequencies + regression_res['new'][ch][1], label=f"Fitted line (slope = {regression_res['new'][ch][0]:.2f})", color='black')
plt.xlabel('Log(Frequency)')
plt.ylabel(f'Log(PSD) for {ROI[ch]}')
plt.legend()
plt.grid(True)
plt.show()


#%%
signal=avgtrialsData[0]
time_scales=[0.1,0.5,1,5,10]
plt.figure(figsize=(10, 8))
def pad_to_length(data, target_length):

    pad_size = target_length - len(data)  # Calculate how many zeros to add
    if pad_size > 0:  # If the data is shorter than the target length
        return np.pad(data, (0, pad_size), 'constant')  # Pad with zeros at the end
    else:
        return data
        
for time_scale in time_scales:

    segment_samples = int(time_scale * Fs)

    # Take a segment of the data
    segment = signal[:segment_samples]
    
    # Pad the segment ( FFT resolution constant)
    padded_segment = pad_to_length(segment, n)

   
    fft_result = np.fft.fft(padded_segment)
    
    
    psd = np.abs(fft_result)**2 / len(padded_segment)

    # frequency axis for f>0
    freqs = np.fft.fftfreq(len(padded_segment), 1/Fs)[:len(padded_segment)//2]

    # log-log of PSD for positive frequencies
    plt.plot(freqs[1:], psd[:len(padded_segment)//2][1:], label=f'{time_scale} seconds')
    
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.title('1/f Distribution Across Different Time Scales')
plt.legend()
plt.grid(True)
plt.show()
#%%
# Take the phase of both datasets 
fft_phases={'original':[],'new':[]}
for fft_result,name in zip([original_freq_data,new_freq_data],['original','new']):
    fft_phases[name] = [np.angle(fft_result[ch]) for ch in range(len(fft_result))] # Magnitude of the FFT

# Plot the phase of both datasets
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(frequencies, fft_phases['original'][ch][:n//2], label=labels['o'], color='blue')
plt.plot(frequencies, fft_phases['new'][ch][:n//2], label=labels['n'], color='red', alpha=0.5)
plt.suptitle(f'Phase Comparison for {ROI[ch]}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True)
plt.show()

plt.subplot(2,1,2)
plt.plot(frequencies, fft_phases['original'][ch][:n//2], label=labels['o'], color='blue')
plt.plot(frequencies, phase_timeshuffle[:n//2], label='Time-shuffled data', color='green', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True)
plt.show()
#%%
# Now let's put it all together and relate it to the cos and sin parts

fft_sin={'original':[],'new':[]}
fft_cosin={'original':[],'new':[]}
for fft_result,name in zip([original_freq_data,new_freq_data],['original','new']):
# Real (cosine) and imaginary (sine) components
    fft_cosin[name]=[np.real(fft_result[ch][:n//2]) for ch in range(35)]
    fft_sin[name]=[np.imag(fft_result[ch][:n//2]) for ch in range(35)]

#%%
ch=random.randint(0,35)
## Plot the real (cosine) and imaginary (sine) parts along with phase
plt.figure(figsize=(10, 8))


# Magnitude and phase plot
plt.subplot(5, 1, 1)
plt.plot(frequencies[1:200],fft_magnitudes['original'][0][ch][1:200] , color='blue', label=labels['o'])
plt.plot(frequencies[1:200],fft_magnitudes['new'][0][ch][1:200], color='red', alpha=0.5, label=labels['n'])
plt.ylabel('Magnitude')
plt.title('FFT Magnitude and Phase Representation')
plt.grid(True)
plt.legend()

# Phase plot
plt.subplot(5, 1, 2)
plt.plot(frequencies[1:200], fft_phases['original'][ch][1:200], label=labels['o'], color='blue')
plt.plot(frequencies[1:200], fft_phases['new'][ch][1:200], label=labels['n'], color='red')
plt.ylabel('Phase (radians)')
plt.ylim([-10,10])
plt.grid(True)
#plt.legend() 
# Real (cosine) component
plt.subplot(5, 1, 3)
plt.plot(frequencies[1:200], fft_cosin['original'][ch][1:200], label='Real Original data', linestyle='-', color='blue')
plt.plot(frequencies[1:200], fft_cosin['new'][ch][1:200], label='Real Phase-shuffled data', linestyle='-', color='red', alpha=0.5)
#plt.xlabel('Frequency (Hz)')
plt.ylabel('Cosine')
plt.grid(True)
plt.legend()

# Imaginary (sine) component
plt.subplot(5, 1, 4)
plt.plot(frequencies[1:200], fft_sin['original'][ch][1:200], label='Imaginary Original data', linestyle='-', color='blue')
plt.plot(frequencies[1:200], fft_sin['new'][ch][1:200], label='Imaginary Phase-shuffled data', linestyle='-', color='red', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sine')
plt.grid(True)
plt.legend()

# Ratio of sine and cosine components
plt.subplot(5, 1, 5)
plt.plot(frequencies[1:200], np.arctan2(fft_sin['original'][ch][1:200],fft_cosin['original'][ch][1:200]), label='Original data', linestyle='-', color='blue')
plt.plot(frequencies[1:200], np.arctan2(fft_sin['new'][ch][1:200],fft_cosin['new'][ch][1:200]), label='Phase-shuffled data', linestyle='-', color='red')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Tan(Sine, Cosine)')
plt.ylim([-10,10])
plt.grid(True)
#plt.legend()
#%%