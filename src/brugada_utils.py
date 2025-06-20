import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals import tools
from scipy import signal
import torch
import wfdb

TARGET_FS = 100
SAMPLES_IN_5_SECONDS_AT_100HZ = 500
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ---------------- PLOT AND SAVING ---------------- 
def plot_ecg(record, path=None, fs=100):
	"""
	Plots and saves ECG for a given record.

	:param record: The ECG record
	:param path: The path to save the image
	:param fs: The sampling frequency of the record
	"""
	# Number of leads
	num_leads = record.shape[1]

	# Create a figure with subplots for each lead
	_, axs = plt.subplots(num_leads, 1, figsize=(12, 10), sharex=True)  # Reduce figure size
	if num_leads == 1:
		axs = [axs]  # Ensure axs is a list even if there is only one lead

	# Time in seconds
	time = np.arange(record.shape[0]) / fs  # Convert samples to seconds

	# Plot each lead
	for i in range(num_leads):
		axs[i].plot(time, record[:, i], linewidth=0.25, color='black')
		
		# Labels and grid
		axs[i].set_ylabel(f'{LEADS[i]}', fontsize=8)  # Reduce label size
		#axs[i].tick_params(axis='y', labelsize=8)
		# Set a denser grid
		axs[i].grid(True, which='major', linestyle='--', linewidth=0.5)
		
	# Adjust the visualization
	axs[-1].set_xlabel('Time (seconds)', fontsize=10)  # Reduce label size
	#fig.suptitle(f"Study {record.record_name} example", fontsize=10)  # Reduce title size

	# Adjust margins to make everything look good
	plt.tight_layout()
	plt.subplots_adjust(top=0.95)  # Adjust to avoid title overlap

	if path:
		plt.savefig(path)
	else:
		plt.show()
	plt.close()

def plot_lead(record, lead, path=None, fs=100, save_text=False):
	"""
	Plots and saves a lead from the given record.

	:param record: The ECG record
	:param lead: The lead to plot
	:param path: The path to save the image
	"""
	
	# Find the index of lead
	lead_index = LEADS.index(lead)
	
	# Check if the record has enough leads
	num_leads = record.shape[1]
	if num_leads <= lead_index:
		print(f"Warning: Record does not have {lead} lead (only has {num_leads} leads)")
		return
	
	# Create a figure for the selected lead
	_, ax = plt.subplots(figsize=(12, 4))

	# Add ECG grid
	ax.grid(True, which='major', color='pink', linewidth=0.8, alpha=0.5)
	ax.grid(True, which='minor', color='pink', linewidth=0.2, alpha=0.5)
	ax.minorticks_on()
	
	time = np.arange(record.shape[0]) / fs
	
	# Plot lead
	ax.plot(time, record[:, lead_index], linewidth=0.5, color='black')
	
	# Labels and grid
	#plot a line at y = 0
	#ax.axhline(y=0, color='black', linewidth=0.5)
	ax.set_xlabel('Time (seconds)', fontsize=10)
	ax.set_ylabel('Voltage (mV)', fontsize=10)
	#ax.set_xticks([])
	#ax.set_yticks([])
	
	# Adjust margins
	plt.tight_layout()
	#plt.box(False)
	
	if path:
		# Save and close
		plt.savefig(path+'.png')
	else:
		plt.show()

	if save_text:
		# convert lead to string
		lead_str = ' '.join(map(str, record[:, lead_index]))
		with open(path+'.txt', 'w') as file:
			file.write(lead_str)

	plt.close()
      
def plot_beat_from_record(record, lead, path=None, fs=100, median=False, save_text=False):
	"""
	Plots and saves a lead from the given record.

	:param record: The ECG record
	:param lead: The lead to plot
	:param path: The path to save the image
	"""

	beat = get_beat(record.T, fs, median)

	# Find the index of lead
	lead_index = LEADS.index(lead)
	beat = beat[lead_index]	

	# Create a figure for the selected lead
	_, ax = plt.subplots(figsize=(5, 5))

	# Add ECG grid
	ax.grid(True, which='major', color='pink', linewidth=0.8, alpha=0.5)
	ax.grid(True, which='minor', color='pink', linewidth=0.2, alpha=0.5)
	ax.minorticks_on()

	# Plot lead
	ax.plot(beat, linewidth=0.5, color='black')
	
	# Labels and grid
	#plot a line at y = 0
	ax.set_ylabel('Voltage (mV)')
	ax.set_xlabel('Time (ms)')
	
	# Adjust margins
	plt.tight_layout()
	#plt.box(False)
	
	if path:
		# Save and close
		plt.savefig(path+'.png', dpi=300)
	else:
		plt.show()

	if save_text:
		# convert beat to string
		beat_str = ' '.join(map(str, beat))
		with open(path+'.txt', 'w') as file:
			file.write(beat_str)
	plt.close()

def plot_ecg_segments(patient_limits, patient_beat, ax=None, save_path=None, dpi=300):
	"""
	Plot ECG segments with labeled P, Q, QRS complex, S and T waves.

	Args:
		patient_id (str): ID of the patient to plot
		samples_icl_peaks (pd.DataFrame): DataFrame containing peak locations
		samples_icl_beats (pd.DataFrame): DataFrame containing beat data
		save_path (str, optional): Path to save the plot. If None, plot is not saved
		dpi (int, optional): DPI for saved plot. Default 300
	"""

	if ax is None:
		fig, ax = plt.subplots(figsize=(5, 5))
	else:
		fig = ax.get_figure()

	# Add ECG grid
	ax.grid(True, which='major', color='pink', linewidth=0.8, alpha=0.5)
	ax.grid(True, which='minor', color='pink', linewidth=0.2, alpha=0.5)
	ax.minorticks_on()

	# Plot P wave
	p_init = int(patient_limits['pinit'].iloc[0])
	p_end = int(patient_limits['pend'].iloc[0])
	p_center = (p_init + p_end) // 2
	ax.plot(range(p_init, p_end), 
			patient_beat[p_init:p_end], 
			color='grey')
	ax.text(p_center, max(patient_beat[p_init:p_end]), 
			'P', horizontalalignment='center')

	# Plot Q wave
	q_init = int(patient_limits['pend'].iloc[0])
	qrs_init = int(patient_limits['qrsinit'].iloc[0])
	q_center = (q_init + qrs_init) // 2
	ax.plot(range(q_init, qrs_init),
			patient_beat[q_init:qrs_init],
			color='red')
	ax.text(q_center, max(patient_beat[q_init:qrs_init]),
			'Q', horizontalalignment='center')

	# Plot QRS complex
	qrs_init = int(patient_limits['qrsinit'].iloc[0])
	qrs_end = int(patient_limits['qrsend'].iloc[0])
	qrs_center = (qrs_init + qrs_end) // 2
	ax.plot(range(qrs_init, qrs_end),
			patient_beat[qrs_init:qrs_end],
			color='blue')
	ax.text(qrs_center, max(patient_beat[qrs_init:qrs_end])-0.05,
			'QRS', horizontalalignment='center')

	# Plot S wave
	s_init = int(patient_limits['qrsend'].iloc[0])
	s_end = int(patient_limits['tinit'].iloc[0])
	s_center = (s_init + s_end) // 2
	ax.plot(range(s_init, s_end),
			patient_beat[s_init:s_end],
			color='green')
	ax.text(s_center, max(patient_beat[s_init:s_end]),
			'S', horizontalalignment='center')

	# Plot T wave
	t_init = int(patient_limits['tinit'].iloc[0])
	t_end = int(patient_limits['tend'].iloc[0])
	t_center = (t_init + t_end) // 2
	ax.plot(range(t_init, t_end),
			patient_beat[t_init:t_end],
			color='orange')
	ax.text(t_center, max(patient_beat[t_init:t_end]),
			'T', horizontalalignment='center')

	# Plot the remaining signal of patient_beats
	ax.plot(range(t_end, len(patient_beat)), patient_beat[t_end:], color='grey')

	# axis are y=mV, x= time(ms)
	ax.set_ylabel('Voltage (mV)')
	ax.set_xlabel('Time (ms)')
	#ax.set_ylim(-1, 1)

	if save_path is not None:
		fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
		plt.close(fig)
	else:
		return fig, ax

def plot_beat(patient_beat, ax=None, save_path=None, dpi=300):
	"""Plot a single beat without segment annotations.

	Args:
		patient_beat: Array containing the beat data
		
	Returns:
		fig, ax: The matplotlib figure and axis objects
	"""
	if ax is None:
		fig, ax = plt.subplots(figsize=(5, 5))
	else:
		fig = ax.get_figure()

	# Add ECG grid
	ax.grid(True, which='major', color='pink', linewidth=0.8, alpha=0.5)
	ax.grid(True, which='minor', color='pink', linewidth=0.2, alpha=0.5)
	ax.minorticks_on()

	# Plot the full beat
	ax.plot(patient_beat, color='black')

	# axis are y=mV, x= time(ms)
	ax.set_ylabel('Voltage (mV)')
	ax.set_xlabel('Time (ms)')
	# set limits to 0-1
	#ax.set_ylim(-1, 1)

	if save_path is not None:
		fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
		plt.close(fig)
	else:
		return fig, ax
# ---------------- END PLOT AND SAVING ------------ 

# ----------------- PREPROCESSING -----------------
def preprocess_ecg(record, current_fs=500, target_fs=100):
	"""
	Preprocess the ECG signal.
	"""
	#record = record/1000 # 1000 is the gain (?) -> apply to convert to mV
	record = record.T # 12, fs*s
	if current_fs != target_fs:
		record = signal.resample(record, int(record.shape[-1] * (target_fs/current_fs)), axis=1)  # resample to 100Hz
	order = int(0.3 * target_fs)
	record, _, _ = tools.filter_signal(signal=record, ftype='FIR', band='bandpass',
									order=order, frequency=[0.05, 47], 
									sampling_rate=target_fs)
	record = record.T
	# Cut to 5 seconds
	record = record.T[:, :SAMPLES_IN_5_SECONDS_AT_100HZ]

	# Replace NaNs with the mean of the signal
	mask = np.isnan(record)
	record = np.where(mask, record[~mask].mean(), record)

	return record

def get_beat(record, fs, median=False):
	"""
	Select beat or calculate median beat for a given sample.
	"""
	pre_window = int(0.35 * fs)
	post_window = int(0.45 * fs)

	# Combine signals of 12 leads
	combined_signal = combine_signals(record)

	# Detect peaks in the combined signal
	peaks = detect_peaks(combined_signal, fs)

	# Extraer segmentos alrededor de los picos detectados
	segments = extract_segments(record, peaks, pre_window=pre_window, post_window=post_window)

	if median:
		# Calculate the median beat
		beat = calculate_median_beat(segments)
	else:
		# Select the beat
		beat = segments[0]

	if len(peaks) <= 0:
		raise ValueError("No peaks found in the signal")

	return beat

def combine_signals(ecg):
    """
    Combine the 12 leads.
    """
    combined_signal = np.zeros(len(ecg[0])-1, dtype='float64')
    for lead in ecg:
        processed_lead = diff_ecg(lead)
        normalized_lead = normalize_signal(processed_lead)
        combined_signal += normalized_lead
    return combined_signal

def diff_ecg(ecg):
    """
    Apply the derivative and square the ECG signal.
    """
    diff_ecg = np.diff(ecg)
    squared_diff_ecg = diff_ecg ** 2
    return squared_diff_ecg

def normalize_signal(signal):
    """
    Normalize the signal.
    """
    return (signal - np.mean(signal)) / np.std(signal)

def detect_peaks(ecg_lead, fs):
    """
    Detect peaks R in an ECG lead using a basic algorithm.
    """
    # select the positions of QRS complexes within the ECG by demanding a distance of at least "distance" samples
    peaks, _ = signal.find_peaks(ecg_lead, distance=fs)
    return peaks

def segment_beats(ecg_lead, r_peaks, window_size):
    """
    Segment the beats of the ECG around the R peaks for a lead.
    """
    segments = []
    half_window = window_size // 2
    for peak in r_peaks:
        if peak - half_window >= 0 and peak + half_window < len(ecg_lead):
            segment = ecg_lead[peak - half_window: peak + half_window]
            segments.append(segment)
    return np.array(segments)

def extract_segments(ecg, peaks, window_size=None, pre_window=None, post_window=None):
    """
    Extract segments around the detected peaks.
    """
    segments = []

    if window_size is not None:
        half_window = window_size // 2
        for peak in peaks:
            if peak - half_window >= 0 and peak + half_window < len(ecg):
                segment = ecg[peak - half_window: peak + half_window]
                segments.append(segment)
    elif pre_window is not None and post_window is not None:
        for peak in peaks:
            if peak - pre_window >= 0 and peak + post_window < ecg.shape[1]:
                segment = ecg[:, peak - pre_window: peak + post_window]
                segments.append(segment)
    return np.array(segments)

def calculate_median_beat(segments):
    """
    Calculate the median beat from the aligned segments.
    """
    median_beat = np.median(segments, axis=0)
    return median_beat

def get_ecg(patient_id, datafile_path, return_tensor=True):
	"""
	Get the ECG signal for a given patient and preprocess it.
	"""
	# Load ecg
	patient_path = datafile_path +'/'+patient_id +'/'+patient_id
	ecg = wfdb.rdrecord(patient_path)
	fs = ecg.fs
	ecg = ecg.p_signal

	# Preprocess ecg
	#ecg = preprocess_ecg(ecg, current_fs=fs, target_fs=TARGET_FS)

	# Convert to torch tensor
	if return_tensor:
		# Reshape to (12*SAMPLES_IN_5_SECONDS_AT_100HZ,)
		ecg = ecg.reshape(-1)
		ecg = torch.from_numpy(ecg.copy()).float()

	return ecg
# ---------------- END PREPROCESSING ----------------

def prepare_data_for_embeddings(metadata, datafile_path):
	"""
	Prepare the data for embeddings.

	Args:
		metadata: Metadata
		datafile_path: Path to the datafile
		cut: Number of samples to cut the ECG to

	Returns:
		samples: Samples
		metadatas: Metadatas
		ids: IDs
	"""
	ids = []
	metadatas = []

	samples = torch.zeros(metadata.shape[0], 12*SAMPLES_IN_5_SECONDS_AT_100HZ)#, device=device)
	for i, patient in metadata.iterrows():
		samples[i, :] = get_ecg(patient['patient_id'], datafile_path)
		metadata_index = {
			"patient_id": patient['patient_id'],
			"label": "normal" if metadata.loc[metadata['patient_id'] == patient['patient_id'], 'diagnosis'].values[0] == 0 else "brugada",
			"label_numeric": int(metadata.loc[metadata['patient_id'] == patient['patient_id'], 'diagnosis'].values[0])
		}
		metadatas.append(metadata_index)
		ids.append(patient['patient_id'])

	return samples, metadatas, ids