
"""
Classification of subjects with Alzheimer's disease,
Frontotemporal dementia and Healthy controls by EEG signals.


04/2026

----------------------------------------

Data Description:

    19 channel EEG signals sampled at 500Hz
    Total duration of signals varies among subjects

    Resting state & closed eyes recordings are captured by bipolar montage,

    3 classes, 88 subjects in total
    36 Alzheimer's, 23 Frontotemporal Dementia, all evaluated by Mini-Mental State Examination (MMSE)
    29 are healthy controls

    Channel names: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2

    Pre-processed signals are located in the folder "derivatives"

    Their pre-processing steps:

        1) BPF 0.5 Hz - 45 Hz
        2) Artifact Subspace Reconstruction routine (ASR) [EEG artifact correction method]
        3) Independent Component Analysis (ICA) to remove artefacts

    Signals are located in ".set" files, while labels are written in "participants.tsv"

    Dataset link: https://doi.org/10.18112/openneuro.ds004504.v1.0.8
"""

#%% 1) Imports


from pathlib import Path
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import stft, welch



#%% 2) Data import and labels



# 2.1) Functions to identify the base directory address 
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
WORKSPACE_DIR = BASE_DIR.parent if BASE_DIR.name == "EEG_FTD_Alzheimer" else BASE_DIR



# 2.2) Directories
DATASET_DIR = WORKSPACE_DIR / "ds004504"
DERIVATIVES_DIR = DATASET_DIR / "derivatives"
PARTICIPANTS_FILE = DATASET_DIR / "participants.tsv"



# 2.3) Dictionary to match the data with labels
group_map = {
    "A": "Alzheimer Disease Group",
    "F": "Frontotemporal Dementia Group",
    "C": "Healthy Group",
}


"""
Groups were written as A,F,C corresponding to the class names. 
These names will be gathered and made readable in the dataframe.
"""



participants_df = pd.read_csv(PARTICIPANTS_FILE, sep="\t")                   # collecting participant (subject) information
participants_df = participants_df.rename(columns={"Group": "group_code"})    # column name change
participants_df["group_name"] = participants_df["group_code"].map(group_map) # new column to match them with labels




# 2.4) Loop to gather the data

records = []

for set_path in sorted(DERIVATIVES_DIR.glob("sub-*/eeg/*.set")):            # look into this directory, by this structure 
    
    participant_id = set_path.parent.parent.name                            # get sub-0xx info
    
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")   # read .set file by using MNE library. 
                                                                            # EEGLAB object, it includes relevant methods and information.


    signals_df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)       # gather 19 channel signals (with column names) into a dataframe 
    
    signals_df.insert(0, "time_sec", raw.times)                             # add time information, will be used to divide the signals


    # to collect all types of information into a list
    
    records.append(                                                         
        {
            "participant_id": participant_id,
            "file_path": str(set_path),
            "sfreq": raw.info["sfreq"],
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
            "n_times": raw.n_times,
            "duration_sec": raw.n_times / raw.info["sfreq"],
            "signals_df": signals_df,
        }
    )



# 2.5) Generate the new dataframe to gather all the information and data together.


eeg_df = pd.DataFrame(records)                                           # convert this ordered list into dataframe

eeg_df = eeg_df.merge(participants_df, on="participant_id", how="left")  # order this dataframe by ID


del records # free 2.73 GB space from the env



#%% 3) EDA: Basic plots


# Selection of the subject
participant_id = "sub-001"
channel_name = "F4"


# 3.1) Select the entire row of eeg_df for selected subject

subject_row = eeg_df.loc[eeg_df["participant_id"] == participant_id].iloc[0]  
 

# 3.2) Extract the full signal for that subject, column names are the channels
subject_signals = subject_row["signals_df"]         
time_sec = subject_signals["time_sec"].to_numpy()


# 3.3) Get the column names
channels = subject_row["channel_names"]


print(
    f"{participant_id} | {subject_row['group_name']} | "
    f"{subject_row['duration_sec']:.2f} second duration | {subject_row['sfreq']} Hz | "
    f"{subject_row['Age']} age | {subject_row['Gender']} gender | {subject_row['MMSE']} MMSE score "
)



# 3.4) Plot all channels of one participant

fig, axes = plt.subplots(19, 1, figsize=(76,38), sharex=True)                # generate subplots

for ax, channel in zip(axes, channels):                                      # select by channel names
    ax.plot(time_sec, subject_signals[channel].to_numpy(), linewidth=0.6)    # extract signal arrays
    ax.set_ylabel(channel)
    ax.set_ylim(-0.00015, 0.00015)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"All EEG channels | {participant_id} | {subject_row['group_name']}", y=1.0)
plt.tight_layout()
plt.show()


#%% 4) Frequency analyses for one channel

signal = subject_signals[channel_name].to_numpy()
fs = 500


# 4.1) Time domain

plt.figure(figsize=(57, 10))
plt.plot(time_sec, signal, linewidth=0.8)
plt.title(f"Time series | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()



# 4.2) Frequency domain: FFT

fft_values = np.abs(fft(signal))
frequencies = fftfreq(len(signal), d=1 / fs)   # linear x-axis for frequencies
positive_mask = frequencies >= 0               # one-sided

frequencies = frequencies[positive_mask]
fft_values = fft_values[positive_mask]
power_db = 10 * np.log10(fft_values**2 + 1e-12)


# Linear plot with magnitudes
plt.figure(figsize=(20, 10))
plt.plot(frequencies, fft_values)
plt.title(f"FFT magnitude | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(-1,60)              # focus on first 60 Hz part, others are filtered
plt.tight_layout()
plt.show()


# Linear plot with (logarithmic) power
plt.figure(figsize=(20, 10))
plt.plot(frequencies, power_db)
plt.title(f"Log power spectrum | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.xlim(-1,60) 
plt.tight_layout()
plt.show()


#%% 4.2) Frequency domain: Periodogram
    
nperseg = 4096
noverlap = nperseg // 4

freq_psd, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")

plt.figure(figsize=(15, 5))
plt.plot(freq_psd, psd)
plt.title(f"Periodogram (Welch PSD) | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(-1,60)  
plt.tight_layout()
plt.show()


#%% 4.3) Time-frequency analysis: Spectrogram

freq_stft, time_stft, zxx = stft(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

plt.figure(figsize=(20, 15))
plt.pcolormesh(time_stft, freq_stft, np.abs(zxx), shading="gouraud")
plt.title(f"Spectrogram | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylim(0,60) 
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()


#%% 4.4) Time-frequency analysis: Scalogram

scales = np.linspace(8, 500, 96)
coefficients, cwt_frequencies = pywt.cwt(signal, scales, "cmor2.0-1.0", sampling_period=1 / fs)
log_coefficients = np.log10(np.abs(coefficients) + 1)

sort_index = np.argsort(cwt_frequencies)
cwt_frequencies = cwt_frequencies[sort_index]
log_coefficients = log_coefficients[sort_index]

plt.figure(figsize=(20, 15))
plt.imshow(
    log_coefficients,
    extent=[time_sec[0], time_sec[-1], cwt_frequencies[0], cwt_frequencies[-1]],
    aspect="auto",
    origin="lower",
    cmap="viridis",
)
plt.title(f"Scalogram | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Log10(Magnitude + 1)")
plt.tight_layout()
plt.show()



#%% 5) Dataset preparation: Splitting and segmentation


"""
eeg_df is a large dataframe that includes 13 columns:

First column is "participant_id" lists 88 subjects with sub-XXX naming convention.

Second column is "file_path" that includes the path of the .set file for each subject, not so important for now.

Third column is "sfreq" that includes the sampling frequency of the signals, which is 500Hz for all subjects. No difference.

Fourth column is "n_channels" that includes the number of channels, which is 19 for all subjects. No difference.

Fifth column is "channel_names" that includes the names of the channels, which are the same for all subjects.

The channel names are: ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'] Their order is same for all, no need to check them. 

Next two columns are important for the division of signals: "n_times" and "duration_sec".

"n_times" is the total number of time points in the signal, which varies among subjects.

"duration_sec" is the total duration of the signal in seconds, which also varies among subjects. It is calculated by dividing "n_times" by "sfreq".

8th column "signals_df" includes the actual EEG signals for corresponding subject, which is another dataframe. 

Each "signals_df" has 20 columns: 1 "time_sec" + 19 EEG channels. "time_sec" corresponding timestamp in seconds for each time point, which is important for the division of signals.

It starts with 0 and ends with the total duration of the signal in seconds, for that subject. The number of rows in "signals_df" is equal to "n_times" for that subject, which is the total number of time points in the signal.

19 EEG channels follow the same channel order as "channel_names". They include our real signals. 

9th column is "Gender" that includes the gender of the subjects. Not important for now.

10th column is "Age" that includes the age of the subjects. Not important for now. 

11th column is "group_code" that includes the group labels in A, F, C format. Their meanings are explained in the last column "group names". They represent our labels.

12th column is "MMSE" that includes the Mini-Mental State Examination scores of the subjects. Not important for now.

13th column is "group_name" that includes the group labels in a readable format.


Your task is to divide the signals into equal parts, for example 1 minute (60 seconds, 60*500=30000 points) segments, and assign the corresponding labels and subject IDs to each segment.

Assignment of subject IDs is crucial to prevent data leakage. 

All the signals are located in the "signals_df" column, which is a dataframe for each subject.

You should be minimalist. You can design a simple for loop with if-else statement to obtain these 1 minute segments, discarding the few remaining points.

All of the resultant data must saved in a NumPy array, not a dataframe anymore. First column will be subject ID. Second column will be the label (A, F, C). 

I will use ordinal encoding for the labels, where A=0, F=1, C=2. I can apply this in a different block of code. 

We will use this NumPy array for train - val - test separation with stratified sampling. We will use subject IDs to ensure that all segments from the same subject are in the same set (train, val, or test) to prevent data leakage.

Train will be 70%, validation will be 10½, and test will be 20½ of the total segments.

As durations of signals vary among each subject, we have different numbers of segments attached to each subject. 

For example, if a subject has 5 minutes of signal, we can obtain 5 segments of 1 minute each. If another subject has 3 minutes of signal, we can obtain 3 segments of 1 minute each. 

This makes the prevention of data leakage in data splitting even harder to implement. We can try to design a minimalist function to handle this separation. 

4 criteria for the separation; 
1) Stratified sampling to maintain class balance in each set, 
2) Subject-wise separation to prevent data leakage, 
3) Randomization to ensure that the segments are randomly distributed across the sets.
4) The total number of segments in each set should be approximately 70%, 10½, and 20½ of the total segments, respectively. 

Only for this complex operation, you can define a minimalist custom function. 

In other cases, do not apply any abstraction, just use simple for loops and if-else statements to achieve the task. 

Follow the style and minimalism of the previous codes in this file. 

Add concise and simple comments to explain each line. 
"""


# 5.1) Set the segment (epoch) length
segment_sec = 60
segment_points = int(segment_sec * 500)                         # 60 sec * 500 Hz = 30000 points
n_channels = 19                                                 # 19 EEG channels




# 5.2) Split the subjects first, by stratified random assignment

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Extract subject IDs and their class labels
subject_ids = eeg_df["participant_id"].to_numpy()              # 88 subject IDs
subject_labels = eeg_df["group_code"].to_numpy()               # 88 labels (A, F, or C)



# First split: 70% train, 30% temporary (which will become val + test in the next part)
train_subject_ids, temp_subject_ids, _, _ = train_test_split(
    subject_ids,
    subject_labels,
    test_size=0.30,
    stratify=subject_labels,             # stratify respect to labels, for maintaining class balance
    random_state=42,
)




# Second split: divide 30% temp into 10% val and 20% test
temp_labels = eeg_df.loc[eeg_df["participant_id"].isin(temp_subject_ids), "group_code"].to_numpy()

val_subject_ids, test_subject_ids, _, _ = train_test_split(
    temp_subject_ids,
    temp_labels,
    test_size=2 / 3,                                            # 2/3 of 30% = 20% test
    stratify=temp_labels,
    random_state=42,
)



# 5.3) Generate containers for segments in each split

train_data_list = []                # will hold (30000, 19) segment signals
val_data_list = []
test_data_list = []

train_segment_ids = []              # will hold subject IDs per segment
val_segment_ids = []
test_segment_ids = []

train_segment_labels = []           # will hold labels per segment (respect to subject ID)
val_segment_labels = []
test_segment_labels = []



# Convert IDs to sets for membership checking during loop (faster)

train_subject_id_set = set(train_subject_ids)                   # {sub-042, sub-061, ...}
val_subject_id_set = set(val_subject_ids)
test_subject_id_set = set(test_subject_ids)




# 5.4) Iterate all 88 subjects, segment each signal, and route to correct split


for _, subject_row in eeg_df.iterrows():    # look for every row of this dataframe

    
    # Extract current subject's information
    participant_id = subject_row["participant_id"]
    group_code = subject_row["group_code"]
    channel_names = subject_row["channel_names"]
    signals = subject_row["signals_df"][channel_names].to_numpy()   # shape: (total_points, 19)
    n_segments = subject_row["n_times"] // segment_points           # determination of total segment count


    # Determine the split of this subject belongs to (train, test, or val)
    
    if participant_id in train_subject_id_set:
        
        target_data_list = train_data_list
        target_id_list = train_segment_ids
        target_label_list = train_segment_labels
        
    elif participant_id in val_subject_id_set:
        
        target_data_list = val_data_list
        target_id_list = val_segment_ids
        target_label_list = val_segment_labels
        
    else:
        target_data_list = test_data_list
        target_id_list = test_segment_ids
        target_label_list = test_segment_labels


    # Prepare all 60-second segments for this subject
    
    for segment_index in range(n_segments):  # total segment count we determined above  (9, 13 etc.)


        # Extract 60*500 points in order
        start = segment_index * segment_points   # index * 30000  (first index=0)
        end = start + segment_points
        segment = signals[start:end, :]          # extract this segment from subject's signals (all 19 channels)     
        
        
        # shape of each segment: (30000, 19)


        # Append segment and metadata to the appropriate split
        target_data_list.append(segment)
        target_id_list.append(participant_id)
        target_label_list.append(group_code)


# We have obtained all the segments/epochs in a list




# 5.5) Convert lists to arrays (3D tensors)
# (n_segments, 30000, 19)

train_data = np.stack(train_data_list) if train_data_list else np.empty((0, segment_points, n_channels))
val_data = np.stack(val_data_list) if val_data_list else np.empty((0, segment_points, n_channels))
test_data = np.stack(test_data_list) if test_data_list else np.empty((0, segment_points, n_channels))


# segment IDs to arrays 
train_segment_ids = np.array(train_segment_ids)
val_segment_ids = np.array(val_segment_ids)
test_segment_ids = np.array(test_segment_ids)


# label arrays 
train_segment_labels = np.array(train_segment_labels)
val_segment_labels = np.array(val_segment_labels)
test_segment_labels = np.array(test_segment_labels)



# 5.6) Final shuffle on segments in each split (prevent order bias & obtain better mixture)

train_data, train_segment_ids, train_segment_labels = shuffle(
    train_data,
    train_segment_ids,
    train_segment_labels,
    random_state=42,              # reproducibility by RNG
)

val_data, val_segment_ids, val_segment_labels = shuffle(
    val_data,
    val_segment_ids,
    val_segment_labels,
    random_state=42,
)

test_data, test_segment_ids, test_segment_labels = shuffle(
    test_data,
    test_segment_ids,
    test_segment_labels,
    random_state=42,
)



print("train_data shape:", train_data.shape)
print("val_data shape:", val_data.shape)
print("test_data shape:", test_data.shape)



# 5.7) Delete temporary variables to free memory


del (
    subject_ids,
    subject_labels,
    temp_subject_ids,
    temp_labels,
    train_subject_ids,
    val_subject_ids,
    test_subject_ids,
    train_subject_id_set,
    val_subject_id_set,
    test_subject_id_set,
    train_data_list,
    val_data_list,
    test_data_list,
    eeg_df
)

 


#%% Optional: To check the IDs manually 


# Any ID cannot be occured twice, to prevent data leakage & provide subject separation
# For manual control, arrays can be saved as Excel files


desktop = Path.home() / "Desktop"    # where you want to save


# Convert to string for clean Excel output
train_ids = np.asarray(train_segment_ids).astype(str)
train_labels = np.asarray(train_segment_labels).astype(str)

val_ids = np.asarray(val_segment_ids).astype(str)
val_labels = np.asarray(val_segment_labels).astype(str)

test_ids = np.asarray(test_segment_ids).astype(str)
test_labels = np.asarray(test_segment_labels).astype(str)


# TRAIN
df_train = pd.DataFrame({"segment_id": train_ids, "segment_label": train_labels})
df_train.to_excel(desktop / "train_segment_ids_and_labels.xlsx", index=False)

# VAL
df_val = pd.DataFrame({"segment_id": val_ids, "segment_label": val_labels})
df_val.to_excel(desktop / "val_segment_ids_and_labels.xlsx", index=False)

# TEST
df_test = pd.DataFrame({"segment_id": test_ids, "segment_label": test_labels})
df_test.to_excel(desktop / "test_segment_ids_and_labels.xlsx", index=False)


print("Saved Excel files to:", desktop)



#%% 6) Data transformation for model input 






