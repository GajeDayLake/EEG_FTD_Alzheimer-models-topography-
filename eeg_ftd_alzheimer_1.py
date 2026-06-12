
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
plt.xlim(-1,60)              # focus on first 60 Hz part, others are not shown
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
plt.ylim(0,30)                     # display up to 30 Hz 
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()


#%% 4.4) Time-frequency analysis: Scalogram

scales = np.linspace(8, 500, 96)
coefficients, cwt_frequencies = pywt.cwt(signal, scales, "gaus1", sampling_period=1 / fs)
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
4 criteria for the separation; 

1) Stratified sampling to maintain class balance in each set, 
2) Subject-wise separation to prevent data leakage, 
3) Randomization to ensure that the segments are randomly distributed across the sets.
4) The total number of segments in each set should be approximately 70%, 10%, and 20% of the total segments, respectively. 

Dataset splitting is handled by 2 steps:
    Train and temporary sets
    Temporary set is splitted into test and validation 
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

train_data = np.stack(train_data_list).astype(np.float32) if train_data_list else np.empty((0, segment_points, n_channels), dtype=np.float32)
val_data   = np.stack(val_data_list).astype(np.float32)   if val_data_list   else np.empty((0, segment_points, n_channels), dtype=np.float32)
test_data  = np.stack(test_data_list).astype(np.float32)  if test_data_list  else np.empty((0, segment_points, n_channels), dtype=np.float32)

# ---- reduced storage size by float64 to float32



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





# 5.7) Rescale the signal values

"""
MNE reads EEGLAB .set files in Volts; but the device sensitivity is 10 uV/mm (see dataset README)
so the native physical unit is microvolts. Multiplying by 1e6 brings the
signal into uV range (approx +-150 uV), which is the standard EEG amplitude scale.

Multiply with 1*10^6 for -> uV scale
Multiply with 1*10^3 for -> mV scale
"""
scale_factor = 1000                                             
train_data   = train_data * scale_factor
val_data     = val_data   * scale_factor
test_data    = test_data  * scale_factor





# 5.8) Delete temporary variables to free memory


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

 


#%% Optional 1: To check the IDs manually 


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



#%% 6) Data transformation for feature extraction: CWT scalograms



# CWT parameters 
scales_cwt        = np.linspace(8, 500, 60)                    
downsample_factor = 10                                         # downsample time axis: 30000 -> 3000
n_time_ds         = segment_points // downsample_factor        # time points after downsampling
n_scales_cwt      = len(scales_cwt)                            # frequency bins



def extract_cwt_scalogram(signal_1d, scales, fs=fs, downsample=downsample_factor):

    # Apply CWT on a single-channel signal 
    
    coefficients, _ = pywt.cwt(signal_1d, scales, "cmor-4.0-1.0", sampling_period=1 / fs)
    log_coefficients = np.log10(np.abs(coefficients) + 1)  
    
    return log_coefficients[:, ::downsample].astype(np.float32) 




# Prepare scalogram arrays for each split

train_scalograms = np.zeros((len(train_data), n_channels, n_scales_cwt, n_time_ds), dtype=np.float32)
val_scalograms   = np.zeros((len(val_data),   n_channels, n_scales_cwt, n_time_ds), dtype=np.float32)
test_scalograms  = np.zeros((len(test_data),  n_channels, n_scales_cwt, n_time_ds), dtype=np.float32)


print("Computing CWT scalograms — this may take several minutes...")

for i in range(len(train_data)):
    for ch in range(n_channels):
        train_scalograms[i, ch] = extract_cwt_scalogram(train_data[i, :, ch], scales_cwt)
    if (i + 1) % 50 == 0:
        print(f"  Train: {i + 1} / {len(train_data)}")

for i in range(len(val_data)):
    for ch in range(n_channels):
        val_scalograms[i, ch] = extract_cwt_scalogram(val_data[i, :, ch], scales_cwt)

for i in range(len(test_data)):
    for ch in range(n_channels):
        test_scalograms[i, ch] = extract_cwt_scalogram(test_data[i, :, ch], scales_cwt)


print("train_scalograms shape:", train_scalograms.shape)       
print("val_scalograms shape:  ", val_scalograms.shape)
print("test_scalograms shape: ", test_scalograms.shape)



# Expected size per segment: n_channels * n_scales_cwt * n_time_ds * 4 bytes (float32)
bytes_per_segment = n_channels * n_scales_cwt * n_time_ds * 4
total_segments    = len(train_scalograms) + len(val_scalograms) + len(test_scalograms)
total_gb          = bytes_per_segment * total_segments / 1e9

print(f"\nSize estimate: {bytes_per_segment / 1e6:.1f} MB/segment × {total_segments} segments = {total_gb:.1f} GB total")
print(f"  → To reduce: lower n_scales_cwt (currently {n_scales_cwt}) or increase downsample_factor (currently {downsample_factor})")



#%% Optional 2: Load from your disk 

# Optional: save to disk to skip recomputation next time
np.save(BASE_DIR / "train_scalograms.npy",     train_scalograms)
np.save(BASE_DIR / "val_scalograms.npy",       val_scalograms)
np.save(BASE_DIR / "test_scalograms.npy",      test_scalograms)
np.save(BASE_DIR / "train_segment_labels.npy", train_segment_labels)
np.save(BASE_DIR / "val_segment_labels.npy",   val_segment_labels)
np.save(BASE_DIR / "test_segment_labels.npy",  test_segment_labels)


"""
# Optional: load from disk (also restores label arrays needed for cell 7)
train_scalograms      = np.load(BASE_DIR / "train_scalograms.npy")
val_scalograms        = np.load(BASE_DIR / "val_scalograms.npy")
test_scalograms       = np.load(BASE_DIR / "test_scalograms.npy")
train_segment_labels  = np.load(BASE_DIR / "train_segment_labels.npy", allow_pickle=True)
val_segment_labels    = np.load(BASE_DIR / "val_segment_labels.npy",   allow_pickle=True)
test_segment_labels   = np.load(BASE_DIR / "test_segment_labels.npy",  allow_pickle=True)
"""


#%% 7) EfficientNet-B3: Training and evaluation


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns


print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")




# ----Hyperparameters
lr           = 0.001
batch_size   = 8
epochs       = 200
patience     = 40
dropout      = 0.1
weight_decay = 0.01



# Label encoding: A=0, F=1, C=2
label_map        = {"A": 0, "F": 1, "C": 2}
train_labels_int = np.array([label_map[l] for l in train_segment_labels], dtype=np.int64)
val_labels_int   = np.array([label_map[l] for l in val_segment_labels],   dtype=np.int64)
test_labels_int  = np.array([label_map[l] for l in test_segment_labels],  dtype=np.int64)



#  Convert to tensors and build DataLoaders
train_tensor = torch.tensor(train_scalograms, dtype=torch.float32)
val_tensor   = torch.tensor(val_scalograms,   dtype=torch.float32)
test_tensor  = torch.tensor(test_scalograms,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor, torch.tensor(train_labels_int)), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_tensor,   torch.tensor(val_labels_int)),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(test_tensor,  torch.tensor(test_labels_int)),  batch_size=batch_size, shuffle=False)



# ----Model: EfficientNet from scratch
"""
Accepts (batch, in_channels, H, W) for any spatial size via AdaptiveAvgPool2d
Width=1.2, Depth=1.4 multipliers over EfficientNet-B0 base
"""


# Function definition 1
class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""
    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)                      # channel-wise attention



# Function definition 2
class MBConv(nn.Module):
    """Mobile inverted bottleneck: depthwise separable conv + SE + SiLU + stochastic depth"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, se_ratio=0.25, sd_prob=0.0):
        super().__init__()
        mid_ch        = in_ch * expand_ratio
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.sd_prob  = sd_prob

        layers = []
        if expand_ratio != 1:                                  # expansion phase (skipped in stage 1)
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
                       nn.SiLU()]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_ch, bias=False),   # depthwise
            nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
            nn.SiLU(),
            SEBlock(mid_ch, max(1, int(in_ch * se_ratio))),    # squeeze dim uses in_ch
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),          # pointwise projection
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            if self.training and self.sd_prob > 0.0:           # stochastic depth: drop residual randomly
                keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) > self.sd_prob
                out  = out * keep.float()
            return x + out
        return out





# Model definition
class EfficientNetB0(nn.Module):
    def __init__(self, in_channels=19, num_classes=3, dropout=0.02):
        super().__init__()
        w, d = 1.0, 1.0                                        # B0 multipliers: smaller capacity 

        def round_ch(c, div=8):                                
            new_c = max(div, int(c + div / 2) // div * div)
            if new_c < 0.9 * c:
                new_c += div
            return new_c

        stem_ch = round_ch(32 * w)                             
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch, momentum=0.01, eps=1e-3),
            nn.SiLU(),
        )

        # (expand_ratio, base_out_ch, kernel_size, stride, base_num_layers)
        stage_cfg = [
            (1,  16,  3, 1, 1),
            (6,  24,  3, 2, 2),
            (6,  40,  5, 2, 2),
            (6,  80,  3, 2, 3),
            (6, 112,  5, 1, 3),
            (6, 192,  5, 2, 4),
            (6, 320,  3, 1, 1),
        ]

        total_blocks = sum(max(1, int(np.ceil(n * d))) for *_, n in stage_cfg)
        block_count  = 0
        in_ch        = stem_ch
        blocks       = []

        for expand, base_out, k, stride, base_n in stage_cfg:
            out_ch = round_ch(base_out * w)
            n      = max(1, int(np.ceil(base_n * d)))
            for i in range(n):
                sd_prob = 0.2 * block_count / total_blocks     # linear stochastic depth schedule
                blocks.append(
                    MBConv(in_ch, out_ch, k, stride if i == 0 else 1, expand, sd_prob=sd_prob)
                )
                in_ch        = out_ch
                block_count += 1

        self.blocks = nn.Sequential(*blocks)

        head_ch = round_ch(1280 * w)                           
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch, momentum=0.01, eps=1e-3),
            nn.SiLU(),
        )

        # Adaptive average pooling - works with any spatial size 
        self.pool       = nn.AdaptiveAvgPool2d(1)           
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(head_ch, num_classes),
        )
        
        
        # Kaiming initialization to initialize the weights of model
        for m in self.modules():                               
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



# Assign this model to use in the code
model = EfficientNetB0(in_channels=n_channels, num_classes=3, dropout=dropout).to(device)


# ----Loss function, optimizer, and cosine LR scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)           # label smoothing can reduce overconfident predictions
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)




#  Training and validation loop
train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []

best_val_loss          = float("inf")       # dummy loss value to decide it is improving 
best_model_state       = None
early_stopping_counter = 0



for epoch in range(epochs):

    # Training phase
    model.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Gradient clipping: prevent gradient spikes

        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_loss     = running_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)


    # Validation phase
    model.eval()
    val_running_loss = 0.0
    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs  = model(inputs)
            loss     = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_loss     = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(y_true_val, y_pred_val) * 100
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    scheduler.step()

    # Early stopping: save best weights based on validation loss  
    if val_loss < best_val_loss:                                  
        best_val_loss          = val_loss                         
        best_model_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    print(f"Epoch {epoch+1}/{epochs}  |  Train Loss: {train_loss:.4f}  Train Acc: {train_accuracy:.2f}%  |  Val Loss: {val_loss:.4f}  Val Acc: {val_accuracy:.2f}%")




# Restore best model weights
print(f"\nRestoring best model — Val Loss: {best_val_loss:.4f}")   
best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
model.load_state_dict(best_model_state)





# Validation set evaluation — use this to compare experiments
model.eval()
y_true_val  = []
y_pred_val  = []
y_proba_val = []

with torch.no_grad():
    for inputs, labels in val_loader:                         # val_loader, not test_loader
        inputs, labels = inputs.to(device), labels.to(device)
        outputs      = model(inputs)
        proba        = torch.softmax(outputs, dim=1)           # probabilities for ROC
        _, predicted = torch.max(outputs, 1)
        y_true_val.extend(labels.cpu().numpy())
        y_pred_val.extend(predicted.cpu().numpy())
        y_proba_val.extend(proba.cpu().numpy())

y_true_val  = np.array(y_true_val)
y_pred_val  = np.array(y_pred_val)
y_proba_val = np.array(y_proba_val)


# Val classification metrics
acc_val  = accuracy_score(y_true_val, y_pred_val)
f1_val   = f1_score(y_true_val, y_pred_val, average="weighted")
prec_val = precision_score(y_true_val, y_pred_val, average="weighted", zero_division=0)
rec_val  = recall_score(y_true_val, y_pred_val, average="weighted")

print(f"\nVal Accuracy:  {acc_val  * 100:.2f}%")
print(f"Val F1:        {f1_val   * 100:.2f}%")
print(f"Val Precision: {prec_val * 100:.2f}%")
print(f"Val Recall:    {rec_val  * 100:.2f}%")


# Val confusion matrix
cm_val = confusion_matrix(y_true_val, y_pred_val)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues",
            xticklabels=["AD", "FTD", "HC"],
            yticklabels=["AD", "FTD", "HC"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Validation")
plt.tight_layout()
plt.show()


# Training and validation loss / accuracy curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Validation")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train")
plt.plot(val_accuracies,   label="Validation")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()


# Val ROC curves (one-vs-rest)
class_names_roc = ["AD (A)", "FTD (F)", "HC (C)"]
y_true_val_bin  = label_binarize(y_true_val, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_val_bin[:, i], y_proba_val[:, i])
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names_roc[i]}  AUC = {auc_score:.3f}")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest) - Validation")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()



#%% 8) Final test evaluation — after all experiments are done




model.eval()
y_true_test  = []
y_pred_test  = []
y_proba_test = []

with torch.no_grad():
    for inputs, labels in test_loader:                         
        inputs, labels = inputs.to(device), labels.to(device)
        outputs      = model(inputs)
        proba        = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())
        y_proba_test.extend(proba.cpu().numpy())

y_true_test  = np.array(y_true_test)
y_pred_test  = np.array(y_pred_test)
y_proba_test = np.array(y_proba_test)


# Test classification metrics
acc_test  = accuracy_score(y_true_test, y_pred_test)
f1_test   = f1_score(y_true_test, y_pred_test, average="weighted")
prec_test = precision_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
rec_test  = recall_score(y_true_test, y_pred_test, average="weighted")

print(f"\nTest Accuracy:  {acc_test  * 100:.2f}%")
print(f"Test F1:        {f1_test   * 100:.2f}%")
print(f"Test Precision: {prec_test * 100:.2f}%")
print(f"Test Recall:    {rec_test  * 100:.2f}%")


# Test confusion matrix
cm_test = confusion_matrix(y_true_test, y_pred_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=["AD", "FTD", "HC"],
            yticklabels=["AD", "FTD", "HC"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Test")
plt.tight_layout()
plt.show()


# Test ROC curves
y_true_test_bin = label_binarize(y_true_test, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_test_bin[:, i], y_proba_test[:, i])
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names_roc[i]}  AUC = {auc_score:.3f}")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest) - Test")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()




"""
model can optionally be saved here:
torch.save(best_model_state, "best_model.pth")
best_model_state = torch.load("best_model.pth", map_location=device)
model.load_state_dict(best_model_state)
"""
