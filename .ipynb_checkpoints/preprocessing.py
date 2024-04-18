import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Directory containing the .wav files
directory1 = "../data/dot/"
directory2 = "../data/dash/"
# List all .wav files in the directory
wav_files1 = [file for file in os.listdir(directory1) if file.endswith(".wav")]
wav_files2 = [file for file in os.listdir(directory2) if file.endswith(".wav")]


def plot_spectrogram(y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

# Process each .wav file
for wav_file in wav_files1:
    audio_data = os.path.join(directory1, wav_file)
    
    # Load audio file
    audio, sample_rate = librosa.load(audio_data)
    
    # Define frameSize and hopSize
    frameSize = 2048
    hopSize = 512
    
    # Compute STFT
    stft_audio = librosa.stft(audio, n_fft=frameSize, hop_length=hopSize)
    
    # Compute magnitude squared of STFT
    y_audio = np.abs(stft_audio) ** 2
    y_audio_log = librosa.power_to_db(y_audio)
    
    # Plot spectrogram
    plot_spectrogram(y_audio_log, sample_rate, hopSize)
    
    # Play audio
    display(Audio(audio_data))

for wav_file in wav_files2:
    audio_data = os.path.join(directory2, wav_file)
    
    # Load audio file
    audio, sample_rate = librosa.load(audio_data)
    
    # Define frameSize and hopSize
    frameSize = 2048
    hopSize = 512
    
    # Compute STFT
    stft_audio = librosa.stft(audio, n_fft=frameSize, hop_length=hopSize)
    
    # Compute magnitude squared of STFT
    y_audio = np.abs(stft_audio) ** 2
    y_audio_log = librosa.power_to_db(y_audio)
    
    # Plot spectrogram
    plot_spectrogram(y_audio_log, sample_rate, hopSize)
    
    # Play audio
    display(Audio(audio_data))
