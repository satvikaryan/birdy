import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Directory containing the .wav files
directory1 = "data/dot/"
directory2 = "data/dash/"

# Create images directory if it doesn't exist
images_directory = "data/images/"
dot_images_directory = os.path.join(images_directory, "dot")
dash_images_directory = os.path.join(images_directory, "dash")

# Create dot and dash images directories if they don't exist
for directory in [dot_images_directory, dash_images_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# List all .wav files in the directories
wav_files1 = [file for file in os.listdir(directory1) if file.endswith(".wav")]
wav_files2 = [file for file in os.listdir(directory2) if file.endswith(".wav")]

def plot_spectrogram(y, sr, hop_length, filename, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(filename)  # Save the plot as an image
    plt.close()

# Process each .wav file in directory1 (dot)
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
    
    # Define filename for saving the image in dot images directory
    image_filename = os.path.join(dot_images_directory, f"{wav_file[:-4]}.png")
    
    # Plot and save spectrogram
    plot_spectrogram(y_audio_log, sample_rate, hopSize, image_filename)
    
    # Play audio
    display(Audio(audio_data))

# Process each .wav file in directory2 (dash)
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
    
    # Define filename for saving the image in dash images directory
    image_filename = os.path.join(dash_images_directory, f"{wav_file[:-4]}.png")
    
    # Plot and save spectrogram
    plot_spectrogram(y_audio_log, sample_rate, hopSize, image_filename)
    
    # Play audio
    display(Audio(audio_data))


