import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

file_path = r"C:\Users\ASUS\Desktop\SPBZ0\UrbanSound8K\audio\fold7\21683-9-0-3.wav"

audio, sr = librosa.load(file_path, sr=22050)

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

plt.figure(figsize=(8, 4))
librosa.display.specshow(mfcc, x_axis='time', cmap='magma', vmax=100, vmin=-100)

plt.colorbar(label='Amplitude')
plt.title("MFCC Feature Representation", fontsize=12)
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")

plt.tight_layout()

plt.savefig("mfcc_real.png", dpi=300)
plt.show()