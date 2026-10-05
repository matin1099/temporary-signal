import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


# =========================
# Settings
# =========================

input_path = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"

sample_rate = 1.0

chunk_size = 1_000_000

nperseg = 4096
noverlap = 3072


# =========================
# Read one chunk
# =========================

print("Reading IQ data...")

iq = np.fromfile(
    input_path,
    dtype=np.complex64,
    count=chunk_size
)

print(f"Loaded samples: {len(iq):,}")


# =========================
# STFT
# =========================

print("Computing STFT...")

f, t, Zxx = stft(
    iq,
    fs=sample_rate,
    nperseg=nperseg,
    noverlap=noverlap,
    return_onesided=False
)


# =========================
# Power spectrogram
# =========================

power = np.abs(Zxx) ** 2

power_db = 10 * np.log10(power + 1e-12)


# =========================
# Shift frequency
# =========================

f_shifted = np.fft.fftshift(f)
power_db_shifted = np.fft.fftshift(power_db, axes=0)


# =========================
# Print information
# =========================

print()
print("===== CHUNK INFORMATION =====")
print(f"Samples       : {len(iq):,}")
print(f"Sample rate   : {sample_rate}")
print(f"nperseg       : {nperseg}")
print(f"noverlap      : {noverlap}")
print(f"STFT shape    : {power_db_shifted.shape}")
print(f"Frequency bins: {len(f_shifted)}")
print(f"Time bins     : {len(t)}")


# =========================
# Plot
# =========================

plt.figure(figsize=(12, 7))

plt.pcolormesh(
    t,
    f_shifted,
    power_db_shifted,
    shading="auto",
    cmap="viridis"
)

plt.xlabel("Time")
plt.ylabel("Normalized Frequency")
plt.title("Test Chunk - STFT 4096 / 3072")

plt.colorbar(label="Power (dB)")

plt.tight_layout()

plt.show()