import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


# =========================
# Settings
# =========================

input_path = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"

output_path = r"data\spectrograms\chunk_0000.png"

sample_rate = 1.0

chunk_size = 1_000_000

nperseg = 4096
noverlap = 3072


# =========================
# Read IQ data
# =========================

print("Reading IQ data...")

iq = np.fromfile(
    input_path,
    dtype=np.complex64,
    count=chunk_size
)

print(f"Loaded {len(iq):,} samples")


# =========================
# Compute STFT
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

f = np.fft.fftshift(f)

power_db = np.fft.fftshift(
    power_db,
    axes=0
)


# =========================
# Fixed dynamic range
# =========================

vmax = np.percentile(power_db, 99)
vmin = vmax - 60

print(f"Display range: {vmin:.2f} to {vmax:.2f} dB")


# =========================
# Create clean image
# =========================

fig = plt.figure(
    figsize=(10, 10),
    dpi=150
)

ax = fig.add_axes([0, 0, 1, 1])

ax.imshow(
    power_db,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)

ax.axis("off")


# =========================
# Save image
# =========================

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
    pad_inches=0
)

plt.close(fig)

print()
print("Saved:")
print(output_path)