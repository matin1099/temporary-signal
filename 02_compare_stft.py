import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


# ==========================================
# Input
# ==========================================

data_path = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"

num_samples = 1_000_000

print("Reading IQ...")

iq = np.fromfile(
    data_path,
    dtype=np.complex64,
    count=num_samples
)

print(f"Loaded {len(iq):,} samples")


# ==========================================
# STFT configurations
# ==========================================

configs = [
    (2048, 1536),
    (4096, 3072),
    (8192, 6144),
]


# ==========================================
# Generate spectrograms
# ==========================================

for nperseg, noverlap in configs:

    print(f"Processing nperseg={nperseg}...")

    f, t, Zxx = stft(
        iq,
        fs=1.0,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )

    power = np.abs(Zxx) ** 2

    power_db = 10 * np.log10(
        power + 1e-12
    )

    # Shift negative frequencies
    f = np.fft.fftshift(f)

    power_db = np.fft.fftshift(
        power_db,
        axes=0
    )

    # ======================================
    # Plot
    # ======================================

    plt.figure(figsize=(14, 6))

    plt.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=[
            t[0],
            t[-1],
            f[0],
            f[-1]
        ]
    )

    plt.xlabel("Time")
    plt.ylabel("Normalized Frequency")

    plt.title(
        f"STFT Spectrogram — nperseg={nperseg}"
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.tight_layout()

    # ======================================
    # Save
    # ======================================

    output_path = (
        f"data/spectrograms/"
        f"compare_{nperseg}.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    print(
        f"Saved: {output_path}"
    )

    plt.show()
    plt.close()


print("Done.")