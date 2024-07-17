import numpy as np
import matplotlib.pyplot as plt

# Generating a sample mass spectrum with a few peaks and a few small noise peaks around them
np.random.seed(0)
x = np.linspace(0, 100, 1000)
y = np.zeros_like(x)
peaks = [(20, 5), (40, 10), (60, 15), (80, 20)]

## generate random peaks
for i in range(20):
    x = np.random.randint(0, 100)
    y = np.random.randint(1, 10)
    ## if already have a peak at that location, skip
    if any(peak[0] == x for peak in peaks):
        continue
    peaks.append((x, y))

for peak in peaks:
    y += peak[1] * np.exp(-((x - peak[0]) ** 2) / (2 * 1 ** 2))

# Plotting the mass spectrum
plt.figure(figsize=(10, 6))

## plot the barplot for peaks
for peak in peaks:
    plt.bar(peak[0], peak[1], 0.5, color="purple", alpha=0.5)


plt.xlabel("Mass-to-charge ratio")
plt.ylabel("Intensity")

plt.savefig("/Users/ericliao/Desktop/dissertation/proposal defense/images/database_mass_spectrum_4.png", dpi=300)

