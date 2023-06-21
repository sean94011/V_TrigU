import pyaudio
import numpy as np

# Set up PyAudio
p = pyaudio.PyAudio()

# freqs = [1000, 1100, 1200, 1300]
freqs = [1000]
duration = 20

# Set the sample rate and duration
sample_rate = 44100  # standard audio sample rate
num_frames = int(sample_rate * duration)

# Generate the sine wave
t = np.linspace(0, duration, num_frames, endpoint=False)
wave = np.zeros_like(t)
for freq in freqs:
    wave += np.sin(2 * np.pi * freq * t)
wave /= len(freqs)

# Convert the wave array to bytes
wave = (wave * 32767).astype(np.int16)
wave_bytes = wave.tobytes()

# Open a new stream and play the sound
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True)
stream.write(wave_bytes)

# Clean up
stream.stop_stream()
stream.close()
p.terminate()

