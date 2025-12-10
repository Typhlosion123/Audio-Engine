import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import os

# --- CONFIGURATION ---
# Input File (Set to None to use the synthetic drone)
# You can use .wav, .mp3, .mp4, etc.
INPUT_MEDIA_FILE = "songs/westside.mp4" 
MAX_DURATION_SEC = 60.0

# Must match your C++ simulation
LISTENER_POS = np.array([-24.0, 0.0, 0.0]) 
LISTENER_RADIUS = 2.0
SPEED_OF_SOUND = 343.0 # m/s
SAMPLE_RATE = 44100    # Hz

# Stereo Settings
LISTENER_RIGHT_VEC = np.array([0.0, 1.0, 0.0]) 

# --- PHYSICS PARAMETERS ---
WALL_THICKNESS = 0.5
WALL_REFLECT_LOW  = 0.5 + (0.49 * WALL_THICKNESS)
WALL_REFLECT_HIGH = 0.1 + (0.50 * WALL_THICKNESS)
AIR_ABSORPTION_LOW  = 0.005
AIR_ABSORPTION_HIGH = 0.100
MUFFLE_CROSSOVER_HZ = 800

def generate_dry_sound(duration_sec=5.0, freq=110):
    """Fallback: Generates a rich Sawtooth drone."""
    print("Generating synthetic sawtooth drone...")
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    audio = signal.sawtooth(2 * np.pi * freq * t)
    
    fade_samples = int(0.1 * SAMPLE_RATE)
    if len(audio) > 2 * fade_samples:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
    return audio

def load_audio_source(filename, max_duration):
    """
    Loads audio from a file (mp3, wav, mp4), resamples it, 
    mixes to mono, and trims it.
    """
    if filename is None:
        return generate_dry_sound()

    try:
        import librosa
    except ImportError:
        print("⚠️  Library 'librosa' not found. Cannot load media files.")
        print("👉 Run: pip install librosa")
        return generate_dry_sound()

    if not os.path.exists(filename):
        print(f"⚠️  File '{filename}' not found. Using fallback drone.")
        return generate_dry_sound()

    print(f"Loading '{filename}'...")
    # librosa.load automatically handles resampling (sr) and mixing to mono
    audio, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True, duration=max_duration)
    
    print(f"Loaded {len(audio)/SAMPLE_RATE:.2f} seconds of audio.")
    return audio

def intersect_segment_sphere(p1, p2, center, radius):
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    if a < 1e-8: return False, None
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b*b - 4*a*c
    if discriminant < 0: return False, None
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)
    if 0 <= t1 <= 1: return True, t1 * np.linalg.norm(d)
    if 0 <= t2 <= 1: return True, t2 * np.linalg.norm(d)
    return False, None

def main():
    csv_path = "../build/paths.csv"
    if not os.path.exists(csv_path):
        print("Error: ../build/paths.csv not found.")
        return

    print("Loading ray paths...")
    df = pd.read_csv(csv_path)
    
    hit_ray_ids = df[df['hit_listener'] == 1]['ray_id'].unique()
    print(f"Processing {len(hit_ray_ids)} rays that touched the listener...")

    # Load Source Audio
    dry_signal = load_audio_source(INPUT_MEDIA_FILE, MAX_DURATION_SEC)
    
    # Calculate IR Length based on signal length + reverb tail (e.g., 2.0s)
    signal_len = len(dry_signal)
    tail_len = int(2.0 * SAMPLE_RATE) 
    total_len = signal_len + tail_len
    
    ir_low  = np.zeros((total_len, 2))
    ir_high = np.zeros((total_len, 2))

    for rid in hit_ray_ids:
        ray_data = df[df['ray_id'] == rid].sort_values('bounce_id')
        coords = ray_data[['x', 'y', 'z']].values
        
        total_distance = 0.0
        energy_low = 1.0
        energy_high = 1.0
        
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            segment_len = np.linalg.norm(p2 - p1)
            if segment_len < 1e-6: continue

            hit, dist_on_seg = intersect_segment_sphere(p1, p2, LISTENER_POS, LISTENER_RADIUS)
            
            if hit:
                final_dist = total_distance + dist_on_seg
                arrival_time = final_dist / SPEED_OF_SOUND
                sample_index = int(arrival_time * SAMPLE_RATE)
                
                if sample_index < total_len:
                    spread_loss = 1.0 / max(final_dist, 1.0)
                    air_loss_L = np.exp(-AIR_ABSORPTION_LOW * final_dist)
                    air_loss_H = np.exp(-AIR_ABSORPTION_HIGH * final_dist)
                    
                    amp_L = energy_low * spread_loss * air_loss_L
                    amp_H = energy_high * spread_loss * air_loss_H
                    
                    t_seg = dist_on_seg / segment_len
                    hit_point = p1 + (p2 - p1) * t_seg
                    rel_vec = hit_point - LISTENER_POS
                    rel_dir = rel_vec / (np.linalg.norm(rel_vec) + 1e-6)
                    pan = np.dot(rel_dir, LISTENER_RIGHT_VEC)
                    
                    gain_R = (pan + 1.0) / 2.0
                    gain_L = (1.0 - pan) / 2.0
                    
                    ir_low[sample_index, 0]  += amp_L * gain_L
                    ir_low[sample_index, 1]  += amp_L * gain_R
                    ir_high[sample_index, 0] += amp_H * gain_L
                    ir_high[sample_index, 1] += amp_H * gain_R
            
            total_distance += segment_len
            energy_low  *= WALL_REFLECT_LOW
            energy_high *= WALL_REFLECT_HIGH

    # --- AURALIZATION ---
    print("Applying filters and convolution...")
    
    sos_low  = signal.butter(4, MUFFLE_CROSSOVER_HZ, 'lp', fs=SAMPLE_RATE, output='sos')
    sos_high = signal.butter(4, MUFFLE_CROSSOVER_HZ, 'hp', fs=SAMPLE_RATE, output='sos')
    
    dry_L = signal.sosfilt(sos_low, dry_signal)
    dry_H = signal.sosfilt(sos_high, dry_signal)
    
    # We use 'full' convolution, but we only need the valid part that matches our calculated length
    # Note: fftconvolve output size is len(dry) + len(ir) - 1. 
    # Since our IR is the full timeline (sparse histogram), convolving it with the signal 
    # effectively places the signal echoes at the right times.
    
    print("Convolving Low Band...")
    wet_L_L = signal.fftconvolve(dry_L, ir_low[:, 0], mode='full')[:total_len]
    wet_L_R = signal.fftconvolve(dry_L, ir_low[:, 1], mode='full')[:total_len]
    
    print("Convolving High Band...")
    wet_H_L = signal.fftconvolve(dry_H, ir_high[:, 0], mode='full')[:total_len]
    wet_H_R = signal.fftconvolve(dry_H, ir_high[:, 1], mode='full')[:total_len]
    
    final_L = wet_L_L + wet_H_L
    final_R = wet_L_R + wet_H_R
    
    wet_signal = np.column_stack((final_L, final_R))
    
    max_val = np.max(np.abs(wet_signal))
    if max_val > 0:
        wet_signal = wet_signal / max_val * 0.9
    
    wet_signal_int = (wet_signal * 32767).astype(np.int16)
    
    filename = f"output/output_simulated_full.wav"
    wav.write(filename, SAMPLE_RATE, wet_signal_int)
    print(f"Success! Saved '{filename}'.")

if __name__ == "__main__":
    main()