import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import os

# --- CONFIGURATION ---
# Must match your C++ simulation
LISTENER_POS = np.array([10, 0.0, 10.0]) 
LISTENER_RADIUS = 2.0
SPEED_OF_SOUND = 343.0 # m/s
SAMPLE_RATE = 44100    # Hz

# Stereo Settings
# Vector pointing to the user's "Right" (Assuming facing +Z)
LISTENER_RIGHT_VEC = np.array([1.0, 0.0, 0.0]) 

# --- PHYSICS PARAMETERS ---
# Wall Thickness (0.0 = Paper, 1.0 = Concrete Bunker)
# Thicker walls reflect more sound (less transmission loss)
WALL_THICKNESS = 0.8 

# Reflection Coefficients (derived from thickness)
# Low Freqs (Bass) reflect better than High Freqs (Treble)
# Thicker walls reflect almost all bass, thin walls absorb/transmit it
WALL_REFLECT_LOW  = 0.5 + (0.49 * WALL_THICKNESS)  # 0.5 to 0.99
WALL_REFLECT_HIGH = 0.1 + (0.50 * WALL_THICKNESS)  # 0.1 to 0.60

# Air absorption (Higher freqs die faster in air)
AIR_ABSORPTION_LOW  = 0.005
AIR_ABSORPTION_HIGH = 0.100

# Crossover Frequency for the Muffling Effect
MUFFLE_CROSSOVER_HZ = 800

def generate_dry_sound(duration_sec=5.0, freq=110):
    """Generates a rich Sawtooth drone (has high freqs to muffle)."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    
    # Use Sawtooth instead of Sine so we have harmonics to filter out
    audio = signal.sawtooth(2 * np.pi * freq * t)
    
    # Add fade in/out to prevent clicking
    fade_samples = int(0.1 * SAMPLE_RATE)
    if len(audio) > 2 * fade_samples:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
    return audio

def intersect_segment_sphere(p1, p2, center, radius):
    """
    Checks if a line segment P1->P2 hits the listener sphere.
    Returns: (True, distance_along_segment) or (False, None)
    """
    d = p2 - p1
    f = p1 - center
    
    a = np.dot(d, d)
    
    # SAFETY CHECK: Ignore zero-length segments (caused by padding in C++)
    if a < 1e-8:
        return False, None

    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2

    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return False, None
    
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)

    # Check if the intersection is actually ON the segment (0 <= t <= 1)
    if 0 <= t1 <= 1:
        return True, t1 * np.linalg.norm(d)
    if 0 <= t2 <= 1:
        return True, t2 * np.linalg.norm(d)
        
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

    # Create TWO Impulse Response buffers (Low Band, High Band)
    # Shape: (Samples, 2 Channels)
    ir_length = 5 * SAMPLE_RATE
    ir_low  = np.zeros((ir_length, 2))
    ir_high = np.zeros((ir_length, 2))

    for rid in hit_ray_ids:
        # Get all bounces for this ray
        ray_data = df[df['ray_id'] == rid].sort_values('bounce_id')
        coords = ray_data[['x', 'y', 'z']].values
        
        total_distance = 0.0
        
        # Track energy separately for Bass and Treble
        energy_low = 1.0
        energy_high = 1.0
        
        # Walk through the path segments
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            
            segment_len = np.linalg.norm(p2 - p1)
            if segment_len < 1e-6: continue

            # Check for Listener Hit
            hit, dist_on_seg = intersect_segment_sphere(p1, p2, LISTENER_POS, LISTENER_RADIUS)
            
            if hit:
                # 1. Timing
                final_dist = total_distance + dist_on_seg
                arrival_time = final_dist / SPEED_OF_SOUND
                sample_index = int(arrival_time * SAMPLE_RATE)
                
                if sample_index < ir_length:
                    # 2. Geometric Spreading (Affects both bands equally)
                    spread_loss = 1.0 / max(final_dist, 1.0)
                    
                    # 3. Air Absorption (Frequency Dependent)
                    air_loss_L = np.exp(-AIR_ABSORPTION_LOW * final_dist)
                    air_loss_H = np.exp(-AIR_ABSORPTION_HIGH * final_dist)
                    
                    # 4. Combine histories
                    amp_L = energy_low * spread_loss * air_loss_L
                    amp_H = energy_high * spread_loss * air_loss_H
                    
                    # 5. Stereo Panning
                    t_seg = dist_on_seg / segment_len
                    hit_point = p1 + (p2 - p1) * t_seg
                    rel_vec = hit_point - LISTENER_POS
                    rel_dir = rel_vec / (np.linalg.norm(rel_vec) + 1e-6)
                    pan = np.dot(rel_dir, LISTENER_RIGHT_VEC)
                    
                    gain_R = (pan + 1.0) / 2.0
                    gain_L = (1.0 - pan) / 2.0
                    
                    # Add to separate frequency bands
                    ir_low[sample_index, 0]  += amp_L * gain_L
                    ir_low[sample_index, 1]  += amp_L * gain_R
                    ir_high[sample_index, 0] += amp_H * gain_L
                    ir_high[sample_index, 1] += amp_H * gain_R
            
            # Prepare for next segment (Bounce)
            total_distance += segment_len
            
            # Apply Wall Absorption (Frequency Dependent)
            energy_low  *= WALL_REFLECT_LOW
            energy_high *= WALL_REFLECT_HIGH

    # --- AURALIZATION ---
    print("Generating Signal...")
    dry_signal = generate_dry_sound()
    
    # Create Filters to split the dry signal into Bass and Treble
    # Butterworth 4th order filter
    sos_low  = signal.butter(4, MUFFLE_CROSSOVER_HZ, 'lp', fs=SAMPLE_RATE, output='sos')
    sos_high = signal.butter(4, MUFFLE_CROSSOVER_HZ, 'hp', fs=SAMPLE_RATE, output='sos')
    
    dry_L = signal.sosfilt(sos_low, dry_signal)
    dry_H = signal.sosfilt(sos_high, dry_signal)
    
    print("Convolving Low Band...")
    wet_L_L = signal.fftconvolve(dry_L, ir_low[:, 0], mode='full')
    wet_L_R = signal.fftconvolve(dry_L, ir_low[:, 1], mode='full')
    
    print("Convolving High Band...")
    wet_H_L = signal.fftconvolve(dry_H, ir_high[:, 0], mode='full')
    wet_H_R = signal.fftconvolve(dry_H, ir_high[:, 1], mode='full')
    
    # Mix Bands
    final_L = wet_L_L + wet_H_L
    final_R = wet_L_R + wet_H_R
    
    # Stack Stereo
    wet_signal = np.column_stack((final_L, final_R))
    
    # Normalize
    max_val = np.max(np.abs(wet_signal))
    if max_val > 0:
        wet_signal = wet_signal / max_val * 0.9
    
    wet_signal_int = (wet_signal * 32767).astype(np.int16)
    
    filename = f"output/output_muffled_thick{WALL_THICKNESS}.wav"
    wav.write(filename, SAMPLE_RATE, wet_signal_int)
    print(f"Success! Saved '{filename}'.")

if __name__ == "__main__":
    main()