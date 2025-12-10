import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import os

# --- CONFIGURATION ---
# Must match your C++ simulation
LISTENER_POS = np.array([7.0, 0.0, 0.0]) 
LISTENER_RADIUS = 2.0
SPEED_OF_SOUND = 343.0 # m/s
SAMPLE_RATE = 44100    # Hz

# Stereo Settings
# Vector pointing to the user's "Right" (Assuming facing +Z)
LISTENER_RIGHT_VEC = np.array([1.0, 0.0, 0.0]) 

# Physics Parameters
# How much volume is lost per meter (Air absorption)
AIR_ABSORPTION_COEF = 0.05 
# How much volume is lost per wall bounce
WALL_REFLECTION_COEF = 0.8  

def generate_dry_sound(duration_sec=5.0, freq=110):
    """Generates a continuous drone sound."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    # Continuous sine wave (Drone) - Removed exponential decay
    audio = np.sin(2 * np.pi * freq * t)
    
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
    
    # Filter only rays that eventually hit the listener to save time
    # (The C++ code marked these with hit_listener=1)
    hit_ray_ids = df[df['hit_listener'] == 1]['ray_id'].unique()
    print(f"Processing {len(hit_ray_ids)} rays that touched the listener...")

    # Create Stereo Impulse Response (IR) buffer (Left, Right)
    ir_length = 5 * SAMPLE_RATE
    impulse_response = np.zeros((ir_length, 2))

    for rid in hit_ray_ids:
        # Get all bounces for this ray
        ray_data = df[df['ray_id'] == rid].sort_values('bounce_id')
        coords = ray_data[['x', 'y', 'z']].values
        
        total_distance = 0.0
        current_energy = 1.0
        
        # Walk through the path segments
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            
            segment_len = np.linalg.norm(p2 - p1)
            
            # Optimization: Skip tiny segments entirely
            if segment_len < 1e-6:
                continue

            # Check if this specific segment passed through the listener
            hit, dist_on_seg = intersect_segment_sphere(p1, p2, LISTENER_POS, LISTENER_RADIUS)
            
            if hit:
                # 1. Calculate Exact Arrival Time
                final_dist = total_distance + dist_on_seg
                arrival_time = final_dist / SPEED_OF_SOUND
                sample_index = int(arrival_time * SAMPLE_RATE)
                
                if sample_index < ir_length:
                    # 2. Calculate Energy (Loudness)
                    spreading_loss = 1.0 / max(final_dist, 1.0)
                    air_loss = np.exp(-AIR_ABSORPTION_COEF * final_dist)
                    
                    base_amplitude = current_energy * spreading_loss * air_loss
                    
                    # 3. Stereo Panning (Intensity Panning)
                    # Find exact hit point on sphere surface
                    t_seg = dist_on_seg / segment_len
                    hit_point = p1 + (p2 - p1) * t_seg
                    
                    # Vector from Center -> Hit Point
                    rel_vec = hit_point - LISTENER_POS
                    rel_dir = rel_vec / (np.linalg.norm(rel_vec) + 1e-6)
                    
                    # Dot product with Right Vector (-1 = Left, 1 = Right)
                    pan = np.dot(rel_dir, LISTENER_RIGHT_VEC)
                    
                    # Simple linear panning gains
                    gain_R = (pan + 1.0) / 2.0
                    gain_L = (1.0 - pan) / 2.0
                    
                    # Add to Stereo IR
                    impulse_response[sample_index, 0] += base_amplitude * gain_L # Left
                    impulse_response[sample_index, 1] += base_amplitude * gain_R # Right
            
            # Prepare for next segment
            total_distance += segment_len
            current_energy *= WALL_REFLECTION_COEF

    # --- AURALIZATION (Stereo Convolution) ---
    print("Convolving with dry sound...")
    
    # Generate a dry "ping"
    dry_signal = generate_dry_sound()
    
    # Convolve Left and Right channels separately
    wet_L = signal.fftconvolve(dry_signal, impulse_response[:, 0], mode='full')
    wet_R = signal.fftconvolve(dry_signal, impulse_response[:, 1], mode='full')
    
    # Stack into stereo array
    wet_signal = np.column_stack((wet_L, wet_R))
    
    # Normalize
    max_val = np.max(np.abs(wet_signal))
    if max_val > 0:
        wet_signal = wet_signal / max_val * 0.9
    
    # Convert to 16-bit PCM
    wet_signal_int = (wet_signal * 32767).astype(np.int16)
    
    output_filename = "output/output_audio_stereo.wav"
    wav.write(output_filename, SAMPLE_RATE, wet_signal_int)
    print(f"Success! Saved '{output_filename}' (Stereo).")

if __name__ == "__main__":
    main()