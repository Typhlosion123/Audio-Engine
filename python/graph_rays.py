import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import time

# --- CONFIGURATION ---
LISTENER_POS = (7.0, 0.0, 0.0) 
LISTENER_RADIUS = 2.0

# 1. Load Data
csv_path = "../build/paths.csv"

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    print("Please run the C++ simulation first: cd build && ./AcousticSim")
    exit()

# Timestamp Check
mod_time = os.path.getmtime(csv_path)
print(f"Loading data from: {csv_path}")
print(f"File generated on: {time.ctime(mod_time)}")

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Detect Source
try:
    source_row = df[(df['ray_id'] == 0) & (df['bounce_id'] == 0)].iloc[0]
    SOURCE_POS = (source_row['x'], source_row['y'], source_row['z'])
except IndexError:
    SOURCE_POS = (-5.0, 0.0, -5.0)

traces = []

# --- Helper: Draw Room Wireframe ---
def get_cube_wireframe(min_c, max_c):
    # Explicitly draw edges to avoid diagonals
    x_b = [min_c, max_c, max_c, min_c, min_c, None]
    y_b = [min_c, min_c, max_c, max_c, min_c, None]
    z_b = [min_c, min_c, min_c, min_c, min_c, None]
    x_t = [min_c, max_c, max_c, min_c, min_c, None]
    y_t = [min_c, min_c, max_c, max_c, min_c, None]
    z_t = [max_c, max_c, max_c, max_c, max_c, None]
    x_v = [min_c, min_c, None, max_c, max_c, None, max_c, max_c, None, min_c, min_c]
    y_v = [min_c, min_c, None, min_c, min_c, None, max_c, max_c, None, max_c, max_c]
    z_v = [min_c, max_c, None, min_c, max_c, None, min_c, max_c, None, min_c, max_c]
    return x_b + x_t + x_v, y_b + y_t + y_v, z_b + z_t + z_v

cx, cy, cz = get_cube_wireframe(-10, 10)
traces.append(go.Scatter3d(
    x=cx, y=cy, z=cz, mode='lines', line=dict(color='white', width=4), name='Room Walls', hoverinfo='none'
))

# --- Helper: Listener & Source ---
def get_sphere_mesh(x_c, y_c, z_c, r, color):
    phi = np.linspace(0, 2*np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    return go.Surface(x=r*np.sin(theta)*np.cos(phi)+x_c, y=r*np.sin(theta)*np.sin(phi)+y_c, z=r*np.cos(theta)+z_c, colorscale=[[0, color], [1, color]], showscale=False, opacity=0.3, name='Listener')

traces.append(get_sphere_mesh(LISTENER_POS[0], LISTENER_POS[1], LISTENER_POS[2], LISTENER_RADIUS, 'yellow'))
traces.append(go.Scatter3d(x=[SOURCE_POS[0]], y=[SOURCE_POS[1]], z=[SOURCE_POS[2]], mode='markers', marker=dict(size=8, color='red'), name='Source'))

# --- Plot Rays (Optimized) ---
ray_ids = df['ray_id'].unique()
hit_count = 0

# Store coordinates for batch rendering
miss_x, miss_y, miss_z = [], [], []
hit_signatures = set() 

print(f"Processing {len(ray_ids)} rays...")

for rid in ray_ids:
    ray_data = df[df['ray_id'] == rid]
    did_hit = ray_data['hit_listener'].iloc[0] == 1
    
    # Extract path coordinates
    rx = ray_data['x'].tolist()
    ry = ray_data['y'].tolist()
    rz = ray_data['z'].tolist()

    if did_hit:
        hit_count += 1
        
        if len(rx) > 1:
            sig = f"{rx[1]:.2f}_{ry[1]:.2f}_{rz[1]:.2f}"
            hit_signatures.add(sig)

        # UPDATED STYLE: Thinner and more transparent
        traces.append(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode='lines',
            line=dict(color='#00FF00', width=2), # Thinner (was 5)
            opacity=0.5,                         # More transparent (was 0.9)
            name=f'Ray {rid}',
            legendgroup="Audible Rays",
            showlegend=False
        ))
    else:
        # Batch misses
        miss_x.extend(rx + [None])
        miss_y.extend(ry + [None])
        miss_z.extend(rz + [None])

# Add the batch trace for misses
traces.append(go.Scatter3d(
    x=miss_x, y=miss_y, z=miss_z,
    mode='lines',
    line=dict(color='cyan', width=1),
    opacity=0.1,
    name='Inaudible Rays',
    hoverinfo='none'
))

# Stats
print("-" * 30)
print(f"Total Hit Count: {hit_count}")
print(f"Unique Hit Paths: {len(hit_signatures)}")
print("-" * 30)

if len(hit_signatures) < hit_count and hit_count > 0:
    print("⚠️  WARNING: DUPLICATE RAYS DETECTED!")
    print(f"You have {hit_count} hits, but only {len(hit_signatures)} unique paths.")

# Legend Dummies
traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='#00FF00', width=2), name='Audible Rays (Hit Listener)'))
traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color='cyan', width=1), name='Inaudible Rays (Missed)'))

layout = go.Layout(
    title=f"Audio Ray Tracer<br>Unique Hits: {len(hit_signatures)} / {hit_count}",
    scene=dict(
        aspectmode='data', 
        xaxis=dict(range=[-12, 12], backgroundcolor="black", showgrid=True),
        yaxis=dict(range=[-12, 12], backgroundcolor="black", showgrid=True),
        zaxis=dict(range=[-12, 12], backgroundcolor="black", showgrid=True),
        bgcolor="black"
    ),
    paper_bgcolor="black",
    font=dict(color="white"),
    margin=dict(r=0, l=0, b=0, t=60)
)

fig = go.Figure(data=traces, layout=layout)
output_file = "output/audio_sim.html"
fig.write_html(output_file)
print(f"Done! Open {output_file}")