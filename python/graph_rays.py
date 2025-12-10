import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import struct

# --- LOAD SCENE FROM BINARY ---
SCENE_FILE = "../build/scene.bin"

if not os.path.exists(SCENE_FILE):
    print("Error: Scene file not found. Run build_scene.py first.")
    exit()

with open(SCENE_FILE, "rb") as f:
    # Read Header (13 floats, 1 int)
    h_data = struct.unpack("13fi", f.read(13*4 + 4))
    
    ROOM_MIN = h_data[0:3]
    ROOM_MAX = h_data[3:6]
    SOURCE_POS = h_data[6:9]
    LISTENER_POS = h_data[9:12]
    LISTENER_RADIUS = h_data[12]
    NUM_OBJECTS = h_data[13]
    
    print(f"Loaded Scene: Room={ROOM_MIN}-{ROOM_MAX}, Objects={NUM_OBJECTS}")

    objects = []
    for _ in range(NUM_OBJECTS):
        # UPDATED: Type(1i), P1(3f), P2(3f), Transmission(1f)
        obj_data = struct.unpack("i7f", f.read(4 + 7*4))
        objects.append({
            'type': obj_data[0],
            'p1': obj_data[1:4],
            'p2': obj_data[4:7],
            'trans': obj_data[7]
        })

# --- LOAD RAYS ---
csv_path = "../build/paths.csv"
if not os.path.exists(csv_path):
    print("Run C++ sim first.")
    exit()

df = pd.read_csv(csv_path)
traces = []

# --- DRAW ROOM ---
def get_box_wireframe(min_c, max_c):
    x0, y0, z0 = min_c
    x1, y1, z1 = max_c
    x = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
    y = [y0, y0, y0, y0, y0, None, y1, y1, y1, y1, y1, None, y0, y1, None, y0, y1, None, y0, y1, None, y0, y1]
    z = [z0, z0, z1, z1, z0, None, z0, z0, z1, z1, z0, None, z0, z0, None, z0, z0, None, z1, z1, None, z1, z1]
    return x, y, z

rx, ry, rz = get_box_wireframe(ROOM_MIN, ROOM_MAX)
traces.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines', line=dict(color='white', width=4), name='Room'))

# --- DRAW OBJECTS ---
def get_sphere_mesh(x, y, z, r, color, name, opacity=0.8):
    phi = np.linspace(0, 2*np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    return go.Surface(
        x=r*np.sin(theta)*np.cos(phi)+x, 
        y=r*np.sin(theta)*np.sin(phi)+y, 
        z=r*np.cos(theta)+z, 
        colorscale=[[0, color], [1, color]], 
        showscale=False, opacity=opacity, name=name
    )

def get_box_mesh(min_c, max_c, color, name, opacity):
    x0, y0, z0 = min_c
    x1, y1, z1 = max_c
    x = [x0, x0, x1, x1, x0, x0, x1, x1]
    y = [y0, y1, y1, y0, y0, y1, y1, y0]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    return go.Mesh3d(x=x, y=y, z=z, color=color, alphahull=0, opacity=opacity, name=name, flatshading=True)

for i, obj in enumerate(objects):
    # Determine color/opacity based on transmission
    # High transmission = Ghostly/White, Low = Solid/Orange
    opacity = 0.9 if obj['trans'] < 0.1 else 0.3
    color = 'orange' if obj['trans'] < 0.5 else 'silver'
    
    if obj['type'] == 0: 
        traces.append(get_sphere_mesh(obj['p1'][0], obj['p1'][1], obj['p1'][2], obj['p2'][0], color, f"Sphere {i}", opacity))
    elif obj['type'] == 1: 
        traces.append(get_box_mesh(obj['p1'], obj['p2'], color, f"Box {i}", opacity))
        bx, by, bz = get_box_wireframe(obj['p1'], obj['p2'])
        traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='white', width=3), name=f"Box Edge {i}", showlegend=False))

# --- DRAW SOURCE/LISTENER ---
traces.append(get_sphere_mesh(LISTENER_POS[0], LISTENER_POS[1], LISTENER_POS[2], LISTENER_RADIUS, 'yellow', 'Listener', opacity=0.3))
traces.append(go.Scatter3d(x=[SOURCE_POS[0]], y=[SOURCE_POS[1]], z=[SOURCE_POS[2]], mode='markers', marker=dict(size=8, color='red'), name='Source'))

# --- DRAW RAYS ---
ray_ids = df['ray_id'].unique()
hit_count = 0
counter = 0

for rid in ray_ids:
    if (counter >= 1000):
        break
    ray_data = df[df['ray_id'] == rid]
    did_hit = ray_data['hit_listener'].iloc[0] == 1
    
    if did_hit and hit_count < 1000: # Limit drawing to prevent lag
        hit_count += 1
        traces.append(go.Scatter3d(
            x=ray_data['x'], y=ray_data['y'], z=ray_data['z'],
            mode='lines', line=dict(color='#00FF00', width=1), opacity=0.1, showlegend=False
        ))
    else:
        traces.append(go.Scatter3d(
            x=ray_data['x'], y=ray_data['y'], z=ray_data['z'],
            mode='lines', line=dict(color='#FF0000', width=1), opacity=0.05, showlegend=False
        ))
    counter += 1

print(f"Hits: {hit_count}")

layout = go.Layout(scene=dict(aspectmode='data', bgcolor='black'))
fig = go.Figure(data=traces, layout=layout)
fig.write_html("output/audio_sim.html")