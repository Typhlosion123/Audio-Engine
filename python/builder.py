import streamlit as st
import plotly.graph_objects as go
import numpy as np
import struct
import os

# --- CONFIGURATION ---
SCENE_FILE = "../build/scene.bin"
HEADER_FMT = "13fi" 
OBJ_FMT = "i7f"

st.set_page_config(page_title="Scene Architect", layout="wide")

# --- INITIALIZE STATE ---
if 'objects' not in st.session_state:
    st.session_state.objects = []
    # Default: A simple wall
    st.session_state.objects.append({
        'name': 'Wall', 'type': 'Box', 
        'p1': [-1.0, -10.0, -5.0], 'p2': [1.0, 10.0, 5.0], 'trans': 0.0
    })

if 'room_dims' not in st.session_state:
    st.session_state.room_dims = {
        'min': [-15.0, -10.0, -15.0],
        'max': [15.0, 10.0, 15.0]
    }

if 'source_pos' not in st.session_state:
    st.session_state.source_pos = [-10.0, 0.0, -10.0]

if 'listener_pos' not in st.session_state:
    st.session_state.listener_pos = [10.0, 0.0, 10.0]

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Global Settings")
    with st.expander("Room Size"):
        col1, col2 = st.columns(2)
        st.session_state.room_dims['min'][0] = col1.number_input("Min X", value=-15.0)
        st.session_state.room_dims['max'][0] = col2.number_input("Max X", value=15.0)
        st.session_state.room_dims['min'][1] = col1.number_input("Min Y", value=-10.0)
        st.session_state.room_dims['max'][1] = col2.number_input("Max Y", value=10.0)
        st.session_state.room_dims['min'][2] = col1.number_input("Min Z", value=-15.0)
        st.session_state.room_dims['max'][2] = col2.number_input("Max Z", value=15.0)

    with st.expander("Positions"):
        st.markdown("**Source (Red)**")
        sx = st.number_input("S_X", value=-10.0)
        sy = st.number_input("S_Y", value=0.0)
        sz = st.number_input("S_Z", value=-10.0)
        st.session_state.source_pos = [sx, sy, sz]
        
        st.markdown("**Listener (Yellow)**")
        lx = st.number_input("L_X", value=10.0)
        ly = st.number_input("L_Y", value=0.0)
        lz = st.number_input("L_Z", value=10.0)
        st.session_state.listener_pos = [lx, ly, lz]

    st.header("2. Add Object")
    
    # Use radio button instead of tabs to ensure single source of truth for preview
    obj_type = st.radio("Object Type", ["AABB", "Sphere"], horizontal=True)
    
    # Holder for the object currently being edited
    preview_obj = None
    
    if obj_type == "AABB":
        b_name = st.text_input("Object Name", "New AABB")
        
        col_min, col_max = st.columns(2)
        with col_min:
            st.markdown("**Min Point**")
            min_x = st.number_input("Min X", value=0.0, key="min_x")
            min_y = st.number_input("Min Y", value=0.0, key="min_y")
            min_z = st.number_input("Min Z", value=0.0, key="min_z")
        
        with col_max:
            st.markdown("**Max Point**")
            max_x = st.number_input("Max X", value=2.0, key="max_x")
            max_y = st.number_input("Max Y", value=2.0, key="max_y")
            max_z = st.number_input("Max Z", value=2.0, key="max_z")
            
        b_trans = st.slider("Transmission", 0.0, 1.0, 0.0, key="b_trans")
        
        # Calculate bounds immediately for preview
        p1 = [min(min_x, max_x), min(min_y, max_y), min(min_z, max_z)]
        p2 = [max(min_x, max_x), max(min_y, max_y), max(min_z, max_z)]
        
        # Construct preview object
        # FIX: Changed type from 'AABB' to 'Box' to match renderer logic
        preview_obj = {
            'name': f"{b_name} (Preview)",
            'type': 'Box', 
            'p1': p1,
            'p2': p2,
            'trans': b_trans,
            'preview': True
        }
        
        if st.button("Add AABB"):
            # Commit the object
            final_obj = preview_obj.copy()
            final_obj['name'] = b_name
            if 'preview' in final_obj: del final_obj['preview']
            st.session_state.objects.append(final_obj)
            st.success(f"Added {b_name}")

    elif obj_type == "Sphere":
        s_name = st.text_input("Sphere Name", "New Sphere")
        sx = st.number_input("Center X", 0.0, key="sx")
        sy = st.number_input("Center Y", 0.0, key="sy")
        sz = st.number_input("Center Z", 0.0, key="sz")
        s_rad = st.number_input("Radius", 3.0)
        s_trans = st.slider("Transmission", 0.0, 1.0, 0.0, key="s_trans")
        
        # Construct preview object
        preview_obj = {
            'name': f"{s_name} (Preview)",
            'type': 'Sphere',
            'p1': [sx, sy, sz], 
            'p2': [s_rad, 0.0, 0.0], 
            'trans': s_trans,
            'preview': True
        }
        
        if st.button("Add Sphere"):
            # Commit the object
            final_obj = preview_obj.copy()
            final_obj['name'] = s_name
            if 'preview' in final_obj: del final_obj['preview']
            st.session_state.objects.append(final_obj)
            st.success(f"Added {s_name}")

    # Save the current preview to session state for the renderer
    st.session_state['preview_object'] = preview_obj

    st.divider()
    st.subheader("Manage Objects")
    for i, obj in enumerate(st.session_state.objects):
        c1, c2 = st.columns([3,1])
        c1.text(f"{obj['name']} ({obj['type']})")
        if c2.button("Del", key=f"del_{i}"):
            st.session_state.objects.pop(i)
            st.rerun()

# --- MAIN VIEW ---
st.title("🏗️ Scene Architect")

# --- SAVE FUNCTION ---
def save_binary():
    # Ensure directory exists
    os.makedirs(os.path.dirname(SCENE_FILE), exist_ok=True)
    
    with open(SCENE_FILE, "wb") as f:
        # Header
        header_data = [
            *st.session_state.room_dims['min'], 
            *st.session_state.room_dims['max'],
            *st.session_state.source_pos, 
            *st.session_state.listener_pos,
            2.0, # Listener Radius
            len(st.session_state.objects)
        ]
        f.write(struct.pack(HEADER_FMT, *header_data))

        # Objects
        for obj in st.session_state.objects:
            type_id = 1 if obj['type'] == 'Box' else 0
            data = [type_id, *obj['p1'], *obj['p2'], obj['trans']]
            f.write(struct.pack(OBJ_FMT, *data))

if st.button("💾 SAVE SCENE (scene.bin)", type="primary"):
    try:
        save_binary()
        st.toast(f"Scene saved to {SCENE_FILE}!")
    except Exception as e:
        st.error(f"Error saving file: {e}")

# --- LIGHTWEIGHT VISUALIZATION ---
# Only draws boxes and spheres. No Rays. Very fast.
def draw_preview():
    traces = []
    
    # Room Wireframe
    rmin = st.session_state.room_dims['min']
    rmax = st.session_state.room_dims['max']
    # Draw simple floor plan line
    traces.append(go.Scatter3d(
        x=[rmin[0], rmax[0], rmax[0], rmin[0], rmin[0], None, rmin[0], rmax[0], rmax[0], rmin[0], rmin[0], None, rmin[0], rmin[0], None, rmax[0], rmax[0], None, rmax[0], rmax[0], None, rmin[0], rmin[0]], 
        y=[rmin[1], rmin[1], rmin[1], rmin[1], rmin[1], None, rmax[1], rmax[1], rmax[1], rmax[1], rmax[1], None, rmin[1], rmax[1], None, rmin[1], rmax[1], None, rmin[1], rmax[1], None, rmin[1], rmax[1]],
        z=[rmin[2], rmin[2], rmax[2], rmax[2], rmin[2], None, rmin[2], rmin[2], rmax[2], rmax[2], rmin[2], None, rmin[2], rmin[2], None, rmin[2], rmin[2], None, rmax[2], rmax[2], None, rmax[2], rmax[2]],
        mode='lines', line=dict(color='white', width=2), name='Room'
    ))

    # FIX: Combine committed objects with the live preview object for rendering
    display_objects = st.session_state.objects.copy()
    if st.session_state.get('preview_object'):
        display_objects.append(st.session_state['preview_object'])

    # Objects
    for obj in display_objects:
        # Style preview objects differently (dotted line or semi-transparent)
        is_preview = obj.get('preview', False)
        
        color = 'orange' if obj['trans'] < 0.5 else 'lightblue'
        # Make preview objects more transparent
        opacity = (1.0 if obj['trans'] < 0.5 else 0.3) if not is_preview else 0.2
        
        obj_name = obj['name']
        
        if obj['type'] == 'Box':
            # Draw Mesh Box
            x0, y0, z0 = obj['p1']
            x1, y1, z1 = obj['p2']
            traces.append(go.Mesh3d(
                x=[x0, x0, x1, x1, x0, x0, x1, x1],
                y=[y0, y1, y1, y0, y0, y1, y1, y0],
                z=[z0, z0, z0, z0, z1, z1, z1, z1],
                color=color, alphahull=0, opacity=opacity, name=obj_name, flatshading=True
            ))
        elif obj['type'] == 'Sphere':
            # Draw Simple Sphere
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obj['p2'][0] * np.outer(np.cos(u), np.sin(v)) + obj['p1'][0]
            y = obj['p2'][0] * np.outer(np.sin(u), np.sin(v)) + obj['p1'][1]
            z = obj['p2'][0] * np.outer(np.ones(np.size(u)), np.cos(v)) + obj['p1'][2]
            traces.append(go.Surface(x=x, y=y, z=z, opacity=opacity, colorscale=[[0, color], [1,color]], showscale=False, name=obj_name))

    # Source & Listener
    src = st.session_state.source_pos
    lst = st.session_state.listener_pos
    traces.append(go.Scatter3d(x=[src[0]], y=[src[1]], z=[src[2]], mode='markers', marker=dict(size=10, color='red'), name='Source'))
    traces.append(go.Scatter3d(x=[lst[0]], y=[lst[1]], z=[lst[2]], mode='markers', marker=dict(size=10, color='yellow'), name='Listener'))

    layout = go.Layout(
        scene=dict(aspectmode='data', bgcolor='black',
                   xaxis=dict(backgroundcolor="black"),
                   yaxis=dict(backgroundcolor="black"),
                   zaxis=dict(backgroundcolor="black")),
        margin=dict(l=0, r=0, b=0, t=0), height=700
    )
    
    st.plotly_chart(go.Figure(data=traces, layout=layout), width = 'stretch')

draw_preview()