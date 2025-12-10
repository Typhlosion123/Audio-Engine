import struct

# --- SCENE DEFINITION ---
ROOM_MIN = (-15, -10, -15)
ROOM_MAX = (15, 10, 15)
SOURCE = (-10, 0, -10)
LISTENER = (10, 0, 10)
LISTENER_RADIUS = 2.0

objects = []

# A large central pillar (Box)
objects.append({
    'type': 1,
    'p1': (-2, -10, -7.5), # Min
    'p2': (2, 10, 7.5)     # Max
})

# A floating sphere blocking the source
objects.append({
    'type': 0,
    'p1': (-5, 0, -5),
    'p2': (3.0, 0, 0) #radius will always be x, otherwise ignore it
})

objects.append({
    'type': 0,
    'p1': (5, 5, 5),
    'p2': (2.0, 0, 0)
})

# Format matches 'SceneHeader' and 'Object' structs in C++
# f = float (4 bytes), i = int (4 bytes)

with open("../build/scene.bin", "wb") as f:
    print(f"Building Scene with {len(objects)} objects...")

    # 1. Write Header
    # Room (6f), Source (3f), Listener (3f), Radius (1f), Count (1i)
    # Total floats: 6 + 3 + 3 + 1 = 13 floats
    header_fmt = "13fi" 
    
    header_data = [
        *ROOM_MIN, *ROOM_MAX,
        *SOURCE,
        *LISTENER,
        LISTENER_RADIUS,
        len(objects)
    ]
    
    f.write(struct.pack(header_fmt, *header_data))

    # 2. Write Objects
    # Type (1i), Param1 (3f), Param2 (3f)
    obj_fmt = "i6f"

    for obj in objects:
        p1 = obj['p1']
        p2 = obj['p2']
        # If sphere, p2 is just radius, pad y/z with 0
        if obj['type'] == 0 and len(p2) == 1: 
             p2 = (p2[0], 0.0, 0.0)
        elif obj['type'] == 0 and isinstance(p2, (int, float)):
             p2 = (p2, 0.0, 0.0)

        data = [obj['type'], *p1, *p2]
        f.write(struct.pack(obj_fmt, *data))

print("Success! Wrote ../build/scene.bin")