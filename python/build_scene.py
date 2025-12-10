import struct

# --- SCENE DEFINITION ---
ROOM_MIN = (-50, -50, -10)
ROOM_MAX = (50, 50, 10)
SOURCE = (-25, 0, 0)
LISTENER = (25, 0, 0)
LISTENER_RADIUS = 2.0

objects = []

# A large central pillar (Box)
objects.append({
    'type': 1,
    'p1': (-2, -50, -10), # Min
    'p2': (2, 40, 10), # Max
    'trans': .1

})


# # A floating sphere blocking the source
# objects.append({
#     'type': 0,
#     'p1': (-5, 0, -5),
#     'p2': (3.0, 0, 0) #radius will always be x, otherwise ignore it
# })

# objects.append({
#     'type': 0,
#     'p1': (5, 5, 5),
#     'p2': (2.0, 0, 0)
# })

# Format matches 'SceneHeader' and 'Object' structs in C++
# f = float (4 bytes), i = int (4 bytes)

with open("../build/scene.bin", "wb") as f:
    print(f"Building Scene with {len(objects)} objects...")

    # Write Header (Unchanged: 13 floats, 1 int)
    header_fmt = "13fi" 
    header_data = [
        *ROOM_MIN, *ROOM_MAX,
        *SOURCE, *LISTENER,
        LISTENER_RADIUS, len(objects)
    ]
    f.write(struct.pack(header_fmt, *header_data))

    # Write Objects (UPDATED)
    # Type (1i), Param1 (3f), Param2 (3f), Transmission (1f)
    # Total = 1 int + 7 floats
    obj_fmt = "i7f"

    for obj in objects:
        p1 = obj['p1']
        p2 = obj['p2']
        # Handle scalar radius for spheres
        if obj['type'] == 0 and len(p2) == 1: p2 = (p2[0], 0.0, 0.0)
        elif obj['type'] == 0 and isinstance(p2, (int, float)): p2 = (p2, 0.0, 0.0)

        data = [obj['type'], *p1, *p2, obj['trans']]
        f.write(struct.pack(obj_fmt, *data))

print("Success! Wrote ../build/scene.bin")