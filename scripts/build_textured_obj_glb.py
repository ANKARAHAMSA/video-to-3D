import cv2
import numpy as np
import json, math, os, struct, zipfile

with open("artifacts/file_generation/ttl=63d/intermediate/web_scene_data.json") as f:
    data = json.load(f)

XC_MIN, XC_MAX = -160.0, 160.0
YC_MIN, YC_MAX = -140.0, 180.0

# 1. Build a 2048 x 2048 Unified Texture Atlas (`campus_texture_atlas.jpg`):
# Top 1536 rows (rows 0..1535, i.e., v_obj in [0.25, 1.0]): 1536x1536 Orthomosaic Ground Map (left) + Roof/Tree/Steel swatches (right)
# Bottom 512 rows (rows 1536..2047, i.e., v_obj in [0.0, 0.25]):
#   Tile 0 (x: 0..512):    3-Story Office/Lab Window Facade Texture
#   Tile 1 (x: 512..1024): High-Bay Industrial Hall Red/White Cladding & Bay Doors Texture
#   Tile 2 (x: 1024..1536): Crystalline Blue Solar PV Module Cell Grid Texture
#   Tile 3 (x: 1536..2048): Dark Monocrystalline Solar PV Cell Grid Texture
atlas = np.zeros((2048, 2048, 3), dtype=np.uint8)

ortho_bgr = cv2.imread("artifacts/file_generation/ttl=63d/intermediate/campus_ortho_clean.jpg")
ortho_1536 = cv2.resize(ortho_bgr, (1536, 1536), interpolation=cv2.INTER_CUBIC)
atlas[0:1536, 0:1536] = ortho_1536

# Right side of top 1536 rows (x: 1536..2048, y: 0..1536): architectural swatches
# (a) y: 0..384 -> Masonry Core Tower texture (dark brown brick with subtle vertical shadow panel)
core_tile = np.full((384, 512, 3), (72, 82, 105), dtype=np.uint8) # BGR
cv2.rectangle(core_tile, (48, 28), (464, 356), (58, 66, 86), -1)
for y in range(0, 384, 16):
    cv2.line(core_tile, (0, y), (512, y), (64, 73, 94), 1)
atlas[0:384, 1536:2048] = core_tile

# (b) y: 384..768 -> Concrete Roof Deck & Parapet texture
roof_tile = np.full((384, 512, 3), (196, 194, 190), dtype=np.uint8)
cv2.rectangle(roof_tile, (16, 16), (496, 368), (176, 174, 170), -1)
atlas[384:768, 1536:2048] = roof_tile

# (c) y: 768..1152 -> Industrial White Skylight Roof texture
ind_roof_tile = np.full((384, 512, 3), (244, 242, 238), dtype=np.uint8)
for y in range(28, 360, 48):
    cv2.rectangle(ind_roof_tile, (24, y), (488, y+16), (205, 198, 188), -1)
atlas[768:1152, 1536:2048] = ind_roof_tile

# (d) y: 1152..1536 -> Tree Foliage (left half) & Bark/Steel (right half)
atlas[1152:1536, 1536:1792] = (22, 102, 64)   # Foliage green BGR
atlas[1152:1344, 1792:2048] = (52, 64, 92)    # Trunk brown BGR
atlas[1344:1536, 1792:2048] = (182, 168, 152) # Galvanized steel BGR

# Bottom strip (y: 1536..2048):
# Tile 0 (x: 0..512): 3-Story Office/Lab Window Facade (matches createFacadeTexture in preview!)
office_tile = np.full((512, 512, 3), (122, 154, 186), dtype=np.uint8) # Tan/sandstone BGR
for floor in range(3):
    y0 = 64 + floor * 144
    cv2.rectangle(office_tile, (0, y0), (512, y0 + 64), (43, 30, 18), -1)
    cv2.rectangle(office_tile, (0, y0 + 8), (512, y0 + 56), (135, 105, 55), -1)
for x in range(0, 512, 48):
    cv2.rectangle(office_tile, (x, 0), (x + 8, 512), (136, 168, 200), -1)
atlas[1536:2048, 0:512] = office_tile

# Tile 1 (x: 512..1024): High-Bay Industrial Hall Cladding & Doors
ind_tile = np.full((512, 512, 3), (55, 62, 185), dtype=np.uint8)
cv2.rectangle(ind_tile, (0, 0), (512, 168), (240, 232, 226), -1)
cv2.rectangle(ind_tile, (0, 168), (512, 256), (27, 27, 153), -1)
for x in range(24, 500, 64):
    cv2.rectangle(ind_tile, (x, 280), (x + 40, 512), (59, 41, 30), -1)
    cv2.rectangle(ind_tile, (x, 280), (x + 40, 512), (139, 116, 100), 2)
atlas[1536:2048, 512:1024] = ind_tile

# Tile 2 (x: 1024..1536): Crystalline Blue Solar PV Panel Grid
solar_blue = np.full((512, 512, 3), (216, 78, 29), dtype=np.uint8) # Deep solar blue BGR
cv2.rectangle(solar_blue, (6, 6), (506, 506), (253, 197, 147), 6)
for x in range(32, 512, 32):
    cv2.line(solar_blue, (x, 6), (x, 506), (235, 150, 95), 2)
for y in range(32, 512, 32):
    cv2.line(solar_blue, (6, y), (506, y), (235, 150, 95), 2)
atlas[1536:2048, 1024:1536] = solar_blue

# Tile 3 (x: 1536..2048): Dark Mono Solar PV Panel Grid
solar_mono = np.full((512, 512, 3), (59, 41, 30), dtype=np.uint8)
cv2.rectangle(solar_mono, (6, 6), (506, 506), (165, 148, 132), 6)
for x in range(32, 512, 32):
    cv2.line(solar_mono, (x, 6), (x, 506), (110, 95, 82), 2)
for y in range(32, 512, 32):
    cv2.line(solar_mono, (6, y), (506, y), (110, 95, 82), 2)
atlas[1536:2048, 1536:2048] = solar_mono

atlas_path = "artifacts/file_generation/ttl=63d/output/campus_texture_atlas.jpg"
cv2.imwrite(atlas_path, atlas, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Helper to convert pixel rect (x_px, y_px) in 2048x2048 image (top-left origin) to OBJ UV (bottom-left origin: v = 1 - y_px/2048)
def px_to_uv(x_px, y_px):
    return (float(x_px) / 2048.0, 1.0 - float(y_px) / 2048.0)

def ortho_uv(x_c, y_c):
    # Ortho is in x_px: [0, 1536], y_px: [0, 1536] where y_px=0 is YC_MAX (+180) and y_px=1536 is YC_MIN (-140)
    fx = np.clip((x_c - XC_MIN) / (XC_MAX - XC_MIN), 0.0, 1.0)
    fy = np.clip((YC_MAX - y_c) / (YC_MAX - YC_MIN), 0.0, 1.0)
    return px_to_uv(fx * 1536.0, fy * 1536.0)

def tile_uv_quad(x0_px, y0_px, x1_px, y1_px):
    # Bottom-left (x0, y1), Bottom-right (x1, y1), Top-right (x1, y0), Top-left (x0, y0)
    return (
        px_to_uv(x0_px, y1_px),
        px_to_uv(x1_px, y1_px),
        px_to_uv(x1_px, y0_px),
        px_to_uv(x0_px, y0_px)
    )

UV_OFFICE_WALL = tile_uv_quad(8, 1544, 504, 2040)
UV_IND_WALL    = tile_uv_quad(520, 1544, 1016, 2040)
UV_SOLAR_BLUE  = tile_uv_quad(1032, 1544, 1528, 2040)
UV_SOLAR_MONO  = tile_uv_quad(1544, 1544, 2040, 2040)
UV_CORE_WALL   = tile_uv_quad(1544, 8, 2040, 376)
UV_ROOF_STD    = tile_uv_quad(1544, 392, 2040, 760)
UV_ROOF_IND    = tile_uv_quad(1544, 776, 2040, 1144)
UV_FOLIAGE     = tile_uv_quad(1544, 1160, 1784, 1528)
UV_TRUNK       = tile_uv_quad(1800, 1160, 2040, 1336)
UV_STEEL       = tile_uv_quad(1800, 1352, 2040, 1528)

# Now build unified mesh (vertices with pos, normal, uv) for both OBJ and GLB!
verts_pos = []
verts_norm = []
verts_uv = []
faces_idx = [] # 0-indexed triangles

obj_lines = [
    "# Wavefront OBJ 3D Model — Industrial Campus & Solar PV Testbed (Full Textured Edition)",
    "# Matches Interactive 3D Preview: Orthomosaic Terrain + 3-Story Window Facades + Solar Cell Grids + Rounded Trees",
    "mtllib factory_campus_3d_model.mtl",
    "usemtl mat_campus_atlas",
    ""
]

def add_quad(p0, p1, p2, p3, norm, uvs):
    base = len(verts_pos)
    for p, uv in zip((p0, p1, p2, p3), uvs):
        verts_pos.append(p)
        verts_norm.append(norm)
        verts_uv.append(uv)
        obj_lines.append(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")
        obj_lines.append(f"vt {uv[0]:.5f} {uv[1]:.5f}")
        obj_lines.append(f"vn {norm[0]:.3f} {norm[1]:.3f} {norm[2]:.3f}")
    faces_idx.append((base, base+1, base+2))
    faces_idx.append((base, base+2, base+3))
    b1 = base + 1
    obj_lines.append(f"f {b1}/{b1}/{b1} {b1+1}/{b1+1}/{b1+1} {b1+2}/{b1+2}/{b1+2}")
    obj_lines.append(f"f {b1}/{b1}/{b1} {b1+2}/{b1+2}/{b1+2} {b1+3}/{b1+3}/{b1+3}")

def add_tri(p0, p1, p2, norm, uv0, uv1, uv2):
    base = len(verts_pos)
    for p, uv in zip((p0, p1, p2), (uv0, uv1, uv2)):
        verts_pos.append(p)
        verts_norm.append(norm)
        verts_uv.append(uv)
        obj_lines.append(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")
        obj_lines.append(f"vt {uv[0]:.5f} {uv[1]:.5f}")
        obj_lines.append(f"vn {norm[0]:.3f} {norm[1]:.3f} {norm[2]:.3f}")
    faces_idx.append((base, base+1, base+2))
    b1 = base + 1
    obj_lines.append(f"f {b1}/{b1}/{b1} {b1+1}/{b1+1}/{b1+1} {b1+2}/{b1+2}/{b1+2}")

def add_cylinder(cx, y0, y1, cz, radius, segments, uvs):
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        x0, z0 = cx + radius * math.cos(a0), cz + radius * math.sin(a0)
        x1, z1 = cx + radius * math.cos(a1), cz + radius * math.sin(a1)
        nx, nz = math.cos((a0 + a1)/2), math.sin((a0 + a1)/2)
        add_quad((x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0), (nx, 0.0, nz), uvs)

# 1. Ground Terrain Plane mapped to Orthomosaic
obj_lines.append("g Terrain_Orthomosaic")
obj_lines.append("o Terrain_Orthomosaic")
NX, NY = 40, 40
xs = np.linspace(XC_MIN, XC_MAX, NX + 1)
ys = np.linspace(YC_MIN, YC_MAX, NY + 1)
for iy in range(NY):
    for ix in range(NX):
        x0, x1 = xs[ix], xs[ix+1]
        y0, y1 = ys[iy], ys[iy+1]
        add_quad(
            (x0, 0.0, -y0), (x1, 0.0, -y0), (x1, 0.0, -y1), (x0, 0.0, -y1),
            (0.0, 1.0, 0.0),
            (ortho_uv(x0, y0), ortho_uv(x1, y0), ortho_uv(x1, y1), ortho_uv(x0, y1))
        )

# 2. All 41 Buildings, Skybridges, Rooftop Units, Carports with exact Preview Facade & Roof Textures
for b in data['buildings']:
    obj_lines.append(f"\ng {b['category']}_{b['id']}")
    obj_lines.append(f"o {b['id']}")
    x0, x1 = b['x0'], b['x1']
    y0, y1 = b['y0'], b['y1']
    z0, z1 = b['z0'], b['z1']
    zs0, zs1 = -y0, -y1

    is_ind = b['category'] == 'industrial_hall'
    is_core = b['category'] in ('building_core', 'rooftop', 'carport', 'auxiliary')
    wall_uvs = UV_IND_WALL if is_ind else (UV_CORE_WALL if is_core else UV_OFFICE_WALL)
    roof_uvs = UV_ROOF_IND if is_ind else UV_ROOF_STD

    # Subdivide long walls into ~25m bays so the 3-story window texture repeats crisply just like Three.js!
    w = x1 - x0
    l = y1 - y0
    nx_bays = max(1, int(round(w / 24.0))) if not is_core else 1
    ny_bays = max(1, int(round(l / 24.0))) if not is_core else 1

    xb = np.linspace(x0, x1, nx_bays + 1)
    yb = np.linspace(y0, y1, ny_bays + 1)

    for i in range(nx_bays):
        xa, xb_i = xb[i], xb[i+1]
        add_quad((xa, z0, zs0), (xb_i, z0, zs0), (xb_i, z1, zs0), (xa, z1, zs0), (0, 0, 1), wall_uvs)
        add_quad((xb_i, z0, zs1), (xa, z0, zs1), (xa, z1, zs1), (xb_i, z1, zs1), (0, 0, -1), wall_uvs)

    for i in range(ny_bays):
        zsa, zsb = -yb[i], -yb[i+1]
        add_quad((x1, z0, zsa), (x1, z0, zsb), (x1, z1, zsb), (x1, z1, zsa), (1, 0, 0), wall_uvs)
        add_quad((x0, z0, zsb), (x0, z0, zsa), (x0, z1, zsa), (x0, z1, zsb), (-1, 0, 0), wall_uvs)

    # Roof mapped to real orthomosaic roof texture (or industrial skylight roof)
    if is_ind:
        add_quad((x0, z1, zs0), (x1, z1, zs0), (x1, z1, zs1), (x0, z1, zs1), (0, 1, 0), roof_uvs)
    else:
        add_quad(
            (x0, z1, zs0), (x1, z1, zs0), (x1, z1, zs1), (x0, z1, zs1),
            (0, 1, 0),
            (ortho_uv(x0, y0), ortho_uv(x1, y0), ortho_uv(x1, y1), ortho_uv(x0, y1))
        )

    if z0 > 0.5:
        add_quad((x0, z0, zs1), (x1, z0, zs1), (x1, z0, zs0), (x0, z0, zs0), (0, -1, 0), roof_uvs)
        cx, cy = (x0 + x1)/2.0, (y0 + y1)/2.0
        for ox, oy in [(-w*0.42, -l*0.4), (w*0.42, -l*0.4), (-w*0.42, l*0.4), (w*0.42, l*0.4)]:
            add_cylinder(cx + ox, 0.0, z0, -(cy + oy), 0.35, 8, UV_STEEL)

# 3. All 49 Tilted 3D Solar PV Arrays with Solar Cell Grid Texture & Cylindrical Support Posts
for s in data['solar_arrays']:
    obj_lines.append(f"\ng Solar_{s['id']}")
    obj_lines.append(f"o {s['id']}")
    suvs = UV_SOLAR_BLUE if s['panel_type'] == 'crystalline_blue' else UV_SOLAR_MONO
    ang = math.radians(s['azimuth_deg'])
    ca, sa = math.cos(ang), math.sin(ang)
    hl, hw = s['length'] / 2.0, s['width'] / 2.0
    zb, zt = s['z_base'], s['z_top']
    mid_z = (zb + zt) / 2.0
    tsign = s.get('tilt_sign', 1)

    def to_world(lx, lz_w, y_off=0.0):
        gx = s['xc'] + lx * ca + lz_w * sa
        gz = -s['yc'] - lx * sa + lz_w * ca
        gy = mid_z - tsign * (lz_w / max(0.1, s['width'])) * (zt - zb) + y_off
        return (gx, gy, gz)

    n_segs = max(1, int(round(s['length'] / 6.0)))
    ls = np.linspace(-hl, hl, n_segs + 1)
    for i in range(n_segs):
        la, lb = ls[i], ls[i+1]
        p0, p1 = to_world(la, -hw), to_world(lb, -hw)
        p2, p3 = to_world(lb,  hw), to_world(la,  hw)
        add_quad(p0, p3, p2, p1, (0.0, 0.92, 0.38), suvs)
        add_quad(p0, p1, p2, p3, (0.0, -0.92, -0.38), UV_STEEL)

    n_posts = max(2, min(6, int(round(s['length'] / 5.5))))
    for px_l in np.linspace(-hl * 0.84, hl * 0.84, n_posts):
        sp = to_world(px_l, 0.0)
        add_cylinder(sp[0], 0.0, mid_z, sp[2], 0.11, 6, UV_STEEL)

# 4. Rounded 3D Campus Trees (Cylindrical Trunks + Spherical/Icosahedral Crowns matching Preview!)
tree_positions = [
    (-115, 118, 9), (-98, 138, 11), (-58, 148, 9),
    (6, 108, 10), (10, 88, 9), (8, 70, 9), (16, -92, 10),
    (82, -132, 8), (142, 92, 9), (145, 62, 8), (-145, -18, 7)
]
obj_lines.append("\ng Campus_Trees")
obj_lines.append("o Campus_Trees")
for tx, ty, tr in tree_positions:
    tz = -ty
    add_cylinder(tx, 0.0, tr * 0.5, tz, 0.55, 7, UV_TRUNK)
    # UV-sphere crown (6 lat bands x 10 lon segments) matching the rounded foliage in preview
    cy_center = tr * 0.65
    rx, ry, rz = tr * 0.60, tr * 0.46, tr * 0.60
    n_lat, n_lon = 6, 10
    for ilat in range(n_lat):
        th0 = math.pi * ilat / n_lat
        th1 = math.pi * (ilat + 1) / n_lat
        for ilon in range(n_lon):
            ph0 = 2.0 * math.pi * ilon / n_lon
            ph1 = 2.0 * math.pi * (ilon + 1) / n_lon
            def sph_pt(th, ph):
                return (
                    tx + rx * math.sin(th) * math.cos(ph),
                    cy_center + ry * math.cos(th),
                    tz + rz * math.sin(th) * math.sin(ph)
                )
            q0, q1, q2, q3 = sph_pt(th0, ph0), sph_pt(th0, ph1), sph_pt(th1, ph1), sph_pt(th1, ph0)
            nx = math.sin((th0+th1)/2)*math.cos((ph0+ph1)/2)
            ny = math.cos((th0+th1)/2)
            nz = math.sin((th0+th1)/2)*math.sin((ph0+ph1)/2)
            add_quad(q0, q1, q2, q3, (nx, ny, nz), UV_FOLIAGE)

# Write MTL referencing campus_texture_atlas.jpg
mtl_content = """# Wavefront MTL Material Library for DJI M30T Factory & Solar Campus 3D Model
newmtl mat_campus_atlas
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.120 0.120 0.120
Ns 30.0
map_Kd campus_texture_atlas.jpg
"""
mtl_path = "artifacts/file_generation/ttl=63d/output/factory_campus_3d_model.mtl"
with open(mtl_path, "w") as f:
    f.write(mtl_content)

obj_path = "artifacts/file_generation/ttl=63d/output/factory_campus_3d_model.obj"
with open(obj_path, "w") as f:
    f.write("\n".join(obj_lines) + "\n")

# Also export Binary glTF 2.0 (.GLB) with the embedded `campus_texture_atlas.jpg` so single-file 3D viewers show all textures automatically!
pos_arr = np.array(verts_pos, dtype=np.float32)
norm_arr = np.array(verts_norm, dtype=np.float32)
# In glTF 2.0, UV (0,0) is top-left of image (so v_gltf = 1.0 - v_obj)
uv_gltf_arr = np.array([[u, 1.0 - v] for u, v in verts_uv], dtype=np.float32)
idx_arr = np.array(faces_idx, dtype=np.uint32).ravel()
img_bytes = open(atlas_path, "rb").read()

pos_b = pos_arr.tobytes()
norm_b = norm_arr.tobytes()
uv_b = uv_gltf_arr.tobytes()
idx_b = idx_arr.tobytes()

def pad4(b):
    return b + b'\x00' * ((4 - len(b) % 4) % 4)

pos_b, norm_b, uv_b, idx_b, img_b = map(pad4, (pos_b, norm_b, uv_b, idx_b, img_bytes))
bin_chunk = pos_b + norm_b + uv_b + idx_b + img_b

offsets = [0]
for part in (pos_b, norm_b, uv_b, idx_b, img_b):
    offsets.append(offsets[-1] + len(part))

gltf_json = {
    "asset": {"version": "2.0", "generator": "DJI M30T Photogrammetry & BIM Digital Twin Exporter"},
    "scene": 0,
    "scenes": [{"name": "Factory & Solar Campus 3D Model", "nodes": [0]}],
    "nodes": [{"name": "FactorySolarCampus", "mesh": 0}],
    "meshes": [{
        "name": "FactorySolarCampusMesh",
        "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3,
            "mode": 4,
            "material": 0
        }]
    }],
    "materials": [{
        "name": "CampusAtlasMaterial",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.08,
            "roughnessFactor": 0.78
        },
        "doubleSided": True
    }],
    "textures": [{"source": 0, "sampler": 0}],
    "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
    "images": [{"bufferView": 4, "mimeType": "image/jpeg", "name": "campus_texture_atlas.jpg"}],
    "buffers": [{"byteLength": len(bin_chunk)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(pos_b), "target": 34962},
        {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(norm_b), "target": 34962},
        {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(uv_b), "target": 34962},
        {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(idx_b), "target": 34963},
        {"buffer": 0, "byteOffset": offsets[4], "byteLength": len(img_bytes)}
    ],
    "accessors": [
        {
            "bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": len(verts_pos), "type": "VEC3",
            "min": pos_arr.min(axis=0).tolist(), "max": pos_arr.max(axis=0).tolist()
        },
        {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": len(verts_norm), "type": "VEC3"},
        {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": len(verts_uv), "type": "VEC2"},
        {"bufferView": 3, "byteOffset": 0, "componentType": 5125, "count": len(idx_arr), "type": "SCALAR"}
    ]
}

jb = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
jb += b' ' * ((4 - len(jb) % 4) % 4)
glb_path = "artifacts/file_generation/ttl=63d/output/factory_campus_3d_model.glb"
with open(glb_path, "wb") as f:
    f.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(jb) + 8 + len(bin_chunk)))
    f.write(struct.pack('<I4s', len(jb), b'JSON') + jb)
    f.write(struct.pack('<I4s', len(bin_chunk), b'BIN\x00') + bin_chunk)

# Update factory_campus_obj_bundle.zip with the textured .obj, .mtl, campus_texture_atlas.jpg, AND self-contained .glb!
zip_path = "artifacts/file_generation/ttl=63d/output/factory_campus_obj_bundle.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(obj_path, "factory_campus_3d_model.obj")
    zf.write(mtl_path, "factory_campus_3d_model.mtl")
    zf.write(atlas_path, "campus_texture_atlas.jpg")
    zf.write(glb_path, "factory_campus_3d_model.glb")

print(f"Updated textured OBJ ({os.path.getsize(obj_path)/1024:.1f} KB), GLB ({os.path.getsize(glb_path)/1024:.1f} KB), and ZIP ({os.path.getsize(zip_path)/1024:.1f} KB)")
print(f"Total vertices: {len(verts_pos)}, Triangles: {len(faces_idx)}")
