import cv2
import numpy as np
import json, math

ortho = cv2.imread("artifacts/file_generation/ttl=63d/output/campus_orthomosaic.jpg")
SCALE_O = 4.0
XC_MIN, YC_MAX = -160.0, 180.0

center = (int((-85 - XC_MIN)*SCALE_O), int((YC_MAX - 10)*SCALE_O))
M = cv2.getRotationMatrix2D(center, 131.5, 1.4)
M[0, 2] += 500 - center[0]
M[1, 2] += 360 - center[1]
M_inv = cv2.invertAffineTransform(M)

# Scale in wa_frame_pixel_grid.jpg: 1 meter = 4.0 * 1.4 = 5.6 pixels!
PX_PER_M = SCALE_O * 1.4

def uv_to_xy(u, v):
    uo = M_inv[0, 0] * u + M_inv[0, 1] * v + M_inv[0, 2]
    vo = M_inv[1, 0] * u + M_inv[1, 1] * v + M_inv[1, 2]
    xc = XC_MIN + uo / SCALE_O
    yc = YC_MAX - vo / SCALE_O
    return xc, yc

solar_arrays = []

def add_from_uv_line(sid, name, zone, u0, v0, u1, v1, width_px, z_base, z_top, ptype, n_panels, face_dir, sub_segments=1, gap_px=0.0):
    # face_dir in wa_frame: "DOWN" (+v), "UP" (-v), "LEFT" (-u), "RIGHT" (+u)
    du, dv = u1 - u0, v1 - v0
    tot_px = math.hypot(du, dv)
    ux, uy = du / tot_px, dv / tot_px
    # Desired facing normal in (u, v):
    if face_dir == "DOWN":
        f_uv = (0.0, 1.0)
    elif face_dir == "UP":
        f_uv = (0.0, -1.0)
    elif face_dir == "LEFT":
        f_uv = (-1.0, 0.0)
    else:
        f_uv = (1.0, 0.0)

    seg_px = (tot_px - gap_px * (sub_segments - 1)) / sub_segments
    panels_per_seg = max(1, int(round(n_panels / sub_segments)))
    width_m = width_px / PX_PER_M
    length_m = seg_px / PX_PER_M

    for i in range(sub_segments):
        dc = i * (seg_px + gap_px) + seg_px / 2.0
        cu = u0 + ux * dc
        cv = v0 + uy * dc
        su0, sv0 = cu - ux * (seg_px / 2.0), cv - uy * (seg_px / 2.0)
        su1, sv1 = cu + ux * (seg_px / 2.0), cv + uy * (seg_px / 2.0)
        x0, y0 = uv_to_xy(su0, sv0)
        x1, y1 = uv_to_xy(su1, sv1)
        xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        az_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
        ang = math.radians(az_deg)
        # Check which tsign (+1 or -1) makes Three.js facing vector match f_uv!
        # In Three.js: when tsign = +1, low edge is local +Z -> in (Xc, Yc) = (+sin(ang), -cos(ang))
        # When tsign = -1, low edge is local -Z -> in (Xc, Yc) = (-sin(ang), +cos(ang))
        fx_pos, fy_pos = math.sin(ang), -math.cos(ang)
        # Convert (xc + fx_pos, yc + fy_pos) to (u, v) and dot with f_uv
        tx, ty = xc + fx_pos, yc + fy_pos
        tuo, tvo = (tx - XC_MIN) * SCALE_O, (YC_MAX - ty) * SCALE_O
        tu = M[0, 0] * tuo + M[0, 1] * tvo + M[0, 2]
        tv = M[1, 0] * tuo + M[1, 1] * tvo + M[1, 2]
        dot_pos = (tu - cu) * f_uv[0] + (tv - cv) * f_uv[1]
        tsign = 1 if dot_pos > 0 else -1

        cur_id = sid if sub_segments == 1 else f"{sid}_s{i+1}"
        cur_name = name if sub_segments == 1 else f"{name} #{i+1}"
        solar_arrays.append({
            "id": cur_id,
            "name": cur_name,
            "zone": zone,
            "xc": round(float(xc), 2),
            "yc": round(float(yc), 2),
            "length": round(float(length_m), 2),
            "width": round(float(width_m), 2),
            "z_base": round(float(z_base), 2),
            "z_top": round(float(z_top), 2),
            "azimuth_deg": round(float(az_deg), 1),
            "panel_type": ptype,
            "panel_count": int(panels_per_seg),
            "tilt_sign": int(tsign)
        })

# 1. Far-Left Circled Standalone 3-Section Solar Rack (6 Panels total: 2 blue + 2 poly + 2 black)
add_from_uv_line("z1_far_left_rack", "Zone 1 Far-Left Standalone Rack", "Zone 1 — NW Solar Canopies", 52, 496, 44, 548, 13, 0.7, 1.8, "crystalline_blue", 6, "LEFT", sub_segments=3, gap_px=2.0)

# 2. Zone 1 — Left Column: 2 Medium Blue Canopies (24 panels each)
add_from_uv_line("z1_med_top", "Zone 1 Medium Canopy Top-Left (24 Panels)", "Zone 1 — NW Solar Canopies", 134, 425, 186, 431, 34, 2.5, 4.1, "crystalline_blue", 24, "DOWN")
add_from_uv_line("z1_med_bot", "Zone 1 Medium Canopy Bot-Left (24 Panels)", "Zone 1 — NW Solar Canopies", 122, 501, 174, 507, 34, 2.5, 4.1, "crystalline_blue", 24, "DOWN")

# 3. Zone 1 — Middle Column: 3 Large Blue Canopies (56 panels each)
add_from_uv_line("z1_large_top", "Zone 1 Large Canopy Top (56 Panels)", "Zone 1 — NW Solar Canopies", 198, 434, 290, 446, 38, 2.5, 4.4, "crystalline_blue", 56, "DOWN")
add_from_uv_line("z1_large_mid", "Zone 1 Large Canopy Mid (56 Panels)", "Zone 1 — NW Solar Canopies", 184, 513, 276, 525, 38, 2.5, 4.4, "crystalline_blue", 56, "DOWN")
add_from_uv_line("z1_large_bot", "Zone 1 Large Canopy Bot (56 Panels)", "Zone 1 — NW Solar Canopies", 170, 593, 262, 605, 38, 2.5, 4.4, "crystalline_blue", 56, "DOWN")

# 4. Zone 1 — 6 Dual-Strip Half-Frame Steel Canopies (2 above Large Top + 4 in Right Column)
add_from_uv_line("z1_hf_top1", "Zone 1 Top-Left Half-Frame Canopy (18 Panels)", "Zone 1 — NW Solar Canopies", 202, 370, 245, 375, 30, 2.4, 3.8, "half_frame_canopy", 18, "DOWN")
add_from_uv_line("z1_hf_top2", "Zone 1 Top-Mid Half-Frame Canopy (18 Panels)",  "Zone 1 — NW Solar Canopies", 256, 376, 299, 381, 30, 2.4, 3.8, "half_frame_canopy", 18, "DOWN")
add_from_uv_line("z1_hf_rt1",  "Zone 1 Right Half-Frame Canopy 1 (18 Panels)",  "Zone 1 — NW Solar Canopies", 316, 384, 359, 389, 30, 2.4, 3.8, "half_frame_canopy", 18, "DOWN")
add_from_uv_line("z1_hf_rt2",  "Zone 1 Right Half-Frame Canopy 2 (18 Panels)",  "Zone 1 — NW Solar Canopies", 302, 456, 345, 461, 30, 2.4, 3.8, "half_frame_canopy", 18, "DOWN")
add_from_uv_line("z1_hf_rt3",  "Zone 1 Right Half-Frame Canopy 3 (18 Panels)",  "Zone 1 — NW Solar Canopies", 288, 530, 331, 535, 30, 2.4, 3.8, "half_frame_canopy", 18, "DOWN")
add_from_uv_line("z1_hf_rt4",  "Zone 1 Right Half-Frame Canopy 4 (18 Panels)",  "Zone 1 — NW Solar Canopies", 274, 606, 317, 611, 30, 2.4, 3.8, "half_frame_canopy", 18, "DOWN")

# 5. Zone 2 — 32 Modular Dual-Pier Test Stands (4 Horizontal Rows x 8 Stands, facing DOWN +v!)
z2_uv_rows = [
    ("r1", 358, 538, 488, 552, 508, 554, 642, 568, "dark_mono", "dark_mono"),
    ("r2", 354, 574, 484, 588, 504, 590, 638, 604, "dark_mono", "dark_mono"),
    ("r3", 350, 610, 480, 624, 500, 626, 634, 640, "dark_mono", "crystalline_blue"),
    ("r4", 346, 646, 476, 660, 496, 662, 630, 676, "dark_mono", "crystalline_blue"),
]
for rtag, u0a, v0a, u1a, v1a, u0b, v0b, u1b, v1b, pta, ptb in z2_uv_rows:
    add_from_uv_line(f"z2_{rtag}_left",  f"Zone 2 Stand {rtag.upper()} Left",  "Zone 2 — 32 Modular Test Stands", u0a, v0a, u1a, v1a, 11, 0.9, 2.1, pta, 20, "DOWN", sub_segments=4, gap_px=8.0)
    add_from_uv_line(f"z2_{rtag}_right", f"Zone 2 Stand {rtag.upper()} Right", "Zone 2 — 32 Modular Test Stands", u0b, v0b, u1b, v1b, 11, 0.9, 2.1, ptb, 16, "DOWN", sub_segments=4, gap_px=8.0)

# 6. Zone 3 — 4 Linear Collector Rows (6 segments per row = 24 segments, facing DOWN +v!)
z3_uv_rows = [
    ("z3_row1", "Zone 3 Collector Row 1", 476, 404, 672, 426),
    ("z3_row2", "Zone 3 Collector Row 2", 472, 437, 668, 459),
    ("z3_row3", "Zone 3 Collector Row 3", 468, 470, 664, 492),
    ("z3_row4", "Zone 3 Collector Row 4", 464, 503, 660, 525),
]
for rid, rname, u0, v0, u1, v1 in z3_uv_rows:
    add_from_uv_line(rid, rname, "Zone 3 — 4 Linear Collector Rows", u0, v0, u1, v1, 13, 1.0, 2.2, "crystalline_blue", 18, "DOWN", sub_segments=6, gap_px=3.0)

# 7. Zone 4 — 2 Vertical Single-Axis Tracker Strings (Left = 8 Dark segments, Right = 7 Blue segments, facing LEFT -u!)
add_from_uv_line("z4_trk_dark", "Zone 4 Dark Tracker String", "Zone 4 — Single-Axis Trackers", 408, 305, 378, 500, 10, 1.1, 2.2, "dark_mono", 24, "LEFT", sub_segments=8, gap_px=4.0)
add_from_uv_line("z4_trk_blue", "Zone 4 Blue Tracker String", "Zone 4 — Single-Axis Trackers", 438, 312, 408, 506, 13, 1.1, 2.3, "crystalline_blue", 28, "LEFT", sub_segments=7, gap_px=4.5)

# 8. Zone 5 — 2 Vertical Trailer Reference Strings (facing LEFT -u!)
add_from_uv_line("z5_trailer_ne", "Zone 5 Trailer Left Array (36 Panels)",  "Zone 5 — Trailer Reference Arrays", 482, 68, 450, 288, 15, 0.9, 2.5, "crystalline_blue", 36, "LEFT")
add_from_uv_line("z5_trailer_sw", "Zone 5 Trailer Right Array (32 Panels)", "Zone 5 — Trailer Reference Arrays", 510, 88, 482, 272, 15, 0.9, 2.5, "crystalline_blue", 32, "LEFT")

# 9. Zone 6 — 6 Long Vertical Central Strings (4 Dual-Half Truss+Blue Strings + 2 Full Dark Strings, facing LEFT -u!)
z6_uv_dual = [
    ("r1", 564, 102, 545, 228, 544, 232, 523, 368),
    ("r2", 602, 108, 583, 234, 582, 238, 561, 374),
    ("r3", 640, 114, 621, 240, 620, 244, 599, 380),
    ("r4", 678, 120, 659, 246, 658, 250, 637, 386),
]
for rtag, u0t, v0t, u1t, v1t, u0b, v0b, u1b, v1b in z6_uv_dual:
    add_from_uv_line(f"z6_{rtag}_truss", f"Zone 6 Row {rtag.upper()} Upper Steel Truss (16 Panels)", "Zone 6 — Central Long Collector Strings", u0t, v0t, u1t, v1t, 14, 1.0, 2.8, "half_frame_canopy", 16, "LEFT")
    add_from_uv_line(f"z6_{rtag}_blue",  f"Zone 6 Row {rtag.upper()} Lower Blue Array (20 Panels)",  "Zone 6 — Central Long Collector Strings", u0b, v0b, u1b, v1b, 16, 1.0, 2.8, "crystalline_blue", 20, "LEFT")

add_from_uv_line("z6_r5_dark", "Zone 6 Row R5 Dark Mono String (44 Panels)", "Zone 6 — Central Long Collector Strings", 714, 126, 673, 396, 15, 1.0, 2.8, "dark_mono", 44, "LEFT")
add_from_uv_line("z6_r6_dark", "Zone 6 Row R6 Wide Dark String (60 Panels)", "Zone 6 — Central Long Collector Strings", 754, 132, 711, 408, 22, 0.9, 3.1, "dark_mono", 60, "LEFT")

# 10. Zone 7 — 5 Stepped South-East Blue Strings (facing LEFT -u!)
z7_uv_rows = [
    ("z7_r1", "Zone 7 Stepped Row 1 (12 Panels)", 598,  2, 586, 56, 12),
    ("z7_r2", "Zone 7 Stepped Row 2 (24 Panels)", 646, -40, 626, 62, 24),
    ("z7_r3", "Zone 7 Stepped Row 3 (24 Panels)", 690, -40, 670, 68, 24),
    ("z7_r4", "Zone 7 Stepped Row 4 (36 Panels)", 742, -95, 712, 74, 36),
    ("z7_r5", "Zone 7 Stepped Row 5 (52 Panels)", 796,-155, 754, 80, 52),
]
for rid, rname, u0, v0, u1, v1, npan in z7_uv_rows:
    add_from_uv_line(rid, rname, "Zone 7 — SE Stepped Solar Strings", u0, v0, u1, v1, 18, 1.0, 2.9, "crystalline_blue", npan, "LEFT")

# Render verification in wa_frame!
vis = ortho.copy()
for s in solar_arrays:
    ang = math.radians(s['azimuth_deg'])
    tsign = s.get('tilt_sign', 1)
    ca, sa = math.cos(ang), math.sin(ang)
    hl, hw = s['length']/2.0, s['width']/2.0
    pts = []
    for lx, lz in [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]:
        gx = s['xc'] + lx * ca + lz * sa
        gy = s['yc'] + lx * sa - lz * ca
        u = int((gx - XC_MIN) * SCALE_O)
        v = int((YC_MAX - gy) * SCALE_O)
        pts.append([u, v])
    col = (255, 220, 0) if s['panel_type'] == 'crystalline_blue' else ((0, 255, 255) if s['panel_type'] == 'half_frame_canopy' else (255, 0, 255))
    cv2.polylines(vis, [np.array(pts, dtype=np.int32)], True, col, 2)
    fx =  sa if tsign > 0 else -sa
    fy = -ca if tsign > 0 else  ca
    cu = int((s['xc'] - XC_MIN) * SCALE_O)
    cv = int((YC_MAX - s['yc']) * SCALE_O)
    eu = int(((s['xc'] + fx * 3.8) - XC_MIN) * SCALE_O)
    ev = int((YC_MAX - (s['yc'] + fy * 3.8)) * SCALE_O)
    cv2.arrowedLine(vis, (cu, cv), (eu, ev), (0, 255, 0), 2, tipLength=0.35)

aligned = cv2.warpAffine(vis, M, (1000, 720), flags=cv2.INTER_CUBIC)
cv2.imwrite("artifacts/file_generation/ttl=63d/intermediate/wa_grid_verified_perfect.jpg", aligned)

with open("artifacts/file_generation/ttl=63d/intermediate/web_scene_data.json") as f:
    scene_data = json.load(f)
scene_data["solar_arrays"] = solar_arrays
with open("artifacts/file_generation/ttl=63d/intermediate/web_scene_data.json", "w") as f:
    json.dump(scene_data, f)

print(f"Saved {len(solar_arrays)} calibrated solar structures ({sum(s['panel_count'] for s in solar_arrays)} panels) to web_scene_data.json!")
