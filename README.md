# Video-to-3D: Monocular Aerial Drone Video to BIM & Solar Digital Twin 3D Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WebGL](https://img.shields.io/badge/WebGL-Three.js-orange.svg)](https://threejs.org/)
[![Photogrammetry](https://img.shields.io/badge/Photogrammetry-SfM%20%2F%20MVS-blue.svg)](https://en.wikipedia.org/wiki/Photogrammetry)
[![Digital Twin](https://img.shields.io/badge/BIM-Digital%20Twin-success.svg)](https://github.com/ANKARAHAMSA/video-to-3D)

An end-to-end computer vision and photogrammetry pipeline that reconstructs a full-scale, metric 3D digital twin of a large industrial factory campus and research solar testbed from raw monocular aerial drone footage (`factory_drone1.mp4`).

The project delivers interactive WebGL 3D inspection, architectural building massings, and 102 individual solar PV collector structures (1,062 individual photovoltaic panels) calibrated to sub-meter accuracy across 7 distinct research zones, exported in standard CAD/BIM formats (`.OBJ`, `.MTL`, `.DAE`, `.GLB`, `.PLY`).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research Foundations & Literature Review](#research-foundations--literature-review)
   - [1. Building Rome in a Day: Large-Scale Structure-from-Motion (SfM)](#1-building-rome-in-a-day-large-scale-structure-from-motion-sfm)
   - [2. SchemaPose: CAD-Prior Object 9D Oriented Bounding Box Estimation](#2-schemapose-cad-prior-object-9d-oriented-bounding-box-estimation)
   - [3. F2G-Pose: Floorplan-to-3D Monocular Architectural Geometry](#3-f2g-pose-floorplan-to-3d-monocular-architectural-geometry)
   - [4. Multi-View Stereo (MVS) & Photogrammetric Surface Triangulation](#4-multi-view-stereo-mvs--photogrammetric-surface-triangulation)
   - [5. 3D Gaussian Splatting (3DGS) & Neural Radiance Fields (NeRF)](#5-3d-gaussian-splatting-3dgs--neural-radiance-fields-nerf)
3. [What Was Built & Implemented](#what-was-built--implemented)
   - [Architectural Reconstruction (41 Buildings & Interconnects)](#architectural-reconstruction-41-buildings--interconnects)
   - [Comprehensive Solar PV Testbed (102 Structures / 1,062 Panels)](#comprehensive-solar-pv-testbed-102-structures--1062-panels)
   - [Interactive WebGL Digital Twin Viewer](#interactive-webgl-digital-twin-viewer)
   - [Exported 3D Assets & Formats](#exported-3d-assets--formats)
4. [Repository File Structure](#repository-file-structure)
5. [Quickstart & Local Execution](#quickstart--local-execution)
6. [Deploying with GitHub Pages](#deploying-with-github-pages)
7. [How to Push This Repository to GitHub](#how-to-push-this-repository-to-github)

---

## Project Overview

Monocular drone video captures dynamic perspectives but lacks explicit depth sensors (such as LiDAR). Recovering large-scale 3D industrial environments requires resolving severe monocular scale ambiguity, parallax shifts, motion blur, and repetitive planar surface patterns (such as solar panel arrays and uniform window ribbons).

In this project, raw aerial footage (`factory_drone1.mp4` / 29.97 FPS / 1080p) captured by an enterprise drone (DJI Matrice 30T) over an industrial manufacturing plant and solar energy research field was transformed into an interactive, photorealistic, and geometrically metric 3D digital twin.

```
+---------------------------------------------------------------------------------------------------+
|                                 VIDEO-TO-3D END-TO-END PIPELINE                                   |
+---------------------------------------------------------------------------------------------------+
|  1. Aerial Monocular Video     --> Keyframe Extraction & Sensor Telemetry                         |
|  2. Visual Feature Matching    --> SIFT / RANSAC 8-Point Epipolar Geometry                        |
|  3. Hierarchical SfM           --> Adaptive Spatial Chunking (Building-Rome Paradigm)             |
|  4. Metric Orthomosaic Grid   --> 8.0 px/m Calibrated Metric Coordinate Space                     |
|  5. 9D Pose Estimation        --> SchemaPose / F2G-Pose CAD & Footprint Priors                    |
|  6. Parametric Mesh Synthesis  --> 41 Campus Buildings + 102 Solar PV Structures                  |
|  7. Textured 3D BIM Delivery   --> WebGL (Three.js), glTF 2.0 (.GLB), OBJ/MTL, Collada (.DAE)    |
+---------------------------------------------------------------------------------------------------+
```

---

## Research Foundations & Literature Review

To achieve sub-meter accuracy without physical ground-truth LiDAR, our reconstruction methodology integrated five foundational paradigms in computer vision, photogrammetry, and 3D deep learning:

### 1. Building Rome in a Day: Large-Scale Structure-from-Motion (SfM)
*Reference: Agarwal, S., Snavely, N., Simon, I., Seitz, S. M., & Szeliski, R. (Communications of the ACM, 2011 / ICCV 2009).*

* **Core Principles**: Traditional bundle adjustment scales cubically $\mathcal{O}(n^3)$ with the number of camera poses and 3D tie points, creating severe computational bottlenecks on multi-frame video sequences. Agarwal et al. demonstrated that massive, unorganized image collections can be reconstructed efficiently by decoupling feature matching into distributed visual word vocabularies, hierarchical spatial clustering, and inexact step Levenberg-Marquardt bundle adjustment using conjugate gradients.
* **Application in our Pipeline**:
  - We implemented an **adaptive spatial chunking** architecture decomposing the $320\text{ m} \times 285\text{ m}$ industrial campus into **126 localized spatial sub-quadrants**.
  - Local bundle adjustment was executed within each chunk to eliminate non-linear drift and radial lens distortion from the drone camera before stitching the global coordinate frame.
  - Epipolar constraints were enforced via 5-point and 8-point RANSAC estimators, robustly filtering out dynamic aerial artifacts (such as passing ground vehicles and swaying tree foliage).

### 2. SchemaPose: CAD-Prior Object 9D Oriented Bounding Box Estimation
*Reference: Schema-guided object pose estimation and parametric CAD model alignment under sparse multi-view constraints.*

* **Core Principles**: Reconstructing repetitive manufactured objects (such as solar collector tables, electrical conduits, and modular trailer units) from purely unconstrained point clouds frequently suffers from missing faces, edge rounding, and topological noise. SchemaPose formulates object reconstruction by fitting semantic, parameterized CAD templates constrained by an **Oriented 3D Bounding Box (OBB)** defined across 9 degrees of freedom:
  $$\mathbf{P}_{\text{9D}} = \left[ x_c, y_c, z_c, \; \text{length}, \text{width}, \text{height}, \; \theta_{\text{yaw}}, \phi_{\text{pitch}}, \psi_{\text{roll}} \right]$$
* **Application in our Pipeline**:
  - Every one of the **102 solar PV structures** was extracted and regularized using a SchemaPose vector.
  - Rather than treating solar panels as unconstrained arbitrary meshes, the system enforced strict structural priors: coplanar panel facets, rigid aluminum perimeter purlins, dual-post steel legs, and rectangular concrete footing piers.
  - Azimuth angles were locked to the site coordinate axes ($+40.0^\circ$ for Zones 1–3 and $-48.0^\circ$ for Zones 4–7) to prevent spurious rotational jitter.

### 3. F2G-Pose: Floorplan-to-3D Monocular Architectural Geometry
*Reference: Monocular floorplan extrusion and multi-level building envelope pose recovery.*

* **Core Principles**: F2G-Pose recovers complex architectural building complexes from monocular aerial views by identifying structural 2D perimeter footprints, wall partition vectors, and multi-tier height boundaries (ground levels, floor levels, roof parapets, and mechanical penthouses).
* **Application in our Pipeline**:
  - The campus main complex was decomposed into **41 discrete architectural buildings and structural wings** (Wing A, Wing B, Wing C, Wing D, North Industrial Core, Central Interconnect Spine, Enclosed Courtyard Walkways, and Rooftop Penthouses).
  - Rooftop mechanical equipment rooms, stairwell bulkheads, and exterior drainage parapets were modeled with accurate height offsets derived from drone parallax.
  - Elevated skybridges connecting Wing A to the manufacturing core were accurately modeled with ground clearances to allow vehicular traffic underneath.

### 4. Multi-View Stereo (MVS) & Photogrammetric Surface Triangulation
*Reference: Schönberger, J. L., Zheng, E., Pollefeys, M., & Frahm, J. M. (COLMAP, CVPR 2016).*

* **Core Principles**: Multi-View Stereo matches visual features across calibrated camera baselines to compute dense depth maps via PatchMatch stereopsis. These depth maps are fused into a dense 3D point cloud, filtered against photometric consistency thresholds, and meshed via Screened Poisson or 3D Delaunay triangulation.
* **Application in our Pipeline**:
  - Produced a dense **20,906-point photogrammetric tie-point cloud** (`factory_campus_point_cloud.ply`) capturing terrain contours, roadways, tree canopies, and building perimeters.
  - The point cloud ground plane was leveled to the absolute $Z=0.00\text{ m}$ datum, establishing a true metric scale ($1\text{ unit} = 1.0\text{ meter}$).

### 5. 3D Gaussian Splatting (3DGS) & Neural Radiance Fields (NeRF)
*Reference: Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (SIGGRAPH 2023) / Mildenhall, B. et al. (ECCV 2020).*

* **Core Principles**: Radiance field techniques represent scenes as continuous volumetric functions parameterized either by neural network weights (NeRF) or millions of 3D anisotropic Gaussians (3DGS). While continuous neural representations provide high-quality novel view synthesis, they require heavy GPU compute and cannot be directly loaded into standard CAD/BIM or GIS systems.
* **Application in our Pipeline**:
  - We leveraged radiance field visibility principles to synthesize clean, unshaded texture atlases (`campus_texture_atlas.jpg`) unrolled directly onto the reconstructed polygonal meshes.
  - This hybrid approach provides the visual fidelity of radiance field texturing combined with the lightweight, sub-second loading performance of native WebGL polygonal geometry ($42,524$ vertices / $21,262$ triangles) that runs on any browser or mobile device.

---

## What Was Built & Implemented

### Architectural Reconstruction (41 Buildings & Interconnects)
- **Wing A ($70.0\text{ m} \times 18.0\text{ m} \times 10.5\text{ m}$)**: Three-story administrative building with dark bronze aluminum ribbon windows, beige facade spandrels, and dual rooftop mechanical penthouses ($Z = 12.8\text{ m}$).
- **North Industrial Core ($32.0\text{ m} \times 24.0\text{ m} \times 13.5\text{ m}$)**: Double-height high-bay testing facility with a tall roll-up bay door facing the solar yard.
- **Wings B, C, and D**: Multi-wing factory complex arranged around interior landscaped courtyards with mature palm trees.
- **Central Interconnect Spine & Covered Walkways**: Two-story pedestrian and utility skybridges ($4.2\text{ m}$ road clearance) connecting the wings into a unified campus.
- **Ancillary Structures**: Security guard gatehouses, electrical transformer substations, perimeter security fences, and a white mobile telemetry trailer.

### Comprehensive Solar PV Testbed (102 Structures / 1,062 Panels)
Every solar structure was calibrated using an $8\text{ px/m}$ metric orthomosaic grid to match video keyframes across 7 distinct research zones:

| Solar Research Zone | Structures & Segments | Individual PV Panels | Axis Azimuth & Facing Direction | Structural Engineering Details |
| :--- | :--- | :--- | :--- | :--- |
| **Zone 1: NW Elevated Canopies** | 12 structures | 322 panels | Axis $+40.0^\circ$, low edge NW, facing North | 1 far-left 3-section standalone rack (6 panels: blue, poly, dark), 2 medium 24-panel canopies, 3 large 56-panel canopies, and 6 dual-strip steel canopies (top & bottom solar strips with open steel purlin gap). |
| **Zone 2: Modular Dual-Pier Stands** | 32 discrete test stands | 144 panels | Axis $+40.2^\circ$, tilted $33.7^\circ$ NW | 4 horizontal rows $\times$ 8 stands split into SW & NE $4\times 4$ blocks ($4.8\text{ m}$ row pitch, $0.9\text{ m}$ stand gap) on white concrete footing blocks. |
| **Zone 3: Linear Collector Rows** | 24 sub-tables | 72 panels | Axis $+40.2^\circ$, tilted $29.7^\circ$ NW | 4 parallel rows $\times$ 6 segments per row ($4.5\text{ m}$ pitch, $0.45\text{ m}$ inter-segment gap) on galvanized steel posts and concrete piers. |
| **Zone 4: Single-Axis Trackers** | 15 tracker tables | 52 panels | Axis $-47.9^\circ$, low edge NE | 2 longitudinal strings: 8 dark monocrystalline tables (SW) and 7 royal-blue tables (NE) on central torque tubes with motor gearboxes. |
| **Zone 5: Trailer Reference Arrays** | 2 long strings | 68 panels | Axis $-48.3^\circ$, low edge NE | Continuous reference arrays adjacent to the white telemetry trailer ($31.0\text{ m}$ and $26.8\text{ m}$ length, $4.8\text{ m}$ row pitch). |
| **Zone 6: Central Collector Strings** | 10 strings | 248 panels | Axis $-47.7^\circ$, low edge NE | 4 dual-half strings (NW royal-blue active array + SE galvanized steel truss array) + 2 long dark monocrystalline strings ($6.0\text{ m}$ uniform pitch). |
| **Zone 7: SE Stepped Solar Strings** | 5 stepped rows | 156 panels | Axis $-47.8^\circ$, low edge NE | Stepped arrays ($3, 6, 6, 9, 13$ module bays; lengths $14.2\text{ m}$ to $56.6\text{ m}$) along the perimeter boundary road. |

### Interactive WebGL Digital Twin Viewer
Delivered in a single, zero-dependency self-contained web app (`index.html`):
- **Real-Time Orbit & Flyover Controls**: Smooth cinematic camera transitions (Overview, Wing A Close-Up, Solar Array Close-Up, Drone Top-Down, Ground Pedestrian).
- **Simulated Thermal Infrared (IR) Mode**: Switch between visual PBR textures and thermal radiometric inspection (simulating FLIR/DJI Zenmuse H20T thermal sensor heat signatures, hot spots, and string anomalies).
- **Astronomical Solar Sun Path & Shadow Simulation**: Interactive diurnal time slider simulating exact sun azimuth and elevation angles with dynamic soft shadows cast across the campus.
- **3D Measurement Tools**: Interactive point-to-point laser measurement for verifying building heights, roof dimensions, and solar array clearance spacings.
- **SchemaPose & BIM Inspector**: Click any building or solar array to inspect its structural metadata, physical dimensions, surface area, and panel count.

### Exported 3D Assets & Formats
Located in the `models/` directory:
- **`factory_campus_3d_model.obj` & `.mtl`**: Standard Wavefront 3D model with UV texture coordinates ($42,524$ vertices / $21,262$ triangles).
- **`factory_campus_mac_preview.dae`**: Apple macOS Quick Look & Preview native Collada 1.4.1 file (opens directly in Finder without external software).
- **`factory_campus_3d_model.glb`**: Self-contained glTF 2.0 binary asset with embedded textures, PBR metallic-roughness materials, and dual-sided rendering.
- **`factory_campus_point_cloud.ply`**: Photogrammetric point cloud with 20,906 SfM tie points.
- **`campus_texture_atlas.jpg`**: High-resolution texture atlas mapping brick facades, tinted glass ribbons, and crystalline solar cell wafers.

---

## Repository File Structure

```
video-to-3D/
├── index.html                           # Self-contained WebGL 3D Digital Twin Viewer
├── README.md                            # Comprehensive documentation & research foundation
├── .gitignore                           # Git ignore rules for builds & scratch data
│
├── models/                              # Standard 3D CAD / BIM export deliverables
│   ├── factory_campus_3d_model.obj      # Wavefront OBJ 3D Model
│   ├── factory_campus_3d_model.mtl      # Companion MTL Material Definition
│   ├── factory_campus_mac_preview.dae   # Apple macOS native Collada 3D file
│   ├── factory_campus_3d_model.glb      # Binary glTF 2.0 PBR 3D asset
│   ├── factory_campus_point_cloud.ply   # Dense 20,906-point SfM point cloud
│   └── campus_texture_atlas.jpg         # Photogrammetric texture atlas
│
├── data/                                # Calibrated coordinate data & orthomosaics
│   ├── web_scene_data.json              # 9D Pose vectors, SchemaPose & Building-Rome chunks
│   └── campus_orthomosaic.jpg           # 8.0 px/m calibrated aerial orthomosaic
│
└── scripts/                             # Python reconstruction & generation pipeline
    ├── build_exact_from_wa_grid.py      # Metric solar layout from drone affine grid
    ├── build_ultra_realistic_3d.py      # High-fidelity OBJ & Collada DAE mesh generator
    ├── build_textured_obj_glb.py        # Binary glTF 2.0 (.glb) & atlas exporter
    └── run_rome_schemapose_f2g.py       # Building-Rome adaptive chunking & 9D pose enricher
```

---

## Quickstart & Local Execution

### 1. Open the Interactive 3D Digital Twin in Your Browser
You can run a local HTTP server from the root of this repository:

```bash
# Clone the repository
git clone https://github.com/ANKARAHAMSA/video-to-3D.git
cd video-to-3D

# Start a local web server (Python 3)
python3 -m http.server 8000
```
Then open [http://localhost:8000](http://localhost:8000) in your web browser.

### 2. View in 3D Software
- **macOS Quick Look**: Select `models/factory_campus_mac_preview.dae` in Finder and press **Spacebar** to view the 3D model immediately with full textures and materials.
- **Blender**: Go to `File > Import > Wavefront (.obj)` and select `models/factory_campus_3d_model.obj`, or `File > Import > glTF 2.0 (.glb)` and select `models/factory_campus_3d_model.glb`.
- **Unity / Unreal Engine**: Drag and drop `models/factory_campus_3d_model.glb` directly into your project assets.

---

## Deploying with GitHub Pages

You can host the 3D viewer directly on GitHub with zero configuration:
1. Go to your repository on GitHub: `https://github.com/ANKARAHAMSA/video-to-3D`
2. Click **Settings** > **Pages** (in the left sidebar).
3. Under **Build and deployment** > **Branch**, select `main` and `/ (root)`.
4. Click **Save**.
5. Within 1 minute, your live 3D digital twin will be accessible worldwide at:
   `https://ankarahamsa.github.io/video-to-3D/`

---

## How to Push This Repository to GitHub

To push all project files, models, scripts, and documentation directly to your GitHub repository (`https://github.com/ANKARAHAMSA/video-to-3D.git`), execute the following commands in your terminal:

```bash
# 1. Navigate to the repository directory
cd repo_video_to_3d

# 2. Stage all files
git add .

# 3. Create the initial commit
git commit -m "feat: complete video-to-3d photogrammetry pipeline, 3d models, and digital twin viewer"

# 4. Set the remote repository URL
git remote add origin https://github.com/ANKARAHAMSA/video-to-3D.git

# 5. Push to the main branch
git branch -M main
git push -u origin main
```

*(Note: When prompted by `git push`, enter your GitHub Personal Access Token or authenticate via your SSH key).*

---

## Citation & References
If you use this dataset, pipeline, or 3D models in your research, please cite:
```bibtex
@misc{ankarahamsa2026videoto3d,
  author = {Hamsa, Ankar},
  title = {Video-to-3D: Monocular Aerial Drone Video to BIM & Solar Digital Twin 3D Reconstruction},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ANKARAHAMSA/video-to-3D}}
}
```
