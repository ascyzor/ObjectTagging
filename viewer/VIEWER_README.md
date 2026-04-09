# ObjectTagging Viewer

SuperSplat-based 3D Gaussian Splat viewer with automatic object label overlays.

## Requirements

Node.js 20+ (install via conda: `conda install -n base -c conda-forge nodejs=20`)

## Usage

```bash
cd viewer
conda run -n base npm run develop   # development server → http://localhost:3000
# or
conda run -n base npm run build     # production build → dist/
```

## Loading your pipeline output

1. Open http://localhost:3000
2. **File → Open** (or drag-and-drop) your `.ply` file  
   e.g. `outputs/<dataset>/<scene>/<timestamp>/point_cloud/iteration_30000/point_cloud_final_web.ply`
3. **File → Load Centroids JSON...** (or click **Load Centroids JSON** in the bottom-left panel)  
   Select `object_centroids.json` from the same iteration directory
4. Labels appear automatically over each object centroid

The **Object Labels** panel (bottom-left) has:
- **Load Centroids JSON** button — load/reload the JSON at any time
- **Show Labels** toggle — hide/show all labels without clearing them

## Notes

- Background label (ID `0`) is shown; filter it in the panel if needed
- Labels stay projected at centroid world positions as you orbit/pan/zoom
- Labels behind the camera are hidden automatically
- This directory (`viewer/`) is self-contained — the ObjectTagging Python pipeline is unchanged