#!/usr/bin/env python3
"""
3DTeethLand Inference Script
Runs instance segmentation and landmark detection on dental meshes
"""
import sys
import os
import argparse
import json
import numpy as np

# Add 3dteethland to path (relative to bite_optimizer project)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # bite_optimizer
TEETHLAND_PATH = os.path.join(PROJECT_ROOT, "3dteethland")
sys.path.insert(0, TEETHLAND_PATH)

import torch
import trimesh

# Color palette for instance segmentation (FDI notation)
INSTANCE_COLORS = np.array([
    [125, 125, 125],  # 0: Background/Gingiva
    [255, 0, 0],      # Instance 1
    [0, 255, 0],      # Instance 2
    [0, 0, 255],      # Instance 3
    [255, 255, 0],    # Instance 4
    [255, 0, 255],    # Instance 5
    [0, 255, 255],    # Instance 6
    [255, 128, 0],    # Instance 7
    [128, 0, 255],    # Instance 8
    [0, 255, 128],    # Instance 9
    [255, 0, 128],    # Instance 10
    [128, 255, 0],    # Instance 11
    [0, 128, 255],    # Instance 12
    [255, 128, 128],  # Instance 13
    [128, 255, 128],  # Instance 14
    [128, 128, 255],  # Instance 15
    [255, 255, 128],  # Instance 16
    [255, 128, 255],  # Instance 17
    [128, 255, 255],  # Instance 18
], dtype=np.uint8)

# Landmark class colors
LANDMARK_COLORS = {
    'Mesial': [255, 0, 0],       # Red
    'Distal': [0, 255, 0],       # Green
    'FacialPoint': [0, 0, 255],  # Blue
    'OuterPoint': [255, 255, 0], # Yellow
    'InnerPoint': [255, 0, 255], # Magenta
    'Cusp': [0, 255, 255],       # Cyan
}


def load_mesh(filepath):
    """Load mesh and extract vertices/normals"""
    mesh = trimesh.load(filepath)

    if hasattr(mesh, 'triangles_center'):
        # Use face centroids for better sampling
        points = mesh.triangles_center
        normals = mesh.face_normals
    else:
        points = mesh.vertices
        normals = mesh.vertex_normals

    return points.astype(np.float32), normals.astype(np.float32), mesh


def normalize_pointcloud(points, sigma=17.3281):
    """Z-score normalization as used in 3DTeethLand"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    points_normalized = points_centered / sigma
    return points_normalized, centroid, sigma


def uniform_downsample(points, normals, voxel_size=0.025):
    """Uniform density downsampling using voxel grid"""
    # Simple voxel-based downsampling
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)

    return points[unique_idx], normals[unique_idx], unique_idx


def run_instance_segmentation(input_path, output_path, checkpoint_path, is_upper=True):
    """Run instance segmentation using 3DTeethLand model"""
    print(f"Loading mesh from {input_path}...")
    points, normals, mesh = load_mesh(input_path)
    original_points = points.copy()

    print(f"Loaded {len(points)} points")

    # Normalize
    points_norm, centroid, sigma = normalize_pointcloud(points)

    # Downsample
    points_ds, normals_ds, ds_idx = uniform_downsample(points_norm, normals, voxel_size=0.025)
    print(f"Downsampled to {len(points_ds)} points")

    # Try to load 3DTeethLand model
    try:
        from teethland.models.fullnet import FullNet
        from teethland.nn.point_tensor import PointTensor

        print(f"Loading model from {checkpoint_path}...")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Create model (simplified - may need config adjustments)
        model = FullNet.load_from_checkpoint(checkpoint_path, map_location='cpu')
        model.eval()

        # Prepare input
        features = np.concatenate([points_ds, normals_ds], axis=1)
        coords = torch.from_numpy(points_ds).float()
        feats = torch.from_numpy(features).float()

        # Create PointTensor
        pt = PointTensor(coords=coords.unsqueeze(0), feats=feats.unsqueeze(0))

        print("Running inference...")
        with torch.no_grad():
            outputs = model(pt)

        # Extract instance predictions
        instances = outputs['instances'].squeeze().cpu().numpy()
        labels = outputs.get('labels', None)

    except Exception as e:
        print(f"3DTeethLand model loading failed: {e}")
        print("Using demo instance segmentation...")

        # Improved demo: DBSCAN-based tooth segmentation
        from sklearn.cluster import DBSCAN

        # Use normalized coordinates for clustering
        pts = points_ds.copy()

        # Step 1: Identify gingiva (background) - lowest Z points (base of dental arch)
        # In normalized space, use percentile
        z_threshold = np.percentile(pts[:, 2], 25)  # Bottom 25% is gingiva
        is_gingiva = pts[:, 2] < z_threshold

        instances = np.zeros(len(points_ds), dtype=np.int32)

        # Step 2: For tooth crown points, use DBSCAN clustering
        tooth_mask = ~is_gingiva
        tooth_points = pts[tooth_mask]
        tooth_indices = np.where(tooth_mask)[0]

        if len(tooth_points) > 100:
            # DBSCAN: eps controls minimum distance between clusters
            # In normalized space (sigma=17.3), typical inter-tooth gap ~1-2mm = 0.06-0.12 units
            # Teeth themselves are ~5-10mm wide = 0.3-0.6 units
            # Start with smaller eps to separate teeth, increase if too many clusters
            min_samples = max(3, len(tooth_points) // 1000)

            # Try progressively smaller eps until we get reasonable clusters
            for eps in [0.08, 0.06, 0.05, 0.04, 0.03]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                tooth_clusters = dbscan.fit_predict(tooth_points)

                # DBSCAN labels: -1 = noise, 0+ = cluster ID
                unique_clusters = set(tooth_clusters) - {-1}
                n_clusters = len(unique_clusters)

                print(f"DBSCAN: eps={eps}, clusters={n_clusters}")

                # Target: 8-16 teeth per jaw (full jaw has 16, but some may be missing)
                if 6 <= n_clusters <= 20:
                    break

            # If too many clusters, merge small ones
            if n_clusters > 20:
                # Count cluster sizes
                cluster_ids, cluster_sizes = np.unique(tooth_clusters[tooth_clusters >= 0], return_counts=True)
                # Keep only clusters with significant size
                min_size = len(tooth_points) // 50  # At least 2% of points
                valid_clusters = cluster_ids[cluster_sizes >= min_size]
                # Remap invalid clusters to nearest valid cluster
                for i in range(len(tooth_clusters)):
                    if tooth_clusters[i] >= 0 and tooth_clusters[i] not in valid_clusters:
                        # Find nearest valid cluster
                        dists = []
                        for vc in valid_clusters:
                            vc_points = tooth_points[tooth_clusters == vc]
                            if len(vc_points) > 0:
                                d = np.min(np.linalg.norm(vc_points - tooth_points[i], axis=1))
                                dists.append((d, vc))
                        if dists:
                            tooth_clusters[i] = min(dists)[1]
                unique_clusters = set(tooth_clusters) - {-1}
                n_clusters = len(unique_clusters)
                print(f"After merging: {n_clusters} clusters")

            # Remap cluster IDs to 1, 2, 3, ...
            cluster_map = {-1: 0}  # Noise -> background
            for new_id, old_id in enumerate(sorted(unique_clusters), start=1):
                cluster_map[old_id] = new_id

            tooth_instances = np.array([cluster_map.get(c, 0) for c in tooth_clusters])

            # Assign instances to tooth points
            for i, ti in enumerate(tooth_indices):
                instances[ti] = tooth_instances[i]

        # Gingiva stays as 0
        labels = list(range(1, int(instances.max()) + 1)) if instances.max() > 0 else []
        # Debug: print downsampled distribution
        unique_ds, counts_ds = np.unique(instances, return_counts=True)
        print(f"Downsampled distribution: {dict(zip(unique_ds.tolist(), counts_ds.tolist()))}")
        print(f"Demo segmentation: found {len(labels)} tooth instances")

    # Map back to original resolution
    full_instances = np.zeros(len(original_points), dtype=np.int32)
    full_instances[ds_idx] = instances

    # Interpolate to all points using nearest neighbor
    if len(ds_idx) < len(original_points):
        from scipy.spatial import cKDTree
        tree = cKDTree(points_norm[ds_idx])
        _, nearest_idx = tree.query(points_norm)
        full_instances = instances[nearest_idx]

    # Create colors based on instances
    colors = INSTANCE_COLORS[full_instances % len(INSTANCE_COLORS)]

    # Save as PLY with colors and instance labels
    save_instance_ply(output_path, original_points, colors, full_instances)

    # Save JSON with instance info
    json_path = output_path.replace('.ply', '.json')
    save_instance_json(json_path, full_instances, labels if labels else list(range(1, int(full_instances.max()) + 1)))

    print(f"Saved instance segmentation to {output_path}")
    return full_instances


def run_landmark_detection(input_path, output_path, checkpoint_path, instance_json=None, is_upper=True):
    """Run landmark detection using 3DTeethLand model"""
    print(f"Loading mesh from {input_path}...")
    points, normals, mesh = load_mesh(input_path)

    # Normalize
    points_norm, centroid, sigma = normalize_pointcloud(points)

    # Downsample for landmark detection (finer than instance seg)
    points_ds, normals_ds, ds_idx = uniform_downsample(points_norm, normals, voxel_size=0.010)
    print(f"Downsampled to {len(points_ds)} points for landmark detection")

    landmarks = []

    try:
        from teethland.models.landmarknet import LandmarkNet

        print(f"Loading landmark model from {checkpoint_path}...")
        model = LandmarkNet.load_from_checkpoint(checkpoint_path, map_location='cpu')
        model.eval()

        # Prepare input
        features = np.concatenate([points_ds, normals_ds], axis=1)
        coords = torch.from_numpy(points_ds).float()
        feats = torch.from_numpy(features).float()

        print("Running landmark inference...")
        with torch.no_grad():
            outputs = model(coords.unsqueeze(0), feats.unsqueeze(0))

        # Extract landmark predictions
        # Output format: seg, mesial_distal, facial, outer, inner, cusps
        landmark_types = ['Mesial', 'Distal', 'FacialPoint', 'OuterPoint', 'InnerPoint', 'Cusp']

        for i, lm_type in enumerate(landmark_types):
            if lm_type in outputs:
                lm_coords = outputs[lm_type].squeeze().cpu().numpy()
                for j, coord in enumerate(lm_coords):
                    # Denormalize coordinates
                    real_coord = coord * sigma + centroid
                    landmarks.append({
                        'class': lm_type,
                        'coord': real_coord.tolist(),
                        'score': 0.9,
                        'instance_id': j
                    })

    except Exception as e:
        print(f"3DTeethLand landmark model loading failed: {e}")
        print("Using demo landmark detection...")

        # Demo: detect landmarks based on curvature/extrema
        # Find local extrema as potential landmarks

        # Load instance info if available
        instances = None
        if instance_json and os.path.exists(instance_json):
            with open(instance_json, 'r') as f:
                inst_data = json.load(f)
                instances = np.array(inst_data['instances'])

        if instances is not None:
            unique_instances = [i for i in np.unique(instances) if i > 0]
        else:
            unique_instances = [1]

        landmark_types = ['Mesial', 'Distal', 'FacialPoint', 'Cusp']

        for inst_id in unique_instances[:8]:  # Limit to 8 teeth for demo
            mask = instances == inst_id if instances is not None else np.ones(len(points), dtype=bool)
            inst_points = points[mask]

            if len(inst_points) < 10:
                continue

            # Find extrema points as landmarks
            # Mesial: min X
            mesial_idx = np.argmin(inst_points[:, 0])
            landmarks.append({
                'class': 'Mesial',
                'coord': inst_points[mesial_idx].tolist(),
                'score': 0.85,
                'instance_id': int(inst_id)
            })

            # Distal: max X
            distal_idx = np.argmax(inst_points[:, 0])
            landmarks.append({
                'class': 'Distal',
                'coord': inst_points[distal_idx].tolist(),
                'score': 0.85,
                'instance_id': int(inst_id)
            })

            # Facial: max Y (or min Y depending on orientation)
            facial_idx = np.argmax(inst_points[:, 1])
            landmarks.append({
                'class': 'FacialPoint',
                'coord': inst_points[facial_idx].tolist(),
                'score': 0.80,
                'instance_id': int(inst_id)
            })

            # Cusp: max Z (highest point)
            cusp_idx = np.argmax(inst_points[:, 2])
            landmarks.append({
                'class': 'Cusp',
                'coord': inst_points[cusp_idx].tolist(),
                'score': 0.90,
                'instance_id': int(inst_id)
            })

    # Save landmarks as JSON
    save_landmarks_json(output_path, landmarks, input_path)

    # Also save as PLY for visualization
    ply_path = output_path.replace('.json', '.ply')
    save_landmarks_ply(ply_path, landmarks)

    print(f"Saved {len(landmarks)} landmarks to {output_path}")
    return landmarks


def save_instance_ply(filepath, points, colors, instances):
    """Save instance segmentation as PLY with colors and labels"""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")

        for i in range(len(points)):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} {instances[i]}\n")


def save_instance_json(filepath, instances, labels):
    """Save instance segmentation metadata as JSON"""
    data = {
        'instances': instances.tolist(),
        'labels': labels,
        'num_instances': int(instances.max())
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_landmarks_json(filepath, landmarks, source_file):
    """Save landmarks in 3DTeethLand format"""
    data = {
        'version': '1.1',
        'description': 'landmarks',
        'key': os.path.basename(source_file),
        'objects': [
            {
                'key': f'landmark_{i}',
                'score': lm['score'],
                'class': lm['class'],
                'coord': lm['coord'],
                'instance_id': lm['instance_id']
            }
            for i, lm in enumerate(landmarks)
        ]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_landmarks_ply(filepath, landmarks):
    """Save landmarks as PLY point cloud for visualization"""
    if not landmarks:
        return

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(landmarks)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")

        for i, lm in enumerate(landmarks):
            color = LANDMARK_COLORS.get(lm['class'], [255, 255, 255])
            coord = lm['coord']
            # Use instance_id as label
            label = lm.get('instance_id', 0)
            f.write(f"{coord[0]} {coord[1]} {coord[2]} "
                   f"{color[0]} {color[1]} {color[2]} {label}\n")


def main():
    parser = argparse.ArgumentParser(description="3DTeethLand Inference")
    parser.add_argument("--input", "-i", required=True, help="Input mesh file")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--mode", "-m", choices=['instances', 'landmarks', 'both'],
                       default='both', help="Inference mode")
    parser.add_argument("--instseg-ckpt",
                       default=os.path.join(PROJECT_ROOT, "instseg_full.ckpt"),
                       help="Instance segmentation checkpoint")
    parser.add_argument("--landmarks-ckpt",
                       default=os.path.join(PROJECT_ROOT, "landmarks_full.ckpt"),
                       help="Landmark detection checkpoint")
    parser.add_argument("--upper", action="store_true", help="Upper jaw (default)")

    args = parser.parse_args()

    base_output = os.path.splitext(args.output)[0]

    instance_json = None

    # For instances or both mode, run instance segmentation and save output
    if args.mode in ['instances', 'both']:
        inst_output = base_output + '_instances.ply' if args.mode == 'both' else args.output
        if not inst_output.endswith('.ply'):
            inst_output = base_output + '.ply'
        run_instance_segmentation(args.input, inst_output, args.instseg_ckpt, args.upper)
        instance_json = inst_output.replace('.ply', '.json')

    if args.mode in ['landmarks', 'both']:
        lm_output = base_output + '_landmarks.json' if args.mode == 'both' else args.output
        if not lm_output.endswith('.json'):
            lm_output = base_output + '.json'

        # For landmarks-only mode, run instance segmentation internally first (2-stage pipeline)
        if args.mode == 'landmarks' and instance_json is None:
            print("Running internal instance segmentation for landmark detection...")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='_instances.ply', delete=False) as tmp:
                tmp_inst_output = tmp.name
            run_instance_segmentation(args.input, tmp_inst_output, args.instseg_ckpt, args.upper)
            instance_json = tmp_inst_output.replace('.ply', '.json')

        run_landmark_detection(args.input, lm_output, args.landmarks_ckpt, instance_json, args.upper)


if __name__ == "__main__":
    main()
