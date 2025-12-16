#!/usr/bin/env python3
"""
Tooth Segmentation Script
Called by ToothViewer C++ app to perform AI segmentation
"""
import sys
import os
import argparse
import numpy as np

# Add script directory to path (for pure Python pointops)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Monkey-patch: Replace CUDA pointops with pure Python version BEFORE importing CrossTooth
try:
    from pointops.functions import pointops as pure_pointops

    # Create fake module structure
    class FakePointopsModule:
        pointops = pure_pointops

    sys.modules['models.PointTransformer.libs.pointops.functions'] = FakePointopsModule()
    sys.modules['models.PointTransformer.libs.pointops'] = type(sys)('pointops')
    sys.modules['models.PointTransformer.libs'] = type(sys)('libs')
    sys.modules['models.PointTransformer'] = type(sys)('PointTransformer')
    print("Using pure Python pointops (CPU)")
except Exception as e:
    print(f"Failed to load pure Python pointops: {e}")

# Add CrossTooth path (relative to bite_optimizer project)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # bite_optimizer
CROSSTOOTH_PATH = os.path.join(PROJECT_ROOT, "CrossTooth_CVPR2025")
sys.path.insert(0, CROSSTOOTH_PATH)

# Try to import torch and model
USE_AI = False
try:
    import torch
    from models.PTv1.point_transformer_seg import PointTransformerSeg38
    from utils import label2color_upper, label2color_lower
    USE_AI = True
    print("AI model loaded successfully")
except ImportError as e:
    print(f"Warning: AI model not available ({e}), using demo segmentation")

# Default color palette for visualization (17 teeth classes + gingiva)
PALETTE = np.array([
    [125, 125, 125],  # 0: gingiva (gray)
    [255, 0, 0],      # 1: tooth (red)
    [0, 255, 0],      # 2: tooth (green)
    [0, 0, 255],      # 3: tooth (blue)
    [255, 255, 0],    # 4: tooth (yellow)
    [255, 0, 255],    # 5: tooth (magenta)
    [0, 255, 255],    # 6: tooth (cyan)
    [255, 128, 0],    # 7: tooth (orange)
    [128, 0, 255],    # 8: tooth (purple)
    [0, 255, 128],    # 9: tooth (spring green)
    [255, 0, 128],    # 10: tooth (rose)
    [128, 255, 0],    # 11: tooth (chartreuse)
    [0, 128, 255],    # 12: tooth (azure)
    [255, 128, 128],  # 13: tooth (light red)
    [128, 255, 128],  # 14: tooth (light green)
    [128, 128, 255],  # 15: tooth (light blue)
    [255, 255, 128],  # 16: tooth (light yellow)
], dtype=np.uint8)


def normalize_pointcloud(points):
    """
    Normalize point cloud: center and scale to unit sphere
    This matches CrossTooth's PointcloudNormalize
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    points_normalized = points_centered / max_dist
    return points_normalized, centroid, max_dist


def load_mesh_as_pointcloud(filepath, num_points=32000):
    """Load OBJ/PLY file and sample face centroids (like CrossTooth)"""
    import trimesh

    mesh = trimesh.load(filepath)

    # Get face centroids and normals (like CrossTooth)
    if hasattr(mesh, 'triangles_center'):
        cell_coords = mesh.triangles_center
        cell_normals = mesh.face_normals
    else:
        # Fallback for non-triangular meshes
        cell_coords = mesh.vertices[mesh.faces].mean(axis=1)
        cell_normals = mesh.face_normals

    n_faces = len(cell_coords)

    # Pad or sample to num_points
    sampled_indices = None
    if n_faces < num_points:
        # Pad with zeros
        padding = np.zeros((num_points - n_faces, 3))
        cell_coords = np.concatenate([cell_coords, padding], axis=0)
        cell_normals = np.concatenate([cell_normals, padding], axis=0)
        sampled_indices = np.arange(n_faces)  # All faces are used
    else:
        # Random sample
        sampled_indices = np.random.permutation(n_faces)[:num_points]
        cell_coords = cell_coords[sampled_indices]
        cell_normals = cell_normals[sampled_indices]

    # Normalize coordinates (critical for model!)
    cell_coords_norm, centroid, scale = normalize_pointcloud(cell_coords)

    # Combine position (3) + normal (3) = 6 channels
    pointcloud = np.concatenate([cell_coords_norm, cell_normals], axis=1).astype(np.float32)

    # Return sampled indices for mapping predictions back to original mesh
    return pointcloud, cell_coords, mesh, sampled_indices


def demo_segment(input_path, output_path, num_points=32000):
    """Demo segmentation without AI model - colors based on position"""
    print(f"Loading mesh from {input_path}...")
    pointcloud, points, mesh, sampled_indices = load_mesh_as_pointcloud(input_path, num_points)

    print("Running demo segmentation (position-based coloring)...")

    # Simple demo: assign colors based on x-position (simulate teeth)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    x_range = x_max - x_min

    # Divide into ~8 regions (simulating teeth)
    num_regions = 8
    region_size = x_range / num_regions

    pred_mask = np.zeros(len(points), dtype=np.uint8)
    for i in range(len(points)):
        region = int((points[i, 0] - x_min) / region_size)
        region = min(region, num_regions - 1)
        # Use alternating colors for teeth, 0 for some as gingiva
        if points[i, 1] > np.median(points[:, 1]):  # Upper part = gingiva
            pred_mask[i] = 0
        else:
            pred_mask[i] = region + 1  # Tooth classes 1-8

    colors = PALETTE[pred_mask]

    # Save as PLY with colors and labels
    save_colored_ply(output_path, points, colors, pred_mask)
    print(f"Saved demo segmentation result to {output_path}")

    return pred_mask


def segment_with_ai(input_path, output_path, model_path, num_points=32000, is_upper=True):
    """Run AI segmentation on mesh file"""
    # Apple Silicon MPS 가속 시도
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS acceleration", flush=True)
        MAX_POINTS = 16000  # MPS는 더 많이 처리 가능
    else:
        device = torch.device("cpu")
        print("Using CPU (slower)", flush=True)
        MAX_POINTS = 6000  # CPU는 더 작게

    if num_points > MAX_POINTS:
        print(f"Reducing points: {num_points} → {MAX_POINTS}", flush=True)
        num_points = MAX_POINTS

    print(f"Loading model from {model_path}...", flush=True)
    model = PointTransformerSeg38(
        in_channels=6,
        num_classes=17 + 2,
        pretrain=False,
        add_cbl=False,
        enable_pic_feat=False
    ).to(device)

    pretrained_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(pretrained_dict)
    model.eval()

    print(f"Loading mesh from {input_path}...")
    pointcloud, points, original_mesh, sampled_indices = load_mesh_as_pointcloud(input_path, num_points)

    pointcloud_tensor = torch.from_numpy(pointcloud).unsqueeze(0)
    pointcloud_tensor = pointcloud_tensor.permute(0, 2, 1).contiguous()
    pointcloud_tensor = pointcloud_tensor.to(device)

    print("Running AI inference (this may take a while on CPU)...")
    with torch.no_grad():
        seg_result, _ = model(pointcloud_tensor)
        pred_softmax = torch.nn.functional.softmax(seg_result, dim=1)
        _, pred_classes = torch.max(pred_softmax, dim=1)

    pred_mask = pred_classes.squeeze(0).cpu().numpy().astype(np.uint8)
    pred_mask[pred_mask == 17] = 0
    pred_mask[pred_mask == 18] = 0

    # Use CrossTooth color palette if available
    if is_upper:
        palette = np.array(
            [[125, 125, 125]] +
            [[label2color_upper[label][2][0],
              label2color_upper[label][2][1],
              label2color_upper[label][2][2]]
             for label in range(1, 17)], dtype=np.uint8)
    else:
        palette = np.array(
            [[125, 125, 125]] +
            [[label2color_lower[label][2][0],
              label2color_lower[label][2][1],
              label2color_lower[label][2][2]]
             for label in range(1, 17)], dtype=np.uint8)

    # Get colors for sampled points
    colors = palette[pred_mask]

    # Save as point cloud with labels (simpler and faster to render)
    save_colored_ply(output_path, points, colors, pred_mask)
    print(f"Saved AI segmentation result to {output_path} ({len(points)} points)")

    return pred_mask


def save_colored_ply(filepath, points, colors, labels=None):
    """Save point cloud as PLY with colors and optional labels"""
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
        if labels is not None:
            f.write("property uchar label\n")
        f.write("end_header\n")

        for i in range(len(points)):
            if labels is not None:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                       f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} {labels[i]}\n")
            else:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                       f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


def save_colored_mesh_ply(filepath, mesh, face_colors):
    """Save mesh with per-face colors as vertex colors PLY"""
    import trimesh

    vertices = mesh.vertices
    faces = mesh.faces

    # Create vertex colors by averaging face colors for adjacent faces
    # For simplicity, we'll duplicate vertices per face (like flat shading)
    new_vertices = []
    new_colors = []
    new_faces = []

    for i, face in enumerate(faces):
        base_idx = len(new_vertices)
        for v_idx in face:
            new_vertices.append(vertices[v_idx])
            new_colors.append(face_colors[i])
        new_faces.append([base_idx, base_idx + 1, base_idx + 2])

    new_vertices = np.array(new_vertices)
    new_colors = np.array(new_colors, dtype=np.uint8)
    new_faces = np.array(new_faces)

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(new_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(new_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(len(new_vertices)):
            f.write(f"{new_vertices[i, 0]} {new_vertices[i, 1]} {new_vertices[i, 2]} "
                   f"{new_colors[i, 0]} {new_colors[i, 1]} {new_colors[i, 2]}\n")

        for face in new_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="Tooth Segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input mesh file (OBJ/PLY)")
    parser.add_argument("--output", "-o", required=True, help="Output PLY file")
    parser.add_argument("--model", "-m",
                       default=os.path.join(CROSSTOOTH_PATH, "models/PTv1/point_best_model.pth"),
                       help="Model path")
    parser.add_argument("--points", "-n", type=int, default=32000, help="Number of sample points")
    parser.add_argument("--upper", action="store_true", help="Use upper jaw color palette")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (no AI)")

    args = parser.parse_args()

    if USE_AI and not args.demo:
        segment_with_ai(args.input, args.output, args.model, args.points, args.upper)
    else:
        demo_segment(args.input, args.output, args.points)


if __name__ == "__main__":
    main()
