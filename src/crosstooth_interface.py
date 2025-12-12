import torch
import numpy as np
import os
import sys
import trimesh

# Add crosstooth to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crosstooth.models.PTv1.point_transformer_seg import PointTransformerSeg38
from crosstooth.dataset.data import ToothData
from crosstooth.utils import label2color_lower

class ScalingPlaceholder:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def segment_mesh(mesh_path: str, output_path: str = None) -> str:
    """
    Segments the mesh at mesh_path.
    Returns path to colored PLY.
    """
    
    device = torch.device("cpu") # Mac M1/M2 usually MPS not fully supported by all Ops in PTv1, stick to CPU for safety or try "mps"
    if torch.backends.mps.is_available():
         # MPS might fail if custom kernels not supported. 
         # CrossTooth uses safe ops mostly? It uses Point Transformer.
         # Let's default to CPU to be safe first.
         pass

    # Model configuration
    # Assuming lower jaw for now or generic?
    # Original predict.py hardcoded num_classes=17+2.
    point_model = PointTransformerSeg38(
        in_channels=6, 
        num_classes=17 + 2, 
        pretrain=False, 
        add_cbl=False, 
        enable_pic_feat=False
    ).to(device)

    # Load weights
    weights_path = os.path.join(os.path.dirname(__file__), 'crosstooth/weights/point_best_model.pth')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    pretrained_dict = torch.load(weights_path, map_location=device)
    point_model.load_state_dict(pretrained_dict)
    point_model.eval()

    # Prepare Args for Dataset
    # num_points=16000? 
    args = ScalingPlaceholder(
        num_points=16000, 
        sample_points=16000, 
        case=mesh_path # Passed to dataset.__getitem__ via case? No, get_by_name uses dataset.__getitem__(name)
    )
    
    dataset = ToothData(args)
    # dataset.get_by_name(mesh_path) -> calls __getitem__(mesh_path)
    
    try:
        pointcloud, point_coords, face_info = dataset.get_by_name(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        raise e

    # Add batch dim
    pointcloud = pointcloud.unsqueeze(0).to(device)
    pointcloud = pointcloud.permute(0, 2, 1).contiguous() # [B, C, N]
    
    with torch.no_grad():
        point_seg_result, _ = point_model(pointcloud)
        pred_softmax = torch.nn.functional.softmax(point_seg_result, dim=1)
        _, pred_classes = torch.max(pred_softmax, dim=1)
        
    # Process result
    pred_mask = pred_classes.squeeze(0).cpu().numpy().astype(np.uint8)
    # Filter background? 17, 18?
    # CrossTooth logic: 
    # pred_mask[pred_mask == 17] = 0
    # pred_mask[pred_mask == 18] = 0
    
    # Palette
    lower_palette = np.array(
        [[125, 125, 125]] +
        [[label2color_lower[label][2][0],
          label2color_lower[label][2][1],
          label2color_lower[label][2][2]]
         for label in range(1, 17)], dtype=np.uint8)
         
    # Handle Out of Bounds?
    # Ensure indices don't exceed palette
    # Palette has 17 entries (0-16). shape (17, 3).
    # If pred_mask has 17 or 18, it will crash.
    pred_mask[pred_mask >= len(lower_palette)] = 0
    
    # Colorize faces
    # pred_mask corresponds to `pointcloud` which was sampled/permuted faces?
    # Wait. `ToothData` returns `pointcloud` which is a subset/permuted set of FACES.
    # The output prediction gives labels for these *sampled faces*.
    # How do we map back to original mesh faces?
    # `face_info` returned by dataset contains the original indices?
    # In `dataset.py`: `face_info = face_info[permute]`
    # So `face_info` tracks which faces were selected?
    # Actually `face_info` in `data.py` (original) was:
    # face_info = np.array(mesh.cells()) -> These are vertex indices for each face.
    # It does NOT track the *index of the face* in the original mesh.
    # So we know the vertex indices of the classified face, but not easily the original face index unless we match?
    # BUT `utils.output_pred_ply` likely reconstructed a mesh from `point_coords` and colorized it?
    # Let's check `utils.output_pred_ply`.
    
    # If we want to visualize in Three.js, we prefer a colored PLY/STL.
    # If `output_pred_ply` generates a mesh based on `point_coords` and `face_info` (which are permuted), 
    # then the output mesh will be defined by these shuffled faces. 
    # `point_coords` passed to `output_pred_ply` is the ORIGINAL vertices (unpermuted).
    # `face_info` passed is PERMUTED face connectivity.
    # So we are constructing a partial mesh?
    
    # Let's rely on `utils.output_pred_ply` logic.
    # But wait, `dataset` returns `point_coords` (original vertices).
    # `face_info` (permuted faces).
    # If we construct mesh from `point_coords` and `face_info`, we effectively reconstruct the sampled mesh.
    
    colors = lower_palette[pred_mask]
    
    # Create colored mesh
    # Trimesh can create a mesh with face colors
    # face_colors = colors
    
    # We need to constructing the mesh using `face_info` as faces and `point_coords` as vertices.
    # `face_info` contains indices into `point_coords`.
    
    reconstructed_mesh = trimesh.Trimesh(vertices=point_coords, faces=face_info)
    reconstructed_mesh.visual.face_colors = colors
    
    if output_path is None:
        output_path = mesh_path.replace('.stl', '_seg.ply').replace('.ply', '_seg.ply')
        
    reconstructed_mesh.export(output_path)
    
    return output_path
