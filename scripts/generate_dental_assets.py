import trimesh
import numpy as np
import os

def create_arch(name="arch", invert=False):
    """
    Create a simplified dental arch.
    
    Args:
        name: Name of the mesh
        invert: If True, teeth point UP (Mandible). If False, point DOWN (Maxilla).
    """
    
    # 1. Create the base "Gum" (U-shape)
    # We'll use a boolean difference of two cylinders to make a generic arch curve? 
    # Or just place boxes along a curve. Placing boxes is easier.
    
    meshes = []
    
    # Arch parameters
    radius = 30.0 # mm
    num_teeth = 14
    angle_span = np.pi # 180 degrees
    
    # Generate positions along a semi-circle
    angles = np.linspace(0, angle_span, num_teeth)
    
    for i, theta in enumerate(angles):
        # Parametric position
        # Let's make it slightly elliptical? x=r*cos, y=0.8*r*sin
        x = radius * np.cos(theta)
        y = radius * 0.8 * np.sin(theta)
        
        # Tooth box
        # Size: 8mm width, 8mm depth, 10mm height
        tooth = trimesh.creation.box(extents=[6, 8, 10])
        
        # Position
        z_pos = 5 if invert else -5 # Teeth origin offset
        
        # Rotation: tangent to the curve
        # Tangent angle is theta + 90 deg approximately
        rot_angle = theta + np.pi/2
        R = trimesh.transformations.rotation_matrix(rot_angle, [0, 0, 1])
        
        T = trimesh.transformations.translation_matrix([x, y, z_pos])
        
        M = trimesh.transformations.concatenate_matrices(T, R)
        tooth.apply_transform(M)
        
        meshes.append(tooth)
        
    # Combine all teeth
    full_mesh = trimesh.util.concatenate(meshes)
    
    # Center the centroid roughly
    full_mesh.apply_translation(-full_mesh.centroid)
    
    # If invert (Mandible), we want it lower than Maxilla
    # But usually 0,0,0 is occlusal plane.
    # If Maxilla (invert=False), teeth point down (Z < 0). Base is at Z > 0 (not drawn yet)
    # If Mandible (invert=True), teeth point up (Z > 0).
    
    # Let's shift them so they barely touch at Z=0
    # Current extents:
    # bounds = full_mesh.bounds
    # Z range is approx [-5, 5] locally for each tooth? No, we set extents=[6,8,10], so -5 to 5.
    
    # Maxilla: We want teeth at Z=[-10, 0] or similar.
    # Mandible: We want teeth at Z=[0, 10].
    
    # Right now, 'z_pos' was just the center of the box.
    # If invert=False (Maxilla), center at -5. Box is 10mm tall. Range: [-10, 0].
    # If invert=True (Mandible), center at 5. Box is 10mm tall. Range: [0, 10].
    # This aligns perfectly at Z=0 (Occlusal plane).
    
    # Add a base plate? 
    # Let's skip base plate for simplicity, just floating teeth is enough visual proxy for now.
    
    return full_mesh

def main():
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    print("Generating Maxilla...")
    maxilla = create_arch("maxilla", invert=False)
    # The 'y' positive is usually 'posterior' for standard dental orientation?
    # Or 'anterior'? Let's assume standard: Y+ is forward? Or Y-?
    # In our loop: theta 0 -> pi. 
    # At theta=0, x=r, y=0. At theta=pi/2, x=0, y=0.8r.
    # This makes a C shape facing -Y? 
    # Let's just generate and see.
    maxilla.export(os.path.join(assets_dir, 'dummy_maxilla.stl'))
    
    print("Generating Mandible...")
    mandible = create_arch("mandible", invert=True)
    mandible.export(os.path.join(assets_dir, 'dummy_mandible.stl'))
    
    print("Done.")

if __name__ == "__main__":
    main()
