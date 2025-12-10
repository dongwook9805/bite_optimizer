import trimesh
import numpy as np
from typing import Tuple, Dict, Optional

class BiteSimulator:
    def __init__(self, maxilla_path: str, mandible_path: str):
        """
        Initialize the BiteSimulator with paths to upper and lower jaw STLs.
        
        Args:
            maxilla_path: Path to the maxilla (upper jaw, fixed) STL file.
            mandible_path: Path to the mandible (lower jaw, movable) STL file.
        """
        # Load meshes
        self.maxilla: trimesh.Trimesh = trimesh.load(maxilla_path, force='mesh')
        self.mandible_initial: trimesh.Trimesh = trimesh.load(mandible_path, force='mesh')
        
        # Working copy of mandible
        self.mandible = self.mandible_initial.copy()
        
        # Cache for performance (e.g. proximity query structure)
        # We query points on mandible against maxilla surface
        self.maxilla_tree = trimesh.proximity.ProximityQuery(self.maxilla)
        
        # Current state
        self.current_rotation = np.eye(3)
        self.current_translation = np.zeros(3)
        self.transform_matrix = np.eye(4)
        
        # Pre-sample points on mandible occlusal surface for faster contact check?
        # For now, use all vertices or a decimated version could be better for speed.
        self.mandible_points = self.mandible.vertices
        
        # Center of rotation (centroid of mandible or specific condyle point?)
        # For simplicity, let's use the centroid of the mandible as the pivot for now,
        # but in dental context, rotation often happens around condyles.
        # User defined behavior: "Action = ΔR, ΔT". 
        # We will rotate around the mandible's centroid by default unless specified.
        self.pivot_point = self.mandible.centroid

    def reset(self):
        """Reset mandible to initial state."""
        self.mandible = self.mandible_initial.copy()
        self.current_rotation = np.eye(3)
        self.current_translation = np.zeros(3)
        self.transform_matrix = np.eye(4)
        return self.get_state()

    def apply_transform(self, delta_r: np.ndarray, delta_t: np.ndarray):
        """
        Apply incremental rotation and translation to the mandible.
        
        Args:
            delta_r: (3,) degree angles (x, y, z) or rotation matrix. 
                     If 3-vector, assumed euler angles in degrees.
            delta_t: (3,) translation vector (x, y, z) in mm.
        """
        # Convert Euler angles to rotation matrix if needed
        if delta_r.shape == (3,):
            # simple euler to matrix (XYZ order)
            rx = trimesh.transformations.rotation_matrix(np.radians(delta_r[0]), [1, 0, 0], point=self.pivot_point)
            ry = trimesh.transformations.rotation_matrix(np.radians(delta_r[1]), [0, 1, 0], point=self.pivot_point)
            rz = trimesh.transformations.rotation_matrix(np.radians(delta_r[2]), [0, 0, 1], point=self.pivot_point)
            matrix_r = trimesh.transformations.concatenate_matrices(rx, ry, rz)
        else:
            matrix_r = delta_r

        # Translation matrix
        matrix_t = trimesh.transformations.translation_matrix(delta_t)
        
        # Combine: Translate then Rotate? Or Rotate then Translate?
        # Usually for small deltas it's applied on top of current.
        # We apply this transform to the MESH directly.
        final_transform = trimesh.transformations.concatenate_matrices(matrix_t, matrix_r)
        
        self.mandible.apply_transform(final_transform)
        
        # Update internal state tracking
        self.current_translation += delta_t
        # Track cumulative transform: new_accum = final_transform @ old_accum
        self.transform_matrix = trimesh.transformations.concatenate_matrices(final_transform, self.transform_matrix)

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate contact scores and other metrics.
        Returns dictionary with 'contact_score', 'balance', 'penetration_depth' etc.
        """
        # 1. Calculate Signed Distance from Mandible vertices to Maxilla surface
        # Positive distance = outside, Negative = inside (penetration) 
        # Trimesh signed_distance returns (distance, index_triangle)
        # Note: Trimesh signed_distance is expensive. 
        # 'trimesh.proximity.signed_distance' requires watertight mesh for correct signs usually.
        # If open surface (dental scans usually open), we might need just 'proximity' (unsigned).
        # But user specified "negative distance = penetration". 
        # We'll assume the maxilla normals point "out" (towards oral cavity).
        # If point is 'behind' surface, dot(normal, vector_to_point) < 0.
        
        # Let's use trimesh.proximity.closest_point for distances
        closest_points, distances, triangle_id = self.maxilla_tree.on_surface(self.mandible.vertices)
        
        # To get sign (penetration), we need normal check.
        # normals at closest points on maxilla
        normals = self.maxilla.face_normals[triangle_id]
        
        # Vector from maxilla surface point to mandible point
        vecs = self.mandible.vertices - closest_points
        
        # Project vector onto normal. If dot product is negative -> penetration (assuming normals point out)
        # However, for 2 shells facing each other, maxilla normals point DOWN, mandible normals point UP.
        # We need to be careful with conventions. 
        # Let's compute a simplified penetration: if distance is small, it's contact.
        # True penetration requires Inside/Outside check which is hard for open meshes.
        # For now, we will use a "Projected Distance" approach.
        
        signed_dists = np.einsum('ij,ij->i', vecs, normals)
        
        # Filter for relevant contact zone (e.g. distance < 2mm) to speed up or reduce noise
        # We optimize for maximizing points with dist ~ 0.
        
        return {
            "distances": signed_dists,
            "points": self.mandible.vertices
        }

    def get_state(self):
        # Placeholder for full state vector
        return np.concatenate([self.current_translation.flatten(), np.zeros(3)]) # simplified
