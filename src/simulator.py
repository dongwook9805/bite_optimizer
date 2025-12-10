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

    def rough_align_icp(self, samples=500, iterations=15):
        """
        Perform rough alignment using Iterative Closest Point (ICP).
        This updates the mandible state to the aligned position.
        
        Args:
            samples: Number of samples to use for ICP from the source mesh.
            iterations: Maximum number of iterations for ICP.
        """
        print("Running ICP alignment...")
        # trimesh.registration.mesh_other(mesh, other, samples=500, scale=False, icp_first=10, icp_final=50)
        # mesh: source (moving), other: target (fixed)
        # returns: transform (4x4), cost
        
        try:
            # We use mesh_other to align mandible to maxilla
            # returns: transform (4x4), cost
            matrix, cost = trimesh.registration.mesh_other(
                self.mandible, 
                self.maxilla, 
                samples=samples, 
                scale=False, 
                icp_first=iterations, 
                icp_final=iterations
            )
            
            # Apply the resulting transform to our state
            # Note: matrix is the transform that moves self.mandible to align with self.maxilla
            
            # We need to decompose this to update current_translation/rotation if we want to track them separately,
            # but our get_state currently just relies on self.transform_matrix if we update it.
            # However, apply_transform updates self.current_translation incrementally.
            # Here we are doing a 'teleport' jump.
            
            # Update the mesh
            self.mandible.apply_transform(matrix)
            
            # Update cumulative transform
            self.transform_matrix = trimesh.transformations.concatenate_matrices(matrix, self.transform_matrix)
            
            # Update translation tracking (approximation, just extracting from matrix)
            # This might break if we strictly rely on accumulated delta_t, but for reset it's fine.
            self.current_translation += matrix[:3, 3] 
            
            # Rotation is harder to track incrementally from proper Euler angles if we just multiply matrices,
            # but if get_state uses the matrix or we don't strictly need Euler history, it's fine.
            
            print(f"ICP converged with cost: {cost}")
            
        except Exception as e:
            print(f"ICP failed: {e}")
            # Fallback or pass (maybe already aligned enough or library issue)
            pass


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

    def get_orthodontic_metrics(self) -> Dict[str, float]:
        """
        Approximates orthodontic metrics from the current mesh state.
        
        Note: Real calculation requires landmarks. We use heuristics:
        - Overjet: Relative Z (or Y depending on orientation) translation vs initial.
        - Overbite: Relative Y/Z overlap.
        - Midline: Relative X translation.
        - Contacts: Derived from simple proximity.
        """
        
        # 1. Decompose current transform
        # We track self.current_translation. 
        # Assume initial state was "ideal" (metrics=ideal) or "neutral" (metrics=0).
        # Let's assume initial state of dummy assets was "End-to-End" (Overjet=0, Overbite=0).
        
        # Our dummy assets are: Maxilla teeth down (Z<0), Mandible teeth up (Z>0). 
        # Gap is Z-axis.
        # Anterior-Posterior is Y-axis (or X?).
        # Lateral is X-axis (or Y?).
        
        # Let's assume:
        # X: Lateral (Midline)
        # Y: Anterior-Posterior (Overjet)
        # Z: Vertical (Overbite/Openbite)
        
        # NOTE: Simulator coordinate system might differ from Dental standard.
        # In dummy gen: 
        # x = radius * cos(theta), y = radius * sin(theta) -> X is lateral, Y is AP.
        # Z is vertical.
        
        tx, ty, tz = self.current_translation
        
        # Overjet: Horizontal overlap. 
        # If mandible moves forward (+Y), overjet decreases (underbite).
        # If mandible moves back (-Y), overjet increases.
        # Let's assume initial overjet was 0.
        overjet_mm = 2.0 - ty  # If we move forward (positive ty), OJ reduces. 
        # Wait, usually Mandible is moving. 
        # Initial: Edge to edge? Let's assume initial is 0.
        # Target: 2mm. So we want ty = -2mm (moved back)? 
        # Or if metrics are absolute:
        overjet_mm = 2.0 + ty # Placeholder logic
        
        # Overbite: Vertical overlap.
        # If mandible moves UP (+Z), bite deepens (Overbite increases).
        # Initial: 0mm (Touching). 
        # Target: 2mm (Overlap). So we want tz = +2mm.
        overbite_mm = tz 
        
        # Midline: Lateral deviation.
        midline_dev_mm = abs(tx)
        
        # Contacts
        # Use existing collision/distance check
        metrics = self.get_metrics()
        distances = metrics['distances']
        points = metrics['points']
        
        # "Contact" defined as distance < 0.2mm
        contact_mask = (distances >= 0) & (distances < 0.2)
        contact_pts = points[contact_mask]
        
        # Ratios
        # Dummy arch: Anterior is max Y? (Tip of U). Posterior is min Y (Legs of U).
        # Centroid Y is roughly 0. 
        # Let's split by Y coordinate.
        y_coords = points[:, 1]
        anterior_mask = y_coords > 0 # Front half
        posterior_mask = y_coords <= 0 # Back half
        
        # Total possible pts in each region (approx)
        total_ant = np.sum(anterior_mask)
        total_post = np.sum(posterior_mask)
        
        # Contact counts in each region
        # We need to map contact_mask to original indices. 
        # get_metrics returns distances for ALL vertices? Yes.
        
        ant_contacts = np.sum(contact_mask & anterior_mask)
        post_contacts = np.sum(contact_mask & posterior_mask)
        
        anterior_contact_ratio = ant_contacts / max(total_ant, 1) * 10.0 # Scale up because contacts are sparse
        posterior_contact_ratio = post_contacts / max(total_post, 1) * 10.0
        
        # Force Balance (Left vs Right)
        # Left: X < 0, Right: X > 0
        left_mask = points[:, 0] < 0
        right_mask = points[:, 0] >= 0
        
        left_force = np.sum(contact_mask & left_mask) # Simply count contacts as "force"
        right_force = np.sum(contact_mask & right_mask)
        
        # Stubs for complex stuff
        working_side_interference = 0.0 # Hard to calc without lateral movement simulation
        nonworking_side_interference = 0.0
        
        # Openbite: Fraction of anterior NOT touching?
        # If anterior_contact_ratio is low, openbite is high?
        # Let's just use 1 - ratio/target
        anterior_openbite_fraction = max(0, 1.0 - anterior_contact_ratio)
        
        # Crossbite: if width mismatch.
        # Stub
        posterior_crossbite_count = 0
        scissors_bite_count = 0
        
        return {
            "overjet_mm": overjet_mm,
            "overbite_mm": overbite_mm,
            "midline_dev_mm": midline_dev_mm,
            "anterior_contact_ratio": min(1.0, anterior_contact_ratio),
            "posterior_contact_ratio": min(1.0, posterior_contact_ratio),
            "left_contact_force": float(left_force),
            "right_contact_force": float(right_force),
            "working_side_interference": working_side_interference,
            "nonworking_side_interference": nonworking_side_interference,
            "anterior_openbite_fraction": anterior_openbite_fraction,
            "posterior_crossbite_count": posterior_crossbite_count,
            "scissors_bite_count": scissors_bite_count,
        }

