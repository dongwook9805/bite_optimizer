import numpy as np

def compute_balance(contact_points: np.ndarray, center_x: float = 0.0) -> float:
    """
    Compute lateral balance of contact points.
    
    Args:
        contact_points: (N, 3) array of points in contact (or near contact).
        center_x: The x-coordinate dividing left and right.
        
    Returns:
        balance_score: Negative value indicating imbalance magnitude, or ratio.
        Ideally 0 difference between left and right input count/area.
    """
    if len(contact_points) == 0:
        return 0.0
        
    left_mask = contact_points[:, 0] < center_x
    right_mask = contact_points[:, 0] >= center_x
    
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    
    # Simple difference metric
    return -abs(n_left - n_right)

def reward_function(distances: np.ndarray, points: np.ndarray, 
                    w_contact=1.0, w_balance=0.5, w_penalty=2.0) -> float:
    """
    Calculate reward based on distances.
    
    Args:
        distances: Signed distances from mandible to maxilla.
                   Pos = separation, Neg = penetration (if normals correct).
                   Actually for this setup, let's assume `distances` is just Euclidean distance 
                   and we handle penetration separately if possible, OR
                   we rely on the signed distance provided by simulator.
    """
    # 1. Contact Score
    # Ideally distance is between 0 and epsilon (e.g. 0.1mm)
    # distance > 0.5mm -> 0 reward
    # distance < 0 (penetration) -> Penalty
    
    eps = 0.1
    limit = 0.5
    
    # Init rewards
    r_contact = 0
    penalty = 0
    contact_indices = []
    
    for i, d in enumerate(distances):
        if d < 0:
            # Penetration
            penalty += abs(d) * w_penalty # Penalty proportional to depth
        elif d <= eps:
            # Good contact
            r_contact += 1.0
            contact_indices.append(i)
        elif d <= limit:
            # Near contact
            r_contact += 0.3
            
    # Normalize contact score? Or sum?
    # Sum encourages MORE contact points (maximization), which is good for occlusion.
    
    # 2. Balance
    if len(contact_indices) > 0:
        contact_pts = points[contact_indices]
        r_balance = compute_balance(contact_pts) * w_balance
    else:
        r_balance = 0
        
    total_reward = (r_contact * w_contact) + r_balance - penalty
    return total_reward
