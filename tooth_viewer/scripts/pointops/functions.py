"""
Pure PyTorch implementation of pointops functions
Slower than CUDA but works on CPU
"""
import torch
import torch.nn.functional as F


def knn(x, k):
    """
    Find k-nearest neighbors
    x: (N, 3) point coordinates
    Returns: (N, k) indices of k nearest neighbors
    """
    inner = -2 * torch.matmul(x, x.transpose(1, 0))
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(1, 0)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each two points.
    src: (N, 3)
    dst: (M, 3)
    Returns: (N, M)
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(0, 1))
    dist += torch.sum(src ** 2, dim=1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=1).unsqueeze(0)
    return dist


def index_points(points, idx):
    """
    Index points by given indices
    points: (N, C)
    idx: (M,) or (M, K)
    Returns: (M, C) or (M, K, C)
    """
    if idx.dim() == 1:
        return points[idx]
    else:
        # (M, K)
        M, K = idx.shape
        C = points.shape[1]
        idx_flat = idx.reshape(-1)
        gathered = points[idx_flat]
        return gathered.reshape(M, K, C)


class pointops:
    """
    Pure PyTorch implementation of pointops
    """

    @staticmethod
    def furthestsampling(xyz, offset, new_offset):
        """
        Farthest Point Sampling
        xyz: (N, 3) point coordinates
        offset: (B,) cumulative point counts per batch
        new_offset: (B,) cumulative sampled point counts per batch
        Returns: (M,) indices of sampled points
        """
        device = xyz.device
        B = offset.shape[0]

        sampled_indices = []

        for b in range(B):
            start = 0 if b == 0 else offset[b - 1].item()
            end = offset[b].item()
            n_sample = new_offset[b].item() - (0 if b == 0 else new_offset[b - 1].item())

            points = xyz[start:end]  # (n, 3)
            n = points.shape[0]

            if n_sample >= n:
                indices = torch.arange(start, end, device=device)
            else:
                # FPS algorithm
                centroids = torch.zeros(n_sample, dtype=torch.long, device=device)
                distance = torch.ones(n, device=device) * 1e10
                farthest = torch.randint(0, n, (1,), device=device).item()

                for i in range(n_sample):
                    centroids[i] = farthest
                    centroid = points[farthest].unsqueeze(0)
                    dist = torch.sum((points - centroid) ** 2, dim=1)
                    mask = dist < distance
                    distance[mask] = dist[mask]
                    farthest = torch.argmax(distance).item()

                indices = centroids + start

            sampled_indices.append(indices)

        return torch.cat(sampled_indices)

    @staticmethod
    def queryandgroup(feat, xyz, offset, new_xyz, new_offset, nsample, use_xyz=True, idx=None):
        """
        Query and group operation
        feat: (N, C) features
        xyz: (N, 3) coordinates
        offset: (B,) batch offsets
        new_xyz: (M, 3) query coordinates
        new_offset: (B,) query batch offsets
        nsample: number of neighbors
        use_xyz: whether to include xyz in output
        idx: precomputed indices (optional)

        Returns:
            grouped_feat: (M, nsample, C+3) if use_xyz else (M, nsample, C)
            idx: (M, nsample)
        """
        device = xyz.device
        B = offset.shape[0]
        M = new_xyz.shape[0]
        C = feat.shape[1] if feat is not None else 0

        if idx is None:
            idx = torch.zeros(M, nsample, dtype=torch.long, device=device)

            for b in range(B):
                # Original points range
                start = 0 if b == 0 else offset[b - 1].item()
                end = offset[b].item()

                # Query points range
                q_start = 0 if b == 0 else new_offset[b - 1].item()
                q_end = new_offset[b].item()

                batch_xyz = xyz[start:end]  # (n, 3)
                batch_query = new_xyz[q_start:q_end]  # (m, 3)

                # Compute distances
                dist = square_distance(batch_query, batch_xyz)  # (m, n)

                # Get k-nearest neighbors
                _, batch_idx = torch.topk(dist, nsample, dim=-1, largest=False)  # (m, nsample)
                batch_idx = batch_idx + start  # Offset to global indices

                idx[q_start:q_end] = batch_idx

        # Gather features
        if use_xyz:
            grouped_xyz = index_points(xyz, idx)  # (M, nsample, 3)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(1)  # Relative coordinates

            if feat is not None:
                grouped_feat = index_points(feat, idx)  # (M, nsample, C)
                grouped_feat = torch.cat([grouped_xyz, grouped_feat], dim=-1)  # (M, nsample, C+3)
            else:
                grouped_feat = grouped_xyz
        else:
            if feat is not None:
                grouped_feat = index_points(feat, idx)
            else:
                grouped_feat = None

        return grouped_feat, idx

    @staticmethod
    def interpolation(xyz1, xyz2, feat1, offset1, offset2):
        """
        3-NN interpolation: interpolate features from xyz1 (source) to xyz2 (target)

        In CrossTooth TransitionUp: interpolation(p2, p1, x2, o2, o1)
        - xyz1 = p2 (source, sparser, M points)
        - xyz2 = p1 (target, denser, N points)
        - feat1 = x2 (features at source positions, M features)
        - offset1 = o2 (batch offsets for source)
        - offset2 = o1 (batch offsets for target)

        For each point in xyz2 (target), find 3 nearest neighbors in xyz1 (source),
        and interpolate features from feat1.

        Returns: (N, C) interpolated features at xyz2 positions
        """
        device = xyz1.device
        M = xyz1.shape[0]  # source points
        N = xyz2.shape[0]  # target points
        C = feat1.shape[1]
        B = offset1.shape[0]

        output = torch.zeros(N, C, device=device, dtype=feat1.dtype)

        for b in range(B):
            # Source points range (sparser, xyz1/feat1 use offset1)
            src_start = 0 if b == 0 else offset1[b - 1].item()
            src_end = offset1[b].item()

            # Target points range (denser, xyz2 uses offset2)
            tgt_start = 0 if b == 0 else offset2[b - 1].item()
            tgt_end = offset2[b].item()

            batch_xyz_src = xyz1[src_start:src_end]  # (m, 3) source
            batch_xyz_tgt = xyz2[tgt_start:tgt_end]  # (n, 3) target
            batch_feat_src = feat1[src_start:src_end]  # (m, C) source features

            m = batch_xyz_src.shape[0]  # source count
            n = batch_xyz_tgt.shape[0]  # target count

            if m == 0 or n == 0:
                continue

            # Compute distances: for each target point, distance to all source points
            dist = square_distance(batch_xyz_tgt, batch_xyz_src)  # (n, m)

            # Find k nearest source neighbors for each target point
            k = min(3, m)
            dist_k, idx = torch.topk(dist, k, dim=-1, largest=False)  # (n, k)

            # Inverse distance weighting
            dist_recip = 1.0 / (dist_k + 1e-8)
            norm = torch.sum(dist_recip, dim=1, keepdim=True)
            weight = dist_recip / norm  # (n, k)

            # Gather source features using LOCAL indices
            idx_flat = idx.reshape(-1)  # (n*k,)
            gathered_feat = batch_feat_src[idx_flat].reshape(n, k, C)  # (n, k, C)

            # Weighted sum
            batch_output = torch.sum(gathered_feat * weight.unsqueeze(-1), dim=1)  # (n, C)

            output[tgt_start:tgt_end] = batch_output

        return output
