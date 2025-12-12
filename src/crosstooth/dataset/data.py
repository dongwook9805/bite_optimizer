import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from . import data_util
import argparse

class ToothData(Dataset):
    def __init__(self, args):
        self.args = args
        self.point_transform = transforms.Compose(
            [
                data_util.PointcloudToTensor(),
                data_util.PointcloudNormalize(radius=1),
                data_util.PointcloudSample(total=args.num_points, sample=args.sample_points)
            ]
        )
    def get_by_name(self, name):
        return self.__getitem__(name)

    def __getitem__(self, item):
        import trimesh
        # item is path to mesh
        
        # Load mesh
        # Force load as mesh (not scene) if multiple
        mesh = trimesh.load(item, force='mesh')
        
        cell_normals = np.array(mesh.face_normals)
        point_coords = np.array(mesh.vertices)
        face_info = np.array(mesh.faces)
        # trimesh has triangles_center
        cell_coords = np.array(mesh.triangles_center)

        pointcloud = np.concatenate((cell_coords, cell_normals), axis=1)

        # Padding if needed
        if pointcloud.shape[0] < self.args.num_points:
            padding = np.zeros((self.args.num_points - pointcloud.shape[0], pointcloud.shape[1]))
            # Pad face_info with index 0 or similar? Original code zero padded face info.
            # face_info has 3 indices per face.
            face_info_padding = np.zeros((self.args.num_points - pointcloud.shape[0], 3), dtype=np.int32)
            face_info = np.concatenate((face_info, face_info_padding), axis=0)
            pointcloud = np.concatenate((pointcloud, padding), axis=0)

        # Sampling
        # Assuming args.num_points is the target input size for the network (probably 16000 faces/cells?)
        # Wait, PointTransformerSeg38 inputs? 
        # The code samples "num_points" from the available cells?
        # See original code: permute = np.random.permutation(self.args.num_points)
        # If shape > num_points, it crashes or slices?
        # Original code used `dataset.data_util.PointcloudSample` later.
        # But here it permutes `num_points`. It implies `pointcloud` MUST be size `num_points`?
        
        # If input mesh has > 16000 faces, we need to sample down?
        # Original code: 
        # if pointcloud.shape[0] < self.args.num_points: pad...
        # permute = np.random.permutation(self.args.num_points) 
        # This implies it EXPECTS size to be EXACTLY num_points after padding.
        # What if > num_points?
        # Original code didn't handle `> num_points` in `__getitem__` explicitly before permute.
        # BUT `PointcloudSample` does sampling.
        # Let's look at `PointcloudSample(total=args.num_points)`.
        
        # If mesh is large, we should sample it down here or handle it.
        # For now, let's assume we implement similar logic.
        
        if pointcloud.shape[0] > self.args.num_points:
             # Simple random choice if too large, or just slice?
             # Better to sample indices.
             # Original code looks like it assumed input was pre-processed or small enough, or `num_points` was large.
             # Default num_points=16000.
             # Let's clamp.
             choice = np.random.choice(pointcloud.shape[0], self.args.num_points, replace=False)
             pointcloud = pointcloud[choice]
             face_info = face_info[choice] # This mismatches faces to points!
             # Wait, `cell_coords` aligns with `cell_normals`. `face_info` is indices.
             # The network takes "points" which are actually "faces" (barycenters).
             # So `face_info` is just carried along metadata?
             pass
        
        # Original:
        # permute = np.random.permutation(self.args.num_points)
        # pointcloud = pointcloud[permute]
        # face_info = face_info[permute]
        
        # If we just replicate logic:
        # Ensure size is exactly num_points
        current_n = pointcloud.shape[0]
        if current_n > self.args.num_points:
            choice = np.random.choice(current_n, self.args.num_points, replace=False)
            pointcloud = pointcloud[choice]
            face_info = face_info[choice]
        elif current_n < self.args.num_points:
            # Already padded above
            pass
            
        # Transform
        pointcloud, face_info = self.point_transform([pointcloud, face_info])

        return pointcloud.to(torch.float), point_coords, face_info


if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--case', type=str, default="")
        parser.add_argument('--num_points', type=int, default=16000)
        parser.add_argument('--sample_points', type=int, default=16000)
        args = parser.parse_args()
        return args

    args = get_args()
    dataset = ToothData(args)
    pointcloud, point_coords, face_info = dataset.get_by_name(args.case)
    print(pointcloud.shape)