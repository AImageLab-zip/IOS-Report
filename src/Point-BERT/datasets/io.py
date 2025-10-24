import h5py
import numpy as np
import open3d
import os
import torch
from pytorch3d.ops import sample_farthest_points


def farthest_point_sample(point, npoint):
    """Farthest Point Sampling using PyTorch3D for efficiency"""
    point = torch.from_numpy(point).unsqueeze(0)  # (1, N, 3)
    sampled_points, _ = sample_farthest_points(point, None, npoint)
    return sampled_points.squeeze(0).numpy()

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.stl', '.STL']:
            return cls._read_stl(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

    @classmethod
    def _read_stl(cls, file_path):
        """Read STL file and return point cloud"""
        mesh = open3d.io.read_triangle_mesh(file_path)
        # Sample points from the mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=32768*2)
        ptcloud = np.array(pcd.points, dtype=np.float32)
        # ptcloud = farthest_point_sample(ptcloud, 32768*2)
        return ptcloud
