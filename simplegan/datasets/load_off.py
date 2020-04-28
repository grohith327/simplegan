import os
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm
import glob


__all__ = ["load_vox_from_off"]


class load_vox_from_off:

    r"""A dataloader classes that loads .off files and renders them into voxels

    Args:
        datadir (str, optional): Local directory to load data from. Defaults to ``None``
        side_length (int): The rendered voxels are converted to a cube of dimension ``side_length``. Defaults to ``64``
    """

    def __init__(self, datadir=None, side_length=64):
        self.data_files = None
        self.datadir = datadir
        self.side_length = side_length

    def __load_modelnet(self):

        os.mkdir("./modelnet")
        command = "wget -O ./modelnet/ModelNet10.zip https://3dshapenets.cs.princeton.edu/ModelNet10.zip"
        os.system(command)
        command = "unzip ./modelnet/ModelNet10.zip -d ./modelnet/"
        os.system(command)
        os.system("rm -rf ./modelnet/ModelNet10.zip ./modelnet/__MACOSX/")

        path = "./modelnet/ModelNet10"
        self.data_files = glob.glob(os.path.join(path, "*/*/*"))

    def __load_custom_datafiles(self, datadir):

        self.data_files = glob.glob(os.path.join(datadir, "*.off"))

        error_message = "files should have extension .off"
        assert len(self.data_files) > 0, error_message

    def load_data(self):

        r"""Load data from ModelNet10 if ``datadir`` is ``None`` or from local directory.

        Return:
            rendered voxels of shape ``(-1, side_length, side_length, side_length, 1)``
        """
        try:
            import trimesh
        except ModuleNotFoundError:
            print("module trimesh not found. install using 'pip install trimesh' command")

        if self.datadir is None:

            self.__load_modelnet()

        else:

            self.__load_custom_datafiles(self.datadir)

        data_voxels = []

        for file in tqdm(self.data_files, desc="rendering data"):

            mesh = trimesh.load(file)
            voxel = mesh.voxelized(0.5)
            (x, y, z) = map(float, voxel.shape)
            zoom_fac = (self.side_length / x, self.side_length / y, self.side_length / z)
            voxel = ndimage.zoom(voxel.matrix, zoom_fac, order=1, mode="nearest")
            data_voxels.append(voxel)

        data_voxels = np.array(data_voxels)
        data_voxels = data_voxels.reshape(
            (-1, self.side_length, self.side_length, self.side_length, 1)
        )
        return data_voxels
