import os
from PIL import Image
from .base_dataset import BaseDataset, get_transform

class PairedDataset(BaseDataset):
    """A dataset class for paired image dataset without concatenation."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, f'{opt.phase}A')
        self.dir_B = os.path.join(opt.dataroot, f'{opt.phase}B')

        self.A_paths = sorted(self.make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(self.make_dataset(self.dir_B, opt.max_dataset_size))

        assert len(self.A_paths) == len(self.B_paths), "The number of images in A and B must be the same"

        self.transform = get_transform(opt)

    def make_dataset(self, dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dir), f'{dir} is not a valid directory'
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        img_name_A = self.A_paths[index]
        img_name_B = self.B_paths[index]

        image_A = Image.open(img_name_A).convert('RGB')
        image_B = Image.open(img_name_B).convert('RGB')

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B, 'A_paths': img_name_A, 'B_paths': img_name_B}