from torch.utils.data import Dataset
import os
from plots.plot_masks import read_image, read_mask


class SegmentationDataset(Dataset):
    """Segmentation dataset.
    """
    def __init__(self, data_path, mask_path, transform, mode, max_size=None):
        """Initializes Segmentation dataset.
        Parameters
        ----------
        data_path : str
            Path to folder with images.
        mask_path : str
            Path to folder with masks.
        transform
            transform applied to image (if mode is test) or image and
        mode : str
            'train', 'val' or 'test'. If set to 'train' or 'val',
            the dataset will contain images and masks.
            If set to 'test', the dataset will only contain images.
        max_size : int
            int or None, optional (default=None)
        """
        assert (mode in ('train', 'val', 'test'))
        if mode == 'test':
            assert (mask_path is None)

        self.data_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        self.max_size = max_size
        self.mode = mode
        self.images_filenames = list(sorted(os.listdir(data_path)))

        if max_size is not None:
            self.images_filenames = self.images_filenames[:max_size]

        # картинок немного, поэтому будем все их хранить в памяти
        self.images = []
        self.masks = []

        for i, image_filename in enumerate(self.images_filenames):
            if self.max_size is not None and i >= self.max_size:
                break

            image = read_image(os.path.join(self.data_path, image_filename))

            if self.mode in ('train', 'val'):
                mask = read_mask(os.path.join(self.mask_path, image_filename.replace('.jpg', '.png')))
            else:
                mask = None

            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform is not None and mask is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        elif self.transform is not None:
            image = self.transform(image)

        return image, mask
