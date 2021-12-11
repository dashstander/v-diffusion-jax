from google.cloud import storage
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pyarrow.dataset as ds
from torch.utils.data import Dataset
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

class BucketDataset(Dataset):

    def __init__(self, bucket, path, transform_fn=None):
        super().__init__()
        client = storage.Client()
        self.index = [blob for blob in client.list_blobs(client.get_bucket(bucket), prefix=path)]
        self.has_downloaded = [False for _ in self.index]
        self.transform_fn = transform_fn
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i: int):
        name = self.index[i].name
        fp = f'./data/{name}'
        if not self.has_downloaded[i]:
            client = storage.Client()
            os.path.makedirs(os.path.dirname(fp), exists_ok=True)
            with open(fp, mode='wb') as file:
                client.download_blob_to_file(self.index[i], file)
        with open(fp, mode='rb') as file:
            image = Image.open(file)
        if self.transform_fn:
            image = self.transform_fn(image)
        return image, None



class WebDataset(Dataset):
    def __init__(self, path, transform_fn=None):
        super().__init__()
        self.path = path
        self.table = ds.dataset(
            path,
            format='parquet',
            exclude_invalid_files=True
        ).to_table(filter=ds.field('status') == 'success')
        self.transform_fn = transform_fn
        self.index = self._make_image_index()


    def __len__(self):
        return len(self.table)

    def _make_image_index(self):
        tar_index = {}
        with ProcessPoolExecutor(max_workers=32) as exec:
            futures = [
                exec.submit(self._get_tar_members, file) 
                for file in os.scandir(self.path) 
                if file.name.endswith('.tar')
            ]
        count = 0
        for result in as_completed(futures):
            name, index = result.result()
            if count % 5 == 0:
                print(f'Added index for {name}')
            tar_index.update({name: index})
            count += 1
        return tar_index

    def _get_tar_members(self, tar_file):
        with tarfile.open(tar_file.path) as tar:
            index = {m.name.split('.')[0]: m for m in tar.getmembers() if not (m.name.endswith('json') or m.name.endswith('txt'))}
        return tar_file.name[:-4], index

    def __getitem__(self, index: int):
        key = self.table['key'][index].as_py()
        text = self.table['caption'][index].as_py()
        tar_fp = f'{self.path}/{key[:5]}.tar'
        tar_member = self.index[key[:5]][key]
        image = None
        with tarfile.open(tar_fp) as tar:
            image = Image.open(tar.extractfile(tar_member))
        if image is None:
            raise ValueError(f'Could not find image {key} at index {index}')
        if self.transform_fn:
            image = self.transform_fn(image)
        return image, text


class ImageDataset(Dataset):
    """ImageDataset is a pytorch Dataset exposing image and text tensors from a folder of image and text"""
    
    def __init__(self, preprocess, folder, enable_text=True, enable_image=True):
        super().__init__()
        path = Path(folder)
        self.enable_text = enable_text
        self.enable_image = enable_image
        if self.enable_text:
            text_files = [*path.glob("**/*.txt")]
            text_files = {text_file.stem: text_file for text_file in text_files}
            if len(text_files) == 0:
                self.enable_text = False
        if self.enable_image:
            image_files = [
                *path.glob("**/*.png"),
                *path.glob("**/*.jpg"),
                *path.glob("**/*.jpeg"),
                *path.glob("**/*.bmp"),
            ]
            image_files = {image_file.stem: image_file for image_file in image_files}
            if len(image_files) == 0:
                self.enable_image = False
        keys = None
        join = lambda new_set: new_set & keys if keys is not None else new_set
        if self.enable_text:
            keys = join(text_files.keys())
        elif self.enable_image:
            keys = join(image_files.keys())

        self.keys = list(keys)
        if self.enable_text:
            self.text_files = {k: v for k, v in text_files.items() if k in keys}
        if self.enable_image:
            self.image_files = {k: v for k, v in image_files.items() if k in keys}
            self.image_transform = preprocess

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        output = {}
        if self.enable_image:
            try:
                image_file = self.image_files[key]
                image_tensor = self.image_transform(Image.open(image_file))
            except (UnidentifiedImageError, OSError):
                print(f"Failed to load image {image_file}. Skipping.")
                return None  # return None to be filtered in the batch collate_fn
            output["image_filename"] = str(image_file)
            output["image_tensor"] = image_tensor
        if self.enable_text:
            text_file = self.text_files[key]
            caption = text_file.read_text()
            output["text"] = caption
        return output

