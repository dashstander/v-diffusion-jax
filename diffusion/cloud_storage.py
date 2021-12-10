from google.cloud import storage
import os
from PIL import Image
import numpy as np
import pyarrow as pa
from pyarrow.csv import ParseOptions, ReadOptions, read_csv
import requests
from torch.utils.data import Dataset


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


class CoCo(Dataset):

    def __init__(self, path, transform_fn=None):
        pops = ParseOptions(delimiter='\t')
        rops = ReadOptions(column_names=['caption', 'url'])
        self.table = read_csv(path, parse_options=pops, read_options=rops)
        self.has_downloaded = np.zeros((len(self.table),)).astype(bool)
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, index):
        fp = f'./data/{index}.jpeg'
        if not self.has_downloaded[index]:
            resp = requests.get(self.table['url'][index].as_py())
            resp.raise_for_status()
            with open(fp, mode='wb') as file:
                file.write(resp.content)
            self.has_downloaded[index] = True
        with open(fp, mode='rb') as file:
            image = Image.open(file)
        if self.transform_fn is not None:
            image = self.transform_fn(image)
        return image, self.table['caption'][index].as_py()
