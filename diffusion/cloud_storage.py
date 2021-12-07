from io import BytesIO
from google.cloud import storage
from PIL import Image
from torch.utils.data import Dataset


class BucketDataset(Dataset):

    def __init__(self, bucket, path, transform_fn=None):
        super().__init__()
        client = storage.Client()
        self.index = [blob for blob in client.list_blobs(client.get_bucket(bucket), prefix=path)]
        self.transform_fn = transform_fn
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i: int):
        client = storage.Client()
        file = BytesIO()
        client.download_blob_to_file(self.index[i], file)
        image = Image.open(file)
        if self.transform_fn:
            image = self.transform_fn(image)
        return image
