import os
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
import pyarrow as pa
from pyarrow.csv import ParseOptions, ReadOptions, read_csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools



def check_image(raw_content):
    file = BytesIO(raw_content)
    is_image = False
    try:
        image = Image.open(file)
        is_image = True
    except UnidentifiedImageError:
        image = None
    return image, is_image


def download_image(url, split, index):
    fp = f'/home/dashiell/coco-3m/{split}/{index}.jpg'
    if os.path.exists(fp):
        return split, index, True
    response = requests.get(url)
    saved_successfully = False
    if response.ok:
        image, is_image = check_image(response.content)
        if index % 1000 == 0:
            print(f'Saving image {index} of the {split} data')
        if is_image:
            with open(fp, mode='wb') as file:
                image.save(file)
            saved_successfully = True
    return split, index, saved_successfully
        

def table_iter(table, split):
    for i, url in enumerate(table['url']):
        yield url.as_py(), split, i


def download_everything(train_table, val_table):
    has_downloaded = {
        'train': np.ones((len(train_table),)).astype(bool),
        'val': np.ones((len(val_table),)).astype(bool)
    }
    chain = itertools.chain(table_iter(train_table, 'train'), table_iter(val_table, 'val'))
    with ThreadPoolExecutor(max_workers=48) as exec:
        image_futures = [
            exec.submit(download_image, *record) for record in chain
        ]
    for future in as_completed(image_futures):
        split, index, saved = future.result()
        if not saved:
            has_downloaded[split][index] = saved
    
    train_table.append_column('downloaded', has_downloaded['train'])
    val_table.append_column('downloaded', has_downloaded['val'])
    pa.parquet.write_table(train_table, '/home/dashiell/coco-3m/train.parquet')
    pa.parquet.write_table(val_table, '/home/dashiell/coco-3m/validation.parquet')
    print('All done')


def main():
    pops = ParseOptions(delimiter='\t')
    rops = ReadOptions(column_names=['caption', 'url'])
    train = read_csv('/home/dashiell/workspace/v-diffusion-jax/Train_GCC-training.tsv', parse_options=pops, read_options=rops)
    val = read_csv('/home/dashiell/workspace/v-diffusion-jax/Validation_GCC-1.1.0-Validation.tsv', parse_options=pops, read_options=rops)
    download_everything(train, val)


if __name__ == '__main__':
    main()

    
