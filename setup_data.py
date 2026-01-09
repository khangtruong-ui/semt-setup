import datasets
import json
import os

def setup_env_from_ds(ds):
    """
    Dataset Format: {
        Dataset({
            features: ['image', 'filename', 'caption'],
        })
    }
    """
    jsonfile = {'images': []}
    for data in ds:
        image = data['image']
        raw = data['caption']
        filename = data['filename']
        filename = filename.replace(filename[-4:], '.png')
        image.save(f"02_NWPU_RESISC45/images/{filename}")
        jsonfile['images'].append({
            'filename': filename,
            'caption': raw,
        })

    json.dump(jsonfile, open('02_NWPU_caption/dataset_nwpu.json', 'w'))

def setup_default():
    setup_env_from_ds(datasets.load_dataset(os.environ['DATASET'])['train'])
