# prompt: List files under ] s3://congo8khz-pnnn/recordings/wav/nn01a using boto3.client

import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config # Import the Config object
from tqdm import tqdm
import random



# Mysterious sounds
#    nn01a_20180215_000000.wav
#    nn03a_20190108_000000.wav
#    nn04e_20180212_000000.wav - strange shapes

### If this exists in the amazon data, it's interesting.
if_these_exist_also_interesting = [
    'nn03c_20180114_000000.wav',
    'nn04a_20180608_000000.wav',
    'nn04a_20180710_000000.wav',  # birds with elephants
    'nn04a_20200314_000000.wav',  # faint and challenging
    'nn06a_20191008_000000.wav', # pretty
    'nn06b_20201005_000100.wav', # nice around 22:02, group around 20:55:00, near and far around 18:36, truck around 17:13
    'nn06b_20201030_000100.wav', # challenging, very faint
    'nn06d_20200118_000000.wav', # ok
    'nn07b_20191022_000000.wav', # thunder?
    'nn07c_20180110_000000.wav', # birds, nice mix; interesting to test.
    'nn10a_20191006_000000.wav', # awesome animal at 400Hz.  Buffalo?
]

interesting_files = [
    'nn06d_20210707_000000.wav',
    'nn10a_20180402_000000.wav',
    'nn06b_20180909_000000.wav',
    'nn01a_20180402_000000.wav',
    'nn01a_20171229_000000.wav', # nice rumble with birds
    'nn01e_20180622_100134.wav', 
    'nn03e_20181015_000000.wav', # nice conversation
    'nn04e_20180212_000000.wav', # wow.
    'nn06a_20220503_000000.wav', # nice mix of rumble shapes
    'nn06a_20220519_000000.wav', # nice rumbles
]

def download_s3_bucket(s3, bucket_name, local_dir):
    s3 = boto3.client('s3')
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    all_objs = []
    paginator = s3.get_paginator('list_objects_v2')
    print("counting objects")
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            batch = [obj['Key'] for obj in page['Contents']]
            all_objs.extend(batch)

    print(f"looks like {len(all_objs)} objects")
    print(all_objs[0:10])

    all_objs = sorted(all_objs, key=lambda x: random.random())


    # List all objects in the bucket
    # looks like 101077 objects
    with tqdm(total=len(all_objs), unit='file') as pbar:
        for relative_path in all_objs:
            local_file_path = os.path.join(local_dir, relative_path)
            local_file_dir = os.path.dirname(local_file_path)
            if os.path.exists(local_file_path):
                print(f"already had {relative_path}")
                continue
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            if not relative_path.endswith('/'):
                s3.download_file(bucket_name, relative_path, local_file_path)
            pbar.update(1)
            pbar.set_description(f"{relative_path}")


def download_test_data():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'congo8khz-pnnn'
    local_directory = os.path.join("tmp","s3",bucket_name)
    download_s3_bucket(s3, bucket_name, local_directory)

