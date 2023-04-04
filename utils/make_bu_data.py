# Adapted from https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/make_bu_data.py

import os
import base64
import numpy as np
import csv
import sys

output_dir = '/import/gameai-01/eey362/datasets/coco/trainval/data/cocobu'
bu_features_location = '/import/gameai-01/eey362/datasets/coco/trainval'

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
input_files = ['karpathy_test_resnet101_faster_rcnn_genome.tsv',
           'karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
           'karpathy_train_resnet101_faster_rcnn_genome.tsv.1',
           'karpathy_val_resnet101_faster_rcnn_genome.tsv']

os.makedirs(output_dir+'_att', exist_ok=True)
os.makedirs(output_dir+'_fc', exist_ok=True)
os.makedirs(output_dir+'_box', exist_ok=True)


def read():
    for infile in input_files:
        print('Reading ' + infile)
        with open(os.path.join(bu_features_location, infile), "r") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                item['image_id'] = int(item['image_id'])
                print(f"Processing {item['image_id']}")
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = item[field] + "=" * (4 - len(item[field]) % 4)
                    item[field] = np.frombuffer(base64.b64decode(item[field].encode('ascii')), 
                            dtype=np.float32).reshape((item['num_boxes'],-1))
                np.savez_compressed(os.path.join(output_dir+'_att', str(item['image_id'])), feat=item['features'])
                np.save(os.path.join(output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
                np.save(os.path.join(output_dir+'_box', str(item['image_id'])), item['boxes'])
    


if __name__ == "__main__":
    read()
