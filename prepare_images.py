import os
import logging
import shutil
import random
import pandas as pd
import argparse

def prepare_random_imgs_from_folder(src_path, dst_dir):
    imgs = os.listdir(src_path)[:]
    random.shuffle(imgs)
    num_folders = len(imgs) // 200
    for i in range(num_folders):
        dst_dir_i = os.path.join(dst_dir, 'random500_' + str(i))
        if not os.path.exists(dst_dir_i):
            os.makedirs(dst_dir_i)
        for img in imgs[i * 200:(i + 1) * 200]:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(dst_dir_i, img))
        logging.info(dst_dir_i + ' made.')



def prepare_material_imgs_from_broden(borden_path, concept, concept_path):
    imgs = os.listdir(borden_path)[:]
    concept_imgs = list(filter(lambda s: concept == s.split('_')[0] and len(s.split('_')) == 2, imgs))
    if len(concept_imgs) != 0:
        dst_dir = os.path.join(concept_path, concept)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in concept_imgs:
            shutil.copy(
                os.path.join(borden_path, img),
                os.path.join(dst_dir, img))
    logging.info(len(concept_imgs), concept, 'concept images found.')

def prepare_class_imgs_from_folder(src_path, truth_file, label, dst_dir):
    df = pd.read_csv(truth_file, header=None)
    imgs = list(map(lambda i: 'ILSVRC2012_val_{:08d}.JPEG'.format(i + 1), df[df[0] == label].index))
    if len(imgs) != 0:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in imgs:
            shutil.copy(
                os.path.join(src_path, img), 
                os.path.join(dst_dir, img))
    logging.info(label, 'label,', str(len(img)), 'images copied.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['material', 'random', 'class'])
    parser.add_argument('source', help='source folder')
    parser.add_argument('output', help='output folder')
    parser.add_argument('--concept')
    parser.add_argument('--truth_file')
    parser.add_argument('--label', type=int)
    args = parser.parse_args()
    if args.type == 'random':
        prepare_random_imgs_from_folder(args.source, args.output)
    elif args.type == 'material':
        prepare_material_imgs_from_broden(args.source, args.concept, args.output)
    else:
        prepare_class_imgs_from_folder(args.source, args.truth_file, args.label, args.output)
