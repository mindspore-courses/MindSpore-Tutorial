"""创建MindRecord数据集"""
import argparse
import json
import os
from collections import Counter
from random import seed, choice, sample

import numpy as np
from PIL import Image
from imageio.v2 import imread
from mindspore.mindrecord import FileWriter
from tqdm import tqdm


def create_input_files(dataset, pickle_path, image_folder, output_folder, captions_per_image, min_word_freq,
                       max_len=100, shard_num=1, write_freq=100):
    """创建MindRecord数据集"""
    assert dataset in {'coco'}

    # 读取词汇表
    with open(pickle_path, 'r',encoding='UTF-8') as pickle:
        data = json.load(pickle)

    # 读取image与caption
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    print("Create word map")

    words = [w for w, freq in word_freq.items() if freq > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(
        min_word_freq) + '_min_word_freq'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save word map to a JSON

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w',encoding='UTF-8') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)

    # Define schema
    schema_json = {
        "image": {"type": "float32", "shape": [256, 256, 3]},
        "captions": {"type": "int32", "shape": [captions_per_image, -1]},
        "lens": {"type": "int32", "shape": [-1]}
    }

    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        f = FileWriter(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.mindrecord'), shard_num,
                       overwrite=True)
        f.add_schema(schema_json)
        print(f"\nReading {split} images and captions, storing to file...\n")

        data = []
        for i, path in enumerate(tqdm(impaths)):
            enc_captions = []
            caplens = []
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            img = imread(impaths[i])
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = np.array(Image.fromarray(img).resize((256, 256))).astype(np.float32)
            # img = img.transpose(2, 0, 1).astype(np.float32)
            img = img / 255.
            # assert img.shape == (3, 256, 256)
            assert np.max(img) <= 1.
            assert img.dtype == np.float32

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c) - 2)

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)
            # print(enc_captions)
            # print(caplens)
            # Save data to MindRecord file
            data.append({
                "image": img,
                "captions": np.array(enc_captions, dtype=np.int32),
                "lens": np.array(caplens, dtype=np.int32),
            })
            if i % write_freq == 0:
                f.write_raw_data(data)
                data = []
            # if i == 100:
            #     break
        if data:
            f.write_raw_data(data)
        f.commit()


def main(_args):
    """主函数"""
    create_input_files(
        _args.dataset,
        _args.json_path,
        _args.image_folder,
        _args.output_folder,
        _args.captions_per_image,
        _args.min_word_freq,
        _args.max_len,
        _args.shard_num,
        _args.write_freq
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create image captioning dataset')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--json_path', type=str, default='../../data/dataset_coco.json')
    parser.add_argument('--image_folder', type=str, default='../../data')
    parser.add_argument('--output_folder', type=str, default='../../data/mindrecord')
    parser.add_argument('--captions_per_image', type=int, default=5)
    parser.add_argument('--min_word_freq', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--shard_num', type=int, default=1)
    parser.add_argument('--write_freq', type=int, default=1)
    args = parser.parse_args()
    main(args)
