import mindspore.dataset as ds
from mindspore.dataset.vision import transforms
from mindspore.dataset.vision.transforms import Normalize, HWC2CHW
from mindspore.dataset.transforms.transforms import Duplicate


def create_dataset(data_path, crop_size, mode='train', batch_size=32, captions_per_image=5,
                   num_parallel_workers=8):
    if mode not in ['train', 'val', 'test']:
        raise ValueError('not support mode: {}'.format(mode))
    if mode == 'train':
        drop_remainder = True
        shuffle = True
    else:
        drop_remainder = False
        shuffle = False

    dataset = ds.MindDataset(data_path, num_parallel_workers=num_parallel_workers, shuffle=False)

    data_size = dataset.get_dataset_size()

    def resize(image, captions, cap_lengths, all_captions, batch_info):
        num = batch_info.get_batch_num() // data_size
        return image, [captions[0][num]], [cap_lengths[0][num]], all_captions

    def resize2(image, captions, cap_lengths, all_captions, batch_info):
        image = [i.squeeze(0) for i in image]
        captions = [i.squeeze(0) for i in captions]
        cap_lengths = [i.squeeze(0) for i in cap_lengths]
        all_captions = [i.squeeze(0) for i in all_captions]
        return image, captions, cap_lengths, all_captions

    img_transforms = [
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        HWC2CHW()]
    cap_transforms = [Duplicate()]
    dataset = dataset.map(img_transforms, input_columns="image", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(cap_transforms,
                          input_columns=["captions"],
                          output_columns=["captions", "all_captions"],
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.project(["image", "captions", "lens", "all_captions"])

    # repeat N times and use batch_map to get different captions
    dataset = dataset.repeat(captions_per_image)
    dataset = dataset.batch(1, input_columns=["image", "captions", "lens", "all_captions"], per_batch_map=resize)

    dataset = dataset.batch(batch_size, input_columns=["image", "captions", "lens", "all_captions"],
                            per_batch_map=resize2,
                            drop_remainder=drop_remainder)
    if shuffle:
        dataset = dataset.shuffle(batch_size)

    if mode == 'train':
        dataset = dataset.project(["image", "captions", "lens"])
    return dataset
