import os
import tarfile
from zipfile import ZipFile
from torchvision import transforms
import random
from PIL import Image
import matplotlib.pyplot as plt
import shutil


def load_meta_tiny_imagenet(root_dir):
    wnids_path = os.path.join(root_dir, 'tiny-imagenet-200', 'wnids.txt')
    words_path = os.path.join(root_dir, 'tiny-imagenet-200', 'words.txt')

    # Read the wnids.txt file to get the WordNet IDs
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f]

    # Create a mapping from WordNet ID to a sequential integer ID
    wnid_to_id = {wnid: i for i, wnid in enumerate(wnids)}
    id_to_wnid = {i: wnid for i, wnid in enumerate(wnids)}

    # Read the words.txt file to get the mapping from WNID to human-readable labels
    words = {}
    with open(words_path, 'r') as f:
        for line in f:
            wnid, label = line.strip().split('\t')
            words[wnid] = label

    # Create a list of interpretable labels ordered by the integer ID
    interpretable_labels = [words[wnid] for wnid in wnids]

    return wnid_to_id, id_to_wnid, interpretable_labels


def get_tiny_imagenet_data(root_dir):
    print("Getting Tiny ImageNet data")
    train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
    val_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val')
    test_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'test')

    # Ensure the directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Load interpretable labels
    wnid_to_id, id_to_wnid, interpretable_labels = load_meta_tiny_imagenet(root_dir)

    # Organize validation data into folders
    def organize_validation_data(val_dir, wnid_to_id):
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')

        if not os.path.exists(val_annotations_path):
            print(f"Validation annotations file not found at {val_annotations_path}")
            return

        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name, label_wnid = parts[0], parts[1]

                # Create class directory if it does not exist
                class_dir = os.path.join(val_dir, label_wnid)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Move the image to the correct class directory
                src_path = os.path.join(val_images_dir, image_name)
                dst_path = os.path.join(class_dir, image_name)
                shutil.move(src_path, dst_path)

        # Remove the now empty 'images' directory
        shutil.rmtree(val_images_dir)

    organize_validation_data(val_dir, wnid_to_id)

    train_data = []
    train_labels = []

    print("Creating training data (paths) and training labels")
    for class_folder in os.listdir(train_dir):
        class_folder_path = os.path.join(train_dir, class_folder, 'images')

        if '.DS_Store' == class_folder:
            continue

        label_id = wnid_to_id[class_folder]

        for image_file in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file)
            train_data.append(image_path)
            train_labels.append(label_id)

    print("Creating validation data (paths) and validation labels")
    val_data = []
    val_labels = []

    val_images_dir = os.path.join(val_dir, 'images')

    val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.exists(val_annotations_path):
        print(f"Validation annotations file not found at {val_annotations_path}")
    else:
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                label_wnid = parts[1]
                label_id = wnid_to_id[label_wnid]
                val_labels.append(label_id)
                val_data.append(os.path.join(val_images_dir, image_name))

    if len(val_data) != len(val_labels):
        print("Warning: The number of validation images does not match the number of labels!")

    return train_data, train_labels, val_data, val_labels, interpretable_labels


def load_tiny_imagenet(config, data_dir):
    print("Loading Tiny ImageNet data")
    train_data, train_labels, val_data, val_labels, interpretable_labels = get_tiny_imagenet_data(data_dir)

    resize_param = 64

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transforms for training
    train_transforms = []
    train_transforms_temp = []
    val_transforms = []

    if bool(config.getint("augmentation", "on")):
        train_transforms.append(transforms.Resize((resize_param, resize_param)))
        val_transforms.append(transforms.Resize((resize_param, resize_param)))

        if bool(config.getint("augmentation", "crop")):
            train_transforms_temp.append(transforms.RandomApply([transforms.RandomResizedCrop(size=(resize_param,
                                                                                                    resize_param),
                                                                                              scale=(0.65, 0.95))],
                                                                p=config.getfloat("augmentation", "crop_probs")))
        if bool(config.getint("augmentation", "horizontal_flip")):
            train_transforms_temp.append(transforms.RandomHorizontalFlip(p=config.getfloat("augmentation", "hf_probs")))
        if bool(config.getint("augmentation", "vertical_flip")):
            train_transforms_temp.append(transforms.RandomVerticalFlip(p=config.getfloat("augmentation", "vf_probs")))
        if bool(config.getint("augmentation", "color_jitter")):
            train_transforms_temp.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.5,
                                                                                        contrast=0.4,
                                                                                        saturation=0.3,
                                                                                        hue=0.2)],
                                                                p=config.getfloat("augmentation", "jitter_probs")))
        if bool(config.getint("augmentation", "blur")):
            train_transforms_temp.append(transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.5, 0.5))],
                                                                p=config.getfloat("augmentation", "blur_probs")))

        if bool(config.getint("augmentation", "adjust_sharpness")):
            train_transforms_temp.append(transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)],
                p=config.getfloat("augmentation", "sharpness_probs")))

        train_transforms.append(transforms.RandomApply(transforms=train_transforms_temp,
                                                       p=config.getfloat("augmentation", "probs")))

        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())

        if config.getint("augmentation", "cutout"):
            cutout_probs = config.getfloat("augmentation", "cutout_probs")
            cutout_patch = config.getint("augmentation", "cutout_patch")
            train_transforms.append(MultipleRandomErasing(num_patches=cutout_patch, p=cutout_probs,
                                                          scale=(0.02, 0.08), ratio=(0.3, 3),
                                                          value=0))

        if bool(config.getint("augmentation", "normalize")):
            train_transforms.append(transforms.Normalize(mean=mean, std=std))
            val_transforms.append(transforms.Normalize(mean=mean, std=std))

    else:
        train_transforms.append(transforms.Resize((resize_param, resize_param)))
        train_transforms.append(transforms.ToTensor())

        val_transforms.append(transforms.Resize((resize_param, resize_param)))
        val_transforms.append(transforms.ToTensor())

    train_transforms = transforms.Compose(transforms=train_transforms)
    val_transforms = transforms.Compose(transforms=val_transforms)

    tiny_imagenet_data = dict()
    tiny_imagenet_data["train_data"] = train_data
    tiny_imagenet_data["train_labels"] = train_labels
    tiny_imagenet_data["val_data"] = val_data
    tiny_imagenet_data["val_labels"] = val_labels
    tiny_imagenet_data["train_transforms"] = train_transforms
    tiny_imagenet_data["val_transforms"] = val_transforms
    tiny_imagenet_data["interpretable_labels"] = interpretable_labels

    return tiny_imagenet_data


class MultipleRandomErasing:
    def __init__(self, num_patches=5, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0):
        """
        Apply RandomErasing multiple times randomly.
        :param num_patches: Number of patches to erase.
        :param p: Probability of erasing a single patch.
        :param scale: Proportion of image to erase per patch.
        :param ratio: Aspect ratio of the erase region.
        :param value: Erasing value.
        """
        self.num_patches = num_patches
        self.eraser = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, img):
        for _ in range(self.num_patches):
            if random.uniform(0, 1) < self.eraser.p:
                img = self.eraser(img)
        return img
