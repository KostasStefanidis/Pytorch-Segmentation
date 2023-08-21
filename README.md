# Pytorch-Segmentation

For a detailed overview of the dataset visit the [Cityscapes website](https://www.cityscapes-dataset.com/) and the [Cityscapes Github repository](https://github.com/mcordts/cityscapesScripts) 

This repository uses the [Pytorch-Lightning](https://lightning.ai/docs/pytorch/latest/) framework for training,evaluation and inference.

&nbsp;

-----
## Script usage

## Dataset Utilities
### 1. File parsing and decoding
Parse files which are under the following directory sctructure

---
    <data_path> : the root directory of the Cityscapes dataset
    |
    ├── gtFine_trainvaltest
    │   └── gtFine
    │       ├── test
    │       │   ├── berlin
    │       │   ├── bielefeld
    │       │   ├── bonn
    │       │   ├── leverkusen
    │       │   ├── mainz
    │       │   └── munich
    │       ├── train
    │       │   ├── aachen
    │       │   ├── bochum
    │       │   ├── bremen
    │       │   ├── cologne
    │       │   ├── darmstadt
    │       │   ├── dusseldorf
    │       │   ├── erfurt
    │       │   ├── hamburg
    │       │   ├── hanover
    │       │   ├── jena
    │       │   ├── krefeld
    │       │   ├── monchengladbach
    │       │   ├── strasbourg
    │       │   ├── stuttgart
    │       │   ├── tubingen
    │       │   ├── ulm
    │       │   ├── weimar
    │       │   └── zurich
    │       └── val
    │           ├── frankfurt
    │           ├── lindau
    │           └── munster
    └── leftImg8bit_trainvaltest
        └── leftImg8bit
            ├── test
            │   ├── berlin
            │   ├── bielefeld
            │   ├── bonn
            │   ├── leverkusen
            │   ├── mainz
            │   └── munich
            ├── train
            │   ├── aachen
            │   ├── bochum
            │   ├── bremen
            │   ├── cologne
            │   ├── darmstadt
            │   ├── dusseldorf
            │   ├── erfurt
            │   ├── hamburg
            │   ├── hanover
            │   ├── jena
            │   ├── krefeld
            │   ├── monchengladbach
            │   ├── strasbourg
            │   ├── stuttgart
            │   ├── tubingen
            │   ├── ulm
            │   ├── weimar
            │   └── zurich
            └── val
                ├── frankfurt
                ├── lindau
                └── munster

Each of the train,val,test directories contain subdirectories with the name of a city. To use a whole split, *`subfolder='all'`* must be passed to the *`Dataset.create()`* method in order to read the images from all the subfolders. For testing purposes a smaller number of images from the dataset can be used by passing `*subfolder='<CityName>'*`. For example, passing *`split='train'`* to the Dataset() constructor, and *`subfolder='aachen'`* to the *`create()`* method will make the Dataset object only read the 174 images in the folder aachen and convert them into a *tf.data.Dataset*. You can choose either all the subfolders or one of them, but not an arbitrary combination of them. After the images `(x)` and the ground truth images `(y)` are read and decoded, they are combined into a single object `(x, y)`.

&nbsp;

### 2. Preprocessing :
Generally images have a shape of `(batch_size, channels, height, width)` "channels_first" format

1. Split the image into smaller patches with spatial resolution `(256, 256)`. Because very image has a spatial resolution of `(1024, 2048)` 32 patches are produced and they comprise a single batch. This means that when the patching technique is used the batch size is fixed to 32. After this operation the images have a shape of `(32, 256, 256, 3)` while the the ground truth images have a shape of `(32, 256, 256, 1)`. To enable patching set the `use_patches` arguement of the `create()` method, to `True`.

&nbsp;

2. Perform data `Augmentation`

    *NOTE : while all augmentations are performed on the images, only horrizontal flip is performed on the ground truth images, because changing the pixel values of the ground truth images means changing the class they belong to.*

&nbsp;

3. Normalize images : 
   - The input pixels values are scaled between **-1** and **1** as default
   - If using a pretrained backbone normalize according to what the pretrained network expects at its input. To determine what type of preprocessing will be done to the images, the name of the pretrained network must be passed as the `preprocessing` arguement of the Dataset constructor. For example, if a model from the EfficientNet model family (i.e EfficientNetB0, EfficientNetB1, etc) is used as a backbone, then `preprocessing = "EfficientNet"` must be passed.

&nbsp;

4. Preprocess ground truth images:
   - Map eval ids to train ids
   - Convert to `one-hot` encoding
   - After this operation ground truth images have a shape of `(batch_size, 1024, 2048, num_classes)`
  
  Finally the dataset which is created is comprised of elements `(image, ground_truth)` with shape `((batch_size, height, width, 3)`, `(batch_size, height, width, num_classes))` 

&nbsp;


------

## **Segmentation Models**