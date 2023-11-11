# AugmentiX
🚀 YOLO Dataset Augmentation package 🖼️

## Project Overview

Welcome to AugmentiX, an image augmentation package designed to enhance your sample dataset for object detection, 
particularly tailored for YOLO-based training datasets. The package provides a convenient GUI interface for specifying 
the number of samples, desired augmentation tasks, and the output directory to save the augmented images.

### Problem Statement

AugmentiX addresses the challenges in efficiently augmenting datasets for object detection. Traditional methods often 
involve coding augmentation tasks for each training, leading to time-consuming processes. This package aims to simplify 
the augmentation process, offering an intuitive GUI and performing tasks such as random rotation, cropping, flipping, 
brightness adjustments, blur, noise, and more.

### Features

- **User-Friendly GUI:** Easily specify the number of samples, augmentation tasks, and output directory through a graphical interface.
- **Comprehensive Augmentation Tasks:** Perform various augmentation tasks, including random rotation, cropping, flipping, brightness adjustments, blur, and noise.
- **Annotation File Generation:** Create annotation files for augmented images, streamlining the dataset preparation process.
- **Performance Enhancement with Rust:** Utilize Rust bindings for improved performance, especially in I/O operations.

## Getting Started

To start using AugmentiX, follow these steps:

1. Install the package using the following command: `pip install augmentiX==0.0.10`
2. Install the required dependencies.
3. To run the CLI, run the following command:
   
    `augmentix --src <source directory of the dataset> --dst <destination directory>`

*Note:** The GUI is currently under development and will be released soon.

Before running the program, make sure the dataset that is being augmented is in the following format:

```
dataset
├── images
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
└── labels
    ├── 1.txt
    ├── 2.txt
    ├── 3.txt
    └── ...
|── data.yaml
```
## Motivation

AugmentiX was born out of a personal need to address the limitations of existing image augmentation libraries,
especially in the context of YOLO-based projects.
As a practitioner in object detection, the creator faced challenges with current libraries,
inspiring the development of a comprehensive solution that streamlines the augmentation process 
and integrates seamlessly with YOLO training.

## Contribution

Join us on this journey to enhance object detection workflows!
Contribute to AugmentiX, share your ideas,
and let's collectively become Legend Gladiators in the world of image augmentation for real-world problem-solving.

Let's code, contribute, and transform datasets together! 🚀🔧
