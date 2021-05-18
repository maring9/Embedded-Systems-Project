# Importing needed libraries

# OS - Library used for OS operations
import os

# CV2 - Library used for computer vision operations
import cv2

# NUMPY - Library used for tensor operations
import numpy as np

# IMGAUG - Library used for image augmentation
from imgaug import augmenters as iaa
import imgaug as ia


def load_images_from_folder(folder, size = (224, 224), start = 0, end = 100000):
    """Helper function to load images from storage

    Args:
        folder (String): Absolute path to the directory from which to load the images
        size (tuple, optional): Size to convert the images to. Defaults to (224, 224).
        start (int, optional): Auxiliary start index for larger image datasets. Defaults to 0.
        end (int, optional): Auxiliary end index for lager datasets. Defaults to 100000.

    Returns:
        np.array: Array of the loaded images
    """
    # Empty list to store images loaded from storage
    images = []

    # Loop over the files in the folder from start to end
    for filename in os.listdir(folder)[start:end]:

        # Read image from the path
        image = cv2.imread(os.path.join(folder,filename))

        # Check if the read was successfull
        if image is not None:
            # Resize the image to the target size
            image = cv2.resize(image, dsize = size)

            # Convert image from standard cv2 BGR color space to tradition RGB color space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Append image to the list of images
            images.append(image)

    # Return images as numpy array
    return np.array(images)


def save_images_in_folder(folder, images, size = (224, 224), start_index = 0):
    """Helper function to save images to storage

    Args:
        folder (String): Absolute path where to save the images
        images (np.array): Array of (augmented) images
        size (tuple, optional): Size to convert the images to. Defaults to (224, 224).
        start_index (int, optional): Start index for image name. Defaults to 0.
    """

    # Loop over the images
    for i, image in enumerate(images):
        # Resize image to target size
        image = cv2.resize(image, dsize = size)
        
        # Convert image back to BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create the path where the image will be save, images will be indexed
        path = os.path.join(folder, str(start_index + i) + '.png')
        
        # Write / Save image 
        cv2.imwrite(path, image)


def augment_images(folder, augmenter, images, size = (224, 224), start_index=0, iterations=1):
    """Main function to augment the images and save them to storage

    Args:
        folder (String): Absolute path of directory where to save the augmented images to
        augmenter (imgaug.augmenters.meta.Sequential): Augmenter that applies transformations to the images
        images (np.array): Array of images on which the transformations will be applied on
        size (tuple, optional): Size to convert the images to. Defaults to (224, 224).
        start_index (int, optional): Start index used as name for images. Defaults to 0.
        iterations (int, optional): Number of iterations to apply to images. Defaults to 1.
    """
    # Get the total number of images
    n = len(images)
    
    # Main iteration that applies random transformations to the images
    for i in range(iterations):
        # Apply transformations to the images
        images_augmented = augmenter(images=images)
    
        # Save the augmented images on the disk
        save_images_in_folder(folder=folder, images=images_augmented, size=size, start_index=i*n)


def get_augmenter():
    """Augmenter that applies transformation to the dataset
    """

    augmenter = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.80, 1.2), "y": (0.80, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-6, 6)
    )
], random_order=True) # apply augmenters in random order

    return augmenter


def main():
    # Base dataset path
    base_path = r'D:\Proiect SI'

    # Red lights path
    red_lights_path = os.path.join(base_path, 'Red')

    # Green lights path
    green_lights_path = os.path.join(base_path, 'Green')

    # Loaded Red light images from the disk
    red_light_images = load_images_from_folder(red_lights_path)

    # Loaded Red light images from the disk
    green_light_images = load_images_from_folder(green_lights_path)

    # Augmented dataset path
    augmented_path = os.path.join(base_path, 'Dataset')

    if not os.path.exists(augmented_path):
        os.mkdir(augmented_path)

    # Path where the augmented red light images will be saved
    red_light_augmented_dir = os.path.join(augmented_path, 'Red Light Augmented')

    if not os.path.exists(red_light_augmented_dir):
        os.mkdir(red_light_augmented_dir)       

    # Path where the augmented green light images will be saved
    green_light_augmented_dir = os.path.join(augmented_path, 'Green Light Augmented')

    if not os.path.exists(green_light_augmented_dir):
        os.mkdir(green_light_augmented_dir)

    augmenter = get_augmenter()

    print('Augmenting images...')

    # Applying the augmentations to the red light images and saving them on the disk
    augment_images(folder = red_light_augmented_dir, augmenter = augmenter,
                   images=red_light_images, size = (224, 224),
                   start_index = 0, iterations = 25)

    # Applying the augmentations to the green light images and saving them on the disk
    augment_images(folder = green_light_augmented_dir, augmenter = augmenter,
                   images=green_light_images, size = (128, 128),
                   start_index = 0, iterations = 25)

    print('Done. Images augmented successfully.')


if __name__ == '__main__':
    main()