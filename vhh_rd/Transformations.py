import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
import cv2
import string

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def get_black_bar_augmenter(percentage: float):
    """
    Adds black bars at the top and the bottom of the image
    Their total height sampled uniformly from [0, percentage*img_height]
    Top and bottom bars have the same height
    """
    def add_black_bars(images, random_state, parents, hooks):
        result = []
        for image in images:
            height, width, _ = image.shape
            bar_height = int(height * percentage * 0.5 * np.random.uniform())
            image[0:bar_height,:,:] = 0
            image[height-bar_height:height,:,:] = 0
        return images 
    return make_augmenter(add_black_bars)

def get_text_augmenter():
    """
    Adds random text to images
    """
    def add_text(images, random_state, parents, hooks):
        result = []
        
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, 
            cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,] 
        letters = [l for l in (" "*10 + string.ascii_letters + string.digits)]

        for image in images:
            font = np.random.choice(fonts)
            height, width, _ = image.shape

            fontsize = np.random.uniform(0.1, 0.6)
            nr_letters = np.random.randint(10, 50)
            text = ''.join(np.random.choice(letters, size = nr_letters))

            x = np.random.randint(0, int(width/3))
            y = np.random.randint(height - int(height/3), height)

            color = [255, 255, 255]
            # Random color
            if np.random.uniform() < 0.25:
                color = [int(np.random.uniform() * 255), int(np.random.uniform() * 255), int(np.random.uniform() * 255)]

            cv2.putText(image, text, (x, y), font, fontsize, color)

            # Add another line of text to the image
            if np.random.uniform() < 0.4:
                image = add_text([image], random_state, parents, hooks)

        return images 
    return make_augmenter(add_text)
    
def func_heatmaps(heatmaps, random_state, parents, hooks):
    return heatmaps
    
def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images

make_augmenter = lambda fct: iaa.Lambda(
    func_images=fct,
    func_heatmaps=func_heatmaps,
    func_keypoints=func_keypoints
)

def get_augmentations():
    return iaa.Sequential(
        [
            # Always crop first
            iaa.Crop(percent=(0, 0.4), keep_size=True, sample_independently=True), 
            iaa.size.Resize((256,256)),
            iaa.Fliplr(0.2), 

            iaa.SomeOf((2, 5),
                [
                iaa.Add((-2, 2), per_channel=0.5),
                iaa.Multiply((0.8, 1.3), per_channel=0.5),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.PiecewiseAffine(scale=(0.01, 0.05))
                 ],
            random_order=True),
            iaa.Sometimes(0.7, get_black_bar_augmenter(0.25)),
            iaa.Sometimes(0.5, get_text_augmenter()),
        ],
        random_order=False
    ).augment_image