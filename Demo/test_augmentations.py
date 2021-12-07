import vhh_rd.Transformations as Transformations
import vhh_rd.RD as RD
import cv2
import numpy as np
import os, glob

from vhh_rd.Transformations import get_augmentations


rows = 4
columns = 2


def main(): 
    config_path = "./config/config_rd.yaml"
    rd = RD.RD(config_path) 
    augmentations = Transformations.get_augmentations()

    image_folder = rd.config["SIAM_TRAIN_PATH"]
    images_path = list(set(glob.glob(os.path.join(image_folder, "**/*.png"))) - set(glob.glob(os.path.join(image_folder, "NA/*.png"))))
    img_selected = np.random.choice(images_path, rows*columns)

    resize = lambda img: cv2.resize(img, (256,256))

    comparisons = []
    for img_path in img_selected:
        img = cv2.imread(img_path).astype(np.uint8)
        img_aug = augmentations(img.copy()).astype(np.float32)
        img_aug2 = augmentations(img.copy()).astype(np.float32)

        comparison = np.hstack((resize(img).astype(np.uint8), resize(img_aug).astype(np.uint8), resize(img_aug2).astype(np.uint8)))
        comparisons.append(comparison)

    comparisons_as_rows = []
    for i in range(0, len(comparisons), columns):
        comparisons_as_rows.append(np.hstack(comparisons[i:i + columns]))

    cv2.imshow("augs", np.vstack(comparisons_as_rows))
    cv2.waitKey(0)



if __name__ == "__main__":
    main()