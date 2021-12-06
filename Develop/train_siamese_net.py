import vhh_rd.Feature_Extractor as FE
import vhh_rd.RD as RD
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os, sys, glob, random
import cv2
import imgaug.augmenters as iaa
import imgaug as ia
from torchvision import transforms
import torch.optim as optim
import wandb

config_path = "./config/config_rd.yaml"
lr = 0.00001

class TriplesDataset(Dataset):
    """
    Dataset of images
    """
    def __init__(self, image_folder, transforms, augmentations):
        """
            image_folder must be a folder containing the dataset as png images
        """
        self.transforms = transforms
        self.augmentations = augmentations
        self.images_paths = glob.glob(os.path.join(image_folder, "**/*.png"))
        # self.i = 0

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx]).astype(np.uint8)
 
        augment = lambda x: self.augmentations(x).astype(np.float32)
        postprocessing = lambda x: self.transforms(torch.from_numpy(x.transpose((2,0,1))))
        processing = lambda x: postprocessing(augment(x))

        # Select a random image as negative sample
        path_neg = random.choice(self.images_paths)
        image_neg = cv2.imread(path_neg).astype(np.uint8)

        # self.i = self.i +1
        # if self.i > 5:
        #     img_aug_neg = cv2.resize(augment(image_neg), (256,256), interpolation = cv2.INTER_AREA)
        #     img_aug_pos = cv2.resize(augment(image), (256,256), interpolation = cv2.INTER_AREA)
        #     img_pos = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        #     cv2.imshow("positives", np.hstack((img_pos.astype(np.uint8), img_aug_pos.astype(np.uint8), img_aug_neg.astype(np.uint8))))
        #     cv2.waitKey(0)

        image_neg = processing(image_neg)
        image_pos = processing(image.copy())
        image = postprocessing(image.astype(np.float32))

        # Note that original is not augmented
        return {"original":image, "positive": image_pos, "negative": image_neg}

    def set_seed(self):
        """
            Call this before iterating through the dataloader wrapper to ensure to always get the same data
        """
        random.seed(0)
        ia.seed(0)

def evaluate(data_loader, model, criterion_cosine, criterion_similarity, device):
    model.eval()
    curr_loss_cos = 0
    curr_loss_sim = 0
    dataset = data_loader.dataset
    dataset.set_seed()
    for i, batch in enumerate(data_loader):
        images = batch["original"].to(device)
        batch_length = images.shape[0]
        out_images = model(images)
        del images
        
        images_pos = batch["positive"].to(device)
        out_images_pos = model(images_pos)
        del images_pos

        images_neg = batch["negative"].to(device)
        out_images_neg = model(images_neg)
        del images_neg

        loss_sim = criterion_similarity(out_images, out_images_pos, out_images_neg).item()
        loss_cos = criterion_cosine(out_images, out_images_pos, torch.ones(batch_length).to(device)).item() + criterion_cosine(out_images, out_images_neg, -torch.ones(batch_length).to(device)).item()

        curr_loss_cos += loss_cos
        curr_loss_sim += loss_sim

    return curr_loss_cos / len(dataset), curr_loss_sim / len(dataset)


def main():
    rd = RD.RD(config_path)
    loss_type = rd.config["LOSS_TYPE"]

    model = FE.FeatureExtractor(rd.config["MODEL"], evaluate=False)
    preprocess = model.get_preprocessing(siamese=True)
    modelPath = os.path.join(rd.models_path, rd.config["MODEL"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.model.to(device) 

    # WandB setup
    wandb.init(
    entity="cvl-vhh-research",
    project="Train Siamese Net (vhh_rd)",
    notes="",
    tags=[],
    config=rd.config
    )

    get_dataset = lambda data_path: TriplesDataset(data_path, preprocess, get_augmentations()) 
    get_data_loader = lambda data_path, batchsize: DataLoader(get_dataset(data_path), batch_size=batchsize)

    train_loader = get_data_loader(rd.config["SIAM_TRAIN_PATH"], rd.config["BATCHSIZE_TRAIN"])
    val_loader = get_data_loader(rd.config["SIAM_VAL_PATH"], rd.config["BATCHSIZE_EVALUATE"])
    test_loader = get_data_loader(rd.config["SIAM_TEST_PATH"], rd.config["BATCHSIZE_EVALUATE"])

    cos = torch.nn.CosineSimilarity()
    def neg_CosineSimilarity(a,b):
        return -1*cos(a,b)

    criterion_cos = nn.CosineEmbeddingLoss(reduction='sum')
    criterion_sim =  torch.nn.TripletMarginWithDistanceLoss(distance_function=neg_CosineSimilarity, margin=1.7, reduction='sum')

    if loss_type == "cosine":
        criterion = criterion_cos
    elif loss_type == "triplet":
        criterion = criterion_sim

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_validation_loss = sys.float_info.max

    for epoch in range(rd.config["NR_EPOCHS"]):
        curr_loss = 0
        print("Epoch {0}".format(epoch))
        model.train()
        for i, batch in enumerate(train_loader):
            images = batch["original"].to(device)
            batch_length = images.shape[0]
            out_images = model(images)
            del images

            images_pos = batch["positive"].to(device)
            out_images_pos = model(images_pos)
            del images_pos
            
            images_neg = batch["negative"].to(device)
            out_images_neg = model(images_neg)
            del images_neg

            if loss_type == "triplet":
                loss = criterion(out_images, out_images_pos, out_images_neg)
                wandb.log({'Train\BatchLoss': loss.item() / batch_length})
            elif loss_type == "cosine":
                c1 = criterion(out_images, out_images_pos, torch.ones(batch_length).to(device))
                c2 = criterion(out_images, out_images_neg, -torch.ones(batch_length).to(device))
                loss = c1 + c2
                wandb.log({'Train\BatchLoss': loss.item() / batch_length, "Train\BatchPossSampleError": c1 , "Train\BatchNegativeSampleError": c2})

            
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
            print("\t{0} / {1}\t Loss {2}".format(i+1, len(train_loader), loss.item()))

        curr_loss = curr_loss / len(train_loader)
        print("Epoch {0}\nTraining Loss {1}".format(epoch, curr_loss))

        # Validate 
        val_cos_loss, val_tri_loss = evaluate(val_loader, model, criterion_cos, criterion_sim, device)
        print("Validation triplet loss: {0}\t Cosine loss: {1}".format(val_tri_loss, val_cos_loss))
        wandb.log({'Train\Loss': curr_loss, 'Val\LossTriplet': val_tri_loss, "Val\LossCos": val_cos_loss})

         # Early stopping
        if ((loss_type == "triplet" and val_tri_loss <= best_validation_loss) or (loss_type == "cosine" and val_cos_loss <= best_validation_loss)):
            if loss_type == "triplet":
                best_validation_loss = val_tri_loss
            elif loss_type == "cosine":
                best_validation_loss = val_cos_loss

            epochs_since_last_improvement = 0

            print("Saving to ", modelPath)
            torch.save(model.state_dict(), modelPath)
        else:
            epochs_since_last_improvement += 1
            if epochs_since_last_improvement >= rd.config["EPOCHS_EARLY_STPOPPING"]:
                print("No improvement since {0} epochs, stopping training.".format(epochs_since_last_improvement))
                print("Loading final model")
                model.load_state_dict(torch.load(modelPath))
                break

    test_cos_loss, test_tri_loss = evaluate(test_loader, model, criterion_cos, criterion_sim, device)
    print("\n\n\nFINAL EVALUATION:")
    val_cos_loss, val_tri_loss = evaluate(val_loader, model, criterion_cos, criterion_sim, device)
    print("Validation triplet loss: {0}\t Cosine loss: {1}".format(val_tri_loss, val_cos_loss))
    print("Test triplet loss: {0}\t Cosine loss: {1}".format(test_tri_loss, test_cos_loss))
    wandb.log({'Test\LossTriplet': test_tri_loss, "Test\LossCos": test_cos_loss})


def get_augmentations():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    return iaa.Sequential(
        [
            # Apply the following augmenters to most images.

            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 2),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ])),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),

                    # Invert each image's channel with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    ),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    ).augment_image

if __name__ == "__main__":
    main()
    