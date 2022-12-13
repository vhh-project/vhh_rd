import vhh_rd.Feature_Extractor as FE
import vhh_rd.RD as RD
import vhh_rd.Transformations as Transformations
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os, sys, glob, random, itertools
import cv2
from torchvision import transforms
import torch.optim as optim
import wandb
import imgaug as ia

config_path = "./config/config_rd.yaml"
lr = 0.000001

# Do not use NA directory
dirs_to_use = ["CU", "ELS", "LS", "MS", "I"]

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
        self.images_paths = list(set(glob.glob(os.path.join(image_folder, "**/*.png"))) - set(glob.glob(os.path.join(image_folder, "NA/*.png"))))
        np.random.shuffle(self.images_paths)

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
        # img_aug_neg = cv2.resize(augment(image_neg), (256,256), interpolation = cv2.INTER_AREA)
        # img_aug_pos = cv2.resize(augment(image), (256,256), interpolation = cv2.INTER_AREA)
        # img_pos = cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
        # cv2.imshow("positives", np.hstack((img_pos.astype(np.uint8), img_aug_pos.astype(np.uint8), img_aug_neg.astype(np.uint8))))
        # cv2.waitKey(0)

        image_neg = processing(image_neg)
        image_pos = processing(image.copy())
        image = processing(image)

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
    for i, batch in enumerate(tqdm(data_loader)):
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

    get_dataset = lambda data_path: TriplesDataset(data_path, preprocess, Transformations.get_augmentations()) 
    get_data_loader = lambda data_path, batchsize: DataLoader(get_dataset(data_path), batch_size=batchsize, num_workers=rd.config["NUM_WORKERS_TRAINING"])

    train_loader = get_data_loader(rd.config["SIAM_TRAIN_PATH"], rd.config["BATCHSIZE_TRAIN"])
    val_loader = get_data_loader(rd.config["SIAM_VAL_PATH"], rd.config["BATCHSIZE_EVALUATE"])
    test_loader = get_data_loader(rd.config["SIAM_TEST_PATH"], rd.config["BATCHSIZE_EVALUATE"])

    cos = torch.nn.CosineSimilarity()
    def neg_CosineSimilarity(a,b):
        return -1*cos(a,b)

    criterion_cos = nn.CosineEmbeddingLoss(reduction='sum')
    criterion_sim =  torch.nn.TripletMarginWithDistanceLoss(distance_function=neg_CosineSimilarity, reduction='sum')

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
            print("\t{0} / {1}\t Loss {2}".format(i+1, len(train_loader), loss.item() / batch_length))

        curr_loss = curr_loss / len(train_loader.dataset)
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

    print("\n\n\nFINAL EVALUATION:")

    val_cos_loss, val_tri_loss = evaluate(val_loader, model, criterion_cos, criterion_sim, device)
    test_cos_loss, test_tri_loss = evaluate(test_loader, model, criterion_cos, criterion_sim, device)

    print("Validation triplet loss: {0}\t Cosine loss: {1}".format(val_tri_loss, val_cos_loss))
    print("Test triplet loss: {0}\t Cosine loss: {1}".format(test_tri_loss, test_cos_loss))
    wandb.log({'Test\LossTriplet': test_tri_loss, "Test\LossCos": test_cos_loss})

if __name__ == "__main__":
    main()
    