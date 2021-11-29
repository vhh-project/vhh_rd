import requests
import os
import glob
import vhh_rd.Configuration as Config
import vhh_rd.Feature_Extractor as FE
import vhh_rd.Helpers as Helpers
import cv2
import csv
from torchvision import transforms
import torch
from tqdm import tqdm

class RD(object):
    """
        Main class of shot type classification (stc) package.
    """

    def __init__(self, config_path: str):
        self.config = Config.Config(config_path)

        # Ensure the data directory has the needed subdirectories
        dirs = ["ExtractedFrames", "FinalResults",  os.path.join("FinalResults", self.config["MODEL"]), "RawResults", "Visualizations"]
        for dir in dirs:
            dir_to_create = os.path.join(self.config["DATA_PATH"], dir)
            if not os.path.isdir(dir_to_create):
                os.mkdir(dir_to_create)

        self.extracted_frames_path = os.path.join(self.config["DATA_PATH"], "ExtractedFrames")
        self.features_path = os.path.join(self.config["DATA_PATH"], "FinalResults", self.config["MODEL"])
        self.raw_results_path = os.path.join(self.config["DATA_PATH"], "RawResults")
        self.visualizations_path = os.path.join(self.config["DATA_PATH"], "Visualizations")
    
    def collect_videos(self):
        """
        Collects the videos from the directory specified in the config
        """
        videos = []
        for video_name in os.listdir(self.config["VIDEO_PATH"]):
            videos.append({"id": video_name.split(".")[0], "path": os.path.join(self.config["VIDEO_PATH"], video_name)})
        
        return videos

    def collect_sbd_results(self, videos):
        """
        Loads the shot information from the directory specified in the config
        """
        for video in videos:
            file_path = glob.glob(os.path.join(self.config["SHOT_PATH"], "{0}.csv".format(video["id"])))[0]
            with open(file_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=';')
                shots = []
                for row in csv_reader:
                    shots.append({"start": int(row["start"]), "end": int(row["end"]), "shot_id": int(row["shot_id"])})
            video["shots"] = shots
        return videos

    def extract_center_frames(self, videos):
        """
        Extracts the center frames from each shot
        :shots: Is a dictionary where the keys are videoIds and values are the shot results
        """
        for video in tqdm(videos):
            cap = None

            for shot in video["shots"]:
                frame = int((shot["end"] - shot["start"]) / 2.) + shot["start"] 

                path = os.path.join(self.extracted_frames_path, "id_{0}_frame_{1}_sid_{2}.png".format(video["id"], frame, shot["shot_id"]))
                if os.path.exists(path):
                    continue

                if cap is None:
                    cap = cv2.VideoCapture(video["path"])

                cap.set(1, frame)
                success, image = cap.read()
                if not success:
                    continue
                cv2.imwrite(path, image)

    def get_feature_path(self, img_name):
        """
        The path at which a feature of a given image will be stored
        """
        return os.path.join(self.features_path, img_name.split(".")[0] + "_model_{0}.pickle".format(self.config["MODEL"]))
    
    def do_feature_extraction(self):
        fe = FE.FeatureExtractor(self.config["MODEL"])
        preprocess = fe.get_preprocessing()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        fe.model.to(device)

        imgs_all = os.listdir(self.extracted_frames_path)

        # Remove images whose features we have already computed
        imgs_all = [x for x in imgs_all if not os.path.exists(self.get_feature_path(x))]

        batchsize = self.config["BATCHSIZE"]
        imgs_as_batches = [imgs_all[i:i+batchsize] for i in range(0, len(imgs_all), batchsize)]

        for img_names in tqdm(imgs_as_batches):
            tensors = []
            for img_name in img_names:
                _, img = Helpers.load_img(img_name, self)
                input_tensor = preprocess(img)
                tensors.append(input_tensor)

            input_batch = torch.stack(tensors)
            input_batch = input_batch.to(device)

            features = fe(input_batch).cpu().detach()
            # Squeeze extra dimensions away
            if len(features.shape) > 2:
                # Check if there are enough dimensions to squeeze away
                if len([x for x in features.shape[2:] if x == 1]) < len(features.shape) - 2:
                    raise ValueError("Output dimensions from model are wrong")

                # Squeeze extra dimension of size 1 away
                while(len(features.shape) > 2):
                    for i in range(2, len(features.shape)):
                        if features.shape[i] == 1:
                            features = torch.squeeze(features, i)
                            break


            features = features.numpy()
            for i, img_name in enumerate(img_names):
                Helpers.do_pickle(features[i,:], self.get_feature_path(img_name))

    def run(self):  
        print("Collect videos")
        videos = self.collect_videos()

        print("Collect SBD results")
        videos = self.collect_sbd_results(videos)

        print("Extracting center frames")
        self.extract_center_frames(videos)

        print("Compute features")
        self.do_feature_extraction()