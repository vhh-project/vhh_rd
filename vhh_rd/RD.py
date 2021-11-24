import requests
import os
import vhh_rd.Configuration as Config
import vhh_rd.API as API
import cv2

class RD(object):
    """
        Main class of shot type classification (stc) package.
    """

    def __init__(self, config_path: str):
        self.config = Config.Config(config_path)

        # Ensure the data directory has the needed subdirectories
        dirs = ["STC", "Videos", "ExtractedFrames"]
        for dir in dirs:
            dir_to_create = os.path.join(self.config["DATA_PATH"], dir)
            if not os.path.isdir(dir_to_create):
                os.mkdir(dir_to_create)

        self.videos_path = os.path.join(self.config["DATA_PATH"], "Videos")
        self.extracted_frames_path = os.path.join(self.config["DATA_PATH"], "ExtractedFrames")

        self.api = API.API(self)

    def download_videos(self, video_list):
        """
        Downloads the videos 
        Returns a list of tuples (videoId, path_to_video)
        """
        videos = []
        for video_dict in video_list:
            try:
                video_format = video_dict["url"].split(".")[-1]
                video_path = os.path.join(self.videos_path, str(video_dict["id"]) + "." + str(video_format))
                self.api.download_video(video_dict["url"], video_path)

                videos.append({"id": video_dict["id"], "path": video_path})
            except Exception as e:
                print("Download failed.", e)
            
            # Todo: REMOVE THIS break to parse all videos
            break
        return videos

    def download_stc_results(self, videos):
        """
        Downloads all STC results for the specified videos
        :videos: list of videos given be download_videos()
        """
        shots = {}
        for video in videos:
            res_json = self.api.get_shot_data(video["id"])
            video["shots"] = res_json
        return videos

    def extract_center_frames(self, videos):
        """
        Extracts the center frames from each shot
        :shots: Is a dictionary where the keys are videoIds and values are the shot results
        """
        for video in videos:
            cap = cv2.VideoCapture(video["path"])
            for shot in video["shots"]:
                # Subtract 1 as shots stored in VhhMMSI have +1 added to all frame counts
                frame = int((shot["outPoint"] - shot["inPoint"]) / 2.) + shot["inPoint"] - 1

                path = os.path.join(self.extracted_frames_path, "id_{0}_frame_{1}.png".format(video["id"], frame))
                if os.path.exists(path):
                    continue

                cap.set(1, frame)
                success, image = cap.read()
                if not success:
                    continue
                cv2.imwrite(path, image)
                
    def run(self):  
        print("Collecting videos ", end = "")
        video_list = self.api.get_videos_list()

        print("Downloading videos")
        videos = self.download_videos(video_list)

        print("Downloading STC results")
        videos = self.download_stc_results(videos)

        print("Extracting center frames")
        self.extract_center_frames(videos)

        print("Resnet 152 (2048 dim layer output)")