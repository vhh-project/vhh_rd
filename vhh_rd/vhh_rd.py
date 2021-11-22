import yaml
import requests
import os

class RD(object):
    """
        Main class of shot type classification (stc) package.
    """

    def __init__(self, config_file: str):
        """
        """
        # Load config
        with open(config_file, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("COULD NOT LOAD YAML FILE")
                quit()

        # Ensure the data directory has the needed subdirectories
        dirs = ["STC", "Videos", "ExtractedFrames"]
        for dir in dirs:
            dir_to_create = os.path.join(self.config["DATA_PATH"], dir)
            if not os.path.isdir(dir_to_create):
                os.mkdir(dir_to_create)

        self.videos_path = os.path.join(self.config["DATA_PATH"], "Videos")

        # Configure endpoints
        self.API_ENDPOINT = self.config["API_ENDPOINT"]
        self.API_VIDEO_SEARCH_ENDPOINT = self.API_ENDPOINT + "/videos/search"

    def collect_videos(self):
        """
        Downloads a list of all processed videos from VhhMMSI
        """

        res = requests.get(self.API_VIDEO_SEARCH_ENDPOINT, params={"processed": True})
        print(res)
        res_json = res.json()
        return res_json

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

                if not os.path.exists(video_path):
                    # Download
                    print("Downloading {0}: ".format(video_dict["url"]), end = "")
                    video_file = requests.get(video_dict["url"]) 
                    open(video_path, 'wb').write(video_file.content)
                    print("DONE")

                videos.append({"id": video_dict["id"], "path": video_path})
            except Exception as e:
                print("FAILED: ", e)
            
            # Todo: REMOVE THIS break
            break
        return videos

    def download_stc_results(self, videoIds):
        """
        Downloads all STC results for the specified videos
        :videoIds: list of video IDs
        """
        pass

    def run(self):  
        print("Collecting videos: ", end = "")
        video_list = self.collect_videos()

        print("Downloading videos")
        self.download_videos(video_list)

        print("Downloading STC results")

        print("Extracting center frames")