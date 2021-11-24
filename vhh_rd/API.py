import requests
import os

class API(object):
    """
    Object that interacts with the VhhMMSI server via REST APIs
    """

    def __init__(self, rd):
        self.rd = rd

        # Configure endpoints
        self.API_ENDPOINT = rd.config["API_ENDPOINT"]
        self.API_VIDEO_SEARCH_ENDPOINT = self.API_ENDPOINT + "/videos/search"

    def get_shot_data(self, videoId):
        res = requests.get(self.API_ENDPOINT + "/videos/{0}/shots/auto".format(videoId))
        res_json = res.json()
        return res_json

    def get_videos_list(self):
        """
        Downloads a list of all processed videos from VhhMMSI
        """
        res = requests.get(self.API_VIDEO_SEARCH_ENDPOINT, params={"processed": True})
        return res.json()

    def download_video(self, url, path_to_store_video):
        """
        Downloads a video from VhhMMSI
        """
        if os.path.exists(path_to_store_video):
            return

        video_file = requests.get(url) 
        open(path_to_store_video, 'wb').write(video_file.content)