import vhh_rd.RD as RD
import vhh_rd.Helpers as Helpers
import os
from sklearn.neighbors import DistanceMetric
import cv2 

"""
Given an image from the image folder, finds similar images
"""

img_name = "id_8238_frame_2824"
config_path = "./config/config_rd.yaml"
top_results = 10

def main():
    rd = RD.RD(config_path)
    query_features = Helpers.load_features(img_name, rd)
    dist = DistanceMetric.get_metric('euclidean')

    similarities = []
    # Compute all similarities
    for key_features_name in os.listdir(rd.features_path):
        features_path = os.path.join(rd.features_path, key_features_name)
        key_img_name = key_features_name.split("_model_")[0]
        key_features = Helpers.load_features(key_img_name, rd)

        similarity = dist.pairwise([query_features, key_features])[0,1]
        similarities.append((similarity, key_img_name))

    # Sort by similarity (0 = highest similarity)
    similarities.sort(key=lambda d: d[0])
    print(similarities[0: top_results+1])

    cv2.imwrite(os.path.join(rd.raw_results_path, "original.png"), Helpers.load_img(img_name + ".png", rd)[1])

    for i in range(top_results):
        cv2.imwrite(os.path.join(rd.raw_results_path, "top_{0}.png".format(i+1)), Helpers.load_img(similarities[i][1] + ".png", rd)[1])

    # Save worst results
    similarities.reverse()
    for i in range(top_results):
        cv2.imwrite(os.path.join(rd.raw_results_path, "top_{0}.png".format(len(similarities) - i)), Helpers.load_img(similarities[i][1] + ".png", rd)[1])

if __name__ == "__main__":
    main()