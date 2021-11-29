import vhh_rd.RD as RD
import vhh_rd.Helpers as Helpers
import os
import cv2 
import vhh_rd.Distance as Dis

"""
Given an image from the image folder, finds similar images
Results will be stored in RawRestults/img_name
"""

img_name = "id_8231_frame_8328_sid_54"
config_path = "./config/config_rd.yaml"
top_results = 10

def main():
    rd = RD.RD(config_path)
    query_features = Helpers.load_features(img_name, rd)
    dist = Dis.Distance(rd.config["DISTANCE_METRIC"], rd.config["METRIC_PARAM"])

    output_path = os.path.join(rd.raw_results_path, img_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    similarities = []
    # Compute all similarities
    for key_features_name in os.listdir(rd.features_path):
        key_img_name = key_features_name.split("_model_")[0]
        key_features = Helpers.load_features(key_img_name, rd)

        similarity = dist(query_features, key_features)
        similarities.append((similarity, key_img_name))

    # Sort by similarity (0 = highest similarity)
    similarities.sort(key=lambda d: d[0])
    print(similarities[0: top_results+1])

    cv2.imwrite(os.path.join(output_path, "original.png"), Helpers.load_img(img_name + ".png", rd)[1])

    for i in range(top_results):
        cv2.imwrite(os.path.join(output_path, "top_{0}.png".format(i+1)), Helpers.load_img(similarities[i][1] + ".png", rd)[1])

    # Save worst results
    similarities.reverse()
    for i in range(top_results):
        cv2.imwrite(os.path.join(output_path, "top_{0}.png".format(len(similarities) - i)), Helpers.load_img(similarities[i][1] + ".png", rd)[1])

if __name__ == "__main__":
    main()