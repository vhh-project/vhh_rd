import vhh_rd.RD as RD
import vhh_rd.Helpers as Helpers
import os
import cv2 
import vhh_rd.Distance as Dis
import math
import numpy as np
import argparse

"""
Given an image from the image folder, finds similar images
Results will be stored in RawRestults/img_name
Run with 
    python Demo/find_similar_images.py -i IMG_NAME
where IMG_NAME is the name of a frame in the ExtractedFrames directory
"""

config_path = "./config/config_rd.yaml"
nr_top_results = 12

# Size of images stored in visualization, 25 means it will have a fourth of the width and height
image_size_in_percent = 12

# Must divide top_results
images_per_row = 6

font = cv2.FONT_HERSHEY_DUPLEX

def main():
    rd = RD.RD(config_path)

    # ARGUMENT PARSING

    parser = argparse.ArgumentParser(description="")

    # Get the folder with the pictures as an argument
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-i', dest='img_name', help =
    "The query image in the ExractedFrames directory", required=True)
    args = parser.parse_args()
    img_name = args.img_name

    # Remove .png ending
    img_name = img_name.replace(".png", "")


    print("\nOriginal image: {0}".format(img_name))

    query_features = Helpers.load_features(img_name, rd)
    dist = Dis.Distance(rd.config["DISTANCE_METRIC"], rd.config["METRIC_PARAM"])

    similarities = []

    # Compute all similarities
    for key_features_name in os.listdir(rd.features_path):
        key_img_name = key_features_name.split("_model_")[0]
        key_features = Helpers.load_features(key_img_name, rd)

        similarity = dist(query_features, key_features)
        similarities.append((similarity, key_img_name))

    # Sort by similarity (0 = highest similarity)
    similarities.sort(key=lambda d: d[0])

    original_img = Helpers.load_img(img_name + ".png", rd)
    original_text = {"vid": img_name.split("_")[1], "sid": img_name.split("_")[5]}

    top_imgs = [Helpers.load_img(similarities[i][1] + ".png", rd) for i in range(nr_top_results)]
    top_text = [{"distance": similarities[i][0], "vid": similarities[i][1].split("_")[1], "sid": similarities[i][1].split("_")[5]}  for i in range(nr_top_results)]

    print("{0} images searched".format(len(similarities)))
    print("Most similar images:")
    for i in range(nr_top_results):
        print("\t{0}, distance: {1}".format(similarities[i][1], similarities[i][0]))


    # Work on worst results (0 = lowest similarity)
    similarities.reverse()
    bot_imgs = [Helpers.load_img(similarities[i][1] + ".png", rd) for i in range(nr_top_results)]
    bot_text = [{"distance": similarities[i][0], "vid": similarities[i][1].split("_")[1], "sid": similarities[i][1].split("_")[5]}  for i in range(nr_top_results)]

    text = {"original": original_text, "top":top_text, "bot": bot_text}
    visualization_path = os.path.join(rd.raw_results_path, img_name + "_model_" + rd.config["MODEL"] + "_metric_" + rd.config["DISTANCE_METRIC"] + ".png")
    visualize(rd, original_img, top_imgs, bot_imgs, text, visualization_path)
    print("Stored visualization at {0}".format(visualization_path))

def get_dimension_after_resizing(img, percentage):
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    return width, height

def resize(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def combine_images(imgs, images_per_row, width, height, text):
    rows = []
    while (len(imgs) > 0):
        # Create a row of images
        row = [imgs.pop(0) for _ in range(min(len(imgs), images_per_row))]
        text_row = [text.pop(0) for _ in range(min(len(text), images_per_row))]
        row = [resize(x, width, height) for x in row]
        for i in range(len(row)):
            # Add text on top
            img = cv2.copyMakeBorder(row[i] ,50,0,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            cv2.putText(img,"Distance: {0}".format(round(text_row[i]["distance"], 2)), (5,15), font, 0.5, [0, 0, 0])
            cv2.putText(img,"VID: {0}, SID: {1}".format(text_row[i]["vid"], text_row[i]["sid"]), (5,35), font, 0.5, [0, 0, 0])
            row[i] = img

        #Combine
        row = np.hstack(row)
        rows.append(row)
    return np.vstack(rows)

def visualize(rd, original, top_imgs, bot_imgs, text, output_path):
    nr_rows = math.ceil(len(top_imgs) / float(images_per_row)) + math.ceil(len(bot_imgs) / float(images_per_row))
    width, height = get_dimension_after_resizing(original, image_size_in_percent)

    # Create the parts that contain the best and worst images
    top_results = combine_images(top_imgs, images_per_row, width, height, text["top"])
    bottom_results = combine_images(bot_imgs, images_per_row, width, height, text["bot"])

    width_bar = top_results.shape[1]
    white_bar = np.zeros([50, width_bar, 3],dtype=np.uint8)
    white_bar.fill(255)

    # Create bars that say "Top/Bottom x results" 
    top_results_bar = white_bar.copy()
    cv2.putText(top_results_bar,"Top {0} results".format(nr_top_results), (5,30), font, 1, [0, 0, 255])
    bot_results_bar = white_bar.copy()
    cv2.putText(bot_results_bar,"Bottom {0} results".format(nr_top_results), (5,30), font, 1, [0, 0, 255])

    right_side = np.vstack([top_results_bar, top_results, bot_results_bar, bottom_results])

    # Add original image on the left side
    og_w, og_h = get_dimension_after_resizing(original, image_size_in_percent * nr_rows)
    original = resize(original, og_w, og_h)
    missing_height = right_side.shape[0] - original.shape[0]
    filler_top = math.ceil(missing_height / 2.)
    filler_bot = math.floor(missing_height / 2.)
    left_side = cv2.copyMakeBorder(original, filler_top,filler_bot,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
    cv2.putText(left_side,"Query image", (5,50), font, 1.5, [0, 0, 255])
    cv2.putText(left_side,"VID {0}, SID: {1}".format(text["original"]["vid"], text["original"]["sid"]), (5,90), font, 0.5, [0, 0, 255])

    cv2.putText(left_side,"Model: {0}, Distance metric: {1}".format(rd.config["MODEL"], rd.config["DISTANCE_METRIC"]), (5,right_side.shape[0] - 10), font, 0.5, [0, 0, 255])

    # Combine everything
    final_visualization = np.hstack([left_side, right_side])
    cv2.imwrite(output_path, final_visualization)

if __name__ == "__main__":
    main()