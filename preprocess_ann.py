import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

from numpy.lib.arraysetops import unique

colors_map = [([5, 247, 131], "BackgroundVegetation", 2, False),
              ([4, 115, 181], "Birch trunk", 4, True),
              ([64, 0, 160], "Ground", 7, False),
              ([225, 126, 94], "Harvester head", 6, True),
              ([102, 95, 178], "LogPile", 8, True),
              ([255, 0, 124], "Pine trunk", 3, True),
              ([46, 94, 161], "Spruce trunk", 1, True),
              ([254, 174, 64], "Tree trunk", 5, True),
              ([98, 165, 118], "Trunk on Harvester", 9, True)]

categories = [
    {
        "id": 1,
        "name": "Spruce trunk",
        "supercategory": "object"
    },
    {
        "id": 2,
        "name": "BackgroundVegetation",
        "supercategory": "background"
    },
    {
        "id": 3,
        "name": "Pine trunk",
        "supercategory": "object"
    },
    {
        "id": 4,
        "name": "Birch trunk",
        "supercategory": "object"
    },
    {
        "id": 5,
        "name": "Tree trunk",
        "supercategory": "object"
    },
    {
        "id": 6,
        "name": "Harvester head",
        "supercategory": "object"
    },
    {
        "id": 7,
        "name": "Ground",
        "supercategory": "background"
    },
    {
        "id": 8,
        "name": "LogPile",
        "supercategory": "object"
    },
    {
        "id": 9,
        "name": "Trunk on Harvester",
        "supercategory": "object"
    }
],


def city_scapes_2_coco(root, ann, out_file, out_folder):

    with open(ann) as coco_file:
        # read file
        gt_data = json.load(coco_file)

    categories = gt_data["categories"]
    images = gt_data["images"]
    filenames = list(map(lambda a: a["file_name"], images))

    data ={}

    data["categories"] = categories
    data["images"] = images
    data["annotations"] = []

    all = []

    for infile in sorted(glob.glob('{}/*.png'.format(root))):
        all.append(os.path.basename(infile))

    imgs_colors = list(filter(lambda im: "color" in im, all))

    imgs_instance = list(filter(lambda im: "instanceIds" in im, all))

    imgs_label = list(filter(lambda im: "labelIds" in im, all))

    # print(list(map(lambda im: im.split("_")[0], imgs_colors)))

    # print(len(all), len(instance_imgs_instance), len(
        # instance_imgs_label), len(instance_imgs_colors))

    # for idx, file in enumerate(imgs_colors):
    for color_im, ins_im, label_im, img in zip(imgs_colors, imgs_instance, imgs_label, filenames):

        color_image = np.array(Image.open(os.path.join(root, color_im))).astype(np.int64)
        ins_image = np.array(Image.open(os.path.join(root, ins_im))).astype(np.int64)
        label_image = np.array(Image.open(os.path.join(root, label_im))).astype(np.int64)
        # print(color_im, ins_im)
        # print(np.where(color_image == [5, 247, 131]), color_image[575, 1718, :])
        # print(np.unique(ins_image))
        new_image = np.zeros_like(ins_image, dtype=np.uint16)

        obj = {"segments_info":[]}
        id2cls = []

        for idx, color in enumerate(colors_map):
            
            locs = np.where(np.all(color_image == color[0], axis=-1))
            if len(locs[0])> 0:
                if color[3]:
                    # print(np.unique(ins_image[locs[0], locs[1]]))
                    new_image[locs[0], locs[1]] = ins_image[locs[0], locs[1]]

                    for val in np.unique(new_image[locs[0], locs[1]]):
                        id2cls.append((val, color[2], True))
                    # continue
                else:
                    new_image[locs[0], locs[1]] = color[2]
                    for val in np.unique(new_image[locs[0], locs[1]]):
                        id2cls.append((val, color[2], False))
                    # obj["segments_info"].append({"area": area, id: val})

        
        # print(np.unique(new_image))
        # print(id2cls)
        im = Image.fromarray(new_image)
        basename = ".".join(img.split(".")[:-1])
        im.save("{}/{}.png".format(out_folder, basename))
        unique_val, areas = np.unique(new_image, return_counts=True)

        for val, area in zip(unique_val, areas):
            # print(val)
            
            if val != 0:
                cat = list(filter(lambda a: a[0] == val, id2cls))[0]
                
                obj["segments_info"].append({"area": int(area), "id": int(val), "category_id": cat[1], "isthing": cat[2]})
        
        data["annotations"].append(obj)

        # print( data["annotations"])
        # break

    with open(out_file, 'w') as outfile:
        json.dump(data, outfile)
        

                

        

    

ann = "submit_pred/pred.json" #From prediction
root = "annotations/cityscapes1.0/gtFine/default" #From cvat cityscapes export
out_file = "gt_city_2.json" #gt cityscapes format
out_folder = "gt_city_2" #output folder
# city_scapes_2_coco(root, ann, out_file, out_folder)



def reverse_ann(json_file, out):

    with open(json_file) as city_file:
        # read file
        gt_data = json.load(city_file)
    
    img_list =  gt_data["images"]
    annotations_list =  gt_data["annotations"]

    img_list.reverse()
    annotations_list.reverse()

    for idx, im in enumerate(img_list):
        img_list[idx]["id"] = idx + 1

    gt_data["images"] = img_list
    gt_data["annotations"] = annotations_list

    with open(out, 'w') as outfile:
        json.dump(gt_data, outfile)

json_file = out_file
out = "gt_city_2_reverse.json"
# reverse_ann(json_file, out_file)