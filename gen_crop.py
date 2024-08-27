import os
import cv2
from tqdm import tqdm
import json
import copy
import numpy as np
import torch

from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from models.dense_heads.yolc_head import YOLCHead

num_classes = 10

def gen_heatmap(gt_bbox, gt_labels, img_shape):
    '''
        gt_bboxes: [x, y, w, h]
    '''
    gt_bbox = torch.tensor(gt_bbox)
    gt_labels = torch.tensor(gt_labels)-1
    H, W = img_shape
    center_heatmap_target = gt_bbox.new_zeros(
            [1, num_classes, H, W]).to(torch.float)
    origin_center_x = gt_bbox[:, [0]] + gt_bbox[:, [2]] / 2
    origin_center_y = gt_bbox[:, [1]] + gt_bbox[:, [3]] / 2
    origin_gt_centers = torch.cat((origin_center_x, origin_center_y), dim=1)

    for j in range(len(origin_gt_centers)):
        ind = gt_labels[j]
        radius = gaussian_radius([gt_bbox[j,2], gt_bbox[j,3]],
                                        min_overlap=0.3)
        radius = max(0, int(radius))
        ori_ctx_int, ori_cty_int = origin_gt_centers[j].int()
        gen_gaussian_target(center_heatmap_target[0, ind],
                                        [ori_ctx_int, ori_cty_int], radius)

    return center_heatmap_target

def convert_to_cocodetection(dir, output_dir):
    train_dir = os.path.join(dir, "VisDrone2019-DET-train")
    train_annotations = os.path.join(train_dir, "annotations")
    train_images = os.path.join(train_dir, "images")

    out_dir = os.path.join(dir, "VisDrone2019-DET-train-crop")

    if not os.path.exists(out_dir + "/images"):
        os.makedirs(out_dir + "/images")

    id_num = 0
    model = YOLCHead(1, 1, 10)
 
    categories = [{"id": 1, "name": "pedestrian"},
                  {"id": 2, "name": "people"},
                  {"id": 3, "name": "bicycle"},
                  {"id": 4, "name": "car"},
                  {"id": 5, "name": "van"},
                  {"id": 6, "name": "truck"},
                  {"id": 7, "name": "tricycle"},
                  {"id": 8, "name": "awning-tricycle"},
                  {"id": 9, "name": "bus"},
                  {"id": 10, "name": "motor"}
                  ]
    for mode in ["train"]:
        images = []
        annotations = []
        print(f"start loading {mode} data...")
        if mode == "train":
            set = os.listdir(train_annotations)
            annotations_path = train_annotations
            images_path = train_images
        cnt = 0
        for i in tqdm(set):
            f = open(annotations_path + "/" + i, "r")
            name = i.replace(".txt", "")
            image = {}
            file = cv2.imread(images_path + "/" + name + ".jpg")
            height, width = file.shape[:2]
            file_name = name + ".jpg"
            image["file_name"] = file_name
            image["height"] = height
            image["width"] = width
            image["id"] = name
            images.append(image)
            cur_anns = []
            
            gt_bboxes, gt_labels = [], []

            for line in f.readlines():
                annotation = {}
                line = line.replace("\n", "")
                if line.endswith(","):  # filter data
                    line = line.rstrip(",")
                line_list = [int(i) for i in line.split(",")]
                bbox_xywh = [line_list[0], line_list[1], line_list[2], line_list[3]]
                annotation["image_id"] = name
                annotation["score"] = line_list[4]
                annotation["bbox"] = bbox_xywh
                annotation["category_id"] = int(line_list[5])
                annotation["id"] = id_num
                annotation["iscrowd"] = 0
                annotation["segmentation"] = []
                annotation["area"] = bbox_xywh[2] * bbox_xywh[3]
                if int(line_list[5]) == 0 or int(line_list[5]) > 10:
                    continue
                else:
                    id_num += 1
                    annotations.append(annotation)
                    cur_anns.append(annotation)
                    gt_bboxes.append(bbox_xywh)
                    gt_labels.append(int(line_list[5]))
                    #cv2.rectangle(file, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 1)
            cv2.imwrite(out_dir + "/images/" + file_name, file)

            # generate crop images
            center_heatmap_target = gen_heatmap(gt_bboxes, gt_labels, (height, width))

            img_metas = [{'border':[0, 0, 0, 0]}]
            subregion_coord = model.LSM(center_heatmap_preds=[center_heatmap_target], img_metas=img_metas)

            areas = []
            for item in subregion_coord:
                x,y,w,h = item
                areas.append(w*h)
            areas = np.array(areas)
            idx = areas.argsort()[::-1]
            saved_crop = 1
            if len(idx) > saved_crop:
                idx = idx[:saved_crop]
            
            subregion_coord = subregion_coord[idx]
            img_np = file
            count = 0
            for i in range(len(subregion_coord)):
                x, y, w, h = subregion_coord[i]
                # filter small crops
                if w*h < 96*96:
                    continue
                count += 1
                x = int(x)
                y = int(y)

                bboxes = [x, y, x+w, y+h]
                box_scale_ratio = 1.2         # box scale factor

                w_half = (bboxes[2] - bboxes[0]) * 0.5
                h_half = (bboxes[3] - bboxes[1]) * 0.5
                x_center = (bboxes[2] + bboxes[0]) * 0.5
                y_center = (bboxes[3] + bboxes[1]) * 0.5

                w_half *= box_scale_ratio
                h_half *= box_scale_ratio
                w, h = img_np.shape[1], img_np.shape[0]

                # scale dense region by 1.2x to avoid truncation
                boxes_scaled = [0, 0, 0, 0]
                boxes_scaled[0] = min(max(x_center - w_half, 0), w - 1)
                boxes_scaled[2] = min(max(x_center + w_half, 0), w - 1)
                boxes_scaled[1] = min(max(y_center - h_half, 0), h - 1)
                boxes_scaled[3] = min(max(y_center + h_half, 0), h - 1)
                boxes_scaled = [int(i) for i in boxes_scaled]

                img_scale_ratio = 1.5   # image scale factor
                w_new = boxes_scaled[2] - boxes_scaled[0]
                h_new = boxes_scaled[3] - boxes_scaled[1]
                # crop and resize sub-image by 1.5x
                img_crop = img_np[boxes_scaled[1]:boxes_scaled[3], boxes_scaled[0]:boxes_scaled[2]]
                img_resize = cv2.resize(img_crop, (int(w_new * img_scale_ratio), int(h_new * img_scale_ratio)))

                # Padding to (1024, 640)
                h_delta = max(640 - img_resize.shape[0], 0)
                w_delta = max(1024 - img_resize.shape[1], 0)
                left= w_delta // 2
                top = h_delta // 2
                right = w_delta - left
                bottom = h_delta - top
                output = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

                new_path = out_dir + "/images/" + name + "_part%d.jpg"%(count)
                
                annos = findincluster(cur_anns, boxes_scaled)
                for i in range(len(annos)):
                    annos[i]["id"] = id_num
                    annos[i]["image_id"] = name + "_part%d"%(count)
                    # bbox resize
                    annos[i]["bbox"] = [int(k) for k in ([j*img_scale_ratio for j in annos[i]["bbox"]])]
                    # transform bbox to new image
                    annos[i]["bbox"][0] += left
                    annos[i]["bbox"][1] += top
                    box = annos[i]["bbox"]
                    id_num += 1
                    annotations.append(annos[i])
                    #cv2.rectangle(output, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 1)
                image_tmp = {}
                image_tmp["file_name"] = name + "_part%d.jpg"%(count)
                image_tmp["height"] = output.shape[0]
                image_tmp["width"] = output.shape[1]
                image_tmp["id"] = name + "_part%d"%(count)
                images.append(image_tmp)
                cv2.imwrite(new_path, output)

        dataset_dict = {}
        dataset_dict["images"] = images
        dataset_dict["annotations"] = annotations
        dataset_dict["categories"] = categories
        json_str = json.dumps(dataset_dict)
        with open(f'{output_dir}/VisDrone2019-DET_{mode}_coco_1crop.json', 'w') as json_file:
            json_file.write(json_str)
    print("json file write done...")


def findincluster(cur_anns, boxes_scaled):
    left, top, right, bot = boxes_scaled 
    anns = []
    for anno in cur_anns:
        bbox_left, bbox_top, w, h = anno["bbox"]
        bbox_right= bbox_left + w
        bbox_bottom = bbox_top + h
        if left <= bbox_left and right >= bbox_right and top <= bbox_top and bot >= bbox_bottom:
            anno_new = copy.deepcopy(anno)
            anno_new["bbox"][0] -= left
            anno_new["bbox"][1] -= top
            anns.append(anno_new)
    return anns

 
if __name__ == '__main__':
    path = "/your_path/data/Visdrone2019"
    convert_to_cocodetection(path, path)
