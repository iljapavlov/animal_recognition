from __future__ import print_function, division
import cv2
import torchvision
import math
import os
import torch
import random
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageEnhance
import matplotlib.patches as patches
import time
from tqdm import tqdm
from iou import intersection_over_union
import torchvision.transforms.functional as TF
import warnings
import torchvision.ops as ops
warnings.filterwarnings('ignore')

# Load Data
class AnimalDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform = None):
        self.ann = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = list(set(self.ann["id"].tolist())) #todo >
        self.class2id = {
            "29": 1,
            "38": 2,
        }
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.root_dir + '/' + img_id

        img = np.array(Image.open(img_path))

        labels = self.ann.loc[self.ann['id']==img_id,'labels'].tolist()
        labels = [self.class2id[str(l)] for l in labels]

        bbox = self.ann.loc[self.ann['id']==img_id, 'xmin':'ymax'].values.tolist()
        sample = (img, {"boxes": bbox, "labels": labels})
        if self.transform:
            sample = self.transform(sample)
        return sample
    def get_id(self, idx):
        return self.ann["id"][idx]

class Normalize(object):
    def __call__(self, sample):
        img, _ = sample
        img = img.astype(np.float32) / 255
        img -= np.min(img)
        img /= np.max(img)
        sample = (img, sample[1])
        return sample

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample[0], sample[1]['boxes']

        h, w = image.shape[:2]
        new_h, new_w = int(self.output_size), int(self.output_size)

        img = cv2.resize(image,(new_h, new_w))

        #resizing bbox
        for i,b in enumerate(bbox):
            bh, bw = b[2]-b[0], b[3]-b[1]
            center = [int(b[0]+bw/2), int(b[1]+bh/2)]
            b = b - np.array(center+center)
            b = np.multiply(b , np.array([new_w / w, new_h / h, new_w / w, new_h / h]))
            center = np.multiply(np.array(center+center), np.array([new_w / w, new_h / h, new_w / w, new_h / h]))
            b = b + center
            bbox[i] = b

        return (img, {'boxes': bbox, 'labels': sample[1]['labels']})

class Random_flip(object):
    def __call__(self, sample):
        image, bbox = sample[0], sample[1]['boxes']
        h, w = image.shape[:2]

        #Random horizontal flip
        if (random.randint(0,1)):
            image = cv2.flip(image, 1)
            for i,b in enumerate(bbox):
                b[0] = w - b[0]
                b[2] = w - b[2]
                b[0],b[2] = b[2],b[0]
                bbox[i] = b

        return (image, {'boxes': bbox, 'labels': sample[1]['labels']})

class ColorJitter(object):
    def __call__(self, sample):
        img = Image.fromarray(np.uint8(sample[0]*255)).convert('RGB')

        rand_c = 0.9 + random.random()/2
        filter = ImageEnhance.Brightness(img)
        img = filter.enhance(rand_c)

        filter = ImageEnhance.Contrast(img)
        img = filter.enhance(rand_c)

        filter = ImageEnhance.Color(img)
        img = filter.enhance(rand_c)

        img = np.array(img, dtype='uint8')/255

        return (img, sample[1])

class ToTensor(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, bbox = sample[0], np.array(sample[1]['boxes'])

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))


        # to be safe
        for i, b in enumerate(bbox):
            if b[0]<0:
                b[0]=0
            if b[1]<0:
                b[1]=0
            if b[2]>self.size:
                b[2] = self.size
            if b[3]>self.size:
                b[3] = self.size
            if b[0]>b[2]:
                b[0],b[2] = b[2],b[0]
            if b[1]>b[3]:
                b[1],b[3] = b[3],b[1]


        return (torch.from_numpy(img).double(), {
            'boxes': torch.from_numpy(bbox).double().to(device),
            'labels': torch.tensor(torch.from_numpy(np.array(sample[1]["labels"])), dtype = torch.long).to(device)
        })

# Training
class tAnimalDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform = None):
        self.ann = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class2id = {
            "29": 1,
            "38": 2,
        }
    def __len__(self):
        return len(self.ann)
    def __getitem__(self, idx):
        img_path = self.root_dir + '/' + self.ann["id"][idx]

        img = np.array(Image.open(img_path))
        label = self.class2id[str(self.ann.loc[idx, 'labels'])]
        bbox = self.ann.loc[idx, 'xmin':'ymax'].tolist()
        sample = (img, {"boxes": bbox, "labels": label})
        if self.transform:
            sample = self.transform(sample)
        return sample
    def get_id(self, idx):
        return self.ann["id"][idx]

class tNormalize(object):
    def __call__(self, sample):
        img, _ = sample
        img = img.astype(np.float32) / 255
        img -= np.min(img)
        img /= np.max(img)
        sample = (img, sample[1])
        return sample

class tRescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample[0], sample[1]['boxes']

        h, w = image.shape[:2]
        new_h, new_w = int(self.output_size), int(self.output_size)

        img = cv2.resize(image,(new_h, new_w))

        #resizing bbox
        bh, bw = bbox[2]-bbox[0], bbox[3]-bbox[1]
        center = [int(bbox[0]+bw/2), int(bbox[1]+bh/2)]
        bbox = bbox - np.array(center+center)
        bbox = np.multiply(bbox , np.array([new_w / w, new_h / h, new_w / w, new_h / h]))
        center = np.multiply(np.array(center+center), np.array([new_w / w, new_h / h, new_w / w, new_h / h]))
        bbox = bbox + center


        return (img, {'boxes': bbox, 'labels': sample[1]['labels']})

class tRandom_flip(object):
    def __call__(self, sample):
        image, bbox = sample[0], sample[1]['boxes']
        h, w = image.shape[:2]

        #Random horizontal flip
        if (random.randint(0,1)):
            image = cv2.flip(image, 1)
            bbox[0] = w - bbox[0]
            bbox[2] = w - bbox[2]
            bbox[0],bbox[2] = bbox[2],bbox[0]

        return (image, {'boxes': bbox, 'labels': sample[1]['labels']})

class tColorJitter(object):
    def __call__(self, sample):
        img = Image.fromarray(np.uint8(sample[0]*255)).convert('RGB')

        rand_c = 0.9 + random.random()/2
        filter = ImageEnhance.Brightness(img)
        img = filter.enhance(rand_c)

        filter = ImageEnhance.Contrast(img)
        img = filter.enhance(rand_c)

        filter = ImageEnhance.Color(img)
        img = filter.enhance(rand_c)

        img = np.array(img, dtype='uint8')/255

        return (img, sample[1])

class tToTensor(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, bbox = sample[0], np.array(sample[1]['boxes'])

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        # to be safe
        if bbox[0]<0:
            bbox[0]=0
        if bbox[1]<0:
          bbox[1]=0
        if bbox[2]>self.size:
          bbox[2] = self.size
        if bbox[3]>self.size:
          bbox[3] = self.size
        if bbox[0]>bbox[2]:
          bbox[0],bbox[2] = bbox[2],bbox[0]
        if bbox[1]>bbox[3]:
            bbox[1],bbox[3] = bbox[3],bbox[1]

        return (torch.from_numpy(img).double(), {
            'boxes': torch.from_numpy(bbox).double(),#.to(device),
            'labels': torch.tensor(torch.from_numpy(np.array(sample[1]["labels"])), dtype = torch.long)#.to(device)
        })

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device

# Global parameters
num_classes = 3
batch_size = 1
learning_rate = 1e-4
dataset_path = '..\images'
num_of_epochs = 1
tr_split = 0.8 #training split


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


if __name__  == '__main__':

    def collate_fn(batch):
        return tuple(zip(*batch))
    def data_prep(data, type):
        images, targets = data
        if type=='train': #todo for some reason?>
            images = list(img.to(device) for img in images)
        elif type  =='test':
            images = list(img.float().to(device) for img in images)

        targets = [{k: v.float().unsqueeze(0).to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            for k,v in t.items():
                if k=='labels':
                    t[k] = torch.tensor(v.to(device), dtype=torch.long)

        return images, targets
    def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
        """
        Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
        """

        # Slicing idx:idx+1 in order to keep tensor dimensionality
        # Doing ... in indexing if there would be additional dimensions
        # Like for Yolo algorithm which would have (N, S, S, 4) in shape
        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # Need clamp(0) in case they do not intersect, then we want intersection to be 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)
    def nms(bboxes,iou_threshold,threshold,box_format='corners', overlap = 0.5):

        bboxes = [box for box in bboxes if box[2] > threshold]
        bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)
        bboxes_after_nms = []

        if  iou_threshold==-1:
            return bboxes
        while bboxes:
            chosen_box = bboxes.pop(0)

            bboxes = [
                box
                for box in bboxes
                if (box[1] != chosen_box[1]
                   or intersection_over_union(
                    torch.tensor(chosen_box[3:]),
                    torch.tensor(box[3:]),
                    box_format=box_format,
                ) > iou_threshold)
            ]


            bboxes_after_nms.append(chosen_box)

        if overlap==-1:
            return bboxes_after_nms

        bboxes_after_nms = sorted(bboxes_after_nms, key=lambda x: x[2], reverse=True)
        boxes_after_nms_overl = []
        while bboxes_after_nms:
            chosen_box = bboxes_after_nms.pop(0)
            bboxes_after_nms = [
                box
                for box in bboxes_after_nms
                if (box[1] == chosen_box[1]) or
                        intersection_over_union(
                            torch.tensor(chosen_box[3:]),
                            torch.tensor(box[3:]),
                            box_format=box_format,
                        ) > overlap
            ]
            boxes_after_nms_overl.append(chosen_box)

        return boxes_after_nms_overl

    def pred_prep(y,idx,type):

        bbx = []
        bbox = y['boxes']
        labels = y['labels']

        if type=="pred":
            scores = y['scores']
            for i in range(len(scores)):
                bbx.append( [idx, labels[i].cpu().item(), scores[i].cpu().item()] + [b.cpu().item() for b in bbox[i]])
        elif type == "gt":
            try:
                bbox = bbox.tolist()[0]
                labels = labels.tolist()[0]
                for i in range(len(bbox)):
                    bbx.append( [idx, labels[i]] + [b for b in bbox[i]])
            except:
                #bbox = bbox.tolist()
                #labels = labels.tolist()
                bbx.append([idx] + [labels] + [b for b in bbox])
        return bbx
    def view(images, boxes, targ):
        class2id = {
            "deer": 1,
            "boar": 2,
        }

        id2class = dict([(value, key) for key, value in class2id.items()])

        im = np.array(images.cpu()).transpose((1,2,0))


        fig, ax = plt.subplots()
        ax.imshow(im)

        if boxes != None:
            for b in boxes:

                rect = patches.Rectangle((b[3],b[4]),b[5]-b[3],b[6]-b[4],
                                         linewidth=1, edgecolor='orange', facecolor='none')
                ax.add_patch(rect)

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(b[5], b[6], id2class[b[1]]+' '+str(round(b[2],2)), bbox=props)

        if targ != None:
            for t in targ:
                rect = patches.Rectangle((t[2], t[3]), t[4] - t[2], t[5] - t[3],
                                         linewidth=1, edgecolor='green', facecolor='none')
                ax.add_patch(rect)

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(t[2], t[3], id2class[t[1]], bbox=props)

        plt.show()
    def show_losses(vis): #todo implement
        def is_num(x):
            if x in ['1','2','3','4','5','6','7','8','9','0','.']:
                return True
            else:
                return False

        with open('losses.txt', 'r') as f:
            losses = [[],[],[],[]] #todo what are the losses?
            for line in f:
                idx = [i+len('tensor(') for i in range(len(line)) if line.startswith('tensor(', i)]
                for j,id in enumerate(idx):
                    a = line[id]
                    num = ''
                    while is_num(a):
                        num+=str(a)
                        a = line[id+len(num)]
                    losses[j].append(float(num))
        if vis:
            read_loss = np.array(losses)
            iter_num = range(len(read_loss[0, :]))
            # plt.scatter(iter_num, read_loss[0, :])
            # plt.scatter(iter_num, read_loss[1, :])
            # plt.scatter(iter_num, read_loss[2, :])
            rsum = read_loss[0, :]+read_loss[1, :]+read_loss[2, :]
            plt.scatter(iter_num, rsum)
            plt.show()
    def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=3):

        """
        Calculates mean average precision
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            true_boxes (list): Similar as pred_boxes except all the correct ones
            iou_threshold (float): threshold where predicted bboxes is correct
            box_format (str): "midpoint" or "corners" used to specify bboxes
            num_classes (int): number of classes
        Returns:
            float: mAP value across all classes given a specific IoU threshold
        """


        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[2:]),
                        box_format=box_format,
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)

            # print("class: ", c, "TP: ", TP_cumsum)
            # print("class: ", c, "FP: ", FP_cumsum)

            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

            #print("Class ",str(c), "average precision:", torch.trapz(precisions, recalls))
        return sum(average_precisions) / 2, average_precisions

    transform = transforms.Compose([Rescale(512), Normalize(), Random_flip(), ColorJitter(), ToTensor(512)])
    transform_tr = transforms.Compose([tRescale(512), tNormalize(), tRandom_flip(), tColorJitter(), tToTensor(512)])
    print("Load dataset")
    dataset_test = AnimalDataset(csv_file='latvian_wmoose.csv',
                            root_dir='../images',
                            transform=transform
                            )

    dataset_train = tAnimalDataset(csv_file='global_labels.csv',
                                     root_dir='E:/dataset_copy',
                                     transform=transform_tr
                                     )
    # data split

    #dataset_train, dataset_test = torch.utils.data.random_split(dataset, [int(len(dataset) * tr_split), len(dataset)-int(len(dataset) * tr_split)])

    data_loader_train = DataLoader(
        dataset = dataset_train,
        batch_size = batch_size,
        shuffle = True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    data_loader_test = DataLoader(
        dataset = dataset_test,
        batch_size = batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )

    # Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

    #Input feature number for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [p for p in model.parameters() if p.requires_grad]

    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate) #todo choose optimiser

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = None

    itr = 0
    loss_hist = Averager()
    model.train()

    #----------------------------------------------TRAIN-----------------------------------------------------------------
    # start_epoch = -1
    # for epoch in range(start_epoch + 1, num_of_epochs):
    #     loss_hist.reset()
    #
    #     print("Epoch - {} Started".format(epoch))
    #     st = time.time()
    #     epoch_loss = []
    #
    #     for i, data in tqdm(enumerate(data_loader_train)):
    #         if i>2000:
    #
    #             itr += 1
    #             images, targets = data_prep(data, type='train')
    #
    #             for j, im in enumerate(images):
    #                 targ = pred_prep(targets[j], dataset_train.get_id(i), type="gt")
    #                 view(im, None, targ)
    #
    #             model = model.double().to(device)
    #             loss_dict = model(images, targets)
    #
    #             losses = sum(loss for loss in loss_dict.values())
    #             loss_hist.send(losses.item())
    #
    #             optimizer.zero_grad()
    #
    #             losses.backward()
    #
    #             optimizer.step()
    #
    #             with open('losses.txt', 'a') as f:
    #                 f.write(','.join([str(v.float().cpu()) for v in loss_dict.values()]) + '\n')
    #
    #             try:
    #                 print(f"Iteration #{itr} from #{len(data_loader_train)} loss: {losses.item()}")
    #             except:
    #                 pass
    #
    #     # update the learning rate
    #     if lr_scheduler is not None:
    #         lr_scheduler.step()
    #
    #     print(f"Epoch #{epoch} loss: {loss_hist.value}")
    #     # #checkpoint
    #     # #todo copy from google colab
    #
    #     (unique, counts) = np.unique(np.array(test_lbls), return_counts=True)
    #     frequencies = np.asarray((unique, counts)).T
    #     print(frequencies)
    # # #
    # #----------------------------------------------TEST-----------------------------------------------------------------
    all_predictions = []
    all_predictions_nonms = []
    all_targets = []

    test_rcnn = False

    retinas = ["retina"+str(i) for i in range(10)]
    rcnns = ["faster" + str(i) for i in range(10)]

    if test_rcnn==False:
        #retina:
        backbone = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        model = torchvision.models.detection.RetinaNet(backbone.backbone, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # lr_scheduler = None
    #
    # Faster-RCNN
    # model.load_state_dict(torch.load('model2epoch.pth'))
    # model.to(device)

    checkpoint= torch.load('retina0epoch_notbalanced.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    times = [] #save execution times
    nms_times = [] #save post processing times

    with torch.no_grad():
        for i,data in tqdm(enumerate(data_loader_test)):
            images, targets = data_prep(data, type = "test")

            # prediction
            start_time = time.time()
            outputs = model(images, )
            times.append(time.time() - start_time)

            for j,out in enumerate(outputs):

                # prediction transformation
                targ = pred_prep(targets[j], dataset_test.get_id(i), type="gt")
                all_targets += targ

                start_time = time.time()
                pred = pred_prep(out, dataset_test.get_id(i), type="pred")
                all_predictions_nonms += pred

                # visualise output
                #view(images[j], pred, targ)

                pred = nms(pred, -1, 0.2, box_format="corners", overlap=-1)
                nms_times.append(time.time() - start_time)
                all_predictions+=pred


                # visualise output
                #view(images[j], pred, targ)


                maps = []
                maps_cl = [[0], [0], [0]]


    times = np.array(times)
    nms_times = np.array(nms_times)
    print("Average prediction time: ", np.mean(times), "s")
    print("Average post-processing time: ", np.mean(nms_times), "s")

    # maps = []
    # maps_cl = [[0],[0],[0]]
    # print("========================NMS=========================")
    # for iou in tqdm(np.arange(0.5,1,0.05)):
    #     map, map_cl = mean_average_precision(all_predictions, all_targets, iou)
    #     maps.append(map.item())
    #     print("IoU = "+str(iou) + " mAP per class: "+ str(map_cl))
    #     print(" Average IoU ", str(map))
    #     for k,m in enumerate(np.array(map_cl)):
    #         maps_cl[k]+=m
    #
    # maps_cl = np.divide(maps_cl,len(maps))
    # maps = np.array(maps)
    # maps = np.mean(maps)
    # print("----------")
    # print("mAP 0.5:0.05:0.95 = "+str(maps))
    # print("mAP per class 0.5:0.05:0.95 = "+str(maps_cl))

    print("========================NO_NMS=========================")
    maps = []
    maps_cl = [[0], [0], [0]]
    for iou in tqdm(np.arange(0.5, 1, 0.05)):
        map, map_cl = mean_average_precision(all_predictions_nonms, all_targets, iou)
        maps.append(map.item())
        print("IoU = " + str(iou) + " mAP per class: " + str(map_cl))
        print(" Average IoU ", str(map))
        for k, m in enumerate(np.array(map_cl)):
            maps_cl[k] += m

    maps_cl = np.divide(maps_cl, len(maps))
    maps = np.array(maps)
    maps = np.mean(maps)
    print("----------")
    print("mAP 0.5:0.05:0.95 = " + str(maps))
    print("mAP per class 0.5:0.05:0.95 = " + str(maps_cl))


# #other species test:
#     all_predictions = []
#     model.eval()
#     with torch.no_grad():
#         model = model.double().to(device)
#
#         class2id = {
#             "deer": 1,
#             "boar": 2,
#         }
#
#         id2class = dict([(value, key) for key, value in class2id.items()])
#
#         for filename in tqdm(os.listdir(r"E:\animal_recogntion_copy\Jaunkalsnava\custom\New folder\other")):
#             with Image.open(r"E:\animal_recogntion_copy\Jaunkalsnava\custom\New folder\other"+"\\"+filename) as im:
#                 #print(im)
#                 img = np.array(im).transpose((2, 0, 1))
#                 #img = np.array(im, dtype='uint8')/255
#                 img = torch.from_numpy(img).unsqueeze_(0).double().to(device)
#                 outputs = model(img)
#                 for j, out in enumerate(outputs):
#                     # prediction transformation
#                     pred = pred_prep(out, "1", type="pred")
#                     pred = nms(pred, 0.8, 0.8, box_format="corners", overlap=1)
#
#
#
#                     # if len(pred)>0 and pred!=None:
#                     #     # print("in")
#                     #     # print(pred[0])
#                     #     boxes = pred
#                     #     fig, ax = plt.subplots()
#                     #     ax.imshow(im)
#                     #
#                     #     for b in boxes:
#                     #         rect = patches.Rectangle((b[3], b[4]), b[5] - b[3], b[6] - b[4],
#                     #                                  linewidth=1, edgecolor='orange', facecolor='none')
#                     #         ax.add_patch(rect)
#                     #
#                     #         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#                     #         ax.text(b[5], b[6], id2class[b[1]] + ' ' + str(round(b[2], 2)), bbox=props)
#                     #
#                     #     plt.show()
#                         # if len(pred)!=0 or pred!=None:
#                         #     plt.figure()
#                         #     plt.imshow(im)
#                         #     plt.show()
#
#                             #view(img, pred, None)
#
#                     all_predictions+=pred
#
#     print(len(all_predictions))
#     print("----")
