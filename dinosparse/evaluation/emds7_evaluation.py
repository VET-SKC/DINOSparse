# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.logger import create_small_table

from .evaluator import DatasetEvaluator


class EMDS7Evaluator(DatasetEvaluator):
    """
    Evaluate EMDS7 AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name, output_dir):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "emds7_trainval_base_anno50"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "EMDS7Xml", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "EMDS7Sets", meta.split + ".txt")
        self._class_names = meta.thing_classes
        # add this two terms for calculating the mAP of different subset
        self._base_classes = meta.base_classes
        self._novel_classes = meta.novel_classes

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # 混淆矩阵
        # 每一行代表每类gt，每一列代表每类预测，各加一表示背景
        self._confusion_matrix_conf = 0.5  # 改变conf
        self._confusion_matrix_AP = 0.5  # 改变AP
        self._confusion_matrix_add_bg = ConfusionMatrix(num_classes=len(self._class_names),
                                                        conf_threshold=self._confusion_matrix_conf,
                                                        iou_threshold=self._confusion_matrix_AP)

        # self._confusion_matrix_add_bg = np.zeros((len(self._class_names) + 1, len(self._class_names) + 1))
        # 每一行代表每类gt，每一列代表每类预测，各加一表示合计数量（MIaMIA）
        # self._confusion_matrix_add_sum = np.zeros((len(self._class_names) + 1, len(self._class_names) + 1))
        # 混淆矩阵png保存路径
        # self._confusion_matrix_save_path = sys.argv[sys.argv.index("OUTPUT_DIR")+1]  # 这样写对pycharm运行配置格式有要求
        self._confusion_matrix_save_path = output_dir
        os.makedirs(self._confusion_matrix_save_path, exist_ok=True)

    def reset(self):
        self._predictions = defaultdict(
            list
        )  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75". # 这是什么key？
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using 2010+ metric. "
            "Note that results do not use the official Matlab API.".format(self._dataset_name)
        )

        # 混淆矩阵
        self._logger.info("Confusion matrix is generating.")

        # 获取图像信息
        with open(self._image_set_path, "r") as f:
            lines = f.readlines()
        image_names = [x.strip().split(".png")[0] for x in lines]  # 从EMDS7Sets读出来带扩展名的字符串，获取主名

        # 制作所需gt框格式
        gt_boxex = {}
        for image_name in image_names:
            gt_boxex[image_name] = parse_rec(self._anno_file_template.format(image_name))  # 获取EMDS7Xml进行解析
        # 统计emds7测试集中456张图片各类gt数量。在所有类微调时的混淆矩阵中，每行gt类别在数量上相符。
        # （对于同一个混淆矩阵，每列pred类别数量也相符，可以通过变量predictions获取。）
        # [sum(1 for gt_list in gt_boxex.values() for gt_dict in gt_list if gt_dict['name'] == class_name)
        #  for class_name in self._class_names]
        # result = [23, 7, 162, 20, 10, 4, 2, 18, 30, 15,  # 1~10
        #           12, 28, 2, 21, 19, 37, 19, 24, 42, 8,  # 11~20
        #           1, 251, 11, 11, 32, 3, 3, 44, 3, 9,  # 21~30
        #           7, 28, 7, 1, 11, 16, 2, 13, 20, 6,  # 31~40
        #           4]  # 41
        new_gt_boxes = {}
        for image_name, gt_list in gt_boxex.items():
            new_gt_list = []
            for i, gt_dict in enumerate(gt_list):
                if gt_dict['name'] != 'unknown' and gt_dict['name'] in self._class_names:
                    bbox = gt_dict['bbox']
                    class_id = self._class_names.index(gt_dict['name'])
                    x1, y1, x2, y2 = bbox
                    new_gt_list.append([class_id, x1, y1, x2, y2])
            new_gt_boxes[image_name] = np.array(new_gt_list)

        # 制作所需预测框格式
        new_predictions_ = {}
        for class_id, prediction_list in predictions.items():
            for i, prediction_str in enumerate(prediction_list):
                prediction_parts = prediction_str.split()
                image_name = prediction_parts[0]
                conf = float(prediction_parts[1])
                x1, y1, x2, y2 = map(float, prediction_parts[2:])
                new_predictions_.setdefault(image_name, []).append([x1, y1, x2, y2, conf, class_id])
        new_predictions = {}
        for key, value in new_predictions_.items():
            new_predictions[key] = np.array(value)

        # 遍历图像，对每一张调用更新
        for image_name in image_names:
            # gt竟然一不注意也能传空值？？？，EMDS7-G011-062-0400里面全是unknown，有10个unknown？？？
            gt = new_gt_boxes[image_name]  # 获取当前图片的gt数据
            if gt.ndim == 1:
                gt = gt.reshape((-1, 5))
            # prediction貌似只有453张，这三个没有预测结果吗？？？（待测试）
            # {'EMDS7-G022-190-0400', 'EMDS7-G022-200-0400', 'EMDS7-G022-214-0400'}
            det = new_predictions.get(image_name, np.empty((0, 6)))  # 获取当前图片的det数据，注意key可能不存在的情况
            self._confusion_matrix_add_bg.process_batch(gt, det)  # 调用处理函数处理当前图片的gt和det数据

        # 输出混淆矩阵 self._confusion_matrix_add_bg
        confusion_matrix_classes = self._class_names
        confusion_matrix_classes.append("bg")
        plot_confusion_matrix(self._confusion_matrix_add_bg.matrix, confusion_matrix_classes,
                              save_dir=self._confusion_matrix_save_path,
                              show=False,
                              title='conf{}_AP{}_bg_percent'
                                    .format(int(self._confusion_matrix_conf*100), int(self._confusion_matrix_AP*100)),
                              mode='percent')
        plot_confusion_matrix(self._confusion_matrix_add_bg.matrix, confusion_matrix_classes,
                              save_dir=self._confusion_matrix_save_path,
                              show=False,
                              title='conf{}_AP{}_bg_count'
                                    .format(int(self._confusion_matrix_conf*100), int(self._confusion_matrix_AP*100)),
                              mode='count')
        confusion_matrix_classes.remove("bg")

        self._logger.info("Confusion matrix is ok.")

        # AP计算
        with tempfile.TemporaryDirectory(prefix="emds7_eval_") as dirname:  # 原本是prefix="pascal_voc_eval_"，不知何用
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            exist_base, exist_novel = False, False

            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=False,
                    )
                    aps[thresh].append(ap * 100)

                    if (
                        self._base_classes is not None
                        and cls_name in self._base_classes
                    ):
                        aps_base[thresh].append(ap * 100)
                        exist_base = True

                    if (
                        self._novel_classes is not None
                        and cls_name in self._novel_classes
                    ):
                        aps_novel[thresh].append(ap * 100)
                        exist_novel = True

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {
            "AP": np.mean(list(mAP.values())),
            "AP50": mAP[50],
            "AP75": mAP[75],
        }

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            ret["bbox"].update(
                {
                    "bAP": np.mean(list(mAP_base.values())),
                    "bAP50": mAP_base[50],
                    "bAP75": mAP_base[75],
                }
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            ret["bbox"].update(
                {
                    "nAP": np.mean(list(mAP_novel.values())),
                    "nAP50": mAP_novel[50],
                    "nAP75": mAP_novel[75],
                }
            )

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}
        self._logger.info("Evaluate per-class AP50:\n" + create_small_table(per_class_res))
        self._logger.info("Evaluate overall bbox:\n" + create_small_table(ret["bbox"]))

        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
##############################################################################


@lru_cache(maxsize=None)
def parse_rec(filename):  # 读取标注的xml文件
    """Parse a EMDS7 xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    # AP的计算，涉及十一点和逐点两种方法：
    # 若use_07_metric=true，则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP；
    # 若use_07_metric=false，则采用更为精确的逐点积分方法。
    if use_07_metric:
        # 2010年以前按recall等间隔取11个不同点处的精度值做平均(0., 0.1, 0.2, …, 0.9, 1.0)
        # 11 点法，用0.1为间隔，将R在 [0,1] 的区间内分成11个点，然后从0到1遍历这11个点，分别统计当R值大于当前值时，对应的最大P值
        # 然后对所有的P值加权求和，即为想要的AP值
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # 取最大值等价于2010以后先计算包络线的操作，保证precise非减
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # 逐点积分法
        # 2010年以后取所有不同的recall对应的点处的精度值做平均
        # 首先需要对在R数组前后插入0和1，在P数组前后插入0
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # 对P数组从后往前，两两比对，然后取最大值
        # 计算包络线，从后往前取最大保证precise非减
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 检测R数组中元素值的变化，并记录
        # 找出所有检测结果中recall不同的点
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # 根据R数组变化位置，计算间隔差，再乘以P值，加和后计算AP
        # 用recall的间隔对精度作加权平均
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    ovthresh=0.5,
    use_07_metric=False,
):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # 主要函数，计算当前类别的recall和precision

    # 计算每个类别对应的AP，mAP是所有类别AP的平均值

    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip().split(".png")[0] for x in lines]  # 从EMDS7Sets读出来带扩展名的字符串，获取主名

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))  # 获取EMDS7Xml进行解析

    # extract gt objects for this class
    # 按类别获取标注文件，recall和precision都是针对不同类别而言的，AP也是对各个类别分别算的。
    class_recs = {}  # 在recs基础上，筛选并构造“当前被调用传入类别”的gt标注dict，class_recs
    npos = 0  # npos标记的目标数量
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]  # 对每张图片过滤，只保留recs中“指定类别”的项，存为R
        bbox = np.array([x["bbox"] for x in R])  # 抽取bbox
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        # treat all "difficult" as GT
        # difficult = np.array([False for x in R]).astype(bool)  # 如果数据集没有difficult结点，则初始化所有项都是0
        det = [False] * len(R)  # len(R)就是当前类别的gt目标个数，det表示是否检测到，初始化为false
        npos = npos + sum(~difficult)  # 自增，非difficult样本数量，如果数据集没有difficult，npos数量就是gt数量，等效npos+=len(R)
        class_recs[imagename] = {
            "bbox": bbox,
            "difficult": difficult,
            "det": det,
        }

    # 获取检测结果
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]  # 检测结果（读取该类临时文件所得）
    image_ids = [x[0] for x in splitlines]  # 图片文件主名
    confidence = np.array([float(x[1]) for x in splitlines])  # 置信度
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)  # 变为float的bbox

    # 将检测结果按置信度排序
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # 遍历检测目标，根据与真值的IOU计算响应的tp和fp
    # go down dets and mark TPs and FPs
    nd = len(image_ids)  # 个人认为就是splitlines的长度
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):  # 遍历所有检测结果，因为已经排序，所以这里是从置信度最高到最低遍历
        R = class_recs[image_ids[d]]  # 当前检测结果所在图像的所有同类别gt
        bb = BB[d, :].astype(float)  # 当前检测结果bbox坐标
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)  # 当前检测结果所在图像的所有同类别gt的bbox坐标

        if BBGT.size > 0:
            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0)
                * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 最大重合率
            jmax = np.argmax(overlaps)  # 最大重合率对应的gt

        if ovmax > ovthresh:  # 如果当前检测结果与真实标注最大重合率满足阈值
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1  # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # 计算相应的PR值
    # compute precision recall
    fp = np.cumsum(fp)  # 积分图，fp长度与splitlines的长度相同
    tp = np.cumsum(tp)  # 积分图，tp长度与splitlines的长度相同
    rec = tp / float(npos)  # 召回率，与splitlines的长度相同
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 准确率，与splitlines的长度相同
    # 计算AP值
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def my_confusion_matrix():
    # 遍历pred

        # 对于每一个pred，遍历相同图片文件中的gt

            # if iou>0.5

                # if 类别一致，计入该类tp位，并删除此gt

                # else 类别不一致，计入相应的fp位，并删除此gt（不再等待后续tp的可能）排序？【疑问1】

            # else 计入相应的fp位（附加行）（gt不可知）

    # 对于依然剩余没被删除的gt，计入相应的fn位（附加列），并删除此gt

    # 检查gt应为0，即每一个pred和gt都被考虑到了

    # 【疑问1】高度重合情况，对于一个预测框，如果在判断后纳入了fp，而遍历下一个gt时本应该是tp但因为已经删除，并不能重新计入tp，这种情况怎么办？

    # 基本与github上找的那个等价，但疑问1依然存在。

    return


# https://github.com/kaanakan/object_detection_confusion_matrix
def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes: int, conf_threshold=0.3, iou_threshold=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = conf_threshold
        self.IOU_THRESHOLD = iou_threshold

    def process_batch(self, labels: np.ndarray, detections):  # 一次一张图片
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            labels (Array[N, 5]), class, x1, y1, x2, y2
            detections (Array[M, 6]), x1, y1, x2, y2, conf, class
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[gt_class, self.num_classes] += 1  # 计入附加列，fn
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)  # tuple 2 分别表示两个维度的索引

        # 两个索引和iou值
        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]  # 按iou降序
            # np.unique..[1]是新列表元素在旧列表中的位置索引。重复的取第一次，即iou高的
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]  # 去重排序，使预测框唯一

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]  # 按iou降序

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]  # 去重排序，使gt框唯一

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[gt_class, detection_class] += 1  # 计入已知类的格子
            else:
                self.matrix[gt_class, self.num_classes] += 1  # 计入附加列，fn

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or (all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
                detection_class = detection_classes[i]
                self.matrix[self.num_classes, detection_class] += 1  # 计入附加行，fp

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          mode='percent',
                          color_theme='plasma'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        mode (str): 自定义是显示百分比还是具体数量。
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]  # 对每行（gt类）所有列的数量进行求和，并构建成列向量形式
    if mode == 'percent':
        confusion_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100  # 得到每个gt类别内归一化的百分比
        text_content = confusion_matrix
        text_format = '{}%'
    else:
        text_content = confusion_matrix
        text_format = '{}'
        confusion_matrix = confusion_matrix.astype(np.float32) / per_label_sums * 100  # 得到每个gt类别内归一化的百分比

    # 创建一个matplotlib的Figure对象和一个Axes对象，用于绘制图像。
    # figsize参数指定图像的大小，根据类别数量动态计算。
    # dpi参数指定图像的分辨率。
    num_classes = len(labels)
    fig, ax = plt.subplots(figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)  # 这里fig其实用处不大
    # 设置混淆矩阵图像的颜色主题，并绘制归一化后的混淆矩阵。
    # cmap变量获取指定颜色主题的颜色映射对象，imshow函数用于绘制矩阵图像，colorbar函数添加颜色条。
    # shrink参数控制colorbar的宽度，它的值是一个0到1之间的浮点数。默认值为1.0。
    # 较小的值会使colorbar变窄，较大的值会使colorbar变宽。
    # aspect参数控制colorbar的高度与宽度的比例。默认值为20。
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax, shrink=1.0, aspect=40)

    # 设置图像的标题、坐标轴标签的字体样式和大小。
    title_font = {'weight': 'bold', 'size': 16}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 14}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    # 设置刻度线的位置，MultipleLocator类用于设置刻度线的间隔。
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    # 绘制网格线，which参数指定绘制哪种类型的网格线，这里是绘制次要刻度线的网格线。
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    # 设置刻度线的位置和标签
    # np.arange(num_classes)生成一个从0到num_classes-1的数组
    # set_xticks和set_yticks函数设置刻度线的位置
    # set_xticklabels和set_yticklabels函数设置刻度线的标签
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # 设置刻度线的样式，tick_params函数设置x轴刻度线的位置和标签的显示方式，setp函数设置x轴刻度线标签的旋转角度和对齐方式。
    ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    # 在混淆矩阵的每个格子中绘制归一化后的百分比值，text函数用于在指定位置绘制文本。
    for i in range(num_classes):
        for j in range(num_classes):
            text = text_format.format(int(text_content[i, j]) if not np.isnan(text_content[i, j]) else -1)
            text_color = 'black' if confusion_matrix[i, j] > 50 else 'w'
            ax.text(j, i, text, ha='center', va='center', color=text_color, size=8)

    # 设置y轴的显示范围，使得混淆矩阵的每个格子都完整显示。
    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    # 调整图像的布局，使得各个元素之间的间距合适。
    # fig.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{title}.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    # plt.show()和plt.savefig()有显示不一致的问题，字号以plt.savefig()为准

    # 手动释放资源
    plt.close('all')
