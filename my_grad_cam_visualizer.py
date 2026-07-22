import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances, Boxes

from dinosparse.config import get_cfg, set_global_cfg
from dinosparse.engine import GradCamPredictor, default_argument_parser, default_setup


def detach_instances(instances):
    """
    Detach all tensors in the Instances object from the computation graph.
    """
    for field in instances.get_fields():
        tensor = instances.get(field)
        if isinstance(tensor, torch.Tensor):
            instances.set(field, tensor.detach())
        elif isinstance(tensor, Boxes):
            tensor.tensor = tensor.tensor.detach()
            instances.set(field, tensor)
    return instances


# 创建配置对象
def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


args = default_argument_parser().parse_args()
cfg = setup(args)

# 设置所需层（代码结构问题，目前只能去defaults.py里改了）
predictor = GradCamPredictor(cfg, "backbone.bottom_up.res5.2.conv3")
# "backbone.bottom_up.res5.2.conv3"  # classical
# "backbone.bottom_up.res4.22.conv3"
# "backbone.bottom_up.res3.3.conv3"
# "backbone.bottom_up.res2.2.conv3"

# 读取输入图像
image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G014-074-0400.png"

# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G001-025-0400.png"  # G001 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G002-002-0400.png"  # G002 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G003-006-0400.png"  # G003 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G004-002-0400.png"  # G004 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G005-053-0400.png"  # G005 done done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G006-001-0400.png"  # G006 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G007-006-0400.png"  # G007 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G008-026-0400.png"  # G008 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G009-001-0400.png"  # G009 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G010-018-0400.png"  # G010 done done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G011-010-0400.png"  # G011 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G012-043-0400.png"  # G012 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G013-006-0400.png"  # G013 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G014-003-0400.png"  # G014 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G015-001-0400.png"  # G015 done done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G016-007-0400.png"  # G016 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G017-037-0400.png"  # G017 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G018-057-0400.png"  # G018 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G019-043-0400.png"  # G019 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G020-002-0400.png"  # G020 ok   ok
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G021-005-0400.png"  # G021 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G022-006-0400.png"  # G022 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G023-002-0400.png"  # G023 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G024-005-0400.png"  # G024 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G025-002-0400.png"  # G025 done done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G026-001-0400.png"  # G026 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G027-003-0400.png"  # G027 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G028-046-0400.png"  # G028 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G029-006-0400.png"  # G029 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G030-019-0400.png"  # G030 done done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G031-018-0400.png"  # G031 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G032-045-0400.png"  # G032 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G033-018-0400.png"  # G033 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G034-007-0400.png"  # G034 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G035-021-0400.png"  # G035 done done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G036-044-0400.png"  # G036 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G037-011-0400.png"  # G037 ok   ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G038-065-0400.png"  # G038 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G039-005-0400.png"  # G039 done done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G040-006-0400.png"  # G040 ok   ok
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G041-003-0400.png"  # G041 ok   ok

image = cv2.imread(image_path)  # BGR

# 确定所需类别，生成Grad-CAM热力图
# emds7_base_anno50_name = ['G001', 'G003', 'G004', 'G005', 'G008', 'G009', 'G010', 'G011', 'G012', 'G014',
#                           'G015', 'G016', 'G017', 'G018', 'G019', 'G022', 'G023', 'G025', 'G028', 'G030',
#                           'G032', 'G035', 'G036', 'G038', 'G039']
# emds7_base_anno50_name.index("G014")
# CAM is generated per object instance, not per class!
# target_instance = int(args.opts[args.opts.index("TEST.DETECTIONS_PER_IMAGE")+1]) - 1
target_instance = 0  # 这个意思大致是说，预测实例的scores从高到低排序的索引？

image_dict, cam_orig = predictor.generate_grad_cam(image, target_instance=target_instance)
# superimposed_img = predictor.visualize_grad_cam(image, cam_visualization)
v = Visualizer(image_dict["image"], MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]), scale=1.0)
out = v.draw_instance_predictions(detach_instances(image_dict["output"]["instances"][target_instance].to("cpu")))

result_image = out.get_image()

plt.imshow(result_image, interpolation='none')
plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)
plt.title(f"CAM for Instance {target_instance} (class {image_dict['label']})")
plt.savefig("my_grad_cam_visualizer_output(plt).png", dpi=100)
plt.show()

cam = cv2.applyColorMap(np.uint8(255 * image_dict["cam"]), cv2.COLORMAP_JET)
result_image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("my_grad_cam_visualizer_output(cv2).png", result_image)
