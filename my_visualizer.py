import cv2
import torch
import numpy as np

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from dinosparse.config import get_cfg, set_global_cfg
from dinosparse.engine import DefaultPredictor, default_argument_parser, default_setup


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
predictor = DefaultPredictor(cfg)

# 读取输入图像
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G014-074-0400.png"
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G014-046-0400.png"

# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G001-025-0400.png"  # G001 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G002-002-0400.png"  # G002 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G003-006-0400.png"  # G003 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G004-002-0400.png"  # G004 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G005-053-0400.png"  # G005 done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G006-001-0400.png"  # G006 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G007-006-0400.png"  # G007 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G008-026-0400.png"  # G008 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G009-001-0400.png"  # G009 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G010-018-0400.png"  # G010 done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G011-010-0400.png"  # G011 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G012-043-0400.png"  # G012 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G013-006-0400.png"  # G013 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G014-003-0400.png"  # G014 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G015-001-0400.png"  # G015 done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G016-007-0400.png"  # G016 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G017-037-0400.png"  # G017 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G018-057-0400.png"  # G018 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G019-043-0400.png"  # G019 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G020-002-0400.png"  # G020 ok
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G021-005-0400.png"  # G021 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G022-006-0400.png"  # G022 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G023-002-0400.png"  # G023 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G024-005-0400.png"  # G024 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G025-002-0400.png"  # G025 done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G026-001-0400.png"  # G026 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G027-003-0400.png"  # G027 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G028-046-0400.png"  # G028 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G029-006-0400.png"  # G029 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G030-019-0400.png"  # G030 done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G031-018-0400.png"  # G031 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G032-045-0400.png"  # G032 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G033-018-0400.png"  # G033 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G034-007-0400.png"  # G034 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G035-021-0400.png"  # G035 done
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G036-044-0400.png"  # G036 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G037-011-0400.png"  # G037 ok
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G038-065-0400.png"  # G038 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G039-005-0400.png"  # G039 done
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G040-006-0400.png"  # G040 ok
#
# image_path = "datasets/EMDS7/EMDS7Images/trainval/EMDS7-G041-003-0400.png"  # G041 ok

# image_path = "datasets/SVIA/Frames_from_original_videos/S_0001/S_0001_0001.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0095/S_0095_0003.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0096/S_0096_0001.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0097/S_0097_0001.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0098/S_0098_0001.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0099/S_0099_0001.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0100/S_0100_0001.png"
# image_path = "datasets/SVIA/Frames_from_original_videos/S_0101/S_0101_0001.png"

# image_path = "datasets/PBCDS/PBCDSImages/test/BA_801290.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/BNE_804375.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/EO_799659.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/ERB_805341.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/IG_864662.jpg"
#
# image_path = "datasets/PBCDS/PBCDSImages/test/LY_807924.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/MMY_791618.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/MO_790080.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/MY_815522.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/NEUTROPHIL_816359.jpg"
#
# image_path = "datasets/PBCDS/PBCDSImages/test/PLATELET_788884.jpg"
# image_path = "datasets/PBCDS/PBCDSImages/test/PMY_811405.jpg"
image_path = "datasets/PBCDS/PBCDSImages/test/SNE_816195.jpg"

image = cv2.imread(image_path)

# 进行预测
outputs = predictor(image)

# 获取预测结果
instances = outputs["instances"].to("cpu")
boxes = instances.pred_boxes if instances.has("pred_boxes") else None
scores = instances.scores if instances.has("scores") else None
classes = instances.pred_classes if instances.has("pred_classes") else None

# 可视化检测结果
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(instances)

# 显示结果图像
result_image = out.get_image()[:, :, ::-1]
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite("my_visualize_output.png", result_image)
