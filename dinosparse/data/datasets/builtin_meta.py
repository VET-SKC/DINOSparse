
# EMDS7 categories
EMDS7_ALL_CATEGORIES = {
    70: ['G001', 'G002', 'G003', 'G004', 'G005', 'G006', 'G007', 'G008', 'G009', 'G010',
         'G011', 'G012', 'G013', 'G014', 'G015', 'G016', 'G017', 'G018', 'G019', 'G020',
         'G021', 'G022', 'G023', 'G024', 'G025', 'G026', 'G027', 'G028', 'G029', 'G030',
         'G031', 'G032', 'G033', 'G034', 'G035', 'G036', 'G037', 'G038', 'G039', 'G040',
         'G041'],
    50: ['G001', 'G002', 'G003', 'G004', 'G005', 'G006', 'G007', 'G008', 'G009', 'G010',
         'G011', 'G012', 'G013', 'G014', 'G015', 'G016', 'G017', 'G018', 'G019', 'G020',
         'G021', 'G022', 'G023', 'G024', 'G025', 'G026', 'G027', 'G028', 'G029', 'G030',
         'G031', 'G032', 'G033', 'G034', 'G035', 'G036', 'G037', 'G038', 'G039', 'G040',
         'G041'],
    30: ['G001', 'G002', 'G003', 'G004', 'G005', 'G006', 'G007', 'G008', 'G009', 'G010',
         'G011', 'G012', 'G013', 'G014', 'G015', 'G016', 'G017', 'G018', 'G019', 'G020',
         'G021', 'G022', 'G023', 'G024', 'G025', 'G026', 'G027', 'G028', 'G029', 'G030',
         'G031', 'G032', 'G033', 'G034', 'G035', 'G036', 'G037', 'G038', 'G039', 'G040',
         'G041']
}

EMDS7_BASE_CATEGORIES = {
    # 20
    70: ['G001', 'G003', 'G004', 'G008', 'G009', 'G011', 'G012', 'G014', 'G015', 'G016',
         'G017', 'G018', 'G019', 'G022', 'G025', 'G028', 'G032', 'G035', 'G036', 'G039'],
    # 25
    50: ['G001', 'G003', 'G004', 'G005', 'G008', 'G009', 'G010', 'G011', 'G012', 'G014',
         'G015', 'G016', 'G017', 'G018', 'G019', 'G022', 'G023', 'G025', 'G028', 'G030',
         'G032', 'G035', 'G036', 'G038', 'G039'],
    # 31
    30: ['G001', 'G002', 'G003', 'G004', 'G005', 'G008', 'G009', 'G010', 'G011', 'G012',
         'G014', 'G015', 'G016', 'G017', 'G018', 'G019', 'G020', 'G022', 'G023', 'G024',
         'G025', 'G028', 'G030', 'G031', 'G032', 'G033', 'G035', 'G036', 'G038', 'G039',
         'G040']
}

EMDS7_NOVEL_CATEGORIES = {
    # 21
    70: ['G002', 'G005', 'G006', 'G007', 'G010', 'G013', 'G020', 'G021', 'G023', 'G024',
         'G026', 'G027', 'G029', 'G030', 'G031', 'G033', 'G034', 'G037', 'G038', 'G040',
         'G041'],
    # 16
    50: ['G002', 'G006', 'G007', 'G013', 'G020', 'G021', 'G024', 'G026', 'G027', 'G029',
         'G031', 'G033', 'G034', 'G037', 'G040', 'G041'],
    # 10
    30: ['G006', 'G007', 'G013', 'G021', 'G026', 'G027', 'G029', 'G034', 'G037', 'G041']
}

# ===== NYU Depth V2 语义分割元数据 =====
# 40 类 superclass 名，顺序严格对应 labels40.mat 的取值 1–40
# 来源：ankurhanda/nyuv2-meta-data 的 classMapping40.mat / className 字段
# labels40 中 0 = void/unlabeled（评测时 ignore，不参与 mIoU）
# 注意：语义分割是"逐像素分类"，本质是 stuff 而非 thing，这里沿用项目现有的 thing_classes 命名以最小化改动，语义上应理解为 stuff_classes
NYU40_CATEGORIES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
    'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
    'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop',
]
assert len(NYU40_CATEGORIES) == 40


def _get_emds7_fewshot_instances_meta():
    return_dict = {
        "EMDS7_ALL_CATEGORIES": EMDS7_ALL_CATEGORIES,
        "EMDS7_BASE_CATEGORIES": EMDS7_BASE_CATEGORIES,
        "EMDS7_NOVEL_CATEGORIES": EMDS7_NOVEL_CATEGORIES,
    }
    return return_dict


def _get_nyu_meta():
    """NYU Depth V2 元数据。深度与分割共用同一份类别名（分割用 40 类；深度不用类别，但为统一接口仍返回）。"""
    return {
        "thing_classes": NYU40_CATEGORIES,   # 40 类（分割用）
        "classes": NYU40_CATEGORIES,
    }


def _get_builtin_metadata(dataset_name):
    if dataset_name == "emds7_fewshot":
        return _get_emds7_fewshot_instances_meta()
    if dataset_name in ("nyu_depth", "nyu_seg"):
        return _get_nyu_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
