import torch
from collections import OrderedDict


def print_nested_dict(d, prefix=''):
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            print(f"{prefix}Dict: {key}")
            print_nested_dict(value, prefix + '  ')
        elif isinstance(value, torch.Tensor):
            print(f"{prefix}Layer: {key}")
            print(f"{prefix}  Shape: {value.shape}")
            # print(f"{prefix}  Type: {value.dtype}")
        else:
            print(f"{prefix}Other: {key} (Type: {type(value)})")
        # print(f"{prefix}" + "-" * 50)


def print_pth_contents(file_path):
    # 加载 .pth 文件
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))

    print(f"File contains: {type(state_dict)}")

    if isinstance(state_dict, (dict, OrderedDict)):
        print("\nContents of the file:")
        print_nested_dict(state_dict)

        # 计算参数总数
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"\n{len(state_dict)} parameters from state_dict")
        print(f"Total number of parameters: {total_params:,}")
    else:
        print("Unable to process this file type.")


# 注意这里只有pth，pkl需要额外包装
print_pth_contents("data/pretrain_weights/swindct_s_fpn_backbone_patch4_window7_renamed.pth")
