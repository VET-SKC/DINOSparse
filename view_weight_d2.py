def view_weight_from_args():
    # from dinosparse.checkpoint import DetectionCheckpointer
    from dinosparse.config import get_cfg, set_global_cfg
    from dinosparse.engine import DefaultTrainer, default_argument_parser, default_setup

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

    model = DefaultTrainer.build_model(cfg)
    # checkpointer = DetectionCheckpointer(model)
    # checkpointer.load(cfg.MODEL.WEIGHTS)
    # print(model)
    for name, param in model.named_parameters():
        print(f"Layer: {name} - Size: {param.size()}")


def view_weight_from_str():
    from dinosparse.config import get_cfg
    from dinosparse.modeling import build_model
    cfg = get_cfg()
    cfg.merge_from_file("configs/emds7/dinov3_vitb_fpn_sparse_base_anno50.yaml")
    cfg.freeze()
    m = build_model(cfg)
    # for name, p in m.backbone.named_parameters():
    for name, p in m.named_parameters():
        left = f'{name:65s}{tuple(p.shape)}'
        right = f'req_grad={p.requires_grad}'
        print(f'{left:85s}{right}')


# --config-file 的来源不同
# view_weight_from_args()
view_weight_from_str()
