import os

from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer

from dinosparse.config import get_cfg, set_global_cfg
from dinosparse.evaluation import DatasetEvaluators, verify_results
from dinosparse.engine import DefaultTrainer, default_argument_parser, default_setup
from dinosparse.utils.gradclip_monitor import install_gradnorm_logger


class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        # 梯度范数监控：在 optimizer.step 内部、裁剪发生前测量全局梯度 L2 范数。
        # 每 every_n 步打印一次（iter / lr / 裁剪前范数 / clip_coef / 触发率）。
        self._gradnorm_stats = install_gradnorm_logger(self, every_n=20)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "emds7":
            from dinosparse.evaluation import EMDS7Evaluator
            return EMDS7Evaluator(dataset_name, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
