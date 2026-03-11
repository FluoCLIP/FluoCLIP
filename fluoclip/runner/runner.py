import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from fluoclip.models import MODELS
from fluoclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef

logger = get_logger(__name__)


class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
        ),
        ckpt_path="",
        stage_num=1,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        model_cfg.update(dict(stage_num=stage_num))
        self.module = MODELS.build(model_cfg)
        self.prompt_learner = self.module.prompt_learner      # <── alias

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.loss_weights = loss_weights
        self.num_ranks = self.module.num_ranks
        self.num_stains = self.prompt_learner.num_stains

        self.register_buffer("stain_output_value_array", torch.arange(0, self.num_stains).float(), persistent=False)
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.score_list = []
        self.pred_score_list = []
        self.stage_num = stage_num

    def switch_to_stage2(self):
        """
        Safely switch Runner + PromptLearner to Stage 2 mode
        WITHOUT reinitializing Stage2-trained parameters.

        This:
            - sets stage_num=2 (Runner + PromptLearner)
            - preserves rank_embeds / rank_context_embeds / stain_embed_learner
            - rebuilds ONLY Stage2 pseudo tokens & sentence templates
            - syncs buffer shapes for correct Stage2 forward
        """
        pl = self.module.prompt_learner

        # ----------------------------------------
        # 1) Switch mode flags
        # ----------------------------------------
        self.stage_num = 2
        pl.stage_num = 2

        # ----------------------------------------
        # 2) Retrieve existing trained Stage2 params
        #    (must NOT reinitialize!)
        # ----------------------------------------
        pl._stage_2_init_f()

        print("[Switch] Now in STAGE 2 mode — rank-related weights PRESERVED, buffers rebuilt.")

    def switch_to_stage1(self):
        """
        Switch Runner + PromptLearner to STAGE 1 (stain-only).
        - Keeps Stage2 weights intact (rank_embeds 등 제거/초기화 안해)
        - Rebuilds only Stage1 buffers (pseudo/sentence/context templates)
        - Forward path, loss path를 Stage1 모드로 변경
        """
        pl = self.prompt_learner

        # 1) 모드 플래그 전환
        self.stage_num = 1
        pl.stage_num = 1

        pl._stage_1_init_f(pl.init_stain_path)

        print("[Switch] Now in STAGE 1 mode (stain-only). Stage2 weights are preserved.")


    # Model Forward
    def forward(self, images, stains):
        return self.module(images, stains)

    def forward_text_only(self, stains):
        return self.module.forward_text_only(stains)
    
    def forward_image(self, img):
        return self.module.encode_image(img)
    
    # Running Steps
    def run_step(self, batch, batch_idx, run_type='train'):
        x, y, stains = batch # images, labels, stains
        logits = self.module(x, stains)

        if self.stage_num == 1:
            # a, b = self.prompt_learner._get_stain_embeds(stains)
            # print(f"stains: {stains}, self.prompt_learner.current_stain_name_to_id_map:{self.prompt_learner.current_stain_name_to_id_map}, get_stan_all_embeds : {a.shape}, {b}" )
            stage1_y = torch.tensor([self.prompt_learner.current_stain_name_to_id_map[s] for s in stains]).to(self.device)
            losses = self.compute_losses(logits, stage1_y, self.num_stains)

            loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

            metrics_exp = self.compute_per_example_metrics(logits, stage1_y, "exp")
            metrics_max = self.compute_per_example_metrics(logits, stage1_y, "max")

        else:
            losses = self.compute_losses(logits, y, self.num_ranks)

            loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

            metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
            metrics_max = self.compute_per_example_metrics(logits, y, "max")

        if run_type == 'test':
            metrics_srcc = self.compute_srcc_per_example_metrics(logits, y, "srcc")
            
        return {"loss": loss, **losses, **metrics_exp, **metrics_max}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, 'test')

        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if run_type == 'test':
            score_list = torch.cat(self.score_list, dim=0).to(dtype=torch.float32)
            pred_score_list = torch.cat(self.pred_score_list, dim=0)
            spearman = SpearmanCorrCoef()
            pearson = PearsonCorrCoef()

            srcc = spearman(pred_score_list.cpu().data, score_list.cpu().data)
            plcc = pearson(pred_score_list.cpu().data, score_list.cpu().data)

            stats["srcc"] = srcc.cpu().item()
            stats["plcc"] = plcc.cpu().item()

            self.log(f"{run_type}_{'srcc'}", stats['srcc'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{run_type}_{'plcc'}", stats['plcc'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")
        predicted_labels = []
        for batch_outputs in outputs:
            predict_y = batch_outputs.get('predict_y', None)
            if predicted_labels is not None:
                predicted_labels.extend(predict_y.cpu().tolist())
        with open(str(self.output_dir / "result_stats.json"), "a") as f:
            f.write(json.dumps(predicted_labels) + "\n")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    # Loss & Metrics
    def compute_losses(self, logits, y, num_labels):
        losses = {}
        losses["ce_loss"] = self.ce_loss_func(logits, y)
        losses["kl_loss"] = self.compute_kl_loss(logits, y, num_labels)

        return losses

    def compute_kl_loss(self, logits, y, num_labels):
        y_t = F.one_hot(y, num_labels).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
        return kl_loss

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        dtype = logits.dtype
        probs = F.softmax(logits, -1)

        if gather_type == "exp":
            if self.stage_num == 1:
                stain_output_value_array = self.stain_output_value_array.type(dtype)
                predict_y = torch.sum(probs * stain_output_value_array, dim=-1)
            else : # 2nd stage
                rank_output_value_array = self.rank_output_value_array.type(dtype)
                predict_y = torch.sum(probs * rank_output_value_array, dim=-1)

        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(logits.dtype)

        return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, "predict_y": predict_y}

    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_text_encoder_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_prompt_learner_stains,
        lr_prompt_learner_rank_context,
        lr_image_encoder,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
        lr_module_adapter,
        lr_prompt_learner_stainembed=None,
    ):
        param_dict_ls = []
        if lr_prompt_learner_context > 0 and self.module.prompt_learner.context_embeds is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.context_embeds,
                    "lr": lr_prompt_learner_context,
                    "init_lr": lr_prompt_learner_context,
                    "name": "lr_prompt_learner_context",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.context_embeds)")
            try:
                freeze_param(self.module.prompt_learner.context_embeds)
            except AttributeError:
                pass

        if lr_prompt_learner_ranks > 0 and self.module.prompt_learner.rank_embeds is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.rank_embeds,
                    "lr": lr_prompt_learner_ranks,
                    "init_lr": lr_prompt_learner_ranks,
                    "name": "lr_prompt_learner_ranks",
                }
            )
        else:
            if self.module.prompt_learner.rank_embeds is not None:
                self._custom_logger.info("freeze_param(self.model.prompt_learner.rank_embeds)")
                try:
                    freeze_param(self.module.prompt_learner.rank_embeds)
                except AttributeError:
                    pass

        if lr_prompt_learner_stains > 0 and self.module.prompt_learner.stain_params is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.stain_params.parameters(),
                    "lr": lr_prompt_learner_stains,
                    "init_lr": lr_prompt_learner_stains,
                    "name": "lr_prompt_learner_stains",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.stain_params)")
            try:
                freeze_param(self.module.prompt_learner.stain_params.parameters())
            except AttributeError:
                pass

        if lr_prompt_learner_rank_context > 0 and self.module.prompt_learner.rank_context_embeds is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.rank_context_embeds,
                    "lr": lr_prompt_learner_rank_context,
                    "init_lr": lr_prompt_learner_rank_context,
                    "name": "lr_prompt_learner_rank_context",
                }
            )
        else:
            if self.module.prompt_learner.rank_context_embeds is not None:

                self._custom_logger.info("freeze_param(self.model.prompt_learner.rank_context_embeds)")
                try:
                    freeze_param(self.module.prompt_learner.rank_context_embeds)
                except AttributeError:
                    pass
        
        if lr_prompt_learner_stainembed is not None and lr_prompt_learner_stainembed > 0 and self.module.prompt_learner.stain_embed_learner is not None:
            print("lr_prompt_learner_stainembed set to {lr_prompt_learner_stainembed}")
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.stain_embed_learner.parameters(),
                    "lr": lr_prompt_learner_stainembed,
                    "init_lr": lr_prompt_learner_stainembed,
                    "name": "lr_prompt_learner_stainembed",
                }
            )
        else:
            if self.module.prompt_learner.stain_embed_learner is not None:
                self._custom_logger.info("freeze_param(self.model.prompt_learner.stain_embed_learner)")
                try:
                    freeze_param(self.module.prompt_learner.stain_embed_learner)
                except AttributeError:
                    pass
        

        if lr_module_adapter > 0 and self.module.adapter is not None:
            param_dict_ls.append(
                {
                    "params": self.module.adapter.parameters(),
                    "lr": lr_module_adapter,
                    "init_lr": lr_module_adapter,
                    "name": "text_encoder",
                }
            )
        else:
            if self.module.adapter is not None:
                self._custom_logger.info("freeze_param(self.model.adapter)")
                freeze_param(self.module.adapter)


        if lr_image_encoder > 0 and self.module.image_encoder is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=self.module.image_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.image_encoder.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "image_encoder",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.image_encoder)")
            freeze_param(self.module.image_encoder)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.logit_scale)")
            freeze_param(self.module.logit_scale)
        return param_dict_ls
    
    def compute_srcc_per_example_metrics(self, logits, y, gather_type="srcc"):
        dtype = logits.dtype

        predict_y = self.convert_logits_to_predicions(logits, gather_type)

        y = y.type(dtype)

        self.score_list.append(y)
        self.pred_score_list.append(predict_y)


    def convert_logits_to_predicions(self, logits, gather_type="max"):
        dtype = logits.dtype
        probs = F.softmax(logits, -1)

        if gather_type == "exp":
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        elif gather_type == "top2":
            score, predict_y = torch.topk(probs, k=2)
        elif gather_type == "top3":
            score, predict_y = torch.topk(probs, k=3)
        elif gather_type == "top5":
            score, predict_y = torch.topk(probs, k=5)
        elif gather_type == "srcc":
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")
        return predict_y
    
    # def load_state_dict(self, state_dict, strict: bool = False):  # default False
    #     # (optional) strip 'module.' prefixes
    #     state_dict = { k: v for k,v in state_dict.items() }
    #     missing, unexpected = super().load_state_dict(state_dict, strict=strict)
    #     # (optional) log them
    #     if missing:
    #         print("[Runner] Ignoring missing keys:", missing)
    #     if unexpected:
    #         print("[Runner] Unexpected keys:", unexpected)
    #     return missing, unexpected