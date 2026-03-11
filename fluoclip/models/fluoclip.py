import os.path as osp

import torch
import torch.nn as nn
import torchvision.models as models
from clip import clip

from fluoclip.utils import get_logger

from . import image_encoders
from .builder import MODELS
from .prompt_learners import PROMPT_LEARNERS
from .adapters import ADAPTERS

logger = get_logger(__name__)


@MODELS.register_module()
class FluoCLIP(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        adapter_cfg,
        stage_num,
        stage2_use=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        # convert to float32
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        prompt_learner_cfg.update(dict(stage_num=stage_num))
        self.prompt_learner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.logit_scale = clip_model.logit_scale
        
        self.out_dims = clip_model.text_projection.shape[1]
        self.embed_dims = clip_model.token_embedding.embedding_dim
        self.num_ranks = self.prompt_learner.num_ranks
        adapter_cfg.update(dict(input_dim=self.out_dims))
        self.adapter = ADAPTERS.build(adapter_cfg)
        self.stage2_use = stage2_use

    def forward(self, images, stains):
        B = images.size(0)
        image_features = self.image_encoder(images)       # (B, D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        all_sentence_embeds, all_pseudo_tokens, stain_ids_tensor = self.prompt_learner(stains)
        assert len(stain_ids_tensor) == B, f"length of stain_ids_tensor {len(stain_ids_tensor)} == batch length {B}"
        
        T = self.prompt_learner.clip_max_num_tokens
        if all_sentence_embeds.ndim == 3: # stage 1
            S = all_sentence_embeds.size(0)
            all_text_feats = self.text_encoder(all_sentence_embeds, all_pseudo_tokens) # (S, out_dims)
            text_features = all_text_feats / all_text_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            text_features = self.adapter(text_features)
            logits = self.logit_scale.exp() * (image_features @ text_features.T) # (B, S)

        elif all_sentence_embeds.ndim == 4:
            S, R = all_sentence_embeds.size(0), all_sentence_embeds.size(1)
            all_text_feats = self.text_encoder(
                all_sentence_embeds.view(-1, T, self.embed_dims),         # → (S*R, T, D)
                all_pseudo_tokens.reshape(-1, T)               # → (S*R, T)
            ).view(S, self.num_ranks, self.out_dims)
            all_text_feats = all_text_feats / all_text_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6) # --> S, R, D
            if self.stage2_use:
                all_text_feats = self.adapter(all_text_feats.view(-1, self.out_dims)).view(S, R, -1)
            text_features = all_text_feats[stain_ids_tensor] # --> B, R, D
        
            # 6) compute per-sample, per-rank logits
            logits = self.logit_scale.exp() * (
                image_features.unsqueeze(1)          # (B, 1, D)
                @ text_features.transpose(1, 2)      # (B, D, R)
            ).squeeze(1)                          # (B, R)
        
        return logits

    def forward_text_only(self, stains):
        # stains = ["Alexa-488", "Hoechst-34860", "ALEXA-647", "Cy3"]
        all_sentence_embeds, all_pseudo_tokens, stain_ids_tensor= self.prompt_learner(stains)
        S, R = all_sentence_embeds.size(0), all_sentence_embeds.size(1)
        T = self.prompt_learner.clip_max_num_tokens
        all_text_feats = self.text_encoder(
                all_sentence_embeds.view(-1, T, self.embed_dims),         # → (S*R, T, D)
                all_pseudo_tokens.reshape(-1, T)               # → (S*R, T)
            ).view(S, self.num_ranks, self.out_dims)
        all_text_feats = all_text_feats / all_text_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6) # --> S, R, D
        if self.stage2_use:
            all_text_feats = self.adapter(all_text_feats.view(-1, self.out_dims)).view(S, R, -1)
        text_features = all_text_feats[stain_ids_tensor]
        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # remove the stochastic back-prop in vgg and alexnet
        MODEL = getattr(image_encoders, image_backbone_name, None)
        if MODEL is None:
            MODEL = getattr(models, image_backbone_name, None)
            logger.warning(f"Try PyTorch Official image model: {image_backbone_name}")
        else:
            logger.info(f"Try Custom image model: {image_backbone_name}")
        if MODEL is None:
            raise ValueError(f"Invalid torchvison model name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model
