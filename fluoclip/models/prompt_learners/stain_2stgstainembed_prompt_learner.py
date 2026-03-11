from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP

from fluoclip.utils import get_logger

from .builder import PROMPT_LEARNERS

logger = get_logger(__name__)
# trainable stain token version
@PROMPT_LEARNERS.register_module()
class Stain2stgEmbedPromptLearner(nn.Module):
    interpolation_functions = {
        "linear": lambda weights, num_ranks: 1.0 - weights / (num_ranks - 1),
        "inv_prop": lambda weights, _, eps=1e-5: 1.0 / (weights + eps),
        "normal": lambda weights, _: torch.exp(-weights * weights),
    }
    rank_tokens_position_candidates = {"tail", "middle", "front"}
    clip_max_num_tokens = 77  # CLIP num_context_tokens = 77

    def __init__(
        self,
        clip_model: CLIP,
        num_base_ranks: int,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        stain_tokens_position: str = "tail",
        init_rank_path: Optional[str] = None,
        init_context: Optional[str] = None,
        rank_specific_context: bool = False,
        init_rank_context: Optional[str] = None,
        num_rank_context_tokens: int = 0,
        interpolation_type: str = "linear",
        init_stain_path: Optional[str] = None,
        stage_num: int=1,
        interpolation_rank : str="rank",
        stain_embed_cfg: Optional[Dict[str, Any]] = None,
        stain_rank_position: str = "sr",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")
        self.clip_model = clip_model
        self.dtype = clip_model.token_embedding.weight.dtype
        self.embeddings_dim = clip_model.token_embedding.embedding_dim
        self.stage_num = stage_num
        
        # freeze text embedding
        for param in self.clip_model.token_embedding.parameters():
            param.requires_grad = False
        self.clip_model.positional_embedding.requires_grad = False
        logger.info("CLIP token_embedding layer is frozen.")
        
        # stage 1 initialization
        self.init_stain_path = init_stain_path
        self.stain_tokens_position =  stain_tokens_position
        self.stain_embed_cache = {}
        self.stain_params = nn.ParameterDict()
        self.num_tokens_per_stain = 0
        self.num_context_tokens = num_context_tokens
        self.init_context = init_context
        context_embeds = self._stage_1_init(init_stain_path)
        self.context_embeds = nn.Parameter(context_embeds)  # (num_context_tokens, embeds_dim)
        self.num_ranks = num_ranks
        self.rank_tokens_position = rank_tokens_position
        self.num_base_ranks = num_base_ranks
        self.num_rank_context_tokens = num_rank_context_tokens
        self.rank_specific_context = rank_specific_context
        self.stain_rank_position = stain_rank_position
        # rank embeds
        self.rank_embeds: Optional[nn.Parameter] = None
        self.rank_context_embeds: Optional[nn.Parameter] = None
        self.stain_embed_learner: Optional[nn.Module] = None
        if stage_num == 2:
            rank_context_embeds, rank_embeds = self._stage_2_init(num_tokens_per_rank, init_rank_context, init_rank_path)
            self.rank_context_embeds = nn.Parameter(rank_context_embeds)
            self.rank_embeds = nn.Parameter(rank_embeds)
            self.create_interpolation_weights(num_base_ranks, num_ranks, interpolation_type, self.dtype)
            logger.info(f"rank_tokens_position : {rank_tokens_position}")
            self.interpolation_rank = interpolation_rank
            self.stain_embed_learner = StainEmbedLearner(self.embeddings_dim, **(stain_embed_cfg or {}))
        
    # initialize init_context, stain embeds, pseudo_sentence_tokens, sentence_embeds template
    def _stage_1_init(self, init_stain_path):
        assert init_stain_path, "init_stain_path must exist"
        logger.info(f"init_stain_path : {init_stain_path}")
        if init_stain_path:
            # update stain embedding, num_tokens_per_stain, num_stains
            self.read_stain_file(init_stain_path)
            logger.info(f"initiated stain embeds with num_stains : {self.num_stains}, length : {self.num_tokens_per_stain}")
        # stage 1 pseudo sentence token, context embeds
        context_embeds, self.num_context_tokens = self.create_context_embeds(self.num_stains, self.num_context_tokens, self.init_context, False, logger, self.dtype)
        pure_sentence_length = self.num_context_tokens + self.num_tokens_per_stain + 3
        pseudo = self.create_pseudo_sentence_tokens(self.num_stains, pure_sentence_length)
        self.register_buffer("pseudo_sentence_tokens", pseudo, persistent=False)
        self.create_sentence_embeds_template(self.num_stains, self.pseudo_sentence_tokens)
        return context_embeds 
    
    def _stage_1_init_f(self, init_stain_path):
        assert init_stain_path, "init_stain_path must exist"
        logger.info(f"init_stain_path : {init_stain_path}")
        if init_stain_path:
            # update stain embedding, num_tokens_per_stain, num_stains
            self.read_stain_file_f(init_stain_path)
            logger.info(f"reinitiated stain embeds with num_stains : {self.num_stains}, length : {self.num_tokens_per_stain}")
        # stage 1 pseudo sentence token, context embeds
        # context_embeds, self.num_context_tokens = self.create_context_embeds(self.num_stains, self.num_context_tokens, self.init_context, False, logger, self.dtype)
        pure_sentence_length = self.num_context_tokens + self.num_tokens_per_stain + 3
        pseudo = self.create_pseudo_sentence_tokens(self.num_stains, pure_sentence_length)
        self.register_buffer("pseudo_sentence_tokens", pseudo, persistent=False)
        self.create_sentence_embeds_template(self.num_stains, self.pseudo_sentence_tokens)
        return 
    
    # initialize init_rank_context, rank embeds, pseudo_sentence_tokens, sentence_embeds template
    def _stage_2_init(self, num_tokens_per_rank, init_rank_context=None, init_rank_path=None):
        if self.rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {self.rank_tokens_position}")
        logger.info(f"num_ranks : {self.num_ranks}, num_base_ranks : {self.num_base_ranks}")

        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * self.num_base_ranks

        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            self.num_base_ranks, num_tokens_per_rank, init_rank_path,
        )
        self.num_tokens_per_rank = torch.tensor(
                [np.max(_num_tokens_per_rank)] * self.num_ranks,
                dtype=torch.long,
            )
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert (
            len(rank_embeds) == self.num_base_ranks
        ), f"len(rank_embeds) {len(rank_embeds)} == num_base_ranks {self.num_base_ranks}"
        
        rank_context_embeds, self.num_rank_context_tokens = self.create_context_embeds(self.num_ranks, self.num_rank_context_tokens, init_rank_context, self.rank_specific_context, logger, self.dtype)
        logger.info(f"self.num_rank_context_tokens : {self.num_rank_context_tokens}, self.num_tokens_per_rank : {self.num_tokens_per_rank}, self.num_context_tokens : {self.num_context_tokens}, self.num_tokens_per_stain : {self.num_tokens_per_stain}")
        def to_len_list(x, N):
            import torch
            if isinstance(x, torch.Tensor):
                if x.dim() == 0:
                    return [int(x.item())] * N
                assert x.dim() == 1 and x.numel() == N, f"expected 1D tensor of len {N}, got {tuple(x.shape)}"
                return [int(v) for v in x.tolist()]
            elif isinstance(x, (list, tuple)):
                assert len(x) == N, f"expected list of len {N}, got {len(x)}"
                return [int(v) for v in x]
            else:  # int
                return [int(x)] * N
        R = self.num_ranks
        pure_sentence_length = [a+b+c+d+e for a,b,c,d,e in zip(to_len_list(self.num_rank_context_tokens, R), to_len_list(self.num_tokens_per_rank, R), to_len_list(self.num_tokens_per_stain, R), to_len_list(self.num_context_tokens, R), to_len_list(3, R))]
        logger.info(f"pure_sentence_length : {pure_sentence_length}")
        pseudo = self.create_pseudo_sentence_tokens(self.num_ranks, pure_sentence_length)
        if hasattr(self, "pseudo_sentence_tokens"):
            # 이미 등록된 버퍼가 있으면 내부 데이터만 갱신
            self.pseudo_sentence_tokens.data = pseudo
        else:
            # 아직 등록 전이면 새로 등록
            self.register_buffer("pseudo_sentence_tokens", pseudo, persistent=False)
        self.create_sentence_embeds_template(self.num_ranks, self.pseudo_sentence_tokens)
        return rank_context_embeds, rank_embeds 

    def _stage_2_init_f(self):
        if self.rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {self.rank_tokens_position}")
        logger.info(f"num_ranks : {self.num_ranks}, num_base_ranks : {self.num_base_ranks}")

        # if isinstance(num_tokens_per_rank, int):
        #     num_tokens_per_rank = [num_tokens_per_rank] * self.num_base_ranks

        # rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
        #     self.num_base_ranks, num_tokens_per_rank, init_rank_path,
        # )
        # self.num_tokens_per_rank = torch.tensor(
        #         [np.max(_num_tokens_per_rank)] * self.num_ranks,
        #         dtype=torch.long,
        #     )
        # self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert (
            len(self.rank_embeds) == self.num_base_ranks
        ), f"len(rank_embeds) {len(self.rank_embeds)} == num_base_ranks {self.num_base_ranks}"
        
        # rank_context_embeds, self.num_rank_context_tokens = self.create_context_embeds(self.num_ranks, self.num_rank_context_tokens, init_rank_context, self.rank_specific_context, logger, self.dtype)
        logger.info(f"self.num_rank_context_tokens : {self.num_rank_context_tokens}, self.num_tokens_per_rank : {self.num_tokens_per_rank}, self.num_context_tokens : {self.num_context_tokens}, self.num_tokens_per_stain : {self.num_tokens_per_stain}")
        def to_len_list(x, N):
            import torch
            if isinstance(x, torch.Tensor):
                if x.dim() == 0:
                    return [int(x.item())] * N
                assert x.dim() == 1 and x.numel() == N, f"expected 1D tensor of len {N}, got {tuple(x.shape)}"
                return [int(v) for v in x.tolist()]
            elif isinstance(x, (list, tuple)):
                assert len(x) == N, f"expected list of len {N}, got {len(x)}"
                return [int(v) for v in x]
            else:  # int
                return [int(x)] * N
        R = self.num_ranks
        pure_sentence_length = [a+b+c+d+e for a,b,c,d,e in zip(to_len_list(self.num_rank_context_tokens, R), to_len_list(self.num_tokens_per_rank, R), to_len_list(self.num_tokens_per_stain, R), to_len_list(self.num_context_tokens, R), to_len_list(3, R))]
        logger.info(f"pure_sentence_length : {pure_sentence_length}")
        pseudo = self.create_pseudo_sentence_tokens(self.num_ranks, pure_sentence_length)
        if hasattr(self, "pseudo_sentence_tokens"):
            self.pseudo_sentence_tokens.data = pseudo
        else:
            self.register_buffer("pseudo_sentence_tokens", pseudo, persistent=False)
        self.create_sentence_embeds_template(self.num_ranks, self.pseudo_sentence_tokens)
        return

    def get_stain_conditioned_rank_embeds(self):
        if self.interpolation_rank == "base":
            base_embeds = self.rank_embeds + self.stain_embed_learner(self.rank_embeds, self.stain_embed_cache) # S, R, T, D
            # rank_embeds = torch.sum(self.t[..., None, None] * base_embeds[None, ...], dim=1)
            weights = self.interpolation_weights[None, ..., None, None]   # (1, num_ranks, num_base_ranks, 1, 1)
            # add singleton axis to base_embeds
            base_embeds = base_embeds[:, None, ...]   # (S, 1, num_base_ranks, T, D)
            # now multiply & sum over num_base_ranks
            rank_embeds = torch.sum(weights * base_embeds, dim=2)   # (S, num_ranks, T, D)
        elif self.interpolation_rank == "rank":
            rank_embeds = torch.sum(self.interpolation_weights[..., None, None] * self.rank_embeds[None, ...], dim=1)
            rank_embeds = rank_embeds + self.stain_embed_learner(rank_embeds, self.stain_embed_cache) # S, R, T, D
        else: 
            logger.error(f"interpolation method '{self.interpolation_rank}' not implemented")
            raise ValueError(f"Unknown method: {self.interpolation_rank}")
        return rank_embeds


    def forward(self, stains) -> torch.Tensor:

        S, R, L, D = self.num_stains, self.num_ranks, self.clip_max_num_tokens, self.embeddings_dim
        device = self.clip_model.positional_embedding.device

        stain_embeds, stain_ids_tensor = self._get_stain_embeds(stains) # S, num_tokens_per_stain, D
        if self.stage_num == 1:
            template = self.sentence_embeds.clone() # S, L, D

            if self.context_embeds.dim() == 2:
                context_embeds = self.context_embeds[None].expand(S, *self.context_embeds.shape)
            len_ctx = torch.as_tensor(self.num_context_tokens, device=device)
            if len_ctx.ndim == 0:
                len_ctx = len_ctx.expand(S)
            len_stain = torch.as_tensor(self.num_tokens_per_stain, device=device)
            if len_stain.ndim ==0:
                len_stain = len_stain.expand(S)

            stage1_len = torch.zeros_like(len_ctx, device=device) # (S, )

            for i in range(S):
                if self.stain_tokens_position == "front":
                    pure_sentence_length = len_ctx[i]+len_stain[i]
                    template[i, 1:1+pure_sentence_length] = torch.cat([stain_embeds[i, :int(len_stain[i].item())], context_embeds[i, :int(len_ctx[i].item())]], dim=0)
                    stage1_len[i] = 1+pure_sentence_length
                elif self.stain_tokens_position == "tail":
                    pure_sentence_length = len_ctx[i]+len_stain[i]
                    template[i, 1:1+pure_sentence_length] = torch.cat([context_embeds[i, :int(len_ctx[i].item())], stain_embeds[i, :int(len_stain[i].item())]], dim=0)
                    stage1_len[i] = 1+pure_sentence_length

            return template, self.pseudo_sentence_tokens, stain_ids_tensor
        # if stage 2
        template = self.sentence_embeds.unsqueeze(0).expand(S, R,-1,-1).clone() # S,R,L,D
        if self.context_embeds.dim() == 2:
            context_embeds = self.context_embeds[None].expand(S, *self.context_embeds.shape)
        len_ctx = torch.as_tensor(self.num_context_tokens, device=device)
        if len_ctx.ndim == 0:
            len_ctx = len_ctx.expand(S)
        len_stain = torch.as_tensor(self.num_tokens_per_stain, device=device)
        if len_stain.ndim ==0:
            len_stain = len_stain.expand(S)
        stage1_len = torch.zeros_like(len_ctx, device=device) # (S, )
        stage1_prefix = torch.zeros(S, L, D, device=device, dtype=self.dtype) # (S, 77)
        for i in range(S):
            if self.stain_tokens_position == "front":
                pure_sentence_length = len_ctx[i]+len_stain[i]
                stage1_prefix[i, 1:1+pure_sentence_length] = torch.cat([stain_embeds[i, :int(len_stain[i].item())], context_embeds[i, :int(len_ctx[i].item())]], dim=0)
                stage1_len[i] = 1+pure_sentence_length
            elif self.stain_tokens_position == "tail":
                pure_sentence_length = len_ctx[i]+len_stain[i]
                stage1_prefix[i, 1:1+pure_sentence_length] = torch.cat([context_embeds[i, :int(len_ctx[i].item())], stain_embeds[i, :int(len_stain[i].item())]], dim=0)
                stage1_len[i] = 1+pure_sentence_length

        rank_embeds = self.get_stain_conditioned_rank_embeds()
        if self.rank_context_embeds.dim() == 2:
            rank_context_embeds = self.rank_context_embeds[None].expand(R, *self.rank_context_embeds.shape)
        else : rank_context_embeds = self.rank_context_embeds
        len_rank_ctx = torch.as_tensor(self.num_rank_context_tokens, device=device, dtype=torch.long)
        if len_rank_ctx.ndim == 0: 
            len_rank_ctx = len_rank_ctx.expand(R)
        len_rank = torch.as_tensor(self.num_tokens_per_rank, device=device)
        if len_rank.ndim ==0:
            len_rank = len_rank.expand(R)
        if self.stain_rank_position == "sr":
            for j in range(S):
                for i in range(R):
                    prev = stage1_len[j]
                    if self.rank_tokens_position == "front":
                        pure_sentence_length = prev+len_rank_ctx[i]+len_rank[i]
                        template[j, i, 1:prev] = stage1_prefix[j, 1:prev]
                        template[j, i, prev:pure_sentence_length] = torch.cat([rank_embeds[j, i, :int(len_rank[i].item())], rank_context_embeds[i, :int(len_rank_ctx[i].item())]], dim=0)
                    elif self.rank_tokens_position == "tail":
                        pure_sentence_length = prev+len_rank_ctx[i]+len_rank[i]
                        template[j, i, 1:prev] = stage1_prefix[j, 1:prev]
                        template[j, i, prev:pure_sentence_length] = torch.cat([rank_context_embeds[i, :int(len_rank_ctx[i].item())], rank_embeds[j, i, :int(len_rank[i].item())]], dim=0)
        elif self.stain_rank_position == "rs":
            for j in range(S):
                for i in range(R):
                    prev = stage1_len[j]
                    if self.rank_tokens_position == "front":
                        pure_sentence_length = prev+len_rank_ctx[i]+len_rank[i] # (1 + stain + ctx) + rank_ctx + rank
                        rank_sentence_length = len_rank_ctx[i]+len_rank[i]
                        template[j, i, 1:1+rank_sentence_length] = torch.cat([rank_embeds[j, i, :int(len_rank[i].item())], rank_context_embeds[i, :int(len_rank_ctx[i].item())]], dim=0)
                        template[j, i, 1+rank_sentence_length:pure_sentence_length] = stage1_prefix[j, 1:prev]
                    elif self.rank_tokens_position == "tail":
                        pure_sentence_length = prev+len_rank_ctx[i]+len_rank[i]
                        rank_sentence_length = len_rank_ctx[i]+len_rank[i]
                        template[j, i, 1:1+rank_sentence_length] = torch.cat([rank_context_embeds[i, :int(len_rank_ctx[i].item())], rank_embeds[j, i, :int(len_rank[i].item())]], dim=0)
                        template[j, i, 1+rank_sentence_length:pure_sentence_length] = stage1_prefix[j, 1:prev]        
        
        return template, self.pseudo_sentence_tokens.unsqueeze(0).expand(S, R, L), stain_ids_tensor

    # stain embedding, num_tokens_per_stain, num_stains
    def read_stain_file(self, init_stain_path: str, update_stain_token=True):
        """
        Load a YAML mapping:  "StainName": "free-form description to tokenize".
        Creates per-stain embeddings (buffer or parameter depending on self.update_stain_token).
        Also updates num_tokens_per_stain and expands existing cached tensors if needed.
        """
        import yaml
        with open(init_stain_path, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f) or {}
        logger.info(f"load init stain from {init_stain_path}.")

        if not isinstance(mapping, dict):
            raise ValueError(f"[read_stain_file] Mapping expected, got {type(mapping)}")

        device = self.clip_model.positional_embedding.device
        dtype  = self.dtype

        token_lists = {}
        max_len = 0
        for stain, desc in mapping.items():
            ids = clip._tokenizer.encode(desc)  # special token
            token_lists[stain] = ids
            max_len = max(max_len, len(ids))
            logger.info(f"for stain {stain}, description will be {desc}, and the max_len will be {len(ids)}")

        assert max_len != 0, "number of initial stains can't be 0"
        assert max_len < self.clip_max_num_tokens, f"number of tokens {max_len} can't be larger then {self.clip_max_num_tokens}"
        D = self.embeddings_dim

        # self.stain_embed_cache = {}
        for stain, ids in token_lists.items():
            token_ids = torch.tensor(ids, device=device)
            with torch.no_grad():
                e = self.clip_model.token_embedding(token_ids).to(dtype=dtype)  # (T,D)

            T = e.size(0)
            pad_len = max_len - T
            if pad_len > 0:
                if update_stain_token is True:
                    # random init ONLY for padded rows
                    pad = torch.empty(pad_len, D, device=device, dtype=dtype)
                    nn.init.normal_(pad, mean=0.0, std=0.02)   # <- your desired init
                else:
                    pad = e.new_zeros(pad_len, D)
                e = torch.cat([e, pad], dim=0)
            elif T > max_len:
                e = e[:max_len]

            if update_stain_token is True:
                self.stain_params[stain] = nn.Parameter(e)
                tensor_ref = self.stain_params[stain]
            else:
                buf_name = f"stain_embed_{stain}"
                self.register_buffer(buf_name, e, persistent=False)
                tensor_ref = getattr(self, buf_name)

            self.stain_embed_cache[stain] = tensor_ref

        self.num_tokens_per_stain = max_len
        self.num_stains = len(self.stain_embed_cache)

    def read_stain_file_f(self, init_stain_path: str, update_stain_token=True):
            """
            Load a YAML mapping:  "StainName": "free-form description to tokenize".
            Creates per-stain embeddings (buffer or parameter depending on self.update_stain_token).
            Also updates num_tokens_per_stain and expands existing cached tensors if needed.
            """
            import yaml
            with open(init_stain_path, "r", encoding="utf-8") as f:
                mapping = yaml.safe_load(f) or {}
            logger.info(f"load init stain from {init_stain_path}.")

            if not isinstance(mapping, dict):
                raise ValueError(f"[read_stain_file] Mapping expected, got {type(mapping)}")

            device = self.clip_model.positional_embedding.device
            dtype  = self.dtype

            token_lists = {}
            max_len_new = 0
            for stain, desc in mapping.items():
                ids = clip._tokenizer.encode(desc)
                token_lists[stain] = ids
                max_len_new = max(max_len_new, len(ids))
                logger.info(f"for stain {stain}, description will be {desc}, and the max_len will be {len(ids)}")
            max_len_old = getattr(self, "num_tokens_per_stain", 0)
            max_len = max(max_len_old, max_len_new)
            self.num_tokens_per_stain = max_len
            assert max_len != 0, "number of initial stains can't be 0"
            assert max_len < self.clip_max_num_tokens, f"number of tokens {max_len} can't be larger then {self.clip_max_num_tokens}"
            D = self.embeddings_dim
            
            updated_params = nn.ParameterDict()
            updated_cache = {}
            
            # self.stain_embed_cache = {}
            for stain, ids in token_lists.items():
                token_ids = torch.tensor(ids, device=device)
                with torch.no_grad():
                    e = self.clip_model.token_embedding(token_ids).to(dtype=dtype)  # (T,D)

                T = e.size(0)
                if stain in self.stain_params:
                    old_e = self.stain_params[stain].data  # (max_len_old, D)
                    old_len = old_e.size(0)

                    new_e = torch.zeros(max_len, D, dtype=dtype, device=device)

                    # 1) copy old weights
                    copy_len = min(old_len, max_len)
                    new_e[:copy_len] = old_e[:copy_len]

                    # 2) update front part with CLIP embedding (policy)
                    #    If you want to preserve old completely, comment this out:
                    # new_e[:T] = e

                    updated_params[stain] = nn.Parameter(new_e)

                # Case B: new stain → initialize + pad
                else:
                    new_e = torch.zeros(max_len, D, dtype=dtype, device=device)
                    new_e[:T] = e
                    updated_params[stain] = nn.Parameter(new_e)

                updated_cache[stain] = updated_params[stain]
                pad_len = max_len - T
                if pad_len > 0:
                    if update_stain_token is True:
                        # random init ONLY for padded rows
                        pad = torch.empty(pad_len, D, device=device, dtype=dtype)
                        nn.init.normal_(pad, mean=0.0, std=0.02)   # <- your desired init
                    else:
                        pad = e.new_zeros(pad_len, D)
                    e = torch.cat([e, pad], dim=0)
                elif T > max_len:
                    e = e[:max_len]

                if update_stain_token is True:
                    self.stain_params[stain] = nn.Parameter(e)
                    tensor_ref = self.stain_params[stain]
                else:
                    buf_name = f"stain_embed_{stain}"
                    self.register_buffer(buf_name, e, persistent=False)
                    tensor_ref = getattr(self, buf_name)

                self.stain_embed_cache[stain] = tensor_ref

            self.num_tokens_per_stain = max_len
            self.num_stains = len(self.stain_embed_cache)

    # return sorted stain embeds
    def _get_stain_embeds(self, stains: List[str]):
        device = self.clip_model.positional_embedding.device
        all_cached_stain_names = sorted(self.stain_embed_cache.keys())
        self.current_stain_name_to_id_map = {
            name: i for i, name in enumerate(all_cached_stain_names)
        }
        stain_ids_for_batch_list = []
        for s in stains:
            try:
                stain_ids_for_batch_list.append(self.current_stain_name_to_id_map[s])
            except KeyError:
                # Handle cases where a stain in the batch is not in the cache.
                logger.error(f"Stain '{s}' from batch not found in stain_embed_cache. Please ensure cache is populated.")
                raise ValueError(f"Unknown stain: {s}")
        # Convert the list of IDs to a PyTorch tensor of shape (B,)
        stain_ids_tensor = torch.tensor(
            stain_ids_for_batch_list,
            device=device, # Put on same device as embeddings
            dtype=torch.long
        ) # Shape: (B,)
        try:
            return torch.stack([self.stain_embed_cache[s] for s in all_cached_stain_names], dim=0).to(device), stain_ids_tensor
        except KeyError as e:
            raise KeyError(f"[get_stain_embeds] unknown stain: {e.args[0]}. "
                        f"Add it to the YAML used in read_stain_file().")

    # creates pseudo_sentence_tokens, called in init
    def create_pseudo_sentence_tokens(
        self,
        num_label, # num_ranks or num_stains
        pure_sentence_length, # list if rank specific context, else int
    ) -> torch.Tensor:
        """
        Construct pseudo sentence token positions for CLIP input:
        <sot> [context] [stain] [rank] <.> <eot>

        Returns:
            pseudo_sentence_tokens: Tensor of shape [num_ranks, max_tokens]
        """
        pseudo_sentence_tokens = torch.zeros(num_label, self.clip_max_num_tokens, dtype=torch.long)
        def to_int(x):
            if isinstance(x, torch.Tensor):
                assert x.dim() == 0, f"length tensor must be scalar, got shape {tuple(x.shape)}"
                return int(x.item())
            return int(x)
        if isinstance(pure_sentence_length, (list, tuple)) or ( isinstance(pure_sentence_length, torch.Tensor) and pure_sentence_length.dim() == 1):
            lengths = [to_int(x) for x in (pure_sentence_length.tolist() if isinstance(pure_sentence_length, torch.Tensor) else pure_sentence_length)]
            assert len(lengths) == num_label, f"num_label={num_label}, but got {len(lengths)} lengths"
            for i, n in enumerate(lengths):
                L = min(1 + n + 2, self.clip_max_num_tokens)  # <sot> + n + <.> + <eot>
                pseudo_sentence_tokens[i, :L] = torch.arange(L, dtype=torch.long, device=pseudo_sentence_tokens.device)
        else:
            n = to_int(pure_sentence_length)
            L = min(1 + n + 2, self.clip_max_num_tokens)
            row = torch.arange(L, dtype=torch.long, device=pseudo_sentence_tokens.device)
            pseudo_sentence_tokens[:, :L] = row
        return pseudo_sentence_tokens

    # return read lines in init_rank_path
    def read_rank_file(self, init_rank_path):
        rank_names = []
        with open(init_rank_path, "r") as f:
            for line in f.readlines():
                line = line.strip().replace("_", " ")
                rank_names.append(line)
        logger.info(f"num rank: {len(rank_names)}:\n\t{rank_names[:5]}\n\t{rank_names[-5:]}")
        return rank_names

    # context_embeds, num_context_tokens 
    def create_context_embeds(
        self,
        num_labels: int, # num_stains or num_ranks
        num_context_tokens: int,
        init_context: Optional[str],
        rank_specific_context: bool,
        logger,
        dtype,
    ):
        # context embeddings
        logger.info("init context token")
        if init_context is not None:
            logger.info(f"rank_specific_context: {rank_specific_context}")
            if rank_specific_context is True:
                _context = self.read_rank_file(init_context)
                if len(_context) != num_labels:
                    raise ValueError(f"The length of init context is {len(_context)}, which should be equal to {num_labels}")
                _c_tokenids = [clip._tokenizer.encode(c) for c in _context]
                _num_context_tokens = [len(c) for c in _c_tokenids]
                logger.info(f"num_context_tokens: {self.num_context_tokens} -> {_num_context_tokens}")
                num_context_tokens = _num_context_tokens
                max_num_tokens_per_rank = np.max(num_context_tokens)
                assert max_num_tokens_per_rank < self.clip_max_num_tokens -3, "Context tokens are too long: {max_num_tokens_per_rank}"
                all_token_ids = torch.zeros(len(num_context_tokens), max_num_tokens_per_rank, dtype=torch.long)
                for i, tid in enumerate(_c_tokenids):
                    all_token_ids[i, : len(tid)] = torch.as_tensor(tid, dtype=torch.long)
                context_embeds = self.clip_model.token_embedding(all_token_ids).type(dtype)
                context_embeds = context_embeds[:, :max_num_tokens_per_rank]
            else :
                init_context = init_context.replace("_", " ")
                logger.info(f"init context: {init_context}")
                
                prompt_tokens = clip.tokenize(init_context)
                prompt_tokens = prompt_tokens[0]  # (num_context_tokens=77)
                _num_context_tokens = torch.argmax(prompt_tokens).item() - 1
                logger.info(f"num_context_tokens: {num_context_tokens} -> {_num_context_tokens}")
                num_context_tokens = _num_context_tokens

                with torch.no_grad():
                    context_embeds = self.clip_model.token_embedding(prompt_tokens).type(dtype)
                context_embeds = context_embeds[1 : 1 + num_context_tokens]

        else:
            embeds_dim = self.clip_model.token_embedding.embedding_dim
            init_context = " ".join(["X"] * num_context_tokens)
            logger.info(f"random context: {init_context}")
            logger.info(f"num context tokens: {num_context_tokens}")
            logger.info(f"rank_specific_context: {rank_specific_context}")

            if rank_specific_context is True:
                context_embeds = torch.empty((self.num_ranks, num_context_tokens, embeds_dim), dtype=dtype)
            else:
                context_embeds = torch.empty((num_context_tokens, embeds_dim), dtype=dtype)
            nn.init.normal_(context_embeds, std=0.02)

        return context_embeds, num_context_tokens



    def create_sentence_embeds_template(self, num_labels, pseudo_sentence_tokens):
        with torch.no_grad():
            null_embed = self.clip_model.token_embedding(torch.LongTensor([0]))[0]
            sot_embed = self.clip_model.token_embedding(torch.LongTensor([49406]))[0]
            eot_embed = self.clip_model.token_embedding(torch.LongTensor([49407]))[0]
            full_stop_embed = self.clip_model.token_embedding(torch.LongTensor([269]))[0]

        sentence_embeds = null_embed[None, None].repeat(
            num_labels, self.clip_max_num_tokens, 1
        )  # not the same null_embed!
        argmax_index = pseudo_sentence_tokens.argmax(dim=-1)
        rank_index = torch.arange(num_labels)
        sentence_embeds[:, 0, :] = sot_embed
        sentence_embeds[rank_index, argmax_index] = eot_embed
        sentence_embeds[rank_index, argmax_index - 1] = full_stop_embed
        if hasattr(self, "sentence_embeds"):
            self.sentence_embeds.data = sentence_embeds
        else:
            self.register_buffer("sentence_embeds", sentence_embeds, persistent=False)



    def create_interpolation_weights(self, num_base_ranks, num_ranks, interpolation_type, dtype):
        if interpolation_type not in self.interpolation_functions:
            raise ValueError(f"Invalide interpolation_type: {interpolation_type}")
        interpolation_func = self.interpolation_functions[interpolation_type]

        interpolation_weights = torch.arange(num_ranks)[..., None].repeat(1, num_base_ranks).to(dtype)
        if num_base_ranks == 1:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, 3)[1:2].to(dtype)
        else:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, num_base_ranks).to(dtype)
        interpolation_weights = torch.abs(interpolation_weights - base_interpolation_weights[None])
        interpolation_weights = interpolation_func(interpolation_weights, num_ranks)
        interpolation_weights = interpolation_weights / interpolation_weights.sum(dim=-1, keepdim=True)
        self.register_buffer("interpolation_weights", interpolation_weights, persistent=False)



    def create_rank_embeds(
        self, num_ranks, num_tokens_per_rank, init_rank_path
    ):
        if init_rank_path is not None:
            logger.info(f"load init rank from: {init_rank_path}.")

            rank_names = self.read_rank_file(init_rank_path)
            if len(rank_names) != num_ranks:
                raise ValueError(
                    f"The length of rank_names is {len(rank_names)}, which is not equal to num_ranks {num_ranks}"
                )

            _rank_tokens = [clip._tokenizer.encode(rank_name) for rank_name in rank_names]
            _num_tokens_per_rank = [len(rank_token) for rank_token in _rank_tokens]
            logger.info(f"num_tokens_per_rank: {num_tokens_per_rank} -> {_num_tokens_per_rank}")
            num_tokens_per_rank = _num_tokens_per_rank
            max_num_tokens_per_rank = np.max(num_tokens_per_rank)

            rank_tokens = torch.zeros(len(_rank_tokens), max_num_tokens_per_rank, dtype=torch.long)
            for i, rank_token in enumerate(_rank_tokens):
                # 3 is <eot>, <sot>, and <full_stop>
                valid_length = self.clip_max_num_tokens - self.num_context_tokens - 3
                if len(rank_token) > valid_length:
                    rank_token = rank_token[:valid_length]
                    raise ValueError(f"rank tokens are too long: {rank_token}")
                rank_tokens[i, : len(rank_token)] = torch.LongTensor(rank_token)
            rank_embeds = self.clip_model.token_embedding(rank_tokens).type(self.dtype)
            rank_embeds = rank_embeds[:, :max_num_tokens_per_rank]

        else:
            logger.info(f"num rank: {num_ranks}")
            logger.info(f"num_tokens_per_rank: {num_tokens_per_rank}")
            embeddings_dim = self.clip_model.token_embedding.embedding_dim
            if isinstance(num_tokens_per_rank, List):
                max_num_tokens_per_rank = np.max(num_tokens_per_rank)
            else:
                max_num_tokens_per_rank = num_tokens_per_rank
            if self.clip_max_num_tokens < self.num_context_tokens + max_num_tokens_per_rank + 3:
                raise ValueError(f"rank tokens are too long: {rank_token}")
            rank_embeds = torch.empty((num_ranks, max_num_tokens_per_rank, embeddings_dim), dtype=self.dtype)
            nn.init.normal_(rank_embeds, std=0.02)

        return (rank_embeds, num_tokens_per_rank)


class StainEmbedLearner(nn.Module):
    """
    Args:
        embed_dim (int): Dimensionality of rank and stain embeddings.
        hidden_dim (int): Dimensionality of the hidden layer.
        num_layers (int): Number of hidden layers (at least 1).
        activation (callable): Activation function (e.g., nn.ReLU()).
    """
    def __init__(
            self,
            embed_dim: int,
            stain_embed_type = "mlp",
            summary_type = "query_attention",
            num_queries = 1,
            reduce = "mean"
        ):
        super().__init__()
        layers = []
        self.summary_type = summary_type
        input_dim = embed_dim * 2  # concatenated token + stain summary
        if stain_embed_type == "mlp":
            layers.append(nn.LayerNorm(input_dim)),
            layers.append(nn.Linear(input_dim, input_dim // 16))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(input_dim // 16, input_dim // 2))
        if summary_type == "query_attention":
            self.summary_net = StainSummaryAttention(embed_dim=embed_dim, num_queries=num_queries, reduce=reduce)
        self.embed_net = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.tensor(0.1)) 

    def forward(
        self,
        rank_embeds: torch.Tensor,                # (R, T, D)
        stain_embed_cache: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        R, T, D = rank_embeds.shape

        stain_names = sorted(stain_embed_cache.keys())
        if self.summary_type == "query_attention":
            summaries = []
            for name in stain_names:
                embed = stain_embed_cache[name] # (T_s, D)
                # print(f"embed.shape : {embed.shape}")
                summary = self.summary_net(embed)
                summaries.append(summary)
            summaries = torch.stack(summaries, dim=0)  # (S, D)
        
        S = summaries.size(0)

        rank_exp = rank_embeds.unsqueeze(0).expand(S, -1, -1, -1)      # (S, R, T, D)
        stain_exp = summaries.view(S, 1, 1, D).expand(-1, R, T, -1)    # (S, R, T, D)
        fused = torch.cat([rank_exp, stain_exp], dim=-1)               # (S, R, T, 2D)

        out = self.embed_net(fused.view(-1, 2*D)).view(S, R, T, D)

        return out*self.alpha  # (S, R, T, D)

class StainSummaryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int = 1, reduce: str = "mean"):
        """
        Args:
            embed_dim (int): Token embedding dimension (D)
            num_queries (int): Number of learnable queries (K)
            reduce (str): How to reduce multiple queries ("mean", "max", "mlp")
        """
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))  # (K,D)
        self.proj = nn.Linear(embed_dim, 1)
        self.reduce = reduce

        if reduce == "mlp":
            self.reducer = nn.Sequential(
                nn.Linear(num_queries * embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        embeds: (T, D)
        mask:   (T,) with 1=valid, 0=pad
        return: (D,)
        """
        summaries = []
        for q in self.queries:  # iterate over K queries
            attn_logits = self.proj(embeds * q)  # (T,1)
            attn_logits = attn_logits.squeeze(-1)  # (T,)
            attn_weights = torch.softmax(attn_logits, dim=0)  # (T,)
            summary = torch.sum(attn_weights.unsqueeze(-1) * embeds, dim=0)  # (D,)
            summaries.append(summary)

        summaries = torch.stack(summaries, dim=0)  # (K,D)

        # reduce back to (D,)
        if self.reduce == "mean":
            return summaries.mean(dim=0)
        elif self.reduce == "max":
            return summaries.max(dim=0).values
        elif self.reduce == "mlp":
            return self.reducer(summaries.view(-1))  # (D,)
        else:
            raise ValueError(f"Unknown reduce method {self.reduce}")