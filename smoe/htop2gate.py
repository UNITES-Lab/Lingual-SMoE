# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from fairseq.utils import print_r0
from typing import Callable, Dict, Tuple, Optional

import math
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

from pytorch_metric_learning import miners, losses, distances
from .utils import convert_langdict_to_groupid
import logging

logger = logging.getLogger(__name__)

gumbel_map: Dict[torch.device, Callable] = {}

# logging
SAMPLE_FRACTION = 0.2


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype)
    output.scatter_(
        len(output.shape) - 1, indices, 1
    )
    return output

def one_hot_topk(indices: torch.Tensor, num_classes: int) -> Tensor:
    output = torch.zeros(indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype)
    output.scatter_(
        len(output.shape) - 1, indices, 1
    )
    return output

def one_hot_group_mask(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype)
    output.scatter_(
        len(output.shape) - 1, indices, 1
    )
    output = torch.sum(output, dim=1)
    return output


def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)


def htop2gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    second_expert_policy='sampling',
    normalize_gate_prob_before_dropping=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    capacity_factor=1.0,
    batch_prioritized_routing=False,
    has_tutel=False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    if has_tutel:
        from tutel import moe as tutel_moe
        fused_cumsum_sub_one=tutel_moe.fast_cumsum_sub_one
    else:
        fused_cumsum_sub_one=lambda mask: torch.cumsum(mask, dim=0) - 1
    metadata = {}

    gates = logits
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    num_2nd_experts = num_experts//4
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = int(2 * math.ceil(num_tokens / num_2nd_experts) * capacity_factor)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_experts)
    if second_expert_policy == 'sampling':
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    else:
        logits_w_noise = logits
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1, keepdim=True)
    mask2 = one_hot(indices2_s, num_experts)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)

    if normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s

    if second_expert_policy == 'random':
        sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
        mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

    # Compute locations in capacity buffer
    if input_mask is not None and input_mask.any():
        nonpadding = ~ input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)
        mask2 = mask2 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    if batch_prioritized_routing:
        # if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = fused_cumsum_sub_one(sorted_mask1) * sorted_mask1
        importance_sorted_locations1 =  sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]

        sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = fused_cumsum_sub_one(sorted_mask2) * sorted_mask2
        importance_sorted_locations2 =  sorted_cumsum2[importance_scores.argsort(dim=0).argsort(dim=0)]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = importance_sorted_locations1, importance_sorted_locations2
    else:
        locations1 = fused_cumsum_sub_one(mask1)
        locations2 = fused_cumsum_sub_one(mask2)
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    
    # l_aux = torch.mean(me * ce)
    l_aux = torch.sum(me * ce)/num_2nd_experts
    l_aux = l_aux * num_experts * num_experts

    # for logging purposes
    metadata["overflow_expert1"] = 100 * torch.sum(mask1 * torch.ge(locations1, capacity)) / torch.sum(mask1)
    metadata["overflow_expert2"] = 100 * torch.sum(mask2 * torch.ge(locations2, capacity)) / torch.sum(mask2)

    # Remove locations outside capacity from mask
    mask1_, mask2_ = mask1, mask2
    mask1 = mask1 * torch.lt(locations1, capacity)
    mask2 = mask2 * torch.lt(locations2, capacity)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    expert2_hist = 100 * torch.histc((indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    expert2_hist = torch.sort(expert2_hist, dim=0, descending=True).values +  torch.finfo(torch.float32).tiny

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()


    metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()

    if not normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        gates1_s = (gates * mask1).sum(dim=1)
        gates2_s = (gates * mask2).sum(dim=1)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

    if has_tutel:
        locations1_s = torch.sum(locations1 * mask1_, dim=1)
        locations2_s = torch.sum(locations2 * mask2_, dim=1)
        return l_aux, metadata, capacity, num_experts, [indices1_s, indices2_s], [locations1_s, locations2_s], [gates1_s, gates2_s]

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    locations2_sc = one_hot(locations2_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1), locations1_sc.to(gates1.dtype).unsqueeze(1)
    )
    combine2_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates2.unsqueeze(-1), locations2_sc.to(gates2.dtype).unsqueeze(1)
    )
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    is_untouted=~(dispatch_mask.any(-1).any(-1))
    if input_mask is not None:
        metadata["unrouted_token_rate"]=(is_untouted.sum()-input_mask.sum())/(is_untouted.size(0)-input_mask.sum())
    else:
        metadata["unrouted_token_rate"]=is_untouted.sum()/is_untouted.size(0)

    return l_aux, combine_weights, dispatch_mask, metadata


class HTop2Gate(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        second_expert_policy='sampling',
        normalize_gate_prob_before_dropping=False,
        moe_eval_capacity_token_fraction=0.25,
        batch_prioritized_routing=False,
        capacity_factor=1.0,
        add_lang_loss=False,
        lang_dict_path=None,
    ) -> None:
        super().__init__()
        self.wg1 = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.wg2 = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.use_fp32 = use_fp32
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing = batch_prioritized_routing
        self.capacity_factor=capacity_factor
        self.expert_group_size=num_experts//4
        self.add_lang_loss=add_lang_loss
        if self.add_lang_loss:
            self.lang_loss_func = losses.ContrastiveLoss(pos_margin = 1, neg_margin = 0, distance=distances.CosineSimilarity())
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
            lang_group_ids = convert_langdict_to_groupid(lang_dict_path)
            self.lang_group_label = torch.tensor(lang_group_ids)
        
    def forward(self, 
                lang_input: torch.Tensor=None, 
                token_input: torch.Tensor=None, 
                global_lang_emb: torch.Tensor=None,
                mask: Optional[torch.Tensor] = None, 
                has_tutel=False, 
                logits:torch.Tensor=None, ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        if logits is None:       
            if lang_input == None:
                logits = self.wg2(token_input)
                orig_dtype = logits.dtype
                lang_group_loss = 0
            else:
                if lang_input.dtype != self.wg1.weight.dtype:
                    lang_input = lang_input.to(dtype=self.wg1.weight.dtype)
                
                # print_r0(f'***********lang input size{lang_input.size()}')
                group_logits = self.wg1(lang_input)
                group_gates = F.softmax(group_logits, dim=1)
                
                group_index = torch.topk(group_gates, self.expert_group_size, dim=1, sorted=False)
                group_mask = one_hot_group_mask(group_index[1], group_gates.shape[1], unsqueeze_indices=True)
                
                if mask is not None and mask.any():
                    nonpadding = ~ mask
                    group_mask = group_mask * nonpadding.unsqueeze(-1).to(group_mask.dtype)
                
                if self.add_lang_loss and self.training:
                    '''
                    contrastive loss for grouping language familiy embeddings
                    - global lang emb is all language embeddings: emb_layer(torch.arange(num_tasks))
                    - self.lang_group_label is corresponding langauge family for each language in global lang emb
                    '''
                    global_lang_emb = global_lang_emb.view(global_lang_emb.shape[0],global_lang_emb.shape[-1])
                    global_logits = self.wg1(global_lang_emb)
                    hard_pairs = self.miner(global_logits, self.lang_group_label)
                    lang_group_loss = self.lang_loss_func(global_logits, self.lang_group_label, hard_pairs)
                else:
                    lang_group_loss = 0

                logits = self.wg2(token_input)

                orig_dtype = logits.dtype
                logits = logits.float() 
                logits = F.softmax(logits.masked_fill(~group_mask.bool(), -1e9),dim=1)
        else:
            orig_dtype = logits.dtype
        
        l_gate, combine_weights, dispatch_mask, metadata = htop2gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            second_expert_policy=self.second_expert_policy,
            normalize_gate_prob_before_dropping=self.normalize_gate_prob_before_dropping,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            capacity_factor=self.capacity_factor,
            batch_prioritized_routing=self.batch_prioritized_routing,
            has_tutel=has_tutel
        )

        l_aux = (l_gate, lang_group_loss)

        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata