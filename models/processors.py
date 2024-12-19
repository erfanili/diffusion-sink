import torch
import torch.nn.functional as F 
from typing import Optional,List, Dict
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention
import math
import numpy as  np 
import matplotlib.pyplot as plt
from pytorch_metric_learning import distances, losses




import torch
import torch.nn.functional as F 
from typing import Optional
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import *
import math
import torch.nn as nn



def get_processor_class(processor_name):
    processor_classes = {
        'processor_sd_x': AttnProcessorSDX,
        'processor_joint_x': JointAttnProcessor2_0X,
        'processor_flux_x' : FluxAttnProcessor2_0X,
        'processor_flux' : FluxAttnProcessor2_0,
}

    return processor_classes[processor_name]



class FluxAttnProcessor2_0X:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
    
    
    def get_attention_scores(self,
            query: torch.Tensor, key: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,  scale = None
        ) -> torch.Tensor:
            dtype = query.dtype
            
            # attention_scores = torch.einsum('bni,bmi->bnm',query,key)*scale
            if attention_mask is None:
                baddbmm_input = torch.empty(
                    query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                )
                beta = 0
            else:
                baddbmm_input = attention_mask
                beta = 1
            # breakpoint()
            attention_scores = torch.baddbmm(
                baddbmm_input,
                query,
                key.transpose(-1, -2),
                beta=beta,
                alpha=scale,
            )
            del baddbmm_input
            
            attention_probs = attention_scores.softmax(dim=-1)
            
            attention_probs = attention_probs.to(dtype)

            return attention_probs
        
        
    def linear_attn(self,query,key,value):
        query = query.squeeze()
        key = key.squeeze()
        key -= torch.mean(key,dim = 1, keepdim=True)
        value = value.squeeze()
        # breakpoint()
        # Optional: Apply kernel transformation for positive scores
        # Linear attention computation
        kv = torch.einsum('blm,blk->bkm', key, value)/4608  # [24, 128, 128]
        output = torch.einsum('blm,bkm->blk', query, kv)/11  +torch.mean(value,dim = 1,keepdim = True)# [24, 4608, 128]

        # normalization = torch.einsum('blm,blm->bl', query, key.sum(dim=1, keepdim=True))  # [24, 4608]
        # normalization = normalization.unsqueeze(-1)  # [24, 4608, 1]

        # Output attention
        # output = z / normalization  # [24, 4608, 128]

        # Reshape back to original dimensions
        output = output.view(1, 24, 4608, 128)
        
        return output
        # [1, 24, 4608, 128]
    
    
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # print('yes')
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)/10
            key = apply_rotary_emb(key, image_rotary_emb)
        # breakpoint()
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states2 = self.linear_attn(query,key,value)
        # dev = 'cuda:2'
        ####
        breakpoint()
        # attention_probs = self.get_attention_scores(query.squeeze().to(dev), key.squeeze().to(dev), attention_mask  , scale = attn.scale)
        self.attn_data_x = torch.zeros(1)
        # torch.mean(attention_probs,dim = 0).cpu()
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states



class JointAttnProcessor2_0X:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    


    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        
        
        
        

        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states












class AttnProcessorSDX:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,positive_prompt:bool = True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.positive_prompt = positive_prompt

    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # breakpoint()
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)

        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        batch = 1 if self.positive_prompt else 0

        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])

        attention_probs = attn_re.reshape(batch_size* attn.heads, shapes[-2], shapes[-1])


        hidden_states = torch.bmm(attention_probs, value)


        hidden_states = attn.batch_to_head_dim(hidden_states)

        ######_x
        # if attention_probs.size()[-1] in [120,77]:
        #     self.attn_data_x = torch.mean(attn_re[batch],dim=0
        if attention_probs.size()[-1] not in [120,77]:
            self.attn_data_x = torch.mean(attn_re[batch],dim = 0)
            
        ######_x


        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)


        # attn.residual_connection = True
        if attn.residual_connection:
        
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states




