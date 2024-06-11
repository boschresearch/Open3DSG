# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import InstructBlipProcessor
from open3dsg.models.custom_instruct_blip import InstructBlipForConditionalGeneration, BaseModelOutput, InstructBlipEncoderLayer

class Dict2Class(object): 
    def __init__(self, my_dict): 
        for key in my_dict: setattr(self, key, my_dict[key]) 

class InstructBlipEncoder(nn.Module):

    def __init__(self,layers=3):
        super().__init__()
        config = {}
        config["attention_dropout"] = 0.0; config["hidden_act"] = "gelu"; config["hidden_size"] = 1408; 
        config["image_size"] = 224; config["initializer_range"] = 1e-10; config["intermediate_size"] = 6144; 
        config["layer_norm_eps"] = 1e-06; config["model_type"] = "instructblip_vision_model"; 
        config["num_attention_heads"] = 16; config["num_hidden_layers"] = 39; config["patch_size"] = 14; 
        config["qkv_bias"] = True; config["transformers_version"] = "4.32.1"; 
        config = Dict2Class(config)
        self.config = config
        self.layers = nn.ModuleList([InstructBlipEncoderLayer(config) for _ in range(layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
