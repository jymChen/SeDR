import logging
import sys
sys.path.append("./")
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM
from transformers.modeling_longformer import LongformerConfig,LongformerSelfAttention
from SeDR_Longformer.model import SeDR_Longformer
from torch import nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

def create_long_model(model, save_model_to, attention_window, max_pos):
    # model = RobertaForMaskedLM.from_pretrained('roberta-base')
    # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    # tokenizer.model_max_length = max_pos
    # tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    # model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    # tokenizer.save_pretrained(save_model_to)
    # return model, tokenizer
    return model

def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model

if __name__ == "__main__":
    attention_window = 256
    max_pos = 2048
    # start to change Roberta checkpoint to the Longformer
    roberta_checkpoint = RobertaForMaskedLM.from_pretrained('./data/star')
    model_path = './data/starlongformer'
    
    model = create_long_model(
        model = roberta_checkpoint, save_model_to = model_path , attention_window= attention_window, max_pos= max_pos)
    
    model = RobertaLongForMaskedLM.from_pretrained(model_path)
    logger.info(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

    # build the star_longformer
    model_class = SeDR_Longformer
    config = LongformerConfig.from_pretrained(model_path) 
    config.attention_window =[256]*12
    config.attention_dilation = [1]*12
    config.attention_mode = 'sliding_chunks'
    config.autoregressive = False
    model = model_class.from_pretrained(
        model_path,
        max_seg_num=4,
        config=config
    )
    model_now_dict = model.state_dict()
    state_dict = torch.load('./data/star/pytorch_model.bin', map_location="cpu")
    new_state_dict = {k: v for k, v in state_dict.items() if "roberta." not in k}
    print(new_state_dict.keys())
    model_now_dict.update(new_state_dict)
    model.load_state_dict(model_now_dict)
    model.save_pretrained(model_path)