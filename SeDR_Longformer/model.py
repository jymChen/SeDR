from copy import deepcopy
import enum
import sys

from torch._C import dtype
sys.path += ['./']
import torch
from torch import nn
import transformers
import torch.nn.functional as F
from torch.cuda.amp import autocast
from SeDR_Longformer.longformerModel import Longformer as LongformerModel

# from transformers.modeling_longformer import LongformerModel
from transformers.modeling_roberta import BertPreTrainedModel

def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask

cache_size = 100
class SeDR_Longformer(BertPreTrainedModel):
    def __init__(self, config, max_seg_num ,model_argobj=None):
        self.pad_to_window_size = pad_to_window_size
        BertPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) ==4 :
            config.return_dict = False
        self.max_seg_num = max_seg_num
        self.roberta = LongformerModel(config)
        self.attention_window = config.attention_window
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        
        # stride
        stride = 512
        self.cls_pos = [0]
        for _ in range(max_seg_num - 1):
            self.cls_pos.append(self.cls_pos[-1] + stride)
        assert len(self.cls_pos) == max_seg_num

        self.apply(self._init_weights)
        
        global cache_size
        self.cache_size = cache_size
        self.doc_cache = []
        self.other_doc_cache = []
        self.query_cache = []
    
    def query_emb(self, input_ids, attention_mask):
        # global  attention
        attention_mask[:, 0] =  2
        input_ids, attention_mask = self.pad_to_window_size(
        input_ids, attention_mask, self.attention_window[0], 1)

        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:,0]
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        cls_pos = torch.LongTensor([pos for pos in self.cls_pos if pos < attention_mask.shape[1]-1]).to(input_ids.device)
        input_ids[:, cls_pos] = input_ids[0,0].clone()
        attention_mask[:, cls_pos] = (attention_mask[:, cls_pos]!=0)*2
        
        input_ids, attention_mask = self.pad_to_window_size(
        input_ids, attention_mask, self.attention_window[0], 1)

        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)

        full_emb = outputs1[0][:, cls_pos]

        full_emb = full_emb * ((attention_mask[:, cls_pos] != 0)*1).unsqueeze(-1)
        
        if full_emb.shape[1] < len(self.cls_pos):
            reps_mat = full_emb.new_zeros(full_emb.shape[0], len(self.cls_pos), full_emb.shape[-1])
            reps_mat[:,:full_emb.shape[1]] = full_emb
            full_emb = reps_mat

        doc_bems = self.norm(self.embeddingHead(full_emb))
        return doc_bems
    
    def body_emb_inference(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        cls_pos = torch.LongTensor(self.cls_pos).to(input_ids.device)
        input_ids[:, cls_pos] = input_ids[0,0].clone()
        attention_mask[:, cls_pos] = (attention_mask[:, cls_pos]!=0)*2
        
        input_ids, attention_mask = self.pad_to_window_size(
        input_ids, attention_mask, self.attention_window[0], 1)

        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)

        full_emb = outputs1[0][:, cls_pos]
        doc_bems = self.norm(self.embeddingHead(full_emb))

        doc_bems = doc_bems * ((attention_mask[:, cls_pos] != 0)*1).unsqueeze(-1)
        return doc_bems.reshape(-1,doc_bems.shape[-1])

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1

    def set_cache_size(x):
        global cache_size
        cache_size = x

    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None, doc_fuse_mat = None, doc_seg_sep = None, other_doc_fuse_mat = None, other_doc_seg_sep = None):
        query_embs = self.query_emb(input_query_ids, query_attention_mask)
        doc_embs = self.body_emb(input_doc_ids, doc_attention_mask)
        other_doc_embs = self.body_emb(other_doc_ids, other_doc_attention_mask)

        batch_size = query_embs.shape[0]
        with autocast(enabled=False):
            cur_cache_size = len(self.query_cache)
            if cur_cache_size > 0:
                cache_query_embs = torch.stack(self.query_cache,dim=0)
                query_embs_cat = torch.cat((query_embs, cache_query_embs), dim=0)
                
                cache_doc_embs = torch.stack(self.doc_cache, dim=0)
                cache_other_doc_embs = torch.stack(self.other_doc_cache, dim=0)
                doc_embs_cat = torch.cat((doc_embs, cache_doc_embs, other_doc_embs, cache_other_doc_embs))
            else:
                query_embs_cat = query_embs
                doc_embs_cat = torch.cat((doc_embs, other_doc_embs))

            # b 1 1 h
            query_embs_cat = query_embs_cat.unsqueeze(1).unsqueeze(1)
            # b 1 1 h @ 1 b h m -> b b 1 m -> b b
            score = (query_embs_cat @ doc_embs_cat.permute(0, 2, 1).unsqueeze(0)).max(3).values.squeeze(-1)

            if rel_pair_mask is not None and hard_pair_mask is not None:
                mask = torch.ones(score.shape, dtype=score.dtype, device=score.device)
                rel_pair_mask = rel_pair_mask + torch.eye(batch_size, dtype=score.dtype, device=score.device)
                mask[:batch_size, :batch_size] = rel_pair_mask
                mask[:batch_size, (batch_size+cur_cache_size):(batch_size*2+cur_cache_size)] = hard_pair_mask
                score = score.masked_fill(mask==0, -10000)
            
            labels = torch.arange(start=0, end=score.shape[0],
                                  dtype=torch.long, device=score.device)
            
            loss = F.cross_entropy(score, labels)
            acc = torch.sum(score[:batch_size].max(1).indices == labels[:batch_size]) / batch_size

            self.query_cache.extend(list(query_embs.detach()))
            self.doc_cache.extend(list(doc_embs.detach()))
            self.other_doc_cache.extend(list(other_doc_embs.detach()))
            
            self.query_cache = self.query_cache[-self.cache_size:]
            self.doc_cache = self.doc_cache[-self.cache_size:]
            self.other_doc_cache = self.other_doc_cache[-self.cache_size:]

            return (loss,  acc)
