from copy import deepcopy
import enum
import sys

from torch._C import dtype
sys.path += ['./']
import torch
from torch import nn
import transformers
# using the SeDR_model
from SeDR_Transformer import RobertaModel as SeDR_MODEL, RobertaPreTrainedModel,RobertaLayer
from SeDR_Transformer_global import RobertaModel as SeDR_global

# from transformers import RobertaModel
import torch.nn.functional as F
from torch.cuda.amp import autocast

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class BaseModelDot(EmbeddingMixin):
    def _text_encode(self, input_ids, attention_mask):
        # TODO should raise NotImplementedError
        # temporarily do this  
        return None 

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)


class RobertaDot(BaseModelDot, RobertaPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        RobertaPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) ==4 :
            config.return_dict = False
        self.roberta = SeDR_MODEL(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1


class SeDR_without_LateCache(RobertaDot):
    def __init__(self, config, max_seg_num, model_argobj=None):
        RobertaDot.__init__(self, config, model_argobj)
        self.max_seg_num = max_seg_num
        self.type_ids = list(range(max_seg_num))
    
    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            # return self.masked_mean(emb_all[0][:,:(-self.max_seg_num+1)], mask)
            print('wrong with use_mean')
            return None
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)
        
        segment_type = []
        start_idx = seg_sep[0]
        for end_idx in seg_sep[1:]:
            segment_type.extend(self.type_ids[0:(end_idx - start_idx)])
            start_idx = end_idx

        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = fuse_mat, segment_type = torch.LongTensor(segment_type).to(input_ids.device))
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        doc1 = self.norm(self.embeddingHead(full_emb))
        
        reps_mat = torch.zeros(len(seg_sep)-1, self.max_seg_num, doc1.shape[-1], device=doc1.device)
        
        start_idx = seg_sep[0]
        for batch_idx,end_idx in enumerate(seg_sep[1:]):
            reps_mat[batch_idx,:(end_idx-start_idx)] = doc1[start_idx:end_idx]
            start_idx = end_idx
        return reps_mat
    
    def body_emb_inference(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        segment_type = None
        if seg_sep is not None:
            segment_type = []
            start_idx = seg_sep[0]
            for end_idx in seg_sep[1:]:
                segment_type.extend(self.type_ids[0:(end_idx - start_idx)])
                start_idx = end_idx
            segment_type = torch.LongTensor(segment_type).to(input_ids.device)

        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = fuse_mat, segment_type = segment_type)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        doc1 = self.norm(self.embeddingHead(full_emb))

        return doc1

    def _text_encode(self, input_ids, attention_mask, fuse_mat = None, segment_type = None):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = fuse_mat, segment_type = segment_type)
        return outputs1
    
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None, doc_fuse_mat = None, doc_seg_sep = None, other_doc_fuse_mat = None, other_doc_seg_sep = None):
        query_embs = self.query_emb(input_query_ids, query_attention_mask)
        doc_embs = self.body_emb(input_doc_ids, doc_attention_mask, fuse_mat= doc_fuse_mat, seg_sep= doc_seg_sep)
        other_doc_embs = self.body_emb(other_doc_ids, other_doc_attention_mask, fuse_mat= other_doc_fuse_mat, seg_sep= other_doc_seg_sep)

        batch_size = query_embs.shape[0]
        
        with autocast(enabled=False):
            # b 1 1 h
            query_embs = query_embs.unsqueeze(1).unsqueeze(1)
            # b 1 1 h @ 1 b h m -> b b 1 m -> b b
            batch_scores = (query_embs @ doc_embs.permute(0, 2, 1).unsqueeze(0)).max(3).values.squeeze(-1)
            other_batch_scores = (query_embs @ other_doc_embs.permute(0, 2, 1).unsqueeze(0)).max(3).values.squeeze(-1)
            score = torch.cat([batch_scores, other_batch_scores], dim=-1)

            if rel_pair_mask is not None and hard_pair_mask is not None:
                rel_pair_mask = rel_pair_mask + torch.eye(score.size(0), dtype=score.dtype, device=score.device)
                mask = torch.cat([rel_pair_mask, hard_pair_mask], dim=-1)
                score = score.masked_fill(mask==0, -10000)
            
            labels = torch.arange(start=0, end=score.shape[0],
                                  dtype=torch.long, device=score.device)
            loss = F.cross_entropy(score, labels)
            acc = torch.sum(score.max(1).indices == labels) / score.size(0)

            return (loss, acc)
    
    
cache_size = 50
class SeDR(SeDR_without_LateCache):
    def __init__(self, config, max_seg_num, model_argobj=None):
        SeDR_without_LateCache.__init__(self, config, max_seg_num, model_argobj=model_argobj)
        global cache_size
        self.cache_size = cache_size
        self.doc_cache = []
        self.other_doc_cache = []
        self.query_cache = []

    def set_cache_size(x):
        global cache_size
        cache_size = x

    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None, doc_fuse_mat = None, doc_seg_sep = None, other_doc_fuse_mat = None, other_doc_seg_sep = None):
        query_embs = self.query_emb(input_query_ids, query_attention_mask)
        doc_embs = self.body_emb(input_doc_ids, doc_attention_mask, fuse_mat= doc_fuse_mat, seg_sep= doc_seg_sep)
        other_doc_embs = self.body_emb(other_doc_ids, other_doc_attention_mask, fuse_mat= other_doc_fuse_mat, seg_sep= other_doc_seg_sep)

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

            return (loss, acc)

class STAR_MaxP(SeDR_without_LateCache):
    def __init__(self, config, max_seg_num, model_argobj=None):
        SeDR_without_LateCache.__init__(self, config, max_seg_num, model_argobj=model_argobj)
    
    def body_emb(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)

        # not use the fuse_mat
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        doc1 = self.norm(self.embeddingHead(full_emb))
        
        reps_mat = torch.zeros(len(seg_sep)-1, self.max_seg_num, doc1.shape[-1], device=doc1.device)
        
        start_idx = seg_sep[0]
        for batch_idx,end_idx in enumerate(seg_sep[1:]):
            reps_mat[batch_idx,:(end_idx-start_idx)] = doc1[start_idx:end_idx]
            start_idx = end_idx
        return reps_mat

    def body_emb_inference(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)

        # not use the fuse_mat
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        doc1 = self.norm(self.embeddingHead(full_emb))

        return doc1
    
    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None, doc_fuse_mat = None, doc_seg_sep = None, other_doc_fuse_mat = None, other_doc_seg_sep = None):
        query_embs = self.query_emb(input_query_ids, query_attention_mask)
        doc_embs = self.body_emb(input_doc_ids, doc_attention_mask, fuse_mat= doc_fuse_mat, seg_sep= doc_seg_sep)

        batch_size = query_embs.shape[0]

        with autocast(enabled=False):
            # b 1 1 h
            query_embs = query_embs.unsqueeze(1).unsqueeze(1)
            # b 1 1 h @ 1 b h m -> b b 1 m -> b b
            batch_scores = (query_embs @ doc_embs.permute(0, 2, 1).unsqueeze(0)).max(3).values.squeeze(-1)

            single_positive_scores = torch.diagonal(batch_scores, 0)
            positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)
            
            if rel_pair_mask is None:
                rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)              
            batch_scores = batch_scores.reshape(-1)
            logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                    batch_scores.unsqueeze(1)], dim=1) 

            lsm = F.log_softmax(logit_matrix, dim=1)
            loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
            first_loss, first_num = loss.sum(), rel_pair_mask.sum()

        if other_doc_ids is None:
            return (first_loss/first_num,)

        other_doc_embs = self.body_emb(other_doc_ids, other_doc_attention_mask, fuse_mat= other_doc_fuse_mat, seg_sep= other_doc_seg_sep)

        with autocast(enabled=False):
            # other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
            # b 1 1 h @ 1 b h m -> b b 1 m -> b b
            other_batch_scores = (query_embs @ other_doc_embs.permute(0, 2, 1).unsqueeze(0)).max(3).values.squeeze(-1)
            other_batch_scores = other_batch_scores.reshape(-1)
            positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_embs.size(0)).reshape(-1)
            other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                    other_batch_scores.unsqueeze(1)], dim=1)  
            # print(logit_matrix)
            other_lsm = F.log_softmax(other_logit_matrix, dim=1)
            other_loss = -1.0 * other_lsm[:, 0]
            # print(loss)
            # print("\n")
            if hard_pair_mask is not None:
                hard_pair_mask = hard_pair_mask.reshape(-1)
                other_loss = other_loss * hard_pair_mask
                second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
            else:
                second_loss, second_num = other_loss.sum(), len(other_loss)

        return ((first_loss+second_loss)/(first_num+second_num), torch.sum(positive_scores > batch_scores)/first_num)


class SeDR_MaxP(SeDR):
    def __init__(self, config, max_seg_num, model_argobj=None):
        SeDR.__init__(self, config, max_seg_num, model_argobj=model_argobj)
    
    def body_emb(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)

        # not use the fuse_mat
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        doc1 = self.norm(self.embeddingHead(full_emb))
        
        reps_mat = torch.zeros(len(seg_sep)-1, self.max_seg_num, doc1.shape[-1], device=doc1.device)
        
        start_idx = seg_sep[0]
        for batch_idx,end_idx in enumerate(seg_sep[1:]):
            reps_mat[batch_idx,:(end_idx-start_idx)] = doc1[start_idx:end_idx]
            start_idx = end_idx
        return reps_mat


class SeDR_Transformer_Head(SeDR):
    def __init__(self, config, max_seg_num, model_argobj=None):
        # config = deepcopy(config)
        SeDR.__init__(self, config, max_seg_num, model_argobj=model_argobj)
        self.trans_head =  RobertaLayer(config)
    
    def body_emb(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)

        # not use the fuse_mat
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        
        reps_mat = torch.zeros(len(seg_sep)-1, self.max_seg_num, full_emb.shape[-1], device=full_emb.device)
        
        start_idx = seg_sep[0]
        for batch_idx,end_idx in enumerate(seg_sep[1:]):
            reps_mat[batch_idx,:(end_idx-start_idx)] = full_emb[start_idx:end_idx]
            start_idx = end_idx
        
        att_mask = torch.ones((reps_mat.shape[0],reps_mat.shape[1]),device=full_emb.device)
        att_mask = self.roberta.get_extended_attention_mask(att_mask, reps_mat.shape[:-1] ,device=full_emb.device)
        reps_mat = self.trans_head(reps_mat,att_mask)[0]

        # outputs head
        reps_mat = self.norm(self.embeddingHead(reps_mat))

        return reps_mat

    def body_emb_inference(self, input_ids, attention_mask, fuse_mat=None, seg_sep=None):

        # not use the fuse_mat
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        
        reps_mat = torch.zeros(len(seg_sep)-1, self.max_seg_num, full_emb.shape[-1], device=full_emb.device)
        
        start_idx = seg_sep[0]
        for batch_idx,end_idx in enumerate(seg_sep[1:]):
            reps_mat[batch_idx,:(end_idx-start_idx)] = full_emb[start_idx:end_idx]
            start_idx = end_idx
        
        att_mask = torch.ones((reps_mat.shape[0],reps_mat.shape[1]),device=full_emb.device)
        att_mask = self.roberta.get_extended_attention_mask(att_mask, reps_mat.shape[:-1] ,device=full_emb.device)
        reps_mat = self.trans_head(reps_mat,att_mask)[0]

        # outputs head
        # todo not tuple
        reps_mat = self.norm(self.embeddingHead(reps_mat))

        reps = reps_mat.new_zeros(full_emb.shape[0], full_emb.shape[1])
        start_idx = seg_sep[0]
        for batch_idx,end_idx in enumerate(seg_sep[1:]):
            reps[start_idx:end_idx] = reps_mat[batch_idx,:(end_idx-start_idx)]
            start_idx = end_idx

        return reps

class SeDR_Global_Attention(SeDR):
    def __init__(self, config, max_seg_num, model_argobj=None):
        # config = deepcopy(config)
        SeDR.__init__(self, config, max_seg_num, model_argobj=model_argobj)
        self.roberta = SeDR_global(config, add_pooling_layer=False)

K = 0
class STAR_Multi(STAR_MaxP):
    def __init__(self, config, max_seg_num, model_argobj=None):
        global K
        SeDR_MaxP.__init__(self, config, K, model_argobj=model_argobj)
        self.K = K
        self.views = None
        self.views_attn = None
    
    @staticmethod
    def set_K(k_input):
        global K
        K = k_input

    def set_prefix(self):
        # expand the unused tokens to embedding layer
        ids_start = self.roberta.embeddings.word_embeddings.weight.shape[0]
        assert ids_start == 50265 or ids_start == 50265 + self.K
        if ids_start == 50265:
            self.roberta.resize_token_embeddings(ids_start + self.K)
        else:
            ids_start -= self.K
        self.views = torch.LongTensor(list(range(ids_start,ids_start+ self.K))).unsqueeze(0).cuda()
        self.views_attn = torch.LongTensor([1]*self.K).unsqueeze(0).cuda()
        assert ids_start == 50265
    
    def body_emb(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if self.views is None:
            self.set_prefix()

        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)
        
        # replace [cls] with [view1, view2, ...]
        input_ids = torch.cat((self.views.expand(input_ids.shape[0],-1),input_ids[:,1:]),dim = 1)
        attention_mask = torch.cat((self.views_attn.expand(input_ids.shape[0],-1),attention_mask[:,1:]),dim = 1)
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        
        # firstK
        emb = outputs1[0][:,:self.K]
        # output head
        emb = self.norm(self.embeddingHead(emb))
        return emb
    
    def body_emb_inference(self, input_ids, attention_mask, fuse_mat = None, seg_sep = None):
        if self.views is None:
            self.set_prefix()

        if seg_sep is None:
            return self.query_emb(input_ids, attention_mask)
        
        # replace [cls] with [view1, view2, ...]
        input_ids = torch.cat((self.views.expand(input_ids.shape[0],-1),input_ids[:,1:]),dim = 1)
        attention_mask = torch.cat((self.views_attn.expand(input_ids.shape[0],-1),attention_mask[:,1:]),dim = 1)
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask, fuse_mat = None)
        # firstK
        emb = outputs1[0][:,:self.K]
        # output head
        emb = self.norm(self.embeddingHead(emb))
        emb = emb.reshape(-1,emb.shape[-1])
        return emb

