from gc import collect
import sys

from regex import P
sys.path += ["./"]
import os
import math
import json
import torch
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List
import copy

logger = logging.getLogger(__name__)


class TextTokenIdsCache:
    def __init__(self, data_dir, prefix):
        meta = json.load(open(f"{data_dir}/{prefix}_meta"))
        self.total_number = meta['total_number']
        self.max_seq_len = meta['embedding_size']
        try:
            self.ids_arr = np.memmap(f"{data_dir}/{prefix}.memmap", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/{prefix}_length.npy")
        except FileNotFoundError:
            self.ids_arr = np.memmap(f"{data_dir}/memmap/{prefix}.memmap", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/memmap/{prefix}_length.npy")
        assert len(self.lengths_arr) == self.total_number
        
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        return self.ids_arr[item, :self.lengths_arr[item]]


class SequenceDataset(Dataset):
    def __init__(self, ids_cache, max_seq_length):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length
        
    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        seq_length = min(self.max_seq_length-1, len(input_ids)-1)
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        attention_mask = [1]*len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item,
        }
        return ret_val

class SeDR_SequenceDataset(Dataset):
    def __init__(self, ids_cache, max_seg_length, max_seg_num):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seg_length*max_seg_num
        
    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        input_ids = input_ids[:self.max_seq_length]

        ret_val = {
            "input_ids": input_ids,
            "id": item
        }
        return ret_val


class SubsetSeqDataset:
    def __init__(self, subset: List[int], ids_cache, max_seq_length):
        self.subset = sorted(list(subset))
        self.alldataset = SequenceDataset(ids_cache, max_seq_length)
        
    def __len__(self):  
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]


def load_rel(rel_path):
    reldict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        reldict[qid].append((pid))
    return dict(reldict)
    

def load_rank(rank_path):
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function(max_seq_length):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  



class TrainInbatchDataset(Dataset):
    def __init__(self, rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length):
        self.query_dataset = SequenceDataset(queryids_cache, max_query_length)
        self.doc_dataset = SequenceDataset(docids_cache, max_doc_length)
        self.reldict = load_rel(rel_file)
        self.qids = sorted(list(self.reldict.keys()))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        return query_data, passage_data


class TrainInbatchWithHardDataset(TrainInbatchDataset):
    def __init__(self, rel_file, rank_file, queryids_cache, 
            docids_cache, hard_num,
            max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length)
        self.rankdict = json.load(open(rank_file))
        assert hard_num > 0
        self.hard_num = hard_num

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        hardpids = random.sample(self.rankdict[str(qid)], self.hard_num)
        hard_passage_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
        return query_data, passage_data, hard_passage_data


class TrainInbatchWithRandDataset(TrainInbatchDataset):
    def __init__(self, rel_file, queryids_cache, 
            docids_cache, rand_num,
            max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length)
        assert rand_num > 0
        self.rand_num = rand_num

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        randpids = random.sample(range(len(self.doc_dataset)), self.rand_num)
        rand_passage_data = [self.doc_dataset[randpid] for randpid in randpids]
        return query_data, passage_data, rand_passage_data

import heapq
class PriorityQueue(object):
    def __init__(self):
        self._queue = [] 
        self._index = 0      
    
    def push(self, item, priority):
        #（priority, index, item)
        heapq.heappush(self._queue, (priority, self._index, item)) 
        self._index += 1
        
    def pop(self):
        # return the higest propority item
        return heapq.heappop(self._queue)[-1]

# Inbatch&Hardneg
class SeDR_Dataset():
    def __init__(self, rel_file, rank_file, queryids_cache, 
            docids_cache, hard_num,
            max_query_length, max_seg_length, max_seg_num, hardneg_topk = 100, max_bsize = None, tokenizer = None):
        # leave space for special tokens
        if max_seg_num > 1:
            max_seg_length -= 12
        else:
            max_seg_length -= 2
        # max_seg_length = max_seg_length - max_seg_num - 1
        self.query_dataset = SequenceDataset(queryids_cache, max_query_length)
        self.doc_dataset = SeDR_SequenceDataset(docids_cache, max_seg_length, max_seg_num)
        self.max_query_length = max_query_length
        self.max_seg_length = max_seg_length
        self.max_seg_num = max_seg_num
        self.max_bsize = max_bsize
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.cls_id = self.tokenizer.cls_token_id
            self.eos_id = self.tokenizer.eos_token_id
            self.pad_id = self.tokenizer.pad_token_id

        self.reldict = load_rel(rel_file)
        self.qids = sorted(list(self.reldict.keys()))
        self.rankdict = json.load(open(rank_file))
        self.hardneg_topk = hardneg_topk
        assert "train" not in rel_file or self.reldict[0][0] not in self.rankdict['0']
        assert hard_num > 0
        self.hard_num = hard_num

        # self.batch_buffer_queue = PriorityQueue()
        self.batch_buffer_queue = []
        self.ignore_len = min(int(0.3*max_seg_length),50)
        # self.ignore_len = int(0.3*max_seg_length)
        self.create_fuse_matrix()
    
    def create_fuse_matrix(self):
        self.segment_len_matrix = []
        self.segment_len_matrix.append([])
        init_que = [0]*(self.max_seg_num-1)
        range_que = list(range(self.max_seg_num))
        for seg_num in range(1,self.max_seg_num+1):
            # ids_mat = torch.LongTensor([init_que]*seg_num)
            mask_mat = torch.LongTensor([init_que]*seg_num)
            ids = []
            for ignore_idx in range(seg_num):
                ids.append(range_que[:ignore_idx]+range_que[ignore_idx+1:])
            ids_mat = torch.LongTensor(ids)
            ids_mat[:,(seg_num-1):] = 0
            mask_mat[:,:(seg_num-1)] = 1
            self.segment_len_matrix.append((ids_mat,mask_mat))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        # todo top200
        hardpids = random.sample(self.rankdict[str(qid)][:self.hardneg_topk], self.hard_num)
        hard_passage_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
        return query_data, passage_data, hard_passage_data
    
    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        if len(self.query_set) > 0 and len(self.batch_buffer_queue) < self.max_bsize:
            sample_num = min(len(self.query_set), self.max_bsize - len(self.batch_buffer_queue))
            batch.extend(random.sample(self.query_set, sample_num))
        self.query_set -= set(batch)
        if len(self.query_set) == 0 and len(self.batch_buffer_queue) == 0 and len(batch) == 0:
            raise StopIteration
        else:
            return self.collate_function([self[idx] for idx in batch])
             
        
    def reset(self):
        self.query_set = set(list(range(len(self.qids))))

    def collate_function(self,batch):
        # batch process
        for x in batch:
            arrlen = len(x[1]['input_ids'])
            seg_num = math.ceil(arrlen / self.max_seg_length)
            if seg_num > 1 and arrlen - (seg_num-1)*self.max_seg_length < self.ignore_len:
                seg_num -= 1
            self.batch_buffer_queue.append((x,seg_num))

        query_batch = []
        doc_batch = []
        hard_doc_batch = []
        max_hard_bsize = self.max_bsize * self.hard_num
        cur_batch = 0
        while len(self.batch_buffer_queue) > 0 and self.batch_buffer_queue[0][1] + cur_batch <= self.max_bsize:
            seg_num = self.batch_buffer_queue[0][1]
            query_data, passage_data, hard_passage_data = self.batch_buffer_queue.pop(0)[0]
            query_batch.append(query_data)
            passage_data['seg_num'] = seg_num
            doc_batch.append(passage_data)
            hard_doc_batch.extend(hard_passage_data)
            cur_batch += seg_num

        # compute the seg_num
        cur_hard_batch = 0
        for idx,doc in enumerate(hard_doc_batch):
            arrlen = len(doc['input_ids'])
            seg_num = math.ceil(arrlen / self.max_seg_length)
            if seg_num > 1 and arrlen -(seg_num-1)*self.max_seg_length < self.ignore_len:
                seg_num -= 1
            seg_num = min(seg_num, max_hard_bsize - cur_hard_batch + idx + 1 - len(hard_doc_batch))
            doc['seg_num'] = seg_num
            cur_hard_batch += seg_num
        assert max_hard_bsize >= cur_hard_batch

        doc_data, doc_ids = self.doc_collate_func(doc_batch)
        hard_doc_data, hard_doc_ids = self.doc_collate_func(hard_doc_batch)

        query_data, query_ids = self.query_collate_func(query_batch)
        rel_pair_mask = [[1 if docid not in self.reldict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        hard_pair_mask = [[1 if docid not in self.reldict[qid] else 0 
            for docid in hard_doc_ids ]
            for qid in query_ids]
        # query_num = len(query_data['input_ids'])
        # hard_num_per_query = self.hard_num
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "other_doc_ids":hard_doc_data['input_ids'],
            "other_doc_attention_mask":hard_doc_data['attention_mask'],
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            "hard_pair_mask":torch.FloatTensor(hard_pair_mask),
            "doc_fuse_mat": doc_data['fuse_mat'],
            "other_doc_fuse_mat":hard_doc_data['fuse_mat'],
            "doc_seg_sep": doc_data['seg_sep'],
            "other_doc_seg_sep":hard_doc_data['seg_sep'],
            }
        return input_data

    def query_collate_func(self,batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=self.pad_id, 
                dtype=torch.int64, length=self.max_query_length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=self.max_query_length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    
    def doc_collate_func(self,batch):
        ids = []
        input_ids_list = []
        mask_list = []
        seg_sep = [0]
        fuse_mat = []
        fuse_attn = []
        for x in batch:
            if 'seg_num' not in x:
                arrlen = len(x['input_ids'])
                seg_num = math.ceil(arrlen / self.max_seg_length)
                if seg_num > 1 and arrlen -(seg_num-1)*self.max_seg_length < self.ignore_len:
                    seg_num -= 1
                x['seg_num'] = seg_num
            ids.append(x['id'])
            input_ids = x['input_ids']
            for idx in range(x['seg_num']):
                start_i = idx*self.max_seg_length
                end_i = min(start_i + self.max_seg_length,len(input_ids))

                seg_input_ids = [self.cls_id] + input_ids[start_i: end_i] + [self.eos_id]
                mask = [1]*(len(seg_input_ids))
                
                input_ids_list.append(seg_input_ids)
                mask_list.append(mask)

            ids_mat, mask_mat = copy.deepcopy(self.segment_len_matrix[x['seg_num']])
            ids_mat += seg_sep[-1]
            fuse_mat.append(ids_mat)
            fuse_attn.append(mask_mat)
        
            seg_sep.append(seg_sep[-1]+x['seg_num'])
            
        fuse_mat = torch.cat(fuse_mat,dim=0)
        fuse_attn = torch.cat(fuse_attn,dim=0)
        input_ids = pack_tensor_2D(input_ids_list, default=self.pad_id, 
                dtype=torch.int64, length=self.max_seg_length+self.max_seg_num+1)

        attention_mask = pack_tensor_2D(mask_list, default=0, 
                dtype=torch.int64, length=self.max_seg_length+self.max_seg_num+1)
        
        if self.max_seg_num > 1:
            attention_mask[:,-(self.max_seg_num-1):] = fuse_attn
        else:
            fuse_mat = None
            # seg_sep = None
        
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "fuse_mat":fuse_mat,
            "seg_sep":seg_sep
        }
        return data, ids



class SeDR_Dataset_doc_inference(SeDR_Dataset):
    def __init__(self, docids_cache,max_seg_length, max_seg_num, max_bsize = None, tokenizer = None,  wo_fuse_mat = False):
        # leave space for special tokens
        if max_seg_num > 1:
            max_seg_length -= 12
        else:
            max_seg_length -= 2
        self.doc_dataset = SeDR_SequenceDataset(docids_cache, max_seg_length, max_seg_num)
        self.max_seg_length = max_seg_length
        self.max_seg_num = max_seg_num
        self.max_bsize = max_bsize
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.cls_id = self.tokenizer.cls_token_id
            self.eos_id = self.tokenizer.eos_token_id
            self.pad_id = self.tokenizer.pad_token_id            

        # self.batch_buffer_queue = PriorityQueue()
        self.batch_buffer_queue = []
        self.ignore_len = min(int(0.3*max_seg_length),50)
        self.create_fuse_matrix()
        self.doc_index = 0
        self.seg_num_list = []
        self.all_seg_num = self.compute_seg_num(docids_cache)

        self.wo_fuse_mat = wo_fuse_mat
    
    def compute_seg_num(self, docids_cache):
        all_seg_num = 0
        for arrlen in docids_cache.lengths_arr:
            seg_num = math.ceil(arrlen / self.max_seg_length)
            if seg_num > self.max_seg_num:
                seg_num = self.max_seg_num
            elif seg_num > 1 and arrlen - (seg_num-1)*self.max_seg_length < self.ignore_len:
                seg_num -= 1
            self.seg_num_list.append(seg_num)
            all_seg_num += seg_num
        return all_seg_num
        
    
    def __len__(self):
        return len(self.doc_dataset)

    def __getitem__(self, item):
        return self.doc_dataset[item]
    
    def __iter__(self):
        self.pbar = tqdm(total=len(self))
        return self

    def __next__(self):
        batch = []
        if self.doc_index < len(self) and len(self.batch_buffer_queue) < self.max_bsize:
            sample_num = min(len(self) - self.doc_index, self.max_bsize - len(self.batch_buffer_queue))
            batch.extend(range(self.doc_index,self.doc_index+sample_num))
            self.doc_index += sample_num
        if self.doc_index >= len(self) and len(self.batch_buffer_queue) == 0 and len(batch) == 0:
            self.pbar.close()
            raise StopIteration
        else:
            return self.collate_function([self[idx] for idx in batch])
    
    def collate_function(self,batch):
        # batch process
        for x in batch:
            self.batch_buffer_queue.append((x,self.seg_num_list[x['id']]))

        doc_batch = []
        cur_batch = 0
        ids = []
        while len(self.batch_buffer_queue) > 0 and self.batch_buffer_queue[0][1] + cur_batch <= self.max_bsize:
            seg_num = self.batch_buffer_queue[0][1]
            passage_data = self.batch_buffer_queue.pop(0)[0]
            doc_batch.append(passage_data)
            cur_batch += seg_num
            ids.extend([passage_data['id']]*seg_num)

        self.pbar.update(len(doc_batch))
        doc_data, doc_ids = self.doc_collate_func(doc_batch)

        # query_num = len(query_data['input_ids'])
        # hard_num_per_query = self.hard_num
        input_data = {
            "input_ids":doc_data['input_ids'],
            "attention_mask":doc_data['attention_mask'],
            "fuse_mat":doc_data['fuse_mat'],
            # addition
            "seg_sep":doc_data['seg_sep']
            }
        if self.wo_fuse_mat:
            input_data['fuse_mat']  = None

        return input_data,ids


def single_get_collate_function(max_seq_length, padding=False):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  


def dual_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            }
        return input_data
    return collate_function  


def triple_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        hard_doc_data, hard_doc_ids = doc_collate_func(sum([x[2] for x in batch], []))
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        hard_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in hard_doc_ids ]
            for qid in query_ids]
        query_num = len(query_data['input_ids'])
        hard_num_per_query = len(batch[0][2])
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "other_doc_ids":hard_doc_data['input_ids'].reshape(query_num, hard_num_per_query, -1),
            "other_doc_attention_mask":hard_doc_data['attention_mask'].reshape(query_num, hard_num_per_query, -1),
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            "hard_pair_mask":torch.FloatTensor(hard_pair_mask),
            }
        return input_data
    return collate_function 