# coding=utf-8
import argparse
import subprocess
import sys

import json
sys.path.append("./")
import faiss
import logging
import os
import numpy as np
import torch
from transformers import RobertaConfig
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from SeDR_Longformer.model import SeDR_Longformer
from star_tokenizer import RobertaTokenizer
import tempfile
from dataset import (
    TextTokenIdsCache, load_rel, pack_tensor_2D, SequenceDataset,
    single_get_collate_function, SeDR_Dataset_doc_inference
)
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
     convert_index_to_gpu
)
from msmarco_eval import compute_metrics2, load_reference, load_candidate
logger = logging.Logger(__name__)
from timeit import default_timer as timer
# from SeDR.adore_train import eval_model, TrainQueryDataset, get_collate_function_query


def get_collate_function_query(max_seq_length):
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
        qids = [x['id'] for x in batch]
        all_rel_pids = [x["rel_ids"] for x in batch]
        return data, all_rel_pids, qids
    return collate_function  

class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, 
            rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_ids'] = self.reldict[item]
        return ret_val

def compute_metrics_from_files(path_to_reference, ranking, trec_eval_bin_path):
    trec_run_fd, trec_run_path = tempfile.mkstemp(text=True)
    try:
        with os.fdopen(trec_run_fd, 'w') as tmp:
            for qid, neighbors in ranking.items():
                for idx, pid in enumerate(neighbors):
                    rank = idx + 1
                    tmp.write(f"{qid} Q0 {pid} {rank} {1/rank} System\n")
        result = subprocess.check_output([
            trec_eval_bin_path, "-c", "-mndcg_cut.10", "-mrecall.100", path_to_reference, trec_run_path])
        logger.info(f'test result {result}')
        return str(result)
    finally:
        os.remove(trec_run_path)

def eval_model(args, model, index, passage_ids):
    test_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "test-query"),
        os.path.join(args.preprocess_dir, "test-qrel.tsv"),
        args.max_seq_length
    )
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset) , 
        batch_size=args.train_batch_size, collate_fn=get_collate_function_query(args.max_seq_length))
    
    dev_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "dev-query"),
        os.path.join(args.preprocess_dir, "dev-qrel.tsv"),
        args.max_seq_length
    )
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset) , 
        batch_size=args.train_batch_size, collate_fn=get_collate_function_query(args.max_seq_length))
    
    # metric
    MaxMRRRank = 100
    Recall_cutoff = [5,10,30,50,100]

    ress = {}
    # dev
    model.eval() 
    epoch_iterator = tqdm(dev_dataloader, desc="Dev eval")
    ranking = {}
    for batch, all_rel_pids, qids in epoch_iterator:
        batch = {k:v.to(args.model_device) for k, v in batch.items()}           
        query_embeddings = model.query_emb(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"])
        I_nearest_neighbor = index.search(
                query_embeddings.detach().cpu().numpy(), 1024)[1]
        I_nearest_neighbor = passage_ids[I_nearest_neighbor]
        for ii,res_seq in enumerate(I_nearest_neighbor.tolist()):
            res_topk = []
            for pid in res_seq:
                if pid not in res_topk:
                    res_topk.append(pid)
                # note this param
                if len(res_topk) == 101:
                    break
            ranking[qids[ii]] = res_topk
    qrels = dev_dataset.reldict
    MRR, Recall = compute_metrics2(qrels,ranking,MRR_cutoff=MaxMRRRank,Recall_cutoff=Recall_cutoff)
    logger.info(f'MRR {MRR} Recall{Recall}')
    ress['dev_MRR'] = MRR
    for i, k in enumerate(Recall_cutoff):
        ress[f'dev_Recall@{k}'] = Recall[i]
    


    ranking = {}
    for batch, all_rel_pids, qids in test_dataloader:
        batch = {k:v.to(args.model_device) for k, v in batch.items()}           
        query_embeddings = model.query_emb(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"])
        I_nearest_neighbor = index.search(
                query_embeddings.detach().cpu().numpy(), 1024)[1]
        I_nearest_neighbor = passage_ids[I_nearest_neighbor]
        for ii,res_seq in enumerate(I_nearest_neighbor.tolist()):
            res_topk = []
            for pid in res_seq:
                if pid not in res_topk:
                    res_topk.append(pid)
                # note this param
                if len(res_topk) == 100:
                    break
            ranking[qids[ii]] = res_topk
    # import json
    # json.dump(ranking, open('./SeDR_data/doc/test_ranking.json', 'w'))
    testres = compute_metrics_from_files(os.path.join(args.preprocess_dir, "test-qrel.tsv") , ranking, "./trec_eval/trec_eval")
    ress['test_res'] = testres

    # 2020 DL track
    test_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "2020test-query"),
        os.path.join(args.preprocess_dir, "2020test-qrel.tsv"),
        args.max_seq_length
    )
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset) , 
        batch_size=args.train_batch_size, collate_fn=get_collate_function_query(args.max_seq_length))
    ranking = {}
    for batch, all_rel_pids, qids in test_dataloader:
        batch = {k:v.to(args.model_device) for k, v in batch.items()}           
        query_embeddings = model.query_emb(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"])
        I_nearest_neighbor = index.search(
                query_embeddings.detach().cpu().numpy(), 1024)[1]
        I_nearest_neighbor = passage_ids[I_nearest_neighbor]
        for ii,res_seq in enumerate(I_nearest_neighbor.tolist()):
            res_topk = []
            for pid in res_seq:
                if pid not in res_topk:
                    res_topk.append(pid)
                # note this param
                if len(res_topk) == 100:
                    break
            ranking[qids[ii]] = res_topk
    # import json
    # json.dump(ranking, open('./SeDR_data/doc/test_ranking.json', 'w'))
    testres = compute_metrics_from_files(os.path.join(args.preprocess_dir, "2020test-qrel.tsv") , ranking, "./trec_eval/trec_eval")
    ress['2020test_res'] = testres

    return MRR,ress 

def index_retrieve(index, query_embeddings, args, batch=None):
    topk = args.topk
    keeptopk = min(args.topk * args.max_seg_num,1000)
    
    print("Query Num", len(query_embeddings))
    start = timer()
    query_offset_base = 0
    pbar = tqdm(total=len(query_embeddings))
    nearest_neighbors = []
    while query_offset_base < len(query_embeddings):
        batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
        batch_nn = index.search(batch_query_embeddings, keeptopk)[1]
        for res_seq in batch_nn.tolist():
            res_topk = []
            for pid in res_seq:
                if pid not in res_topk:
                    res_topk.append(pid)
                # if len(res_topk) == topk:
                #     break
            nearest_neighbors.append(res_topk)
        query_offset_base += len(batch_query_embeddings)
        pbar.update(len(batch_query_embeddings))
    pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors

def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size*args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    
    write_index = 0
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for step, (inputs, ids) in enumerate(test_dataloader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
        with torch.no_grad():
            logits = model.query_emb(**inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)

def doc_prediction(model, args, doc_dataset, embedding_memmap, ids_memmap):
    os.makedirs(args.output_dir, exist_ok=True)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    write_index = 0
    for step, (inputs, ids) in enumerate(doc_dataset):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model.body_emb_inference(**inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap), f"write_index:{write_index} {len(embedding_memmap)} {len(ids_memmap)}"


def query_inference(model, args, embedding_size):
    if os.path.exists(args.query_memmap_path):
        print(f"{args.query_memmap_path} exists, skip inference")
        return
    query_collator = single_get_collate_function(args.max_query_length)
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )
    query_memmap = np.memmap(args.query_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))
    queryids_memmap = np.memmap(args.queryids_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(query_dataset), ))
    try:
        prediction(model, query_collator, args,
                query_dataset, query_memmap, queryids_memmap, is_query=True)
    except:
        subprocess.check_call(["rm", args.query_memmap_path])
        subprocess.check_call(["rm", args.queryids_memmap_path])
        raise


def doc_inference(model, args, embedding_size, tokenizer):
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages")
    doc_dataset = SeDR_Dataset_doc_inference(ids_cache, args.max_doc_length, 1, max_bsize = args.max_bsize, tokenizer = tokenizer)
    
    if os.path.exists(args.doc_memmap_path):
        print(f"{args.doc_memmap_path} exists!! note it!")
        return

    # assert not os.path.exists(args.doc_memmap_path)
    doc_memmap = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="w+", shape=(doc_dataset.all_seg_num, embedding_size))
    docid_memmap = np.memmap(args.docid_memmap_path, 
        dtype=np.int32, mode="w+", shape=(doc_dataset.all_seg_num, ))
    try:
        doc_prediction(model, args, doc_dataset, doc_memmap, docid_memmap)
    except:
        subprocess.check_call(["rm", args.doc_memmap_path])
        subprocess.check_call(["rm", args.docid_memmap_path])
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--max_seg_num", type=int, default=4)
    parser.add_argument("--max_bsize", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_class", type=str, default="SeDR")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gen_hardneg", action="store_true")
    parser.add_argument("--faiss_gpus", type=int, default=None, nargs="+")
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.preprocess_dir = f"./data/preprocess"
    # args.model_path = f"./data/models/{args.model}"
    args.output_dir = f"./data/evaluate/{os.path.split(args.model_path)[-1]}-inf{args.max_doc_length}-{args.max_seg_num}"

    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    args.fass_index_path = os.path.join(args.output_dir, "index.faiss")
    # logger.info(args)
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)

    config = RobertaConfig.from_pretrained(args.model_path, gradient_checkpointing=False)

    model_class = SeDR_Longformer

    # config.max_seg_num = data_args.max_seg_num
    config.attention_window =[256]*12
    config.attention_dilation = [1]*12
    config.attention_mode = 'sliding_chunks'
    # config.attention_mode = 'tvm'
    config.autoregressive = False

    model = model_class.from_pretrained(args.model_path, config=config,max_seg_num = args.max_seg_num)

    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case = True, cache_dir=None)

    doc_inference(model, args, output_embedding_size, tokenizer)
    
    # model = None
    torch.cuda.empty_cache()

    doc_embeddings = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, 
        dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)
    
    index = construct_flatindex_from_embeddings(doc_embeddings)
    index = convert_index_to_gpu(index, args.faiss_gpus, False)
    
    args.model_device = args.device
    args.max_seq_length=args.max_query_length
    args.train_batch_size = args.eval_batch_size
    mrr, res_data = eval_model(args, model, index, doc_ids)

    with open(os.path.join(args.output_dir, f"res.json"),'a+') as f:
        f.write(json.dumps(res_data))
    

if __name__ == "__main__":
    main()