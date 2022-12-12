import sys
sys.path += ["./"]
import os
import time
import torch
import random
import faiss
import logging
import argparse
import subprocess
import tempfile
import numpy as np
from tqdm import tqdm, trange
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
    RobertaConfig)

from dataset import TextTokenIdsCache, SequenceDataset, load_rel, pack_tensor_2D
from SeDR_model import SeDR

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)    
import pickle
import torch.nn.functional as F

from retrieve_utils import (
    construct_flatindex_from_embeddings, 
     convert_index_to_gpu
)
from msmarco_eval import compute_metrics2

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args, optimizer=None):
    print(f'save model to {save_name}')
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))


class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, 
            rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_ids'] = self.reldict[item]
        return ret_val


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
        qids = [x['id'] for x in batch]
        all_rel_pids = [x["rel_ids"] for x in batch]
        return data, all_rel_pids
    return collate_function 

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
    print('MS MARCO results:')
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
    # json.dump(ranking, open('./data/test_ranking.json', 'w'))
    print('TREC-DL 2019 results:')
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
    # json.dump(ranking, open('./data/test_ranking.json', 'w'))
    print('TREC-DL 2020 results:')
    testres = compute_metrics_from_files(os.path.join(args.preprocess_dir, "2020test-qrel.tsv") , ranking, "./trec_eval/trec_eval")
    ress['2020test_res'] = testres

    return (MRR+Recall[-1]),ress 

gpu_resources = []

def train(args, model):
    """ Train the model """
    # tb_writer = SummaryWriter(os.path.join(args.log_dir, 
    #     time.strftime("%b-%d_%H:%M:%S", time.localtime())))
    passage_embeddings = np.memmap(os.path.join(args.pembed_dir, "passages.memmap"), dtype=np.float32, mode="r"
        ).reshape(-1, model.output_embedding_size)
    passage_ids = np.memmap(os.path.join(args.pembed_dir, "passages-id.memmap"), dtype=np.int32, mode="r"
        )
    pid2embids = None
    pid2embidspath = os.path.join(args.pembed_dir, "pid2embids.pkl")

    if os.path.exists(pid2embidspath):
        print(f'loads pid2embids from {pid2embidspath}')
        with open(pid2embidspath,'rb') as f:
            pid2embids = pickle.load(f)
    else:
        pid2embids = []
        pre_pid = -1
        for embid,pid in enumerate(passage_ids):
            if pid != pre_pid:
                assert len(pid2embids) == pid
                pid2embids.append(embid)
                pre_pid = pid
        pid2embids.append(passage_embeddings.shape[0])
        with open(pid2embidspath,'wb') as f:
            pickle.dump(pid2embids, f, pickle.HIGHEST_PROTOCOL)

    # loading faiss index of passage embeddings
    # index = load_index(passage_embeddings, args.faiss_gpu_index)
    index = construct_flatindex_from_embeddings(passage_embeddings)
    index = convert_index_to_gpu(index, args.faiss_gpu_index, False)
    

    args.train_batch_size = args.per_gpu_batch_size
    train_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "train-query"),
        os.path.join(args.preprocess_dir, "train-qrel.tsv"),
        args.max_seq_length
    )

    train_sampler = RandomSampler(train_dataset) 
    collate_fn = get_collate_function(args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_mrr, logging_mrr = 0.0, 0.0
    tr_recall, logging_recall = 0.0, 0.0
    tr_acc, logging_acc = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)  


    best_mrr,_ = eval_model(args, model, index, passage_ids)
    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, all_rel_pids) in enumerate(epoch_iterator):

            batch = {k:v.to(args.model_device) for k, v in batch.items()}
            model.train()            
            query_embeddings = model.query_emb(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
            I_nearest_neighbor = index.search(
                    query_embeddings.detach().cpu().numpy(), 1024)[1]
            
            loss = 0
            for retrieve_embids, cur_rel_pids, qembedding in zip(
                I_nearest_neighbor, all_rel_pids, query_embeddings):
                retrieve_pids = passage_ids[retrieve_embids]

                # uniqe
                retrieve_pids_list = []
                retrieve_embids_list = []
                for idx, pid in enumerate(retrieve_pids):
                    if pid not in retrieve_pids_list:
                        retrieve_pids_list.append(pid)
                        retrieve_embids_list.append(retrieve_embids[idx])
                    if len(retrieve_pids_list) == args.metric_cut:
                        break
                retrieve_pids  = np.array(retrieve_pids_list)
                retrieve_embids = np.array(retrieve_embids_list)
                # ----
                

                target_labels = np.isin(retrieve_pids, cur_rel_pids).astype(np.int32)

                # change to MRR100
                first_rel_pos = np.where(target_labels[:100])[0] 
                mrr = 1/(1+first_rel_pos[0]) if len(first_rel_pos) > 0 else 0

                tr_mrr += mrr/args.train_batch_size
                recall = 1 if mrr > 0 else 0
                tr_recall += recall / args.train_batch_size

                first_rel_pos = np.where(target_labels)[0]
                if len(first_rel_pos) > 0:
                    rel_embid = retrieve_embids[first_rel_pos[0]]
                else:
                    pid = cur_rel_pids[0]
                    pidemb = passage_embeddings[pid2embids[pid]:pid2embids[pid+1]]
                    pidemb = torch.FloatTensor(pidemb).to(args.model_device)
                    max_index = (qembedding.unsqueeze(0) * pidemb).sum(-1).max(0).indices.item()
                    rel_embid = pid2embids[pid]+max_index

                neg_retrieve_embids = retrieve_embids[target_labels==False]

                retrieve_embids = np.hstack([rel_embid, neg_retrieve_embids])

                topK_passage_embeddings = torch.FloatTensor(
                    passage_embeddings[retrieve_embids]).to(args.model_device)
                y_pred = (qembedding.unsqueeze(0) * topK_passage_embeddings).sum(-1, keepdim=False).unsqueeze(0)
                label = torch.LongTensor([0]).to(args.model_device)

                curloss = F.cross_entropy(y_pred, label)
                tr_acc += (1 if y_pred[0].max(0).indices.item() == 0 else 0 )/ args.train_batch_size

                loss += curloss
            
            loss /= (args.train_batch_size * args.gradient_accumulation_steps)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    # tb_writer.add_scalar('train/all_loss', cur_loss, global_step)
                    logging_loss = tr_loss

                    cur_mrr =  (tr_mrr - logging_mrr)/(
                        args.logging_steps * args.gradient_accumulation_steps)
                    # tb_writer.add_scalar('train/mrr_100', cur_mrr, global_step)
                    logging_mrr = tr_mrr

                    cur_recall =  (tr_recall - logging_recall)/(
                        args.logging_steps * args.gradient_accumulation_steps)
                    # tb_writer.add_scalar('train/recall_100', cur_recall, global_step)
                    logging_recall = tr_recall

                    cur_acc =  (tr_acc - logging_acc)/(
                        args.logging_steps * args.gradient_accumulation_steps)
                    # tb_writer.add_scalar('train/acc', cur_acc, global_step)
                    logging_acc = tr_acc
                    logger.info(f'{global_step} step --- loss:{cur_loss} mrr_100:{cur_mrr} recall_100:{cur_recall} acc:{cur_acc}')

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                if global_step % args.save_steps == 0: 
                    cur_mrr, _ = eval_model(args, model, index, passage_ids)
                    if cur_mrr > best_mrr:
                        save_model(model, args.model_save_dir, '{}-ckpt-{}'.format(os.path.split(args.init_path)[-1],global_step), args)
                        best_mrr = cur_mrr
        cur_mrr, _ = eval_model(args, model, index, passage_ids)
        if cur_mrr > best_mrr:
            save_model(model, args.model_save_dir, '{}-epoch-{}'.format(os.path.split(args.init_path)[-1], epoch_idx+1), args)
            best_mrr = cur_mrr


def run_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_cut", type=int, default=200)
    parser.add_argument("--init_path", type=str, required=True)
    parser.add_argument("--pembed_dir", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--neg_topk", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--per_gpu_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", default=2000, type=int)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--save_steps", type=int, default=3000) # not use
    parser.add_argument("--logging_steps", type=int, default=1000)

    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=6, type=int)

    parser.add_argument("--model_gpu_index", type=int, default=0)
    parser.add_argument("--faiss_gpu_index", type=int, default=[], nargs="+")
    parser.add_argument("--faiss_omp_num_threads", type=int, default=32)
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.faiss_omp_num_threads)

    return args


def main():
    args = run_parse_args()
    # Setup CUDA, GPU 
    args.model_device = torch.device(f"cuda:{args.model_gpu_index}")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logger.warning("Model Device: %s, n_gpu: %s", args.model_device, args.n_gpu)

    # Set seed
    set_seed(args)

    logger.info(f"load from {args.init_path}")
    config = RobertaConfig.from_pretrained(args.init_path)
    # config.max_seg_num = 4
    model = SeDR.from_pretrained(args.init_path, config=config, max_seg_num = config.max_seg_num)

    model.to(args.model_device)
    logger.info("Training/evaluation parameters %s", args)
    
    os.makedirs(args.model_save_dir, exist_ok=True)
    train(args, model)
    

if __name__ == "__main__":
    main()
