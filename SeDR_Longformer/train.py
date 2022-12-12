# coding=utf-8
from email.policy import default
import imp
import sys
sys.path.append("./")
import logging
import os
from dataclasses import dataclass, field
import transformers
from transformers import (
    RobertaConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from SeDR_Longformer.model import SeDR_Longformer
from dataset import TextTokenIdsCache, load_rel
from dataset import (
    SeDR_Dataset
)
# from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import (
    TrainingArguments, 
    )
from transformers import AdamW, get_linear_schedule_with_warmup
from lamb import Lamb
from star_tokenizer import RobertaTokenizer

logger = logging.Logger(__name__)

def create_optimizer_and_scheduler(args, model, num_training_steps: int):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer_str == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
    elif args.optimizer_str == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
    else:
        raise NotImplementedError("Optimizer must be adamw or lamb")

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler

def is_main_process(local_rank):
    return local_rank in [-1, 0]

@dataclass
class DataTrainingArguments:
    max_query_length: int = field() # Max query length
    max_doc_length: int = field() # Max document length
    max_seg_num: int = field() # Max segment number (document length = max segment length ✖️ max segment number)
    preprocess_dir: str = field() # "./data/passage"
    hardneg_path: str = field() # use prepare_hardneg.py to generate
    hardneg_topk: int = field(default=200) # hardness

@dataclass
class ModelArguments:
    init_path: str = field() # please use warmup model or roberta-base
    use_gradient_checkpointing: bool = field(default=False)


@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./data/models") # where to output
    logging_dir: str = field(default="./data/log")
    padding: bool = field(default=False)
    optimizer_str: str = field(default="lamb")
    overwrite_output_dir: bool = field(default=False)
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},)

    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=25.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})

    logging_steps: int = field(default=200, metadata={"help": "Log every X updates steps."})
    
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    max_bsize: int = field(default=512, metadata={"help": "Batch size of segments"})
    cache_size: int =field(default=50, metadata={"help": "Cache size of Late-Cache Negative"})
    postfix: str = field(default='')

def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logging.info(f"Training/evaluation parameters {training_args}")
    logging.info(f"data parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = RobertaConfig.from_pretrained(
        model_args.init_path,
        finetuning_task="msmarco",
        gradient_checkpointing=model_args.use_gradient_checkpointing
    )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case = True, cache_dir=None)
    config.gradient_checkpointing = model_args.use_gradient_checkpointing
    
    data_args.label_path = os.path.join(data_args.preprocess_dir, "train-qrel.tsv")

    train_dataset = SeDR_Dataset(
        rel_file=data_args.label_path,
        rank_file=data_args.hardneg_path,
        queryids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="train-query"),
        docids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="passages"),
        max_query_length=data_args.max_query_length,
        # max_doc_length=data_args.max_doc_length,
        hard_num=1,
        max_seg_length = data_args.max_doc_length, max_seg_num  = 1, 
        hardneg_topk=data_args.hardneg_topk,
        max_bsize = training_args.max_bsize, 
        tokenizer = tokenizer
    )

    SeDR_Longformer.set_cache_size(training_args.cache_size)
    model_class = SeDR_Longformer

    # config.max_seg_num = data_args.max_seg_num
    config.attention_window =[256]*12
    config.attention_dilation = [1]*12
    config.attention_mode = 'sliding_chunks'
    # config.attention_mode = 'tvm'
    config.autoregressive = False

    model = model_class.from_pretrained(
        model_args.init_path,
        config=config,max_seg_num = data_args.max_seg_num
    )
    model = model.cuda()

    all_step_num = len(train_dataset)*training_args.num_train_epochs //((training_args.max_bsize)*training_args.gradient_accumulation_steps)
    optimizer, scheduler = create_optimizer_and_scheduler(training_args, model, all_step_num)
    
    global_step = 0

    for ep in range(int(training_args.num_train_epochs)):
        loss_sum, acc_sum, batch_num  = 0.0, 0.0, 0.0
        model.train()
        # reset for random sample
        train_dataset.reset()
        for step, batch in enumerate(train_dataset):
            batch = {k:v.cuda() if 'sep' not in k and v is not None else v for k, v in batch.items()}
            loss, acc = model(batch['input_query_ids'], batch['query_attention_mask'],\
                batch['input_doc_ids'], batch['doc_attention_mask'], \
                other_doc_ids=batch['other_doc_ids'], other_doc_attention_mask=batch['other_doc_attention_mask'],\
                rel_pair_mask=batch['rel_pair_mask'], hard_pair_mask=batch['hard_pair_mask'], doc_fuse_mat = batch['doc_fuse_mat'], \
                doc_seg_sep = batch['doc_seg_sep'], other_doc_fuse_mat = batch['other_doc_fuse_mat'], other_doc_seg_sep = batch['other_doc_seg_sep'])
            
            if training_args.gradient_accumulation_steps > 1:
                loss /= training_args.gradient_accumulation_steps

            loss.backward()

            loss_sum += loss.item()   
            acc_sum += acc.item()
            batch_num +=  len(batch['doc_seg_sep']) -1

            if training_args.max_grad_norm != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm) 
            
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                global_step += 1
                optimizer.zero_grad()

            if (step+1) % training_args.logging_steps == 0:
                logging.info('step:{}, lr:{}, avg_batch:{}, loss: {:.5f}, acc:{:.5f}'.
                                format(global_step+1,
                                    optimizer.param_groups[0]['lr'],
                                    batch_num / training_args.logging_steps * training_args.gradient_accumulation_steps,
                                    loss_sum / training_args.logging_steps,
                                    acc_sum/training_args.logging_steps))
                loss_sum, acc_sum, batch_num  = 0.0, 0.0, 0.0

        logging.info('epoch {} finish'.format(ep+1))
        save_model(model, training_args.output_dir, 'epoch-{}-{}-{}-{}-{}{}'.format(ep+1, data_args.max_doc_length, data_args.max_seg_num, training_args.max_bsize,model_class.__name__, training_args.postfix), training_args)

if __name__ == "__main__":
    main()