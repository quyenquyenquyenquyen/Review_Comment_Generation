# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import glob
import logging
import math
import multiprocessing
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import add_args, set_dist, set_seed
from evaluator import smooth_bleu
from evaluator.bleu import _bleu
from evaluator.CodeBLEU import calc_code_bleu
from models import build_or_load_gen_model
from transformers import get_linear_schedule_with_warmup
from utils import get_elapse_time, get_filenames, load_and_cache_gen_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
        num_workers=4, pin_memory=True
    )
    logger.info("  ***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    target_ids=target_ids,
                    target_mask=target_mask
                )
            else:
                outputs = model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    labels=target_ids,
                    decoder_attention_mask=target_mask
                )
                loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        batch_num += 1

    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on %s data*****", split_tag)
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_sampler = SequentialSampler(eval_data)
    kwargs = {
        'sampler': eval_sampler,
        'batch_size': args.eval_batch_size
    }
    if args.data_num == -1:
        kwargs.update({'num_workers': 4, 'pin_memory': True})
    eval_dataloader = DataLoader(eval_data, **kwargs)

    model.eval()
    pred_ids = []

    for batch in tqdm(
            eval_dataloader,
            total=len(eval_dataloader),
            desc=f"Eval bleu for {split_tag} set"
    ):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(
                    source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    num_beams=args.beam_size,
                    early_stopping=args.task == 'summarize',
                    max_length=args.max_target_length
                )
                top_preds = list(preds.cpu().numpy())

        pred_ids.extend(top_preds)

    pred_nls = [
        tokenizer.decode(
            idx, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ) for idx in pred_ids
    ]

    output_fn = os.path.join(args.res_dir, f"test_{criteria}.output")
    gold_fn = os.path.join(args.res_dir, f"test_{criteria}.gold")
    src_fn = os.path.join(args.res_dir, f"test_{criteria}.src")

    if args.task == 'defect':
        target_map = {0: 'false', 1: 'true'}
        golds = [target_map[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as fo, open(gold_fn, 'w') as fg, open(src_fn, 'w') as fs:
            for pred, ex in zip(pred_nls, eval_examples):
                fo.write(pred.strip() + '\n')
                fg.write(target_map[ex.target] + '\n')
                fs.write(ex.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs = []
        with open(output_fn, 'w') as fo, open(gold_fn, 'w') as fg, open(src_fn, 'w') as fs:
            for pred, ex in zip(pred_nls, eval_examples):
                dev_accs.append(pred.strip() == ex.target.strip())
                if args.task == 'summarize':
                    fo.write(f"{ex.idx}\t{pred.strip()}\n")
                    fg.write(f"{ex.idx}\t{ex.target.strip()}\n")
                    fs.write(f"{ex.idx}\t{ex.source.strip()}\n")
                else:
                    fo.write(pred.strip() + '\n')
                    fg.write(ex.target.strip() + '\n')
                    fs.write(ex.source.strip() + '\n')

        if args.task == 'summarize':
            gold_map, pred_map = smooth_bleu.computeMaps(predictions=[], gold_fn=gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(gold_map, pred_map)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang) * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)

    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    pool = multiprocessing.Pool(args.cpu_cont)
    (
        args.train_filename,
        args.dev_filename,
        args.test_filename
    ) = get_filenames(
        args.data_dir,
        args.task,
        args.sub_task
    )

    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = f"{args.summary_dir}/{args.output_dir.strip('/').replace('/', '_')}"
            tb_writer = SummaryWriter(summary_fn)

        train_examples, train_data = load_and_cache_gen_data(
            args, args.train_filename, pool, tokenizer, 'train'
        )
        train_sampler = (
            RandomSampler(train_data)
            if args.local_rank == -1
            else DistributedSampler(train_data)
        )
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=4,
            pin_memory=True
        )

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': args.weight_decay
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_train_optimization_steps
        )

        logger.info("args.warmup_steps %s", args.warmup_steps)
        logger.info("num_train_optimization_steps: %s", num_train_optimization_steps)
        print("args.warmup_steps",args.warmup_steps) # <-- Lệnh print từ file 1
        print("num_train_optimization_steps:", num_train_optimization_steps) # <-- Lệnh print từ file 1
        global_step = 0
        best_bleu_em = -1
        best_ppl = float('inf')

        ckpt_paths = glob.glob(os.path.join(
            args.output_dir, 'checkpoint-epoch-*', 'checkpoint.pt'
        ))
        if ckpt_paths:
            latest_ckpt = max(ckpt_paths, key=os.path.getctime)
            logger.info("Resume from %s", latest_ckpt)
            ckpt = torch.load(latest_ckpt, map_location=args.device)
            print("Loaded checkpoint from epoch:", ckpt.get('epoch'))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            global_step = ckpt.get('global_step', 0)
            args.start_epoch = ckpt.get('epoch', 0) + 1
        else:
            args.start_epoch = 0

        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        logger.info("Model type: %s", args.model_type)
        print( args.model_type) # <-- Lệnh print từ file 1
        dev_dataset = {}
        not_loss_dec_cnt = 0
        not_bleu_em_inc_cnt = 0 if args.do_eval_bleu else int(1e6)
        ce_cir = nn.CrossEntropyLoss(ignore_index=-100)
        kld = nn.KLDivLoss(reduction='none')

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            logger.info("Starting epoch %d", cur_epoch)
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(
                        source_ids=source_ids,
                        source_mask=source_mask,
                        target_ids=target_ids,
                        target_mask=target_mask
                    )
                else:
                    outputs = model(
                        input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,
                        decoder_attention_mask=target_mask
                    )
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    avg_loss = tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1)
                    bar.set_description(f"[{cur_epoch}] Train loss {avg_loss:.3f}")

            if args.do_eval:
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(
                        args, args.dev_filename, pool, tokenizer, 'dev'
                    )
                    dev_dataset['dev_loss'] = (eval_examples, eval_data)

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                logger.info("  eval_ppl = %s", eval_ppl)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)
                print("eval_ppl:",eval_ppl) # <-- Lệnh print từ file 1

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl: %s", eval_ppl)
                    best_ppl = eval_ppl

                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    os.makedirs(output_dir, exist_ok=True)
                    if args.always_save_model:
                        torch.save(
                            model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            os.path.join(output_dir, "pytorch_model.bin")
                        )
                        logger.info("Saved best ppl model to %s", output_dir)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl did not decrease for %d epochs", not_loss_dec_cnt)

                torch.cuda.empty_cache()

                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(
                        args, args.dev_filename, pool, tokenizer, 'dev',
                        only_src=True, is_sample=True
                    )
                    result = eval_bleu_epoch(
                        args, eval_data, eval_examples, model, tokenizer,
                        'dev', f'e{cur_epoch}'
                    )
                    dev_bleu, dev_em = result['bleu'], result['em']
                    dev_bleu_em = (
                        dev_bleu if args.task == 'summarize' else
                        dev_em if args.task == 'defect' else
                        dev_bleu + dev_em
                    )
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                    print("dev_bleu:",dev_bleu)
                    print("dev_em:", dev_em)
                    print("dev_bleu_em:",dev_bleu_em)
                    print("best_bleu_em:", best_bleu_em)

                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        best_bleu_em = dev_bleu_em
                        logger.info(
                            "  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                            cur_epoch, dev_bleu_em, dev_bleu, dev_em
                        )
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        os.makedirs(output_dir, exist_ok=True)
                        if args.data_num == -1 or args.always_save_model:
                            torch.save(
                                model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                                os.path.join(output_dir, "pytorch_model.bin")
                            )
                            logger.info("Saved best bleu model to %s", output_dir)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info(
                            "Bleu+em did not improve for %d epochs", not_bleu_em_inc_cnt
                        )

            # Save checkpoint each epoch
            checkpoint_dir = os.path.join(
                args.output_dir, f'checkpoint-epoch-{cur_epoch}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = {
                'epoch': cur_epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, 'checkpoint.pt'))
            logger.info("Saved checkpoint for epoch %d to %s", cur_epoch, checkpoint_dir)
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  ***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-bleu']:
            model_path = os.path.join(
                args.output_dir, f'checkpoint-{criteria}', 'pytorch_model.bin'
            )
            logger.info("Reload model from %s", model_path)
            model.load_state_dict(torch.load(model_path))
            eval_examples, eval_data = load_and_cache_gen_data(
                args, args.test_filename, pool, tokenizer,
                'test', only_src=True, is_sample=False
            )
            result = eval_bleu_epoch(
                args, eval_data, eval_examples, model, tokenizer,
                'test', criteria
            )
            logger.info(
                "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f",
                criteria, result['bleu'], result['em'], result.get('codebleu', 0)
            )
            fa.write(
                f"[{criteria}] bleu-4: {result['bleu']:.2f}, em: {result['em']:.4f}, codebleu: {result.get('codebleu', 0):.4f}\n"
            )
            if args.res_fn:
                with open(args.res_fn, 'a+') as f_out:
                    f_out.write(
                        f"[Time: {get_elapse_time(t0)}] {model_path}\n"
                    )
                    f_out.write(
                        f"[{criteria}] bleu-4: {result['bleu']:.2f}, em: {result['em']:.4f}, codebleu: {result.get('codebleu', 0):.4f}\n"
                    )

    logger.info("Finish and take %s", get_elapse_time(t0))
    fa.write("Finish and take %s" % get_elapse_time(t0))
    fa.close()


if __name__ == "__main__":
    main()
