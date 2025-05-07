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


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    # ... (giữ nguyên hàm này) ...
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
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
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
        
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    # ... (giữ nguyên hàm này) ...
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0 # codebleu có vẻ chưa được dùng đúng cách ở đây, cần xem xét lại nếu task là concode
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
        
        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode': # Cần đảm bảo calc_code_bleu được gọi đúng
            # Ví dụ: codebleu_score = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)
            # result['codebleu'] = codebleu_score * 100
            # Hiện tại codebleu vẫn là 0.0 như khởi tạo
            result['codebleu'] = 0 # Giữ nguyên như code gốc nếu chưa sửa

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
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        global_step = 0
        best_bleu_em = -1
        best_ppl = 1e6 # Sử dụng giá trị lớn cho best_ppl ban đầu
        # Khởi tạo args.start_epoch ở đây để đảm bảo nó có giá trị nếu không có checkpoint
        args.start_epoch = 0

        # --- BEGIN CHECKPOINT LOADING LOGIC ---
        # Tìm các checkpoint đã lưu dạng 'checkpoint-epoch-*'
        # glob.glob trả về danh sách các path, ví dụ: ['./outputs/checkpoint-epoch-0', './outputs/checkpoint-epoch-1']
        # Chúng ta cần tìm file checkpoint.pt bên trong các thư mục đó
        potential_ckpt_dirs = glob.glob(os.path.join(args.output_dir, 'checkpoint-epoch-*'))
        ckpt_files = []
        for d in potential_ckpt_dirs:
            ckpt_file = os.path.join(d, 'checkpoint.pt')
            if os.path.exists(ckpt_file):
                ckpt_files.append(ckpt_file)
        
        if ckpt_files:
            latest_ckpt_file = max(ckpt_files, key=os.path.getctime)
            logger.info("Resuming training from checkpoint %s", latest_ckpt_file)
            print(f"DEBUG: Attempting to resume from {latest_ckpt_file}", flush=True)
            try:
                # Thêm weights_only=True để an toàn hơn, nếu checkpoint chỉ chứa tensors và dicts cơ bản
                # Nếu gây lỗi, quay lại False nhưng cần cẩn trọng với file checkpoint không tin cậy
                checkpoint = torch.load(latest_ckpt_file, map_location=args.device) #, weights_only=True) 
                
                model_to_load = model.module if hasattr(model, 'module') else model
                model_to_load.load_state_dict(checkpoint['model_state_dict'])
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                args.start_epoch = checkpoint['epoch'] + 1
                global_step = checkpoint['global_step']
                # Khôi phục các giá trị best nếu có trong checkpoint (tùy chọn, nếu bạn muốn duy trì chúng)
                best_ppl = checkpoint.get('best_ppl', 1e6)
                best_bleu_em = checkpoint.get('best_bleu_em', -1)

                logger.info("Successfully resumed from epoch %d (next epoch to run: %d). Global step: %d",
                            checkpoint['epoch'], args.start_epoch, global_step)
                print(f"DEBUG: Successfully resumed. Next epoch: {args.start_epoch}, Global step: {global_step}", flush=True)

            except Exception as e:
                logger.error("Failed to load checkpoint %s. Starting training from scratch. Error: %s",
                             latest_ckpt_file, e, exc_info=True)
                print(f"ERROR: Failed to load checkpoint {latest_ckpt_file}. Error: {e}. Starting from scratch.", flush=True)
                args.start_epoch = 0
                global_step = 0
                best_bleu_em = -1
                best_ppl = 1e6
        else:
            logger.info("No existing 'checkpoint-epoch-*' found. Starting training from scratch.")
            print("DEBUG: No 'checkpoint-epoch-*' found. Starting from scratch.", flush=True)
        # --- END CHECKPOINT LOADING LOGIC ---

        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        logger.info("  Starting epoch = %d", args.start_epoch) # Log epoch bắt đầu
        logger.info("  Model type: %s", args.model_type) # Đổi từ print sang logger.info

        dev_dataset = {}
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        
        # ce_cir = nn.CrossEntropyLoss(ignore_index=-100) # Biến này được định nghĩa nhưng không thấy dùng trong CE loss block
        # kld = nn.KLDivLoss(reduction='none') # Tương tự

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            logger.info(f"Starting epoch {cur_epoch}/{int(args.num_train_epochs) - 1}")
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {cur_epoch} Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
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
                    # Sửa lại cách tính avg_loss cho bar description
                    avg_loss = round(tr_loss / (step + 1), 4) # Chia cho số step đã qua trong epoch này
                    bar.set_description(f"Epoch {cur_epoch} Train loss {avg_loss:.3f}")


            # --- BEGIN SAVE CHECKPOINT PER EPOCH ---
            if args.local_rank in [-1, 0]: # Chỉ lưu checkpoint ở master process nếu dùng distributed
                checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{cur_epoch}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_data = {
                    'epoch': cur_epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'best_ppl': best_ppl, # Lưu cả best_ppl và best_bleu_em
                    'best_bleu_em': best_bleu_em,
                    'args': args # Lưu args để tham khảo (tùy chọn)
                }
                output_checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pt")
                torch.save(checkpoint_data, output_checkpoint_file)
                logger.info(f"Saved checkpoint for epoch {cur_epoch} to {output_checkpoint_file}")
                print(f"DEBUG: Saved checkpoint for epoch {cur_epoch} to {output_checkpoint_file}", flush=True)
            # --- END SAVE CHECKPOINT PER EPOCH ---

            if args.do_eval:
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1 and args.local_rank in [-1, 0]: # Thêm check local_rank
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)
                
                # Bỏ save 'checkpoint-last' vì đã có 'checkpoint-epoch-*'
                # logger.info("eval_ppl: %s", eval_ppl) # Đổi print sang logger.info

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl: %s (decreased from %s)", eval_ppl, best_ppl)
                    best_ppl = eval_ppl
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    
                    # Lưu checkpoint 'best-ppl' (vẫn giữ logic này nếu bạn muốn)
                    # Nếu checkpoint-epoch-* đã đủ, có thể bỏ phần này
                    if args.always_save_model and args.local_rank in [-1, 0]:
                        output_dir_best_ppl = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                        os.makedirs(output_dir_best_ppl, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        # Lưu đầy đủ thông tin cho best-ppl checkpoint nếu muốn resume từ nó
                        # Hoặc chỉ lưu model state nếu chỉ dùng để test
                        # Ở đây, ví dụ vẫn lưu đầy đủ như checkpoint-epoch-*
                        best_ppl_checkpoint_data = {
                            'epoch': cur_epoch, # Epoch mà best_ppl đạt được
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'best_ppl': best_ppl,
                            'best_bleu_em': best_bleu_em,
                            'args': args
                        }
                        output_model_file = os.path.join(output_dir_best_ppl, "checkpoint.pt") # Đổi tên thành checkpoint.pt
                        torch.save(best_ppl_checkpoint_data, output_model_file)
                        logger.info("Saved best PPL model checkpoint to %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl did not decrease for %d epochs. Current PPL: %s, Best PPL: %s",
                                not_loss_dec_cnt, eval_ppl, best_ppl)
                    if all([x >= args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]): # Sửa > thành >=
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d (>=%d), and not_loss_dec_cnt=%d (>=%d)\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, args.patience, not_loss_dec_cnt, args.patience)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break # Thoát vòng lặp epoch

                logger.info("***** Clearing CUDA cache after PPL eval *****")
                torch.cuda.empty_cache()

                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)
                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    
                    if args.data_num == -1 and args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                    
                    # logger.info("dev_bleu: %s, dev_em: %s, dev_bleu_em: %s, best_bleu_em: %s",
                    #             dev_bleu, dev_em, dev_bleu_em, best_bleu_em) # Đổi print

                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f) (increased from %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em, best_bleu_em)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        
                        # Lưu checkpoint 'best-bleu' (tương tự best-ppl)
                        if (args.data_num == -1 or args.always_save_model) and args.local_rank in [-1, 0]:
                            output_dir_best_bleu = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                            os.makedirs(output_dir_best_bleu, exist_ok=True)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            best_bleu_checkpoint_data = {
                                'epoch': cur_epoch,
                                'model_state_dict': model_to_save.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'global_step': global_step,
                                'best_ppl': best_ppl,
                                'best_bleu_em': best_bleu_em,
                                'args': args
                            }
                            output_model_file = os.path.join(output_dir_best_bleu, "checkpoint.pt") # Đổi tên
                            torch.save(best_bleu_checkpoint_data, output_model_file)
                            logger.info("Saved best BLEU model checkpoint to %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu+em did not improve for %d epochs. Current: %.2f, Best: %.2f",
                                    not_bleu_em_inc_cnt, dev_bleu_em, best_bleu_em)
                        # fa.write(...) # Giữ nguyên log này nếu muốn
                        if all([x >= args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]): # Sửa > thành >=
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d (>=%d), and not_loss_dec_cnt=%d (>=%d)\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, args.patience, not_loss_dec_cnt, args.patience)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break # Thoát vòng lặp epoch
            
            logger.info("***** Clearing CUDA cache at end of epoch %d *****", cur_epoch)
            torch.cuda.empty_cache()
            # Kết thúc vòng lặp epoch

        if args.local_rank in [-1, 0] and args.data_num == -1 and 'tb_writer' in locals(): # Check tb_writer tồn tại
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-bleu', 'best-ppl']: # Có thể thêm 'checkpoint-epoch-X' nếu muốn test epoch cụ thể
            # Điều chỉnh đường dẫn để tải checkpoint.pt từ thư mục criteria
            checkpoint_file_to_test = os.path.join(args.output_dir, f'checkpoint-{criteria}', 'checkpoint.pt')
            
            if not os.path.exists(checkpoint_file_to_test):
                # Nếu criteria là best-bleu/best-ppl mà không có checkpoint.pt, thử pytorch_model.bin (legacy)
                # Hoặc chỉ báo lỗi nếu muốn thống nhất dùng checkpoint.pt
                legacy_model_file = os.path.join(args.output_dir, f'checkpoint-{criteria}', 'pytorch_model.bin')
                if os.path.exists(legacy_model_file):
                    logger.info("Found legacy model file %s for testing.", legacy_model_file)
                    checkpoint_to_load = torch.load(legacy_model_file, map_location=args.device)
                    # Nếu là legacy, nó chỉ có model_state_dict
                    model_to_load = model.module if hasattr(model, 'module') else model
                    model_to_load.load_state_dict(checkpoint_to_load) # Giả sử checkpoint_to_load là state_dict
                else:
                    logger.warning(f"Checkpoint file {checkpoint_file_to_test} (nor legacy pytorch_model.bin) not found for criteria {criteria}. Skipping test.")
                    continue
            else:
                logger.info("Reloading model from checkpoint %s", checkpoint_file_to_test)
                checkpoint_to_load = torch.load(checkpoint_file_to_test, map_location=args.device)
                model_to_load = model.module if hasattr(model, 'module') else model
                model_to_load.load_state_dict(checkpoint_to_load['model_state_dict'])

            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result.get('codebleu', 0) # Sử dụng .get để tránh KeyError nếu codebleu không có
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f_res: # Đổi tên biến f thành f_res để tránh trùng
                    f_res.write('[Time: {}] {}\n'.format(get_elapse_time(t0), checkpoint_file_to_test))
                    f_res.write(result_str)

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
