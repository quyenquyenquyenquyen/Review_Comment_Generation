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

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import glob
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
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
    all_pred_ids = [] # Đổi tên để tránh nhầm lẫn, sẽ chứa tất cả các beam
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                # Roberta thường là model phân loại hoặc encoder-only, không dùng generate() với beam
                # Giả sử nó trả về top-1 prediction trực tiếp
                preds = model(source_ids=source_ids, source_mask=source_mask)
                # Cần đảm bảo preds này có dạng phù hợp cho việc decode sau đó
                # Ví dụ, nếu preds là logits, cần argmax và xử lý padding
                # Trong code gốc của run_gen.py, nó lấy pred[0]
                current_batch_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length,
                                       num_return_sequences=args.beam_size) # Vẫn lấy tất cả beam_size sequences
                current_batch_preds = list(preds.cpu().numpy())
            all_pred_ids.extend(current_batch_preds)

    # Decode tất cả các predictions từ các beam
    all_pred_nls_decoded = [tokenizer.decode(pred_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for pred_id in all_pred_ids]

    # Lấy ra top-1 prediction cho mỗi input example để tính EM
    # và cũng để ghi vào file output cho BLEU (thường BLEU cũng tính trên top-1)
    top1_pred_nls_for_em_and_bleu = []
    # Gộp các beam lại để ghi vào file output nếu cần (ví dụ cho summarize hoặc phân tích)
    # nhưng không dùng để tính EM tiêu chuẩn
    merged_beam_predictions_for_file = []

    num_examples = len(eval_examples)
    for i in range(num_examples):
        # Lấy dự đoán đầu tiên (top-1) từ nhóm beam_size dự đoán cho ví dụ thứ i
        top1_pred_nls_for_em_and_bleu.append(all_pred_nls_decoded[i * args.beam_size])
        
        # Nếu bạn vẫn muốn ghi tất cả các beam vào một dòng trong file output (như code cũ)
        # bạn có thể tạo merged_beam_predictions_for_file ở đây
        current_example_beams = all_pred_nls_decoded[i * args.beam_size : (i + 1) * args.beam_size]
        merged_beam_predictions_for_file.append('\t'.join(current_example_beams))


    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    # Phần xử lý cho task 'defect' vẫn giữ nguyên vì nó thường là phân loại, không dùng beam search phức tạp kiểu này
    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        # Đối với defect, pred_nls nên là top1_pred_nls_for_em_and_bleu nếu model.generate trả về nhiều beam
        # Tuy nhiên, logic của roberta ở trên có thể đã xử lý để chỉ có top-1
        # Nếu không, bạn cần đảm bảo pred_nls cho defect là top-1
        eval_acc = np.mean([int(p == g) for p, g in zip(top1_pred_nls_for_em_and_bleu, golds)]) # Sử dụng top-1
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(top1_pred_nls_for_em_and_bleu, eval_examples): # Ghi top-1
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        # Tính EM dựa trên top-1 predictions
        dev_accs = []
        for pred_top1, gold_ex in zip(top1_pred_nls_for_em_and_bleu, eval_examples):
            dev_accs.append(pred_top1.strip() == gold_ex.target.strip())
        
        em_score = np.mean(dev_accs) * 100

        # Ghi file output và gold để tính BLEU
        # File output sẽ chứa top-1 predictions
        predictions_for_smooth_bleu = [] # Cho task summarize
        with open(output_fn, 'w') as f_out, open(gold_fn, 'w') as f_gold, open(src_fn, 'w') as f_src:
            for i, gold_ex in enumerate(eval_examples):
                pred_to_write = top1_pred_nls_for_em_and_bleu[i] # Ghi top-1
                # Nếu bạn muốn ghi chuỗi gộp các beam (merged_beam_predictions_for_file[i]) vào file thì thay ở đây
                # Tuy nhiên, để tính BLEU chuẩn, nên dùng top-1

                f_src.write(gold_ex.source.strip() + '\n')
                f_gold.write(gold_ex.target.strip() + '\n')
                f_out.write(pred_to_write.strip() + '\n')

                if args.task in ['summarize']:
                    predictions_for_smooth_bleu.append(str(gold_ex.idx) + '\t' + pred_to_write.strip())


        if args.task == 'summarize':
            # Cần đảm bảo gold_fn cho summarize cũng có định dạng idx + \t + text
            # Đoạn ghi file ở trên cần điều chỉnh cho summarize nếu gold_fn yêu cầu idx
            # Hiện tại, gold_fn được ghi không có idx, computeMaps có thể cần predictions là list các text
            # Hoặc gold_fn được chuẩn bị sẵn có idx
            # Giả sử gold_fn đã đúng định dạng cho computeMaps
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions_for_smooth_bleu, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2) # Tính BLEU trên file output (chứa top-1)

        result = {'em': em_score, 'bleu': bleu}
        if args.task == 'concode':
            # calc_code_bleu.get_codebleu cũng nên chạy trên output_fn chứa top-1
            codebleu_val = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)
            result['codebleu'] = codebleu_val * 100


    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

# Phần còn lại của file (main,...) giữ nguyên
def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer): # Copy từ file gốc nếu cần
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
        num_workers=4, pin_memory=True
    )
    # logger.info("  ***** Running ppl evaluation *****")
    # logger.info("  Num examples = %d", len(eval_examples))
    # logger.info("  Batch size = %d", args.eval_batch_size)

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
    if batch_num == 0: # Tránh chia cho 0 nếu eval_dataloader rỗng
        return float('inf')
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl

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
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.res_dir): # Thêm dòng này để tạo res_dir
        os.makedirs(args.res_dir)


    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            # summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            # Đảm bảo summary_dir tồn tại
            if not os.path.exists(args.summary_dir):
                os.makedirs(args.summary_dir)
            # Sửa lại cách tạo summary_fn để tránh lỗi nếu output_dir không có '/'
            output_dir_parts = args.output_dir.strip('/').split('/')
            summary_fn_suffix = '_'.join(output_dir_parts) if len(output_dir_parts) > 0 else "run"
            summary_fn = os.path.join(args.summary_dir, summary_fn_suffix)
            tb_writer = SummaryWriter(summary_fn)


        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        # Sửa lỗi num_train_optimization_steps có thể bằng 0
        if len(train_dataloader) == 0 and args.num_train_epochs > 0 :
             logger.warning("Train dataloader is empty. Training might not proceed as expected.")
             num_train_optimization_steps = 1 # Hoặc một giá trị mặc định khác
        elif len(train_dataloader) > 0:
            num_train_optimization_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
        else:
            num_train_optimization_steps = 0


        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Train Batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)


        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
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
                # nb_tr_steps += 1 # Đã có step từ enumerate
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    # train_loss = round(tr_loss / global_step, 4) # tr_loss là tổng loss, cần tính loss trung bình cho batch hiện tại
                    # Tính loss trung bình cho các bước gradient accumulation gần nhất
                    avg_loss_over_accum = tr_loss / (step + 1) # Đây là loss trung bình từ đầu epoch
                                                              # Có thể bạn muốn loss trung bình của (gradient_accumulation_steps) step gần nhất
                    current_avg_loss = loss.item() * args.gradient_accumulation_steps # Ước lượng loss cho lần update này
                    
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(current_avg_loss, 3)))


            if args.do_eval:
                if 'dev_loss' not in dev_dataset :
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                else:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                
                if len(eval_data) > 0: # Chỉ đánh giá nếu có dữ liệu dev
                    eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                    result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                    logger.info("  " + "*" * 20)
                    if args.data_num == -1 and 'tb_writer' in locals():
                        tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                    if args.save_last_checkpoints:
                        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_ppl < best_ppl:
                        not_loss_dec_cnt = 0
                        logger.info("  Best ppl:%s", eval_ppl)
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                        best_ppl = eval_ppl

                        output_dir_ppl = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                        if not os.path.exists(output_dir_ppl):
                            os.makedirs(output_dir_ppl)
                        if args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_loss_dec_cnt += 1
                        logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                        if args.patience > 0 and all([x >= args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(early_stop_str)
                            fa.write(early_stop_str)
                            break
                else:
                    logger.warning("Dev data is empty. Skipping PPL evaluation.")

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                if args.do_eval_bleu:
                    # Load dev data for BLEU (có thể khác với data cho PPL)
                    if 'dev_bleu' not in dev_dataset:
                        eval_examples_bleu, eval_data_bleu = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                        only_src=True, is_sample=True) # is_sample=True thường cho dev
                        dev_dataset['dev_bleu'] = eval_examples_bleu, eval_data_bleu
                    else:
                        eval_examples_bleu, eval_data_bleu = dev_dataset['dev_bleu']

                    if len(eval_data_bleu) > 0:
                        result = eval_bleu_epoch(args, eval_data_bleu, eval_examples_bleu, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                        dev_bleu, dev_em = result['bleu'], result['em']
                        if args.task in ['summarize']:
                            dev_bleu_em = dev_bleu
                        elif args.task in ['defect']:
                            dev_bleu_em = dev_em
                        else:
                            dev_bleu_em = dev_bleu + dev_em # Hoặc một công thức kết hợp khác
                        
                        if args.data_num == -1 and 'tb_writer' in locals():
                            tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)

                        if dev_bleu_em > best_bleu_em:
                            not_bleu_em_inc_cnt = 0
                            logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                        cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                            logger.info("  " + "*" * 20)
                            best_bleu_em = dev_bleu_em
                            fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, dev_bleu, dev_em))
                            
                            output_dir_bleu = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                            if not os.path.exists(output_dir_bleu):
                                os.makedirs(output_dir_bleu)
                            if args.data_num == -1 or args.always_save_model:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info("Save the best bleu model into %s", output_model_file)
                        else:
                            not_bleu_em_inc_cnt += 1
                            logger.info("Bleu+em does not improve for %d epochs", not_bleu_em_inc_cnt)
                            # fa.write(... # Ghi log tương tự)
                            if args.patience > 0 and all([x >= args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                                stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                    cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                                logger.info(stop_early_str)
                                fa.write(stop_early_str)
                                break
                    else:
                        logger.warning("Dev data for BLEU is empty. Skipping BLEU/EM evaluation.")
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1 and 'tb_writer' in locals():
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        for criteria_test in ['best-bleu']: # Có thể thêm 'best-ppl' nếu bạn muốn test cả checkpoint đó
            file_to_load = os.path.join(args.output_dir, f'checkpoint-{criteria_test}', "pytorch_model.bin")
            if os.path.exists(file_to_load):
                logger.info(f"Loading model from {file_to_load}")
                model.load_state_dict(torch.load(file_to_load, map_location=args.device))
            else:
                logger.warning(f"Checkpoint {file_to_load} not found. Testing with the current model state.")
                # Hoặc bạn có thể chọn không chạy test nếu checkpoint không tồn tại
                # continue 

            eval_examples_test, eval_data_test = load_and_cache_gen_data(
                args, args.test_filename, pool, tokenizer,
                'test', only_src=True, is_sample=False # is_sample=False cho test
            )
            if len(eval_data_test) > 0:
                result_test = eval_bleu_epoch(
                    args, eval_data_test, eval_examples_test,
                    model, tokenizer, 'test', criteria_test
                )
                
                logger.info(f"=== Test metrics ({criteria_test}) ===")
                logger.info(f"  BLEU-4          : {result_test['bleu']:.2f}")
                logger.info(f"  Exact Match (EM): {result_test['em']:.2f}%")
                if 'codebleu' in result_test:
                     logger.info(f"  CodeBLEU        : {result_test.get('codebleu', 0):.2f}")
            
                result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                    criteria_test, result_test['bleu'], result_test['em'], result_test.get('codebleu', 0)
                )
                fa.write(result_str)
            
                if args.res_fn:
                    with open(args.res_fn, 'a+') as f_res:
                        f_res.write(f"[Time: {get_elapse_time(t0)}] {file_to_load}\n")
                        f_res.write(result_str)
            else:
                logger.warning("Test data is empty. Skipping testing.")
      
        logger.info("Finish and take %s", get_elapse_time(t0))
        fa.write("Finish and take %s\n" % get_elapse_time(t0))
        fa.close()


if __name__ == "__main__":
    main()
