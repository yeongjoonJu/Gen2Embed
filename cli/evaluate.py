import json
import sys
from dataclasses import dataclass, field
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoConfig

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from src.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

@dataclass
class SamplingDataArguments(DataArguments):
    apply_represent_prompt : bool = field(default=False)
    in_one_word_D : bool = field(default=False)
    in_one_word_Q : bool = field(default=False)


def main():
    # Handle torch.distributed local rank arg
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)

    parser = HfArgumentParser((ModelArguments, SamplingDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.encode_output_path, exist_ok=True)
    
    # system
    apply_represent_prompt = data_args.apply_represent_prompt
    in_one_word_D = data_args.in_one_word_D
    in_one_word_Q = data_args.in_one_word_Q

    processor = AutoProcessor.from_pretrained(
        model_args.model_name if model_args.checkpoint_path is None else model_args.checkpoint_path,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    with open(os.path.join(data_args.encode_output_path, "system_prompt.txt"), "w") as fout:
        if data_args.Q_prompt is not None:
            fout.write(data_args.Q_prompt+"\n")
        if data_args.D_prompt is not None:
            fout.write(data_args.D_prompt)

    # Loop through each subset, encode and immediately compute score
    for idx, subset in enumerate(data_args.subset_name):
        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")

        # Skip encoding if both exist
        if not (os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path)):
            # Encode query
            eval_qry_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="qry_text",
                img_path_field="qry_img_path",
                apply_represent_prompt = apply_represent_prompt,
                in_one_word_D = in_one_word_D,
                in_one_word_Q = in_one_word_Q
            )
            eval_qry_loader = DataLoader(
                eval_qry_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
            )
            encoded_qry = []
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc="Encode query"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = model(qry=batch)
                    encoded_qry.append(output["qry_reps"].cpu().float().numpy())
            encoded_qry = np.concatenate(encoded_qry)
            with open(encode_qry_path, 'wb') as f:
                pickle.dump((encoded_qry, eval_qry_dataset.paired_data), f)

            # Encode target
            eval_tgt_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="tgt_text",
                img_path_field="tgt_img_path",
                apply_represent_prompt = apply_represent_prompt,
                in_one_word_D = in_one_word_D,
                in_one_word_Q = in_one_word_Q
            )
            eval_tgt_loader = DataLoader(
                eval_tgt_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
            )
            encoded_tgt = []
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                    batch = batch_to_device(batch, training_args.device)
                    output = model(tgt=batch)
                    encoded_tgt.append(output["tgt_reps"].cpu().float().numpy())
            encoded_tgt = np.concatenate(encoded_tgt)
            with open(encode_tgt_path, 'wb') as f:
                pickle.dump((encoded_tgt, eval_tgt_dataset.paired_data), f)
        else:
            print(f"Already encoded {subset}, loading saved tensors.")

     
        # Immediate scoring for this subset
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)

        qry_dict = { (tt['text'], tt['img_path']): vec for vec, tt in zip(qry_tensor, qry_index) }
        tgt_dict = { (tt['text'], tt['img_path']): vec for vec, tt in zip(tgt_tensor, tgt_index) }

        eval_data = load_dataset(
            data_args.dataset_name,
            subset,
            split=data_args.dataset_split,
        )
        
        if 'Y-J-Ju' in data_args.dataset_name:
            task_inst = eval_data[0]['task_instruction'].strip()
            task_inst = task_inst + ' in one word' if in_one_word_Q else task_inst
            if subset!='MSCOCO':
                eval_data = eval_data.map(
                    lambda x: {
                        'qry_text': task_inst.strip() + ":\n" + x['qry_text'].strip(),
                    }
                )
            if apply_represent_prompt or subset=='RefCOCO-Matching':
                represent_prompt = eval_data[0]['represent_prompt']
                if subset=='RefCOCO-Matching':
                    eval_data = eval_data.map(
                        lambda x: {
                            'tgt_text': [represent_prompt.strip() +"\n"+ tgt.strip() for tgt in x['tgt_text']],
                        }
                    )
                else:
                    represent_prompt = represent_prompt + ' in one word' if in_one_word_D else represent_prompt
                    eval_data = eval_data.map(
                        lambda x: {
                            'tgt_text': [tgt.strip()+"\n"+represent_prompt.strip() for tgt in x['tgt_text']],
                        }
                    )
                
        n_correct = 0
        all_pred = []
        with torch.no_grad():
            for row in eval_data:
                qry_vec = qry_dict[(row['qry_text'], row['qry_img_path'])]
                tgt_vecs = np.stack([tgt_dict[(t, p)] for t, p in zip(row['tgt_text'], row['tgt_img_path'])], axis=0)
                
                scores, pred = get_pred(qry_vec, tgt_vecs, normalization=model_args.normalize)
                if pred == 0:
                    n_correct += 1
                # Fix: append a tuple of text and path correctly
                all_pred.append((row['tgt_text'][pred], row['tgt_img_path'][pred]))

        # Write predictions
        pred_file = os.path.join(data_args.encode_output_path, f"{subset}_pred.txt")
        with open(pred_file, 'w') as f:
            for text, path in all_pred:
                f.write(f"{text}\t{path}\n")

        # Write score
        score = n_correct / len(eval_data)
        score_dict = {"acc": score, "num_correct": n_correct, "num_pred": len(eval_data)}
        score_file = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        with open(score_file, 'w') as f:
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {score}\033[0m")

if __name__ == "__main__":
    main()
