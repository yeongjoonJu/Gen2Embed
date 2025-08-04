import json
import sys
import random
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
import faiss
import pickle
import os
from datasets import load_dataset
from src.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name


# already labeled subsets (classification datasets)
except_list = [
    "ImageNet_1K",
    "Place365",
    "N24News",
    "HatefulMemes",
    "VOC2007",
    "SUN397",
    "ImageNet-A",
    "ImageNet-R",
    "ObjectNet",
    "Country211",
]


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
    top_k: int = field(default=7, metadata={"help": "Hard negative K"})
    pool_mult: int = field(default=5, metadata={"help": "Multiplier for pool size"})
    
def construct_D2Q(data):
    d2q = {}
    q2rid = {}
    for i, row in enumerate(data):
        q_key = (row['qry'], row['qry_image_path'])
        # true pos1
        positive = (row['pos_text'], row['pos_image_path'])
        if positive in d2q:
            d2q[positive].add(q_key)
        else:
            d2q[positive] = set([q_key])
        q2rid[q_key] = i
    
    return d2q, q2rid

def couple_samples(eval_data, used_q, D2Q, Q2RID, subset,
                   qry_dict, cand_texts, cand_map, cand_vecs,
                   training_args, data_args, first_n_cands=56, pool_size=56):
    
    hnegs=[]
    total=acc=acc5=acc10=0
    
    # Indexing        
    cand_vecs = cand_vecs.astype(np.float32)
    dim = cand_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cand_vecs)
    
    if used_q:
        allow_redundant = True
        reused = set()
    else:
        allow_redundant = False
        
    with torch.no_grad():
        for row in tqdm(eval_data, desc="Scoring"):
            cur_q_key = (row['qry'], row['qry_image_path'])
            if cur_q_key in used_q:
                continue
            
            qry_vec = qry_dict[cur_q_key]
            positive = (row['pos_text'], row['pos_image_path'])
            
            D,I = index.search(qry_vec[None].astype(np.float32), pool_size)
            scores, idxs = D[0], I[0]

            # true pos
            pos_idx = cand_map.get(positive)
            if pos_idx in idxs:
                rank = int(np.where(idxs==pos_idx)[0][0])
                # pscore = scores[rank]
            else:
                rank = pool_size
                # pscore = float(qry_vec @ cand_dict[positive].T)
                
            # accuracy
            total+=1
            if rank==0: acc+=1
            if rank<5: acc5+=1
            if rank<10: acc10+=1
                            
            # hard-negs
            negs = np.array([j for j in idxs if j!=pos_idx])
            negs = negs[negs!=-1].tolist()
            
            qry_cands = []
            q_keys = []
            qry_vec_tensor = torch.tensor(qry_vec).unsqueeze(0).to(training_args.device)
                            
            for j in negs:
                D_key = cand_texts[j]
                if allow_redundant:
                    Q_keys = D2Q[D_key].difference(reused)
                else:
                    Q_keys = D2Q[D_key].difference(used_q)
                Q_keys = list(Q_keys)
                
                if not Q_keys:
                    continue
                
                if len(Q_keys)==1:
                    Q_key = Q_keys[0]
                else:
                    q_first_cand = torch.cat([torch.tensor(qry_dict[key]).unsqueeze(0) for key in Q_keys]).to(training_args.device)
                    ind = torch.argmax(qry_vec_tensor @ q_first_cand.T, dim=1)[0]
                    Q_key = Q_keys[ind.item()]
                                        
                qry_cands.append(torch.tensor(qry_dict[Q_key]).unsqueeze(0))
                q_keys.append(Q_key)
                
                if len(q_keys) >= first_n_cands:
                    break
                
            if not qry_cands:
                continue
            
            used_q.add(cur_q_key)
            
            if subset not in except_list:
                qry_cands_tensor = torch.cat(qry_cands).to(training_args.device)
                q_scores = (qry_vec_tensor @ qry_cands_tensor.T)[0].cpu().tolist()
                q_keys = [xx[1] for xx in sorted(zip(q_scores, q_keys), key=lambda x: x[0])]
            
            final_neg_keys = q_keys[:data_args.top_k]
            cv_rids = [Q2RID[qk] for qk in final_neg_keys]
            
            hnegs.append(row)
            hnegs[-1].update({
                'neg_samples': [
                    eval_data[sel]
                    for sel in cv_rids
                ]
                # 'neg_scores': [float(scores[j]) for j in sel]
            })
            
            final_neg_keys = set(final_neg_keys)
            used_q.update(final_neg_keys)
            if allow_redundant:
                reused.update(final_neg_keys)
        
    return hnegs

    
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

    # sampling params
    top_k   = data_args.top_k
    mult    = data_args.pool_mult

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

    # write prompts
    spath = os.path.join(data_args.encode_output_path, "system_prompt.txt")
    with open(spath, 'w') as fout:
        if data_args.Q_prompt: fout.write(data_args.Q_prompt + "\n")
        if data_args.D_prompt: fout.write(data_args.D_prompt)
            

    # Loop through each subset, encode and immediately compute score with hard negatives
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
                text_field="qry",
                img_path_field="qry_image_path",
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
                text_field="pos_text",
                img_path_field="pos_image_path",
            )
            eval_tgt_loader = DataLoader(
                eval_tgt_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
            )
            print('#Q:', len(eval_qry_dataset))
            print('#D:', len(eval_tgt_dataset))
                
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
            

        # Immediate scoring and hard negative sampling for this subset
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)

        # Build dicts
        qry_dict = { (tt['text'], tt['img_path']): vec for vec, tt in zip(qry_tensor, qry_index) }
        cand_vecs  = tgt_tensor
        cand_texts = [(tt['text'], tt['img_path']) for tt in tgt_index]
        cand_map = {text: i for i, text in enumerate(cand_texts)}
               
        eval_data = load_dataset(
            data_args.dataset_name,
            subset,
            split=data_args.dataset_split,
        )
        
        D2Q, Q2RID = construct_D2Q(eval_data)
        
        used_q = set()
        first_n_cands = top_k * mult
        pool_size = first_n_cands *2
        
        print("# Queres:", len(qry_dict.keys()), "# Candidates:", len(cand_texts))       
        
        hnegs = couple_samples(eval_data, used_q, D2Q, Q2RID, subset,
                       qry_dict, cand_texts, cand_map, cand_vecs, 
                       training_args, data_args, first_n_cands=first_n_cands, pool_size=pool_size)

        
        # 6. 남은 샘플 처리 (Phase 2: Leftover Processing)
        print(f"\nPhase 1 complete. {len(hnegs)} samples created.")
        all_query_keys = set(Q2RID.keys())
        leftover_keys = list(all_query_keys - used_q)
        
        if not leftover_keys:
            print("No leftover samples to process.")
        else:
            print(f"Phase 2: Found {len(leftover_keys)} leftover samples. Building temporary target index...")
            
        hnegs.extend(couple_samples(eval_data, used_q, D2Q, Q2RID, subset,
                        qry_dict, cand_texts, cand_map, cand_vecs,
                        training_args, data_args, first_n_cands=first_n_cands*2, pool_size=pool_size))
                
        print(len(hnegs))
        print(len(list(used_q)))
        
        # save
        base=data_args.encode_output_path
        with open(os.path.join(base,f"{subset}_hard_negatives.json"),'w') as f:
            json.dump(hnegs, f, indent=4)

if __name__ == "__main__":
    main()
