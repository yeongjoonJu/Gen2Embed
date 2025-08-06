from collections import defaultdict, OrderedDict, deque
import torch
import random
from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets, Features, Value
from PIL import Image
import os
import re, math
from torch.jit import isinstance

from src.model_utils import PHI3V, vlm_image_tokens, GEMMA3
from src.utils import print_master, print_rank

from torch.utils.data import Dataset, Sampler, BatchSampler
import json
 
TASK_MAP = {
    'ImageNet_1K': 'I_T_CLS',
    'N24News': 'I_T_CLS',
    'SUN397': 'I_T_CLS',
    'VOC2007': 'I_T_CLS',
    'MSCOCO_i2t': 'I_T_RET',
    'VisualNews_i2t': 'I_T_RET',
    'MSCOCO_t2i': 'T_I_RET',
    'NIGHTS': 'T_I_RET',
    'VisDial': 'T_I_RET',
    'VisualNews_t2i': 'T_I_RET',
    'Wiki-SS-NQ': 'T_I_RET',
    'CIRR': 'IT_I_RET',
    'FashionIQ': 'IT_I_RET',
    'EDIS': 'T_IT_RET',
    'WebQA': 'T_IT_RET',
    'A-OKVQA': 'IT_T_VQA',
    'ChartQA': 'IT_T_VQA',
    'DocVQA': 'IT_T_VQA',
    'InfographicsVQA': 'IT_T_VQA',
    'OK-VQA': 'IT_T_VQA',
    'Visual7W': 'IT_T_VQA',
    'MSCOCO': 'grounding',
}

def process_image(image, resolution, max_dim=1024):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid_high":
        image = image.resize((768, 768))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "mid_low":
        image = image.resize((448, 448))
    elif resolution == "clip":
        image = image.resize((336, 336))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        w, h = image.size
        
        if max(w, h) > 2 * min(w, h):
            if w > h:
                h = w / 2
            else:
                w = h / 2
                
        min_dim = min(w, h)
        max_dim = max(w, h)

        scale = 1.0
        if max_dim > 992:
            scale = 992 / max_dim
        if min_dim < 336:
            scale = 336 / min_dim

        w = int(w * scale)
        h = int(h * scale)
        
        image = image.resize((w, h), Image.LANCZOS)
        
    return image

def groupby_indices(subset_ids):
    groups = OrderedDict()
    for idx, sid in enumerate(subset_ids):
        groups.setdefault(sid, []).append(idx)
    return groups.items()        # [(sid, [i1,i2,…]), …]


class SameSubsetBatchSampler(BatchSampler):
    """
    Yield lists of indices so that each batch is taken entirely
    from one subset_id.
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool, seed: int = 42):
        # Build a mapping subset_id -> all indices with that ID
        self.rng = random.Random(seed)
        self.subset_to_indices = defaultdict(list)
        for idx, sid in enumerate(dataset.train_data["subset_id"]):
            self.subset_to_indices[sid].append(idx)
        
        for sid, idxs in self.subset_to_indices.items():
            print(f"Subset {sid} has {len(idxs)} samples")

        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Flatten to a list of (subset_id, shuffled indices)
        total = 0
        self.batches = []
        for sid, idxs in self.subset_to_indices.items():
            self.rng.shuffle(idxs)
            n = len(idxs)
            nb = n // batch_size if drop_last else math.ceil(n / batch_size)
            for i in range(nb):
                start = i * batch_size
                end = start + batch_size
                chunk = idxs[start:end]
                if len(chunk) < batch_size and drop_last:
                    continue
                total += len(chunk)
                self.batches.append(chunk)
        
        print(f"Total {len(self.batches)} batches with {total} samples")

        self.rng.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    
TEXT_RET = [
  'msmarco_passage',
  'hotpotqa',
  'msmarco_document',
  'eli5',
  'squad',
  'fiqa',
  'nq',
  'arguana',
  'trivial',
  'fever',
  'quora',
  'stack_overflow_dup_questions',
  'scidocsrr'
]

MM_RET = [
    "VisDial",
    "CIRR",
    "VisualNews_t2i",
    "VisualNews_i2t",
    "MSCOCO_t2i",
    "MSCOCO_i2t",
    "NIGHTS",
    "WebQA",
    "OVEN",
    "FashionIQ",
    "EDIS",
    "Wiki-SS-NQ"
]

GROUNDING = [
    "MSCOCO", "RefCOCO", "Visual7W-Pointing", "RefCOCO-Matching"
]
    
class TrainTextImageDataset(Dataset):
    def __init__(self, data_args, model_args, hard_negatives=0):
        self.data_args = data_args
        self.model_args = model_args
        self.hard_negatives = hard_negatives
        
        train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        
        for subset in data_args.subset_name:
            if hard_negatives > 0:
                with open(f"{data_args.dataset_name}/{subset}_hard_negatives.json", "r") as f:
                    subset_data = json.load(f)
                train_data.extend(subset_data)
            else:
                subset_data = load_dataset(self.data_args.dataset_name, subset, split=data_args.split_name)[0]
                if len(subset_data) > data_args.num_sample_per_subset:
                    subset_data = subset_data.shuffle(seed=42).select(range(data_args.num_sample_per_subset))
                    
                subset_data = subset_data.add_column(
                    "subset_id", [subset] * len(subset_data)
                )
                train_data.append(subset_data)
        
        if self.hard_negatives > 0:
            self.train_data = train_data
        else:
            self.train_data = concatenate_datasets(train_data)

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        
        if self.hard_negatives > 0:
            neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
            if len(neg_texts) < self.hard_negatives:
                mult = round(self.hard_negatives / len(neg_texts)) + 2
                neg_texts = neg_texts * mult
                neg_image_paths = neg_image_paths * mult
                
            neg_texts = neg_texts[:self.hard_negatives]
            neg_image_paths = neg_image_paths[:self.hard_negatives]
        
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            # neg_texts = [neg_texts]
            # neg_image_paths = [neg_image_paths]
        _qry_texts, _qry_images, _pos_texts, _pos_images = [], [], [], [] # , _neg_texts, _neg_images, [], []
        backbone = self.model_args.model_backbone #, neg_text, neg_image_path 
        for qry_text, qry_image_path, pos_text, pos_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):#, neg_texts, neg_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                # neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            # neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            # _neg_texts.append(neg_text)
            # _neg_images.append(neg_image)
        
        out =  {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images}
        
        if self.hard_negatives > 0:
            if backbone != PHI3V:
                neg_texts = [t.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) for t in neg_texts]
            neg_image = [self._get_image(ii) for ii in neg_image_paths]
            out.update({'neg_text': neg_texts, 'neg_image': neg_image})
            
        return out
    
    
class EffTrainTextImageDataset(Dataset):
    def __init__(self, data_args, model_args, hard_negatives=0):
        self.data_args = data_args
        self.model_args = model_args
        self.hard_negatives = hard_negatives
        
        train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        
        for subset in data_args.subset_name:
            if hard_negatives > 0:
                with open(f"{data_args.dataset_name}/{subset}_hard_negatives.json", "r") as f:
                    subset_data = json.load(f)
                train_data.extend(subset_data)
            else:
                subset_data = load_dataset(self.data_args.dataset_name, subset, split=data_args.split_name)[0]
                if len(subset_data) > data_args.num_sample_per_subset:
                    subset_data = subset_data.shuffle(seed=42).select(range(data_args.num_sample_per_subset))
                    
                subset_data = subset_data.add_column(
                    "subset_id", [subset] * len(subset_data)
                )
                train_data.append(subset_data)
        
        if self.hard_negatives > 0:
            self.train_data = train_data
        else:
            self.train_data = concatenate_datasets(train_data)

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            # neg_texts = [neg_texts]
            # neg_image_paths = [neg_image_paths]
        
        if self.hard_negatives > 0:
            neg_samples = self.train_data[data_idx]['neg_samples']
            for i in range(len(neg_samples)):
                qry_texts.append(neg_samples[i]["qry"])
                qry_image_paths.append(neg_samples[i]['qry_image_path'])
                pos_texts.append(neg_samples[i]['pos_text'])
                pos_image_paths.append(neg_samples[i]['pos_image_path'])
        
        _qry_texts, _qry_images, _pos_texts, _pos_images = [], [], [], [] # , _neg_texts, _neg_images, [], []
        backbone = self.model_args.model_backbone #, neg_text, neg_image_path 
        for qry_text, qry_image_path, pos_text, pos_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):#, neg_texts, neg_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                # neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            # neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            # _neg_texts.append(neg_text)
            # _neg_images.append(neg_image)
        
        out =  {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images}
        
        return out
    
                
class TrainTextNegativeDataset(Dataset):
    def __init__(self, data_args, model_args, hard_negatives=0):
        self.data_args = data_args
        self.model_args = model_args
        self.hard_negatives = hard_negatives
        train_data = []
        
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        
        for subset in data_args.subset_name:
            with open(f"{data_args.dataset_name}/{subset}_hard_negatives.json", "r") as f:
                subset_data = json.load(f)
                
            train_data.extend(subset_data)
            
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        qry_texts = self.train_data[data_idx]["qry"]
        if 'pos_text' in self.train_data[data_idx]:
            pos_texts = self.train_data[data_idx]["pos_text"]
        else:
            pos_texts = self.train_data[data_idx]["pos"]
            
        if 'neg_text' in self.train_data[data_idx]:
            neg_texts = self.train_data[data_idx]["neg_text"][:self.hard_negatives]
        else:
            neg_texts = self.train_data[data_idx]["negs"][:self.hard_negatives]
        
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            pos_texts = [pos_texts]
            # neg_image_paths = [neg_image_paths]
            
        _qry_texts, _pos_texts = [], [] # , _neg_texts, _neg_images, [], []
        for qry_text, pos_text in zip(qry_texts, pos_texts): #, neg_texts, neg_image_paths):
            _qry_texts.append(qry_text)
            _pos_texts.append(pos_text)

        return {"query_text": _qry_texts, "pos_text": _pos_texts, "neg_text": neg_texts}
        
        
class TrainJointDataset(Dataset):
    def __init__(self, data_args, model_args, apply_prompt=True):
        self.data_args = data_args
        self.model_args = model_args
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name} split: {data_args.split_name}")
        
        not_labeled = [
            'msmarco_passage',
            'hotpotqa',
            'msmarco_document',
            'eli5',
            'squad',
            'fiqa',
            'nq',
            'arguana',
            'trivial',
            'fever',
            'quora',
            'stack_overflow_dup_questions',
            'scidocsrr',
            "MSCOCO",
            "VisDial",
            "CIRR",
            "VisualNews_t2i",
            "VisualNews_i2t",
            "MSCOCO_t2i",
            "MSCOCO_i2t",
            "NIGHTS",
            "WebQA"
        ]
        
        with open(self.data_args.dataset_name, "r") as f:
            data = json.load(f)
        
        self.train_data = []
        for subset in data_args.subset_name:
            print_rank(f"Loading subset: {subset}")
            subset_data = data[subset]
            if len(subset_data) > data_args.num_sample_per_subset:# and subset in not_labeled:
                subset_data = random.sample(subset_data, data_args.num_sample_per_subset)
                
            for sample in subset_data:
                if apply_prompt:
                    task_inst = sample['task_instruction'].strip()
                    # if not sample['qry_image_path']:
                    #     task_inst = task_inst.replace(" in one word", "")
                    # if sample['pos_image_path']:
                    represent_prompt = '\n'+sample['represent_prompt'].strip()
                    # else:
                    #     represent_prompt = ""
                        
                    sample['qry'] = sample['qry'].strip() + "\n" + task_inst
                    sample['pos_text'] = sample['pos_text'].strip() + represent_prompt
                    
                sample['subset_id'] = subset
                if not 'answer' in sample:
                    sample['answer'] = ""
                    
                self.train_data.append(sample)

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        qry_texts, qry_image_paths, pos_texts, pos_image_paths, answers = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"],
            self.train_data[data_idx]['answer']
        )
        # if 'neg_text' in self.train_data.column_names:
        #     neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
        # else:
        #     neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            
            # neg_texts = [neg_texts]
            # neg_image_paths = [neg_image_paths]
            
        _qry_texts, _qry_images, _pos_texts, _pos_images = [], [], [], [] # , _neg_texts, _neg_images , [], []
        backbone = self.model_args.model_backbone
        for qry_text, qry_image_path, pos_text, pos_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths): # , neg_texts, neg_image_paths
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                # neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            # neg_image = self._get_image(neg_image_path) if neg_image_path else None
            # if (not qry_text and not qry_image) or (not pos_text and not pos_image):
            #     print("empty inputs")
            #     continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            
            # _neg_texts.append(neg_text)
            # _neg_images.append(neg_image)

        return {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images}#, 'answer': answers}
                # "neg_text": _neg_texts, "neg_image": _neg_images}


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field,
                apply_represent_prompt=False, in_one_word_Q=False, in_one_word_D=False):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        
        # if 'Y-J-Ju' in self.data_args.dataset_name:
        #     task_inst = self.eval_data[0]['task_instruction'].strip()
        #     task_inst = task_inst + ' in one word' if in_one_word_Q else task_inst
        #     if subset!='MSCOCO':
        #         self.eval_data = self.eval_data.map(
        #             lambda x: {
        #                 'qry_text': task_inst.strip() + ":\n" + x['qry_text'].strip(),
        #             }
        #         )
        #     if apply_represent_prompt or subset=='RefCOCO-Matching':
        #         represent_prompt = self.eval_data[0]['represent_prompt']
        #         if subset=='RefCOCO-Matching':
        #             self.eval_data = self.eval_data.map(
        #                 lambda x: {
        #                     'tgt_text': [represent_prompt.strip() +"\n"+ tgt.strip() for tgt in x['tgt_text']],
        #                 }
        #             )
        #         else:
        #             represent_prompt = represent_prompt + ' in one word' if in_one_word_D else represent_prompt
        #             self.eval_data = self.eval_data.map(
        #                 lambda x: {
        #                     'tgt_text': [tgt.strip()+"\n"+represent_prompt.strip() for tgt in x['tgt_text']],
        #                 }
        #             )
        
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })
        if 'qry' in text_field or 'matching' in subset.lower():
            self.system_prompt = data_args.Q_prompt
            self.type = 'Q'
        else:
            self.system_prompt = data_args.D_prompt
            self.type = 'D'
            
    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])

        return text, self._get_image(img_path), self.system_prompt, self.type

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if type(row[img_path_field]) is list:
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data
        
    
class EvalTextDataset(Dataset):
    def __init__(self, data_args, model_args, subset_data, text_field):
        
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone
        
        if text_field=='pos':
            self.system_prompt = data_args.D_prompt
            self.type = 'D'
        else:
            self.system_prompt = data_args.Q_prompt
            self.type = 'Q'

        self.dataset = set()
        for i in range(len(subset_data)):
            if self.type=='Q':
                self.dataset.add(
                    subset_data[i]['prompt'] + '\n' + subset_data[i][text_field]
                )
            else:
                self.dataset.add(subset_data[i][text_field][0])
                self.dataset.update(set(subset_data[i]['neg']))
        
        self.dataset = list(self.dataset)
        print("# Removed redundancy", len(subset_data) - len(self.dataset))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]

        return text, None, self.system_prompt, self.type



class FlickrDataset(Dataset):
    def __init__(self, modality, model_backbone):
        self.model_backbone = model_backbone
        self.modality = modality
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()
        else:
            self.eval_data, self.image_names = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def get_image_data(self):
        eval_data, image_names = [], []
        inst = "<|image_1|> Find an image caption describing the given image."

        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        eval_data, image_names = [], []
        inst = ""
        for row in self.raw_data:
            for caption in row["caption"]:
                # eval_data.append((caption, None))
                eval_data.append((inst + caption, None))
                image_names.append(row["filename"])
        return eval_data, image_names
