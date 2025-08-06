import json
import sys
import os
import numpy as np
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import HfArgumentParser, AutoProcessor, AutoConfig, ProcessorMixin

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.collator import EvalCollator
from src.utils import print_rank
from src.model_utils import get_backbone_name

class SugarCrepePlusPlusDataset(Dataset):
    def __init__(self, data_dir, image_dir, name):
        self.name = name
        self.image_root = image_dir
        self.dataset = json.load(open(f"{data_dir}/{name}.json", 'r', encoding='utf-8'))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]
        
        caption1 = data_item['caption']
        caption2 = data_item['caption2']
        negative_caption = data_item['negative_caption']
        
        image_path = os.path.join(self.image_root, data_item['filename'])
        image = Image.open(image_path).convert("RGB")
        
        return image, caption1, caption2, negative_caption

class SugarCrepePlusPlusAdapterCollator:
    def __init__(self, eval_collator: EvalCollator):
        self.eval_collator = eval_collator
        self.system_prompt = "Given an image, summarize the provided image in one word. Given only text, describe the text in one word."

    def __call__(self, batch: list[tuple]):
        images, captions1, captions2, neg_captions = zip(*batch)

        img_tuples = [('<|image_pad|>', img, self.system_prompt, 'D') for img in images]
        cap1_tuples = [("Find me an image that matches the given caption: " + txt, None, self.system_prompt, 'Q') for txt in captions1]
        cap2_tuples = [("Find me an image that matches the given caption: " + txt, None, self.system_prompt, 'Q') for txt in captions2]
        neg_tuples = [("Find me an image that matches the given caption: " + txt, None, self.system_prompt, 'Q') for txt in neg_captions]
        
        img_batch = self.eval_collator(img_tuples)
        cap1_batch = self.eval_collator(cap1_tuples)
        cap2_batch = self.eval_collator(cap2_tuples)
        neg_batch = self.eval_collator(neg_tuples)

        return {
            "img_batch": img_batch,
            "cap1_batch": cap1_batch,
            "cap2_batch": cap2_batch,
            "neg_batch": neg_batch,
        }

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    output_dir = os.path.join(data_args.encode_output_path, "sugarcrepe_results")
    os.makedirs(output_dir, exist_ok=True)
    
    data_name = ['swap_obj', 'swap_att','replace_obj', 'replace_att', 'replace_rel']
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_name if model_args.checkpoint_path is None else model_args.checkpoint_path,
        trust_remote_code=True,
    )

    hf_config = AutoConfig.from_pretrained(
        model_args.model_name if model_args.checkpoint_path is None else model_args.checkpoint_path,
        trust_remote_code=True
    )
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    base_eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    sugarcrepe_pp_collator = SugarCrepePlusPlusAdapterCollator(eval_collator=base_eval_collator)

    overall_results = {}

    for subset in data_name:
        print_rank(f"===== Evaluating SugarCrepe subset: {subset} =====")
        
        eval_dataset = SugarCrepePlusPlusDataset(data_dir=data_args.dataset_name, image_dir=data_args.image_dir, name=subset)

        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=sugarcrepe_pp_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        
        correct_img_text = 0
        correct_text_only = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Encoding and Scoring {subset}"):
                img_batch = batch_to_device(batch['img_batch'], training_args.device)
                cap1_batch = batch_to_device(batch['cap1_batch'], training_args.device)
                cap2_batch = batch_to_device(batch['cap2_batch'], training_args.device)
                neg_batch = batch_to_device(batch['neg_batch'], training_args.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    img_reps = model(qry=img_batch)['qry_reps']
                    cap1_reps = model(qry=cap1_batch)['qry_reps']
                    cap2_reps = model(qry=cap2_batch)['qry_reps']
                    neg_reps = model(qry=neg_batch)['qry_reps']

                # 1. Image-to-Text Task
                cos_img_p1 = (img_reps * cap1_reps).sum(dim=-1)
                cos_img_p2 = (img_reps * cap2_reps).sum(dim=-1)
                cos_img_neg = (img_reps * neg_reps).sum(dim=-1)
                correct_img_text += ((cos_img_p1 > cos_img_neg) & (cos_img_p2 > cos_img_neg)).sum().item()

                # 2. Text-Only Task
                cos_p1_p2 = (cap1_reps * cap2_reps).sum(dim=-1)
                cos_p1_neg = (cap1_reps * neg_reps).sum(dim=-1)
                cos_p2_neg = (cap2_reps * neg_reps).sum(dim=-1)
                correct_text_only += ((cos_p1_p2 > cos_p1_neg) & (cos_p1_p2 > cos_p2_neg)).sum().item()

                total_samples += img_reps.size(0)

        acc_img_text = (correct_img_text / total_samples) * 100 if total_samples > 0 else 0
        acc_text_only = (correct_text_only / total_samples) * 100 if total_samples > 0 else 0
        overall_results[subset] = {'img_text': acc_img_text, 'text_only': acc_text_only}
                
        score_dict = {
            "Image-to-Text Accuracy": acc_img_text, 
            "Text-Only Accuracy": acc_text_only,
            "correct_img_text": correct_img_text,
            "correct_text_only": correct_text_only,
            "total": total_samples
        }
        score_file = os.path.join(output_dir, f"{subset}_score.json")
        with open(score_file, 'w') as f:
            json.dump(score_dict, f, indent=4)
        
        with open(score_file, 'w') as f: json.dump(score_dict, f, indent=4)
        
        print_rank(f"\033[91m{subset} Image-to-Text Accuracy: {acc_img_text:.2f}%\033[0m")
        print_rank(f"\033[91m{subset} Text-Only Accuracy: {acc_text_only:.2f}%\033[0m\n")

    # print_rank("===== SugarCrepe Evaluation Summary =====")
    # total_acc = [acc for acc in overall_results.values()]
    # average_accuracy = np.mean(total_acc) if total_acc else 0
    # for subset, acc in overall_results.items():
    #     print_rank(f"{subset:<15}: {acc:.2f}%")
    
    # print_rank("-----------------------------------------")
    # print_rank(f"\033[92m{'Average Accuracy':<15}: {average_accuracy:.2f}%\033[0m")

if __name__ == "__main__":
    main()