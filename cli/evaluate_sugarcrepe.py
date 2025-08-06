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



class SugarCrepeDataset(Dataset):
    def __init__(self, data_dir, name, image_dir):
        self.name = name
        self.image_root = image_dir
        self.dataset = json.load(open(f"{data_dir}/{name}.json", 'r', encoding='utf-8'))
        self.dataset_keys = list(self.dataset.keys())
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item_key = self.dataset_keys[index]
        data_item = self.dataset[item_key]
        
        positive_caption = data_item['caption']
        negative_caption = data_item['negative_caption']
        image_path = os.path.join(self.image_root, data_item['filename'])
        image = Image.open(image_path).convert("RGB")
        
        return image, positive_caption, negative_caption

# +++ 새로운 어댑터 클래스 정의 +++
class SugarCrepeAdapterCollator:
    def __init__(self, eval_collator: EvalCollator):
        self.eval_collator = eval_collator
        self.system_prompt = "Given an image, summarize the provided image in one word. Given only text, describe the text in one word."

    def __call__(self, batch: list[tuple]):
        images, pos_captions, neg_captions = zip(*batch)

        image_tuples = [('<|image_pad|>', img, self.system_prompt, 'D') for img in images]
        pos_tuples = [("Find me an image that matches the given caption: " + txt, None, self.system_prompt, 'Q') for txt in pos_captions]
        neg_tuples = [("Find me an image that matches the given caption: " + txt, None, self.system_prompt, 'Q') for txt in neg_captions]
        
        image_batch = self.eval_collator(image_tuples)
        pos_batch = self.eval_collator(pos_tuples)
        neg_batch = self.eval_collator(neg_tuples)

        return {
            "image_batch": image_batch,
            "pos_batch": pos_batch,
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
    
    data_name = ['replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att', 'add_obj', 'add_att']
    
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

    sugar_crepe_final_collator = SugarCrepeAdapterCollator(eval_collator=base_eval_collator)

    overall_results = {}

    for subset in data_name:
        print_rank(f"===== Evaluating SugarCrepe subset: {subset} =====")
        
        eval_dataset = SugarCrepeDataset(
            data_dir=data_args.dataset_name,
            name=subset,
            image_dir=data_args.image_dir,
        )
        
        # 3. DataLoader에 최종 어댑터 collator를 인자로 전달
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=sugar_crepe_final_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        n_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Encoding and Scoring {subset}"):
                image_batch = batch_to_device(batch['image_batch'], training_args.device)
                pos_batch = batch_to_device(batch['pos_batch'], training_args.device)
                neg_batch = batch_to_device(batch['neg_batch'], training_args.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    img_reps = model(qry=image_batch)['qry_reps']
                    pos_reps = model(qry=pos_batch)['qry_reps']
                    neg_reps = model(qry=neg_batch)['qry_reps']


                pos_scores = (img_reps * pos_reps).sum(dim=-1)
                neg_scores = (img_reps * neg_reps).sum(dim=-1)
                
                n_correct += (pos_scores > neg_scores).sum().item()
                total_samples += img_reps.size(0)

        accuracy = (n_correct / total_samples) * 100 if total_samples > 0 else 0
        overall_results[subset] = accuracy
        
        score_dict = {"accuracy": accuracy, "correct": n_correct, "total": total_samples}
        score_file = os.path.join(output_dir, f"{subset}_score.json")
        with open(score_file, 'w') as f:
            json.dump(score_dict, f, indent=4)
        
        print_rank(f"\033[91m{subset} Accuracy: {accuracy:.2f}%\033[0m\n")

    print_rank("===== SugarCrepe Evaluation Summary =====")
    total_acc = [acc for acc in overall_results.values()]
    average_accuracy = np.mean(total_acc) if total_acc else 0
    for subset, acc in overall_results.items():
        print_rank(f"{subset:<15}: {acc:.2f}%")
    
    print_rank("-----------------------------------------")
    print_rank(f"\033[92m{'Average Accuracy':<15}: {average_accuracy:.2f}%\033[0m")

if __name__ == "__main__":
    main()