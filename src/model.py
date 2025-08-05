from typing import Dict, Optional
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel, PromptTuningConfig, TaskType, PromptTuningInit
from src.arguments import ModelArguments, TrainingArguments
from src.model_utils import LLAVA_NEXT, QWEN2_VL, PHI3V, QWEN2_5_VL, get_backbone_name, print_master, backbone2model
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from typing import Any, List, Union
from src.model_utils import process_vlm_inputs_fns
# from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from PIL import Image
from torch.utils.data import DataLoader
from mteb.encoder_interface import PromptType
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss, VarNegDistributedContrastiveLoss

class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 hard_negatives: int = 0,
                 Q_prompt: str = None,
                 D_prompt: str = None,
                ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.Q_prompt = Q_prompt
        self.D_prompt = D_prompt
        
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
        loss_fn_cls = SimpleContrastiveLoss
        if self.is_ddp:
            # if hard_negatives > 0:
            #     loss_fn_cls = VarNegDistributedContrastiveLoss
            # else:
            loss_fn_cls = DistributedContrastiveLoss
            
        self.loss_fn = loss_fn_cls(temperature=self.temperature)
            
    def encode_input(self, input):
        # input = {k: v.to(self.encoder.device) for k, v in input.items()}
        outputs = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        labels = input.get('labels', None)
        pooled_output = self._pooling(hidden_states, input['attention_mask'], labels=labels)
        
        if labels is not None:
            return pooled_output, outputs.loss
        
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask, labels=None):
        batch_size, seq_len, dim = last_hidden_state.size()
                
        if labels is not None:
            label_mask = labels != -100
            
            has_label = label_mask.any(dim=1) # (B,) bool
            
            first_label_idx = label_mask.int().argmax(dim=1)
            eos_idx = attention_mask.sum(dim=1) - 1
            
            pooling_idx = torch.where(
                has_label, first_label_idx, eos_idx
            ) # (B,)
            
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), pooling_idx,
            ] # (B, D)
        
        else:
            if self.pooling == 'last' or self.pooling == 'eos':
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                batch_size = last_hidden_state.shape[0]
                if left_padding:
                    # Get the vectors at the last position
                    reps = last_hidden_state[torch.arange(batch_size), -1, :]
                else:
                    # Calculate last 1 position in the original tensor
                    eos_indices = attention_mask.sum(dim=1) - 1
                    # Get the vectors at the last 1 position of each attention mask
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
            else:
                raise NotImplementedError
            
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    
    @classmethod
    def prepare_soft_prompting(cls, base_model, model_args, processor=None):
        prompt_tuning_init_text="Given an image, summarize the provided image in one word. Given only text, describe the text in one word."
        # <|im_start|>system\n  <|im_end|>\n
        if processor is None:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name,
                trust_remote_code=True,
                num_crops=model_args.num_crops,
            )
        init_ids = processor.tokenizer(prompt_tuning_init_text).input_ids
        num_virtual_tokens = len(init_ids)
        new_tokens = [f'<|sys_{k}|>' for k in range(num_virtual_tokens)]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        virtual_prompt = "".join(new_tokens)
        emb_weight = base_model.get_input_embeddings().weight
        
        if emb_weight.shape[0] < len(processor.tokenizer):
            base_model.resize_token_embeddings(len(processor.tokenizer))
        
        new_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
        
        with torch.no_grad():
            for k, init_id in enumerate(init_ids):
                emb_weight.data[new_ids[k]] = emb_weight.data[init_id].clone()
        
        emb_weight.requires_grad = True
        
        def mask_grad_hook(grad):
            # grad: [vocab_size, hidden_dim]
            mask = torch.zeros_like(grad)
            mask[new_ids, :] = 1.0
            return grad * mask
        
        emb_weight.register_hook(mask_grad_hook)
        
        for n, p in base_model.named_parameters():
            if not 'embed_tokens' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        
        return base_model, processor, virtual_prompt

    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments=None, processor=None, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')
        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
                
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            # model_args.model_name,
            base_model = backbone2model[model_backbone].from_pretrained(
                getattr(model_args, 'from_ckpt', model_args.model_name),
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)
            
        if model_args.prompt_tuning:
            print_master(f'Loading prompt adapter from {base_model}')
            base_model, processor, virtual_prompt = cls.prepare_soft_prompting(base_model, model_args, processor)
            
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                Q_prompt=virtual_prompt,
                D_prompt=virtual_prompt
            )
            
            cls.processor = processor
            
        elif model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            target_modules = model_args.lora_target_modules.split(',')
            
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                hard_negatives=training_args.hard_negatives
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                hard_negatives=training_args.hard_negatives
            )
        
        if not model_args.prompt_tuning:
            cls.processor = AutoProcessor.from_pretrained(
                model_args.model_name,
                trust_remote_code=True,
                num_crops=model_args.num_crops,
            )
        
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, **kwargs):
        # Loading the base model
        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')

        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL}:
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
                
        elif model_args.model_backbone == PHI3V:
            # Loading the base model
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                checkpoint_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # Building the model on top of the base
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)

            merged_model = lora_model.merge_and_unload()
            model = cls(
                encoder=merged_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize
            )

        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        
    def forward_across_layers(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps_across_layers = []
        tgt_reps_across_layers = []
        
        if qry is not None:
            qry_reps = self.encoder(**qry, return_dict=True, output_hidden_states=True)
            for emb in qry_reps.hidden_states:
                pooled_output = self._pooling(emb, qry['attention_mask'])
                qry_reps_across_layers.append(pooled_output.cpu())
        if tgt is not None:
            tgt_reps = self.encoder(**tgt, return_dict=True, output_hidden_states=True)
            for emb in tgt_reps.hidden_states:
                pooled_output = self._pooling(emb, tgt['attention_mask'])
                tgt_reps_across_layers.append(pooled_output.cpu())
                
        return {"qry_reps": qry_reps_across_layers, "tgt_reps": tgt_reps_across_layers}
    
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()
    

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        if type(qry_reps) is tuple:
            qry_reps, causal_loss = qry_reps
        else:
            causal_loss = None
        
        counts = None
        if tgt is not None and 'counts' in tgt:
            counts = tgt['counts']
            del tgt['counts']
            
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            outs =  {"qry_reps": qry_reps, "tgt_reps": tgt_reps}
            if counts is not None:
                outs.update({"counts": counts})
                
            return outs
        
        if counts is None:
            loss = self.loss_fn(qry_reps, tgt_reps)
        else:
            loss = self.loss_fn(qry_reps, tgt_reps, counts)
            
        return loss
        

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))


class VOEQwen2VLModel(nn.Module):
    def __init__(self, model_name, ckpt_path=None, lora=False, use_flash_attn=True, gen=False, **kwargs):
        super().__init__()
        config = AutoConfig.from_pretrained(ckpt_path if ckpt_path is not None else model_name, trust_remote_code=True)
        
        config._attn_implementation = "flash_attention_2" if use_flash_attn else 'eager'
        config.vision_config._attn_implementation = "flash_attention_2"
        self.system_prompt = "Given an image, summarize the provided image in one word. Given only text, describe the text in one word."
        
        if gen:
            from transformers import Qwen2VLForConditionalGeneration as Qwen2VL
        else:
            from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration as Qwen2VL
            
        base_model = Qwen2VL.from_pretrained(
            ckpt_path if ckpt_path is not None else model_name, # model_name, #
            torch_dtype=torch.bfloat16,
            config=config
        )
        
        if lora and ckpt_path is not None:
            print(f"Loading LoRA adapter from {ckpt_path}")
            lora_config = LoraConfig.from_pretrained(ckpt_path)
            lora_model = PeftModel.from_pretrained(base_model, ckpt_path, config=lora_config)
            self.model = lora_model.merge_and_unload()
        else:
            self.model = base_model
        
        self.model = self.model.to(dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(
            ckpt_path if ckpt_path is not None else model_name, # model_name, #
            trust_remote_code=True,
        )
        
        self.process_vlm_fn = process_vlm_inputs_fns['qwen2_vl']
        self.pooling = "last"
        self.normalize = True
        self.max_len = 2048
        
    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    
    def encode_input(self, input, output_attentions=False):
        input = {k: v if isinstance(v, list) else v.to(self.model.device) for k, v in input.items()}
        outputs = self.model(**input, return_dict=True, output_hidden_states=True,
                                   output_attentions=output_attentions)
        hidden_states = outputs.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        
        if output_attentions:
            return pooled_output, outputs.attentions
        
        return pooled_output
    
    def generate(self, query, image=None, system_prompt=None, do_sample=False, **kwargs):
        """
        Generate text based on the input query and image.
        Args:
            query (str): The input text query.
            image (Image.Image, optional): The input image.
            system_prompt (str, optional): System prompt to guide the generation.
        Returns:
            str: Generated text response.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
                
        if image is not None:
            query = "<|image_pad|>" + query
        
        prompt = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'
                
        inputs = self.processor(text=[prompt], images=[image] if image is not None else None, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, do_sample=do_sample,
                                            max_new_tokens=256, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
    def forward(self, texts=None, images=None, output_attentions=False, return_inputs=False):
        assert texts is not None or images is not None
        if texts is not None and type(texts) is str:
            texts = [texts]
        if images is not None and type(images) is Image:
            images = [images]
            
        if images is None:
            images = [None for _ in range(len(texts))]
            
        with torch.no_grad():
            inputs = self.process_vlm_fn({'text': texts, 'image': images},
                                        processor = self.processor, max_length = self.max_len,
                                        system_prompt=self.system_prompt)
            outputs = self.encode_input(inputs, output_attentions)
            
            if return_inputs:
                return inputs, outputs
            
            return outputs
    
    def get_probs(self, texts=None, images=None, temperature=1.0):
        assert texts is not None or images is not None
        if texts is not None and type(texts) is str:
            texts = [texts]
        if images is not None and type(images) is Image:
            images = [images]
            
        if images is None:
            images = [None for _ in range(len(texts))]
            
        with torch.no_grad():
            inputs = self.process_vlm_fn({'text': texts, 'image': images},
                                        processor = self.processor, max_length = self.max_len,
                                        system_prompt=self.system_prompt)
            
            inputs = {k: v if isinstance(v, list) else v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=False,
                                    output_attentions=False)
            logits = outputs.logits[:,-1]
            
            hidden_states = outputs.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            
            return torch.softmax(logits/temperature, dim = 1), pooled_output
            
    def encode(
        self,
        sentences: List[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.get_text_embeddings(sentences, **kwargs)
    
    def get_text_embeddings(self, texts: List[str], batch_size: int=32, **kwargs: Any) -> torch.Tensor:
        """
        Get text embeddings for a list of texts.
        Args:
            texts (List[str]): List of input texts.
            batch_size (int): Batch size for processing.
            **kwargs: Additional arguments for the model.
        Returns:
            torch.Tensor: Tensor of text embeddings.
        """
                
        if kwargs['prompt_type']==PromptType.query:
            role = 'Q'
        else:
            role = 'D'
        
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i : i + batch_size]
                inputs = self.process_vlm_fn({'text': batch_texts, 'image': [None for _ in range(len(batch_texts))]}, 
                                             processor = self.processor, max_length = self.max_len,
                                             system_prompt=self.system_prompt, role=role)
                batch_embeddings = self.encode_input(inputs)
                all_embeddings.append(batch_embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)
    
    def get_image_embeddings(
        self,
        images: Union[List[Image.Image], DataLoader],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        이미지 리스트 또는 DataLoader를 입력받아 배치 단위로 이미지 임베딩을 생성합니다.
        DaVidInternVL2_5 모델의 encode(image= ...) 메서드를 활용합니다.
        """
        
        if kwargs['prompt_type']==PromptType.query:
            role = 'Q'
        else:
            role = 'D'
            
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            # DataLoader의 경우 배치 단위로 이미 제공됨
            if isinstance(images, DataLoader):
                for batch in tqdm(images, desc="Encoding images"):
                    inputs = self.process_vlm_fn({'text': ["<|image_pad|>"]*len(batch), 'image': batch}, 
                                             processor = self.processor, max_length = self.max_len,
                                             system_prompt=self.system_prompt, role=role)
                    batch_embeddings = self.encode_input(inputs)
                    all_embeddings.append(batch_embeddings.cpu())
            else:
                for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
                    batch_imgs = images[i : i + batch_size]
                    inputs = self.process_vlm_fn({'text': ["<|image_pad|>"]*len(batch_imgs), 'image': batch},
                                             processor = self.processor, max_length = self.max_len,
                                             system_prompt=self.system_prompt, role=role)
                    batch_embeddings = self.encode_input(inputs)
                    all_embeddings.append(batch_embeddings.cpu())
                    
        return torch.cat(all_embeddings, dim=0)
    
    
    def get_fused_embeddings(
        self,
        texts: List[str],
        images: Union[List[Image.Image], DataLoader],
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        
        if kwargs['prompt_type']==PromptType.query:
            role = 'Q'
        else:
            role = 'D'
        
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            # DataLoader 형태라면 텍스트 리스트도 배치 단위로 나누어 전달합니다.
            if isinstance(images, DataLoader):
                idx = 0
                for batch in tqdm(images, desc="Encoding fused embeddings"):
                    batch_size_actual = len(batch)
                    batch_texts = texts[idx : idx + batch_size_actual]
                    idx += batch_size_actual
                    batch_texts = [f"<|image_pad|> {te}" for te in batch_texts]
                    inputs = self.process_vlm_fn({'text': batch_texts, 'image': batch},
                                                processor = self.processor, max_length = self.max_len,
                                                system_prompt=self.system_prompt, role=role)
                    batch_embeddings = self.encode_input(inputs)
                    all_embeddings.append(batch_embeddings.cpu())
            else:
                assert len(texts) == len(images), "#Text and #Image should be matched"
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding fused embeddings"):
                    batch_texts = texts[i : i + batch_size]
                    batch_imgs = images[i : i + batch_size]
                    batch_texts = [f"<|image_pad|> {te}" for te in batch_texts]
                    inputs = self.process_vlm_fn({'text': batch_texts, 'image': batch_imgs},
                                                processor = self.processor, max_length = self.max_len,
                                                system_prompt=self.system_prompt, role=role)
                    batch_embeddings = self.encode_input(inputs)
                    all_embeddings.append(batch_embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)