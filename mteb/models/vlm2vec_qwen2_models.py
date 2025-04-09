from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal
import traceback
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor #, Qwen2VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from src.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration

import pdb
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)

logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


class VLM2VecQwen2Wrapper:
    """Adapted from https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/model.py"""

    def __init__(
        self,
        model_name: str = "TIGER-Lab/VLM2Vec-Qwen2VL-2B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        requires_image_dependencies()
        if suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        ):
            import flash_attn  # noqa

        requires_package(self, "peft", model_name, "pip install 'mteb[peft]'")
        from peft import LoraConfig, PeftModel  # noqa

        self.pooling = "last"
        self.normalize = True
        self.temperature = 1.0
        self.hidden_size = 4096
        self.device = device

        # Loading the base model
        base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        config.use_cache = False
        # config.padding_side = "left"

        checkpoint_path = model_name if model_name else base_model_name
        #pdb.set_trace()
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # base_model.padding_side = "left"
        #pdb.set_trace()
        lora_config = LoraConfig.from_pretrained(model_name)
        lora_model = PeftModel.from_pretrained(
            base_model, model_name, config=lora_config
        )
        merged_model = lora_model.merge_and_unload()
        #pdb.set_trace()
        model = merged_model.to(torch.bfloat16)  # propagate dtype.
        #pdb.set_trace()
        # # Building the model on top of the base
        # if "LoRA" in model_name or 1 :
        #     lora_config = LoraConfig.from_pretrained(model_name)
        #     lora_model = PeftModel.from_pretrained(
        #         base_model, model_name, config=lora_config
        #     )
        #     merged_model = lora_model.merge_and_unload()
        #     model = merged_model.to(torch.bfloat16)  # propagate dtype.
        # else:
        #     model = base_model.to(torch.bfloat16)

        model.eval()
        model.to(device)
        self.mdl = model
        # print(base_model_name)
        image_processor = Qwen2VLImageProcessor.from_pretrained(base_model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(base_model_name)
        self.processor = Qwen2VLProcessor.from_pretrained(
            base_model_name,
            image_processor=image_processor, tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )

        # self.processor = AutoProcessor.from_pretrained(
        #     base_model_name,
        #     trust_remote_code=True,
        #     min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        # )

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        return self.get_text_embeddings(texts=sentences)

    def encode_input(self, input):
        try :
            hidden_states = self.mdl(**input, return_dict=True, output_hidden_states=True)
        except :
            print(traceback.format_exc())
            #pdb.set_trace()
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input["attention_mask"])
        return pooled_output

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

        # if self.pooling == "last":
        #     sequence_lengths = attention_mask.sum(dim=1) - 1
        #     batch_size = last_hidden_state.shape[0]
        #     reps = last_hidden_state[
        #         torch.arange(batch_size, device=last_hidden_state.device),
        #         sequence_lengths,
        #     ]
        # else:
        #     raise NotImplementedError
        # if self.normalize:
        #     reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        # return reps

    # reference: https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/collator.py
    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        text = "<|image_pad|> Represent the given image."
        all_image_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch:
                        inputs = self.processor(
                            text = text,
                            images = F.to_pil_image(b.to("cpu")),
                            return_tensors="pt",
                            # max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1)) # 1x256 -> 256x1
                        pixel_values.append(inputs["pixel_values"].unsqueeze(0))
                        image_grid_thw.append(inputs["image_grid_thw"].unsqueeze(0))
                    #pdb.set_trace()
                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2) # 1x256
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id) # 1x256
                    #pdb.set_trace()
                    pixel_values = torch.cat(pixel_values, dim=0) # [ [19276, 1176] ] -> [19276, 1176]
                    image_grid_thw = torch.cat(image_grid_thw, dim=0) #[ [1x3] ] -> [1x3]
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }
                    #pdb.set_trace()
                    image_outputs = self.encode_input(inputs)
                    all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch_images:
                        inputs = self.processor(
                            text = [text],
                            images = [b],
                            return_tensors="pt",
                            # max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"].unsqueeze(0))
                        image_grid_thw.append(inputs["image_grid_thw"].unsqueeze(0))

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }
                    try:
                        #pdb.set_trace()
                        image_outputs = self.encode_input(inputs)
                    except :
                        #pdb.set_trace()
                        print("ERROR")
                    all_image_embeddings.append(image_outputs.cpu().to(torch.float32))

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        self.all_image_embeddings = all_image_embeddings
        self.images = images
        return all_image_embeddings

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                input_ids = []
                batch_texts = texts[i : i + batch_size]
                for text in batch_texts:
                    inputs = self.processor(
                        text = [text],
                        images = None,
                        return_tensors="pt",
                        # max_length=256,
                        truncation=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))

                input_ids = torch._C._nn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                ).squeeze(2)
                attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                text_outputs = self.encode_input(inputs)
                all_text_embeddings.append(text_outputs.cpu().to(torch.float32))

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        self.all_text_embeddings = all_text_embeddings
        self.texts = texts
        return all_text_embeddings

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None
        kwargs.update(
            task_name=task_name, prompt_type=prompt_type, batch_size=batch_size
        )

        if texts is not None and images is None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)
            return text_embeddings

        if images is not None and texts is None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)
            return image_embeddings

        # text_embeddings is not None and image_embeddings is not None
        texts = iter(texts)
        all_fused_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in images:
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch:
                        text = next(texts)
                        inputs = self.processor(
                            f"<|image_pad|> Represent the given image with the following question: {text}",
                            [F.to_pil_image(b.to("cpu"))],
                            return_tensors="pt",
                            # max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"].unsqueeze(0))
                        image_grid_thw.append(inputs["image_grid_thw"].unsqueeze(0))

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    input_ids, pixel_values, image_grid_thw = [], [], []
                    for b in batch_images:
                        text = next(texts)
                        inputs = self.processor(
                            text = [f"<|image_pad|> Represent the given image with the following question: {text}"],
                            images= [b],
                            return_tensors="pt",
                            # max_length=256,
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                        pixel_values.append(inputs["pixel_values"].unsqueeze(0))
                        image_grid_thw.append(inputs["image_grid_thw"].unsqueeze(0))

                    input_ids = torch._C._nn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.processor.tokenizer.pad_token_id,
                    ).squeeze(2)
                    attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

                    pixel_values = torch.cat(pixel_values, dim=0)
                    image_grid_thw = torch.cat(image_grid_thw, dim=0)
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                    }

                    outputs = self.encode_input(inputs)
                    all_fused_embeddings.append(outputs.cpu().to(torch.float32))

        fused_embeddings = torch.cat(all_fused_embeddings, dim=0)
        return fused_embeddings


vlm2vec_training_datasets = {
    # MMEB-train
}

vlm2vec_qwen2 = ModelMeta(
    loader=partial(
        VLM2VecQwen2Wrapper,
        model_name="TIGER-Lab/VLM2Vec-Qwen2VL-2B",
    ),
    name="TIGER-Lab/VLM2Vec-Qwen2VL-2B",
    languages=["eng_Latn"],
    revision="7717deedf0631e6f520b7c83c8f82dcbc2c4c21e",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    framework=["PyTorch"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-LoRA",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
)

vlm2vec_qwen7 = ModelMeta(
    loader=partial(
        VLM2VecQwen2Wrapper,
        model_name="TIGER-Lab/VLM2Vec-Qwen2VL-7B",
    ),
    name="TIGER-Lab/VLM2Vec-Qwen2VL-7B",
    languages=["eng_Latn"],
    revision="f2f1c2194823b780632c628548d85a03939d896c",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=4_150_000_000,
    memory_usage_mb=7909,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TIGER-AI-Lab/VLM2Vec",
    public_training_data="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    framework=["PyTorch"],
    reference="https://huggingface.co/TIGER-Lab/VLM2Vec-Full",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
)
