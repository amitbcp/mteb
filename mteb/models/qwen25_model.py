import pdb
import logging

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)
from functools import partial
from typing import Any, Literal
from PIL import Image
import bm25s
import Stemmer

import pytrec_eval
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import math
import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import io
import base64
from tqdm import tqdm
from .oci_utils import *

logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]
from torchvision.transforms import InterpolationMode

class ResizeIfLarger:
        def __init__(self, max_size):
            self.max_height, self.max_width = max_size

        def __call__(self, img: Image.Image):
            if img.height > self.max_height or img.width > self.max_width:
                # Resize while maintaining aspect ratio
                img.thumbnail((self.max_width, self.max_height), Image.ANTIALIAS)
            return img

def get_default_transform(max_size=(1024, 1024)):
    requires_image_dependencies()
    from torchvision import transforms


    return transforms.Compose([
        transforms.Resize(max_size, interpolation=InterpolationMode.BILINEAR),
        transforms.PILToTensor(),
    ])
    # return transforms.Compose([transforms.PILToTensor()])



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_column_name = image_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column_name]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            # Assume the image is already in a usable format (e.g., PIL Image)
            image = image
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

def custom_collate_fn(batch):
    return batch

class Qwen25BM25Wrapper:
    def __init__(
            self,
            model="qwen25bm25",
            model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
            previous_results: str = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            transform=None,
            **kwargs,
        ):

            self.model_name = model
            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )
            self.model =  bm25s.BM25()
            if transform is None:
                self.transform = get_default_transform()

            # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
            self.ml_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            # default processer
            self.processor = AutoProcessor.from_pretrained(model_path)



    # @classmethod
    # def name(self):
    #     return "qwen25bm25"

    def search(
        self,
        corpus: dict[str, dict[str, str | Image.Image]],
        queries: dict[str, dict[str, str | Image.Image]],

        top_k: int,
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        logger.info("Encoding Corpus...")
        corpus_ids = list(corpus.keys())
        # pdb.set_trace()
        corpus_with_ids = [
            {
                "doc_id": cid,
                **(
                    {"text": corpus[cid]}
                    if isinstance(corpus[cid], str)
                    else corpus[cid]
                ),
            }
            for cid in corpus_ids
        ]

        corpus_texts = [
            "\n".join([doc.get("title", ""), doc["text"]])
            for doc in corpus_with_ids
        ]  # concatenate all document values (title, text, ...)
        encoded_corpus = self.encode(corpus_texts)
        pdb.set_trace()
        logger.info(
            f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
        )

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(encoded_corpus)

        logger.info("Encoding Queries...")
        # pdb.set_trace()
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries_texts = [queries[qid] for qid in queries]

        query_token_strs = self.encode(queries_texts, return_ids=False)

        logger.info(f"Retrieving Results... {len(queries):,} queries")

        queries_results, queries_scores = retriever.retrieve(
            query_token_strs, corpus=corpus_with_ids, k=top_k
        )

        # Iterate over queries
        for qi, qid in enumerate(query_ids):
            doc_id_to_score = {}
            query_results = queries_results[qi]
            scores = queries_scores[qi]
            doc_id_to_score = {}

            # Iterate over results
            for ri in range(len(query_results)):
                doc = query_results[ri]
                score = scores[ri]
                doc_id = doc["doc_id"]

                doc_id_to_score[doc_id] = float(score)

            self.results[qid] = doc_id_to_score
        # pdb.set_trace()
        return self.results

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
        ):
        """Encode a list of sentences using the model.
        Args:
            sentences: The list of sentences to encode.
            task_name: The name of the task to use for building the encoding prompt.
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        if not sentences:
            return None
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list of strings")
        if not all(isinstance(s, str) for s in sentences):
            raise TypeError("sentences must be a list of strings")
        print(prompt_type)
        # pdb.set_trace()
        return bm25s.tokenize(sentences, stopwords=self.stopwords, stemmer=self.stemmer)


    def search_image(
        self,
        corpus: Dataset,  # solve memoery issues
        queries: Dataset,  # solve memoery issues
        top_k: int,
        score_function: str,
        encode_kwargs: dict[str, Any],
        task_name: str,
        return_sorted: bool = False,
        **kwargs,
        ) -> dict[str, dict[str, float]]:


        # if score_function not in self.score_functions:
        #     raise ValueError(
        #         f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
        #     )
        logger.info(f"Corpus Modality : {corpus[0]['modality']}")
        logger.info(f"Query Modality : {queries[0]['modality']}")

        logger.info("Preparing Corpus...")
        corpus_ids = list(corpus["id"])
        top_k = min(1000, len(corpus_ids))
        corpus_modality = corpus[0]["modality"]
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info(
            f"Scoring Function:)"# {self.score_function_desc[score_function]} ({score_function})"
        )

        if corpus_modality == "text":
            # pdb.set_trace()
            corpus_texts = corpus["text"]
            sub_corpus_embeddings = self.get_text_embeddings(
                texts=corpus_texts,
                task_name=task_name,
                prompt_type=PromptType.passage,
                **encode_kwargs,
            )
        else:
            # pdb.set_trace()
            corpus_dataset = ImageDataset(
                corpus, image_column_name="image", transform=self.transform
            )
            corpus_image_dataloader = DataLoader(
                corpus_dataset,
                batch_size=encode_kwargs["batch_size"],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=min(math.floor(os.cpu_count() / 2), 16),
                # num_workers=min(0, 16),
            )
            # pdb.set_trace()
            if corpus_modality == "image":
                sub_corpus_embeddings = self.get_image_embeddings(
                    images=corpus_image_dataloader,
                    task_name=task_name,
                    prompt_type=PromptType.passage,
                    **encode_kwargs,
                )
            elif corpus_modality == "image,text":
                corpus_texts = corpus["text"]
                sub_corpus_embeddings = self.get_fused_embeddings(
                    texts=corpus_texts,
                    images=corpus_image_dataloader,
                    task_name=task_name,
                    prompt_type=PromptType.passage,
                    **encode_kwargs,
                )
            else:
                raise ValueError(f"Unsupported modality: {corpus_modality}")
        pdb.set_trace()
        self.model.index(sub_corpus_embeddings)


        logger.info("Encoding Queries...")
        query_ids = list(queries["id"])
        self.results = {qid: {} for qid in query_ids}
        q_modality = queries[0]["modality"]

        if q_modality == "text":
            # pdb.set_trace()
            query_texts = queries["text"]
            query_embeddings = self.get_text_embeddings(
                texts=query_texts,
                task_name=task_name,
                prompt_type=PromptType.query,
                **encode_kwargs,
            )
        else:
            # pdb.set_trace()
            queries_dataset = ImageDataset(
                queries, image_column_name="image", transform=self.transform
            )
            query_image_dataloader = DataLoader(
                queries_dataset,
                batch_size=encode_kwargs["batch_size"],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=min(math.floor(os.cpu_count() / 2), 16),
                # num_workers=min(0, 16),
            )
            if q_modality == "image":
                query_embeddings = self.get_image_embeddings(
                    images=query_image_dataloader,
                    task_name=task_name,
                    prompt_type=PromptType.query,
                    **encode_kwargs,
                )
            elif q_modality == "image,text":
                query_texts = queries["text"]
                query_embeddings = self.get_fused_embeddings(
                    texts=query_texts,
                    images=query_image_dataloader,
                    task_name=task_name,
                    prompt_type=PromptType.query,
                    **encode_kwargs,
                )
            else:
                raise ValueError(f"Unsupported modality: {q_modality}")
        # pdb.set_trace()
        logger.info(f"Retrieving Results... {len(queries):,} queries")
        queries_results, queries_scores = self.model.retrieve(
                # query_embeddings, corpus=corpus_with_ids, k=top_k
                query_embeddings,  k=top_k
            )
        # pdb.set_trace()
        # Iterate over queries
        for qi, qid in enumerate(query_ids):
            doc_id_to_score = {}
            query_results = queries_results[qi]
            scores = queries_scores[qi]
            doc_id_to_score = {}

            # Iterate over results
            for ri in range(len(query_results)):
                doc = query_results[ri]
                score = scores[ri]
                # pdb.set_trace()
                doc_id = corpus_ids[doc] #doc["doc_id"]

                doc_id_to_score[doc_id] = float(score)

            self.results[qid] = doc_id_to_score
        # pdb.set_trace()
        return self.results


    def get_text_embeddings(
        self,
        texts: list[str],
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        """Get text embeddings for a list of texts using the model.
        Args:
            texts: The list of texts to encode.
            task_name: The name of the task to use for building the encoding prompt.
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        # pdb.set_trace()
        # texts = ["My Name is Amit", "My Name is Amit"]
        return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

        # return self.encode(
        #     texts, task_name=task_name, prompt_type=prompt_type, **kwargs
        # )

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
        ):
        """Get image embeddings for a list of images using the model.
        Args:
            images: The list of images to encode.
            task_name: The name of the task to use for building the encoding prompt.
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        # pdb.set_trace()

        all_image_texts = []
        image_list = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in images:
                    for image_tensor in batch:
                        img_data_uri = tensor_to_base64(image_tensor)
                        model_response = self.get_model_inference(img_data_uri)
                        all_image_texts.append(model_response)
                #         image_list.append(img_data_uri)
                #         if len(image_list) >= 10000:
                #             # pdb.set_trace()
                #             image_texts = run_parallel(image_list,max_threads=100)
                #             all_image_texts.extend(image_texts)
                #             image_list = []
                # # for the remaining images
                # if len(image_list) > 0:
                #     image_texts = run_parallel_progress(image_list,max_threads=100)
                #     all_image_texts.extend(image_texts)

                # pdb.set_trace()
                # all_image_texts = run_parallel(image_list,max_threads=100)
                # all_image_texts.extend(["Sample text for image"]* len(batch))
        else:
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch_images = images[i : i + batch_size]
                    for image_tensor in batch_images:
                        img_data_uri = tensor_to_base64(image_tensor,pil_image=False)
                        model_response = self.get_model_inference(img_data_uri)
                        all_image_texts.append(model_response)
                #         image_list.append(img_data_uri)
                #         if len(image_list) >= 10000:
                #             # pdb.set_trace()
                #             image_texts = run_parallel(image_list,max_threads=100)
                #             all_image_texts.extend(image_texts)
                #             image_list = []

                # if len(image_list) > 0:
                #     image_texts = run_parallel_progress(image_list,max_threads=100)
                #     all_image_texts.extend(image_texts)
                # pdb.set_trace()
                # all_image_texts = run_parallel(image_list,max_threads=100)
                # all_image_texts.extend(["Sample text for image"]* len(batch_images))
        # pdb.set_trace()
        return bm25s.tokenize(all_image_texts, stopwords=self.stopwords, stemmer=self.stemmer)

        # return self.encode(
        #     images, task_name=task_name, prompt_type=prompt_type, **kwargs
        # )

    def get_fused_embeddings(
        self,
        texts: list[str]  | None = None,
        images: list[Image.Image]| DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        """Get fused embeddings for a list of texts and images using the model.
        Args:
            texts: The list of texts to encode.
            images: The list of images to encode.
            task_name: The name of the task to use for building the encoding prompt.
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        # pdb.set_trace()
        import torchvision.transforms.functional as F

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        all_image_texts = []
        image_list = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in images:
                    for image_tensor in batch:
                        img_data_uri = tensor_to_base64(image_tensor)
                        model_response = self.get_model_inference(img_data_uri)
                        all_image_texts.append(model_response)
                #         image_list.append(img_data_uri)
                #         if len(image_list) >= 10000:
                #             # pdb.set_trace()
                #             image_texts = run_parallel(image_list,max_threads=100)
                #             all_image_texts.extend(image_texts)
                #             image_list = []

                # if len(image_list) > 0:
                #     image_texts = run_parallel_progress(image_list,max_threads=100)
                #     all_image_texts.extend(image_texts)


                # all_image_texts = run_parallel(image_list,max_threads=100)
                # all_image_texts.extend(["Sample text for image"]* len(batch))
        else:
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch_images = images[i : i + batch_size]
                    for image_tensor in batch_images:
                        img_data_uri = tensor_to_base64(image_tensor,pil_image=False)
                        model_response = self.get_model_inference(img_data_uri)
                        all_image_texts.append(model_response)

                #         image_list.append(img_data_uri)
                #         if len(image_list) >= 10000:
                #             # pdb.set_trace()
                #             image_texts = run_parallel(image_list,max_threads=100)
                #             all_image_texts.extend(image_texts)
                #             image_list = []

                # if len(image_list) > 0:
                #     image_texts = run_parallel_progress(image_list,max_threads=100)
                #     all_image_texts.extend(image_texts)

                # all_image_texts = run_parallel(image_list,max_threads=100)


        fused_text = [f"{t}. {i}" for t, i in zip(texts, all_image_texts)]
        # pdb.set_trace()
        return bm25s.tokenize(fused_text, stopwords=self.stopwords, stemmer=self.stemmer)

    def get_model_inference(self,img_data_uri):
        return "Amit Kumar" #text
        messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                                    "image":img_data_uri,
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.ml_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]







qwen25_3b_bm25 = ModelMeta(
    loader=partial(
        Qwen25BM25Wrapper,
        model_name="qwen25_3b_bm25",
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    name="qwen25_3b_bm25",
    languages=["eng-Latn"],
    revision="0_1_10",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=5120,
    license="apache-2.0",
    open_weights=True,
    public_training_code="",
    public_training_data="https://github.com/xhluca/bm25s",
    framework=["PyTorch"],
    reference="https://github.com/xhluca/bm25s",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=None,
)

qwen25_7b_bm25 = ModelMeta(
    loader=partial(
        Qwen25BM25Wrapper,
        model_name="qwen25_7b_bm25",
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    ),
    name="qwen25_7b_bm25",
    languages=["eng-Latn"],
    revision="0_1_10",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=5120,
    license="apache-2.0",
    open_weights=True,
    public_training_code="",
    public_training_data="https://github.com/xhluca/bm25s",
    framework=["PyTorch"],
    reference="https://github.com/xhluca/bm25s",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=None,
)