from datetime import datetime
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
from tqdm import tqdm

# from scripts.webqa.utils import KEYWORDS, LOGGING_FORMAT, SAVE_DIR_PATH, get_dataset, get_filtered_dataset
sys.path.append("/home/mattrowe/code/VLM2Vec")

import logging
import mteb

from datasets import Dataset

from mteb.models.qwen25_model import Qwen25BM25Wrapper
from mteb.models.oci_utils import tensor_to_base64
from torchvision.transforms.functional import pil_to_tensor
from datasets import Dataset, load_dataset, load_from_disk

MODEL_NAME = "qwen25_3b_bm25"
SEED = 42
DS_NAME = "MRBench/mbeir_webqa_task2"
KEYWORDS = [
    "graph (i.e. graph showing a relation between two variables)",
    "numerical chart (i.e. a sheet of information in the form of a table, graph, or diagram)",
    "data table (i.e. a set of facts or figures systematically displayed, especially in columns)",
    "schematic diagram or flowchart (i.e. a schematic representation)",
    "data entry form (i.e. a document with blank spaces for information to be inserted)"
]
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SAVE_DIR_PATH = Path("/home/mattrowe/data/mteb/datasets/webqa")

logger = logging.getLogger(__name__)


def get_dataset(split: str,
                config: str) -> Dataset:
    dataset = load_dataset(path=DS_NAME,
                           name=config,
                           split=split)
    return dataset


def get_filtered_dataset(dataset: Dataset,
                         refresh: bool = False) -> Dataset:
    save_filtered_ds_path = SAVE_DIR_PATH / "filtered_ds.hf"
    if not save_filtered_ds_path.exists() or refresh:
        filtered_ds = dataset.filter(lambda x: any(keyword in x['text'].lower() for keyword in KEYWORDS))
        filtered_ds.save_to_disk(dataset_path=save_filtered_ds_path)
        logger.info(f"Wrote filtered DS to: {save_filtered_ds_path}")
    else:
        filtered_ds = load_from_disk(dataset_path=save_filtered_ds_path)
        logger.info(f"Loaded filtered DS from: {save_filtered_ds_path}")
    return filtered_ds


def _dump_sample_images(dataset: Dataset,
                        n_samples_per_type: int = 5) -> None:
    save_images_dir_path = SAVE_DIR_PATH / "images"
    if not save_images_dir_path.exists():
        save_images_dir_path.mkdir()

    for keyword in KEYWORDS:
        keyword_ds = dataset.filter(lambda x: keyword in x['text'].lower()).shuffle()[:n_samples_per_type]
        for _id, image in zip(keyword_ds['id'], keyword_ds['image']):
            _id = _id.replace(":", "_")
            image_file_name = f"{keyword}_{_id}.jpg"
            image_file_path = save_images_dir_path / image_file_name
            image.save(image_file_path)

        print(keyword_ds)


def _load_model() -> Qwen25BM25Wrapper:
    model = mteb.get_model(MODEL_NAME)
    logger.info(f"Loaded Model: {model}")
    return model


def _get_prompt_text() -> str:
    image_types_str = ", ".join(KEYWORDS + ['other'])
    prompt_text = ("If the image is any of the following types then return the type. "
                   f"The types are: {image_types_str}. Return a one word answer.")
    logger.info(f"Using VL Model prompt: {prompt_text}")
    return prompt_text


def _get_image_descs(dataset: Dataset,
                     model: Qwen25BM25Wrapper,
                     prompt_text: str) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for sample in tqdm(dataset):
        sample_id = sample['id']
        sample_text = sample['text']
        image_pil = sample['image']
        image_tensor = pil_to_tensor(pic=image_pil)
        img_data_uri = tensor_to_base64(image_tensor)
        text_description: Qwen25BM25Wrapper = model.get_model_inference(img_data_uri=img_data_uri,
                                                                        prompt_text=prompt_text)
        results[sample_id] = {
            'text': sample_text,
            'image_type': text_description
        }
        # if len(results) > 5:
        #     break

    logger.info(f"Got image descriptions for N={len(results)} images")
    return results


def _save_results(results) -> None:
    results_df = pd.DataFrame(results).T
    results_df.index.name = "id"

    now = datetime.now()
    timestamp_date_str = f"{str(now.year).zfill(4)}-{str(now.month).zfill(2)}-{str(now.day).zfill(2)}"
    timestamp_time_str = f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
    timestamp_str = f"{timestamp_date_str}_{timestamp_time_str}"
    results_file_name = f"image_descs_{MODEL_NAME}_{timestamp_str}.csv"

    results_df_file_path = SAVE_DIR_PATH / results_file_name
    results_df.to_csv(path_or_buf=results_df_file_path)
    logger.info(f"Saved results with shape: {results_df.shape} to file: {results_df_file_path}")


def main() -> None:
    corpus_ds = get_dataset(split="corpus",
                            config="corpus")
    filtered_ds = get_filtered_dataset(dataset=corpus_ds)

    sample = False
    if sample:
        _dump_sample_images(dataset=filtered_ds)

    model = _load_model()
    prompt_text = _get_prompt_text()
    results = _get_image_descs(dataset=filtered_ds,
                               model=model,
                               prompt_text=prompt_text)
    _save_results(results=results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    main()

# TODO: get collection of negative samples to include to make the problem harder
# TODO: add option to refresh to argsparse
# TODO: add option to sample images
