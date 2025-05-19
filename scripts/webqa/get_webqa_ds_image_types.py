from datetime import datetime
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
from tqdm import tqdm

from scripts.webqa.mllms import ask_gpt

sys.path.append("/home/mattrowe/code/VLM2Vec")

import logging
import mteb

from datasets import Dataset

from mteb.models.qwen25_model import Qwen25BM25Wrapper
from mteb.models.oci_utils import tensor_to_base64
from torchvision.transforms.functional import pil_to_tensor
from datasets import Dataset, load_dataset, load_from_disk

SEED = 42
DS_NAME = "MRBench/mbeir_webqa_task2"
IMAGE_TYPE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "chart": {
        "definition": "(i.e. a graph or chart of data)",
        "synonyms": ["graph", "plot", "chart"]
    },
    "table": {
        "definition": "(i.e. a data table consisting of rows and columns)",
        "synonyms": ["table"]
    },
    "diagram": {
        "definition": "(i.e. a schematic diagram or representation)",
        "synonyms": ["diagram", "schematic"]
    },
    "form": {
        "definition": "(i.e. a completed data form consisting of information)",
        "synonyms": ["form"]
    }
}
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
    image_type_synomyms = [synonym
                           for _, def_dict in IMAGE_TYPE_DEFINITIONS.items()
                           for synonym in def_dict['synonyms']]
    logger.info(f"Filtering using image types synomyms: {image_type_synomyms}")
    if not save_filtered_ds_path.exists() or refresh:
        filtered_ds = dataset.filter(lambda x: any(keyword in x['text'].lower() for keyword in image_type_synomyms))
        filtered_ds.save_to_disk(dataset_path=save_filtered_ds_path)
        logger.info(f"Wrote filtered DS to: {save_filtered_ds_path}")
    else:
        filtered_ds = load_from_disk(dataset_path=save_filtered_ds_path)
        logger.info(f"Loaded filtered DS from: {save_filtered_ds_path}")
    return filtered_ds


def _load_qwen_model() -> Qwen25BM25Wrapper:
    MODEL_NAME = "qwen25_3b_bm25"
    model = mteb.get_model(MODEL_NAME)
    logger.info(f"Loaded Model: {model}")
    return model


def _get_prompt_text() -> str:
    image_types_str = ""
    for image_type, def_dict in IMAGE_TYPE_DEFINITIONS.items():
        image_types_str += f"a {image_type} {def_dict['definition']}, "
    image_types_str += " or other (i.e. none of the previous types)"

    prompt_text = f"""
You are an image type detection system. Given an image you need to return the TYPE of the image.

If you cannot decide on the TYPE of the image then return 'other'.

The TYPE of images are as follows: 
{image_types_str}

Respond with a one word answer with the TYPE of the following image.
"""
    logger.info(f"Using VL Model prompt: {prompt_text}")
    return prompt_text


def _get_image_descs_qwen(dataset: Dataset,
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

    logger.info(f"Got image descriptions for N={len(results)} images")
    return results


def _get_image_descs_llama4(dataset: Dataset,
                            prompt_text: str) -> Dict[str, Dict[str, str]]:
    model_name = "llama3.2"
    results: Dict[str, Dict[str, str]] = {}
    total = dataset.num_rows
    for ix, sample in tqdm(enumerate(dataset), total=total):
        sample_id = sample['id']
        sample_text = sample['text']
        image_pil = sample['image']
        image_tensor = pil_to_tensor(pic=image_pil)
        img_data_uri = tensor_to_base64(image_tensor)

        image_type = ask_gpt(query=prompt_text,
                             img_data_uri=img_data_uri,
                             model_name=model_name)
        image_type = image_type.replace(".", "")
        results[sample_id] = {
            'text': sample_text,
            'image_type': image_type
        }
        # if ix > 1000:
        #     break

    logger.info(f"Got image descriptions for N={len(results)} images")
    return results


def _save_results(results: Dict[str, Dict[str, str]],
                  model_type: str) -> None:
    results_df = pd.DataFrame(results).T
    results_df.index.name = "id"

    now = datetime.now()
    timestamp_date_str = f"{str(now.year).zfill(4)}-{str(now.month).zfill(2)}-{str(now.day).zfill(2)}"
    timestamp_time_str = f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
    timestamp_str = f"{timestamp_date_str}_{timestamp_time_str}"
    results_file_name = f"image_descs_{model_type}_{timestamp_str}.csv"

    results_df_file_path = SAVE_DIR_PATH / results_file_name
    results_df.to_csv(path_or_buf=results_df_file_path)
    logger.info(f"Saved results with shape: {results_df.shape} to file: {results_df_file_path}")


def main() -> None:
    corpus_ds = get_dataset(split="corpus",
                            config="corpus")
    filtered_ds = get_filtered_dataset(dataset=corpus_ds)

    prompt_text = _get_prompt_text()

    model_type = "llama3.2"

    if model_type == "qwen":
        model = _load_qwen_model()
        results = _get_image_descs_qwen(dataset=filtered_ds,
                                        model=model,
                                        prompt_text=prompt_text)
    elif model_type == "llama3.2":
        results = _get_image_descs_llama4(dataset=filtered_ds,
                                          prompt_text=prompt_text)
    _save_results(results=results,
                  model_type=model_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    main()

# TODO: switch to using llama4
# TODO: get the descriptions for the entire dataset
# TODO: refactor llama4 call to be within this module
# TODO: get collection of negative samples to include to make the problem harder
# TODO: add option to refresh to argsparse
# TODO: add option to sample images
