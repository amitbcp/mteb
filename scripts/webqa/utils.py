import logging
from datasets import Dataset, load_dataset, load_from_disk
from pathlib import Path


DS_NAME = "MRBench/mbeir_webqa_task2"
KEYWORDS = [
    "plot (i.e. graph showing a relation between two variables)", 
    "chart (i.e. a sheet of information in the form of a table, graph, or diagram)", 
    "table (i.e.. a set of facts or figures systematically displayed, especially in columns)", 
    "diagram (i.e. a schematic representation)", 
    "form (i.e. a document with blank spaces for information to be inserted)"
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
