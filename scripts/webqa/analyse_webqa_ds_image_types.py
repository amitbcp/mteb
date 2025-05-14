import logging
import pandas as pd

from datasets import Dataset
from pathlib import Path

from tqdm import tqdm

from scripts.webqa.utils import KEYWORDS, LOGGING_FORMAT, SAVE_DIR_PATH, get_dataset, get_filtered_dataset

logger = logging.getLogger(__name__)


def _dump_sample_images(keyword_ds: Dataset,
                        keyword: str) -> None:
    save_images_dir_path = SAVE_DIR_PATH / "images"
    if not save_images_dir_path.exists():
        save_images_dir_path.mkdir()

    for _id, image in zip(keyword_ds['id'], keyword_ds['image']):
        _id = _id.replace(":", "_")
        image_file_name = f"{keyword}_{_id}.jpg"
        image_file_path = save_images_dir_path / image_file_name
        image.save(image_file_path)
    logger.info(f"Dumped sample images for {keyword}")


def _get_canonical_image_type_name(image_type: str) -> str:
    if "table" in image_type:
        return "table"
    elif "graph" in image_type:
        return "graph"
    elif "diagram" in image_type:
        return "diagram"
    elif "table" in image_type:
        return "table"
    elif "form" in image_type:
        return "form"
    else:
        return "other"


def main() -> None:
    images_desc_path = Path("webqa/image_descs_qwen25_3b_bm25_2025-05-14_19-14-29.csv")
    images_df = pd.read_csv(filepath_or_buffer=images_desc_path, index_col=[0])
    images_df.loc[:, 'original_image_type'] = images_df['image_type'].str.lower()
    images_df.loc[:, 'image_type'] = images_df['original_image_type'].apply(_get_canonical_image_type_name)

    corpus_ds = get_dataset(split="corpus",
                            config="corpus")
    filtered_ds = get_filtered_dataset(dataset=corpus_ds)

    n_samples_per_type = 5
    unique_image_types = images_df['image_type'].unique()

    for keyword in unique_image_types:
        keyword_ids = images_df[images_df['image_type'] == keyword].index.tolist()
        if len(keyword_ids) > 0:
            keyword_ids = pd.Series(keyword_ids).sample(n_samples_per_type).tolist()
            keyword_ds = filtered_ds.filter(lambda x: x['id'] in keyword_ids)
            _dump_sample_images(keyword_ds=keyword_ds,
                                keyword=keyword)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    main()
