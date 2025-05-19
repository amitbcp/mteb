from pathlib import Path
from datasets import concatenate_datasets

from scripts.webqa.get_webqa_ds_image_types import get_dataset, get_filtered_dataset

SAVE_DIR_PATH = Path("/mnt/shared/genai/mteb/webqa")

def main() -> None:
    corpus_ds = get_dataset(split="corpus",
                            config="corpus")
    filtered_ds = get_filtered_dataset(dataset=corpus_ds)
    filtered_ids = filtered_ds['id']
    n_samples = filtered_ds.num_rows

    filtered_negative_ds = corpus_ds.filter(lambda x: x['id'] not in filtered_ids)
    filtered_negative_ds = filtered_negative_ds.shuffle().select(range(n_samples))

    full_ds = concatenate_datasets(dsets=[filtered_ds, filtered_negative_ds]).shuffle()
    
    save_downsampled_ds_path = SAVE_DIR_PATH / "downsampled_ds.hf"
    full_ds.save_to_disk(dataset_path=save_downsampled_ds_path)


if __name__ == "__main__":
    main()