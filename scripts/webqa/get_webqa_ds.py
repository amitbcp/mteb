from typing import List, Tuple
from datasets import load_dataset


def main() -> None:
    ds_name = "MRBench/mbeir_webqa_task2"
    split_config_pairs: List[Tuple[str, str]] = [
        ("corpus", "corpus"),
        ("test", "query"),
        ("test", "qrels")
    ]

    for split, config in split_config_pairs:
        dataset = load_dataset(path=ds_name,
                               name=config,
                               split=split)
        print(split, config, dataset.dataset_size)


if __name__ == "__main__":
    main()
