from pathlib import Path

from datasets import load_dataset, load_from_disk

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _local_corpus_dir(subdir: str) -> Path:
    """Resolve the local directory containing the stripped corpus."""
    return Path(__file__).resolve().parents[5] / "notebooks" / "table_embedding_expt" / subdir


class FinanceBenchNoNumbersRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinanceBenchNoNumbersRetrieval",
        description=(
            "FinanceBench retrieval with corpus numeric tokens stripped. Corpus documents are derived "
            "from the FinanceBench dataset by removing standalone numeric values, while queries and qrels "
            "remain unchanged."
        ),
        reference="https://huggingface.co/datasets/embedding-benchmark/FinanceBench",
        dataset={
            "path": "embedding-benchmark/FinanceBench",
            "revision": "e68478442112cae36b70a216f52cc2777acf0a7e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a financial question, retrieve the most relevant numeric-free document excerpt."
        },
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        # Load stripped corpus from disk
        corpus_path = _local_corpus_dir("financebench_no_numbers")
        corpus_ds = load_from_disk(str(corpus_path))

        # Queries and qrels from original dataset
        queries_ds = load_dataset(
            self.metadata.dataset["path"],
            "queries",
            revision=self.metadata.dataset["revision"],
        )["queries"]
        qrels_ds = load_dataset(
            self.metadata.dataset["path"],
            "default",
            revision=self.metadata.dataset["revision"],
        )["test"]

        corpus = {}
        queries = {}
        relevant_docs = {}

        for item in corpus_ds:
            corpus[item["id"]] = {"title": "", "text": item["text"]}

        for item in queries_ds:
            queries[item["id"]] = item["text"]

        for item in qrels_ds:
            query_id = item["query-id"]
            relevant_docs.setdefault(query_id, {})[item["corpus-id"]] = int(item["score"])

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}

        self.data_loaded = True
