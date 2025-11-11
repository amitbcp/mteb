from pathlib import Path

from datasets import load_dataset, load_from_disk

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _local_dir(subdir: str) -> Path:
    return Path(__file__).resolve().parents[5] / "notebooks" / "table_embedding_expt" / subdir


class FinQANumTokenRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinQANumTokenRetrieval",
        description=(
            "FinQA retrieval with corpus numeric tokens replaced by 'NUM'. Corpus stored locally; queries and qrels "
            "remain from the base FinQA dataset."
        ),
        reference="https://huggingface.co/datasets/embedding-benchmark/FinQA",
        dataset={
            "path": "embedding-benchmark/FinQA",
            "revision": "ed3e1639ae0da10e56574efb10396bcd98efb725",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        prompt={
            "query": "Given a financial question, retrieve evidence where numbers are replaced with 'NUM'."
        },
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        corpus_ds = load_from_disk(str(_local_dir("finqa_num_token")))
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
            qid = item["query-id"]
            relevant_docs.setdefault(qid, {})[item["corpus-id"]] = int(item["score"])

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True
