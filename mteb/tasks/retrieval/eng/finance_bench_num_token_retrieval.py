from pathlib import Path

from datasets import load_dataset, load_from_disk

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _local_dir(subdir: str) -> Path:
    return Path(__file__).resolve().parents[5] / "notebooks" / "table_embedding_expt" / subdir


class FinanceBenchNumTokenRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinanceBenchNumTokenRetrieval",
        description=(
            "FinanceBench retrieval with corpus numeric tokens replaced by the literal token 'NUM'. Corpus is "
            "derived locally; queries and qrels come from the canonical FinanceBench dataset."
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
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="expert-annotated",
        sample_creation="found",
        prompt={
            "query": "Given a financial question, retrieve the relevant document excerpt where numbers are replaced with 'NUM'."
        },
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        corpus_ds = load_from_disk(str(_local_dir("financebench_num_token")))
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
