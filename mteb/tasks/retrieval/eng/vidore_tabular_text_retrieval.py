from __future__ import annotations

from pathlib import Path

from datasets import load_dataset, load_from_disk

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _local_corpus_dir(subdir: str) -> Path:
    """Resolve the local directory containing the OCR-derived corpus variant."""
    return Path(__file__).resolve().parents[5] / "notebooks" / "table_embedding_expt" / subdir


class _VidoreTextBase(AbsTaskRetrieval):
    corpus_subdir: str

    def load_data(self) -> None:
        if self.data_loaded:
            return

        corpus_path = _local_corpus_dir(self.corpus_subdir)
        corpus_ds = load_from_disk(str(corpus_path))

        queries_ds = load_dataset(
            self.metadata.dataset["path"],
            "queries",
            split="test",
            revision=self.metadata.dataset["revision"],
        )
        qrels_ds = load_dataset(
            self.metadata.dataset["path"],
            "qrels",
            split="test",
            revision=self.metadata.dataset["revision"],
        )

        corpus = {
            str(item["corpus-id"]): {
                "title": "",
                "text": item["text"],
            }
            for item in corpus_ds
        }
        queries = {str(item["query-id"]): item["query"] for item in queries_ds}
        relevant_docs = {}
        for row in qrels_ds:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            relevant_docs.setdefault(qid, {})[did] = int(row["score"])

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}
        self.data_loaded = True


class VidoreTabfquadTextNoNumbersRetrieval(_VidoreTextBase):
    corpus_subdir = "vidore_tabfquad_test_subsampled_beir_fusion_text_no_numbers"
    metadata = TaskMetadata(
        name="VidoreTabfquadTextNoNumbersRetrieval",
        description=(
            "ViDoRe TabFQuAD retrieval using OCR-converted documents with numeric tokens removed. "
            "Queries and relevance labels follow the original dataset."
        ),
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/tabfquad_test_subsampled_beir",
            "revision": "61a2224bcd29b7b261a4892ff4c8bea353527a31",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={
            "query": "Given a question, retrieve the most relevant OCR text snippet stripped of numeric values."
        },
    )


class VidoreTabfquadTextNumTokenRetrieval(_VidoreTextBase):
    corpus_subdir = "vidore_tabfquad_test_subsampled_beir_fusion_text_num_token"
    metadata = TaskMetadata(
        name="VidoreTabfquadTextNumTokenRetrieval",
        description=(
            "ViDoRe TabFQuAD retrieval using OCR-converted documents where numeric tokens are replaced "
            "with the literal token NUM."
        ),
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/tabfquad_test_subsampled_beir",
            "revision": "61a2224bcd29b7b261a4892ff4c8bea353527a31",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={
            "query": "Given a question, retrieve the OCR text snippet where numeric values are replaced with NUM."
        },
    )


class VidoreTatdqaTextNoNumbersRetrieval(_VidoreTextBase):
    corpus_subdir = "vidore_tatdqa_test_beir_fusion_text_no_numbers"
    metadata = TaskMetadata(
        name="VidoreTatdqaTextNoNumbersRetrieval",
        description=(
            "ViDoRe TatDQA retrieval using OCR-converted documents with numeric tokens removed. "
            "Queries and relevance labels follow the original dataset."
        ),
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/tatdqa_test_beir",
            "revision": "5feb5630fdff4d8d189ffedb2dba56862fdd45c0",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={
            "query": "Given a table-focused question, retrieve the OCR text snippet stripped of numeric values."
        },
    )


class VidoreTatdqaTextNumTokenRetrieval(_VidoreTextBase):
    corpus_subdir = "vidore_tatdqa_test_beir_fusion_text_num_token"
    metadata = TaskMetadata(
        name="VidoreTatdqaTextNumTokenRetrieval",
        description=(
            "ViDoRe TatDQA retrieval using OCR-converted documents where numeric tokens are replaced "
            "with the literal token NUM."
        ),
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/tatdqa_test_beir",
            "revision": "5feb5630fdff4d8d189ffedb2dba56862fdd45c0",
        },
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={
            "query": "Given a table-focused question, retrieve the OCR text snippet where numerics become NUM."
        },
    )
