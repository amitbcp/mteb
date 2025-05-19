from __future__ import annotations
from collections import defaultdict
import json
import logging
from typing import List
from datasets import Dataset, load_from_disk
from pathlib import Path

import pandas as pd

from mteb.abstasks.AbsTaskRetrieval import HFDataLoader
from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


logger = logging.getLogger(__name__)


DATASET_SAVE_DIR_PATH = Path("/mnt/shared/genai/mteb/webqa")


class WebQAT2ITDownsampledRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="WebQAT2ITDownsampledRetrieval",
        description="Retrieve sources of information based on questions with per-image results.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html",
        dataset={
            "path": "MRBench/mbeir_webqa_task2",
            "revision": "53db4c9f9c93cb74926a1c9d04dea7d7acac2f21",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{chang2022webqa,
      title={Webqa: Multihop and multimodal qa},
      author={Chang, Yingshan and Narang, Mridu and Suzuki, Hisami and Cao, Guihong and Gao, Jianfeng and Bisk, Yonatan},
      booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
      pages={16495--16504},
       year={2022}
      }""",
        prompt={"query": "Find a Wikipedia image that answers this question."},
        descriptive_stats={
            "n_samples": {"test": 2511},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 403196,
                    "num_queries": 2511,
                    "average_relevant_docs_per_query": 1.4,
                }
            },
        },
    )

    def _load_downsampled_ids(self) -> List[str]:
        downsampled_ds_path = DATASET_SAVE_DIR_PATH / "downsampled_ds.hf"
        downsampled_ds = load_from_disk(dataset_path=downsampled_ds_path)
        logger.info(f"Loaded downsampled corpus from: {downsampled_ds_path} with num_rows={downsampled_ds.num_rows}")
        downsampled_ids = downsampled_ds['id']
        return downsampled_ids

    def _load_downsampled_corpus(self,
                                 downsampled_ids: List[str],
                                 downsampled_corpus_path: Path) -> None:
        if not downsampled_corpus_path.exists():
            corpus_ds: Dataset = self.corpus['test']
            logger.info(f"Loaded corpus with num_rows={corpus_ds.num_rows}")
            corpus_ds = corpus_ds.filter(lambda x: x['id'] in downsampled_ids)
            self.corpus['test'] = corpus_ds
            logger.info(f"Downsampled corpus to num_rows={corpus_ds.num_rows}")
            corpus_ds.save_to_disk(dataset_path=downsampled_corpus_path)
            logger.info(f"Saved downsampled corpus to: {downsampled_corpus_path}")
        else:
            corpus_ds: Dataset = load_from_disk(dataset_path=downsampled_corpus_path)
            self.corpus['test'] = corpus_ds
            logger.info(f"Downsampled corpus loaded from: {downsampled_corpus_path} with num_rows={corpus_ds.num_rows}")

    def _load_downsampled_qrels(self,
                                downsampled_ids: List[str],
                                downsampled_qrels_path: Path) -> defaultdict:
        if not downsampled_qrels_path.exists():
            relevant_docs: defaultdict = self.relevant_docs['test']
            logger.info(f"Loaded relevant docs with N={len(relevant_docs)}")
            relevant_docs_updated: defaultdict = {}
            for query_id, docs_dict in relevant_docs.items():
                docs_dict = {corpus_doc_id: corpus_doc_hit
                             for corpus_doc_id, corpus_doc_hit in docs_dict.items()
                             if corpus_doc_id in downsampled_ids}
                if docs_dict:
                    relevant_docs_updated[query_id] = docs_dict
            self.relevant_docs['test'] = relevant_docs_updated
            logger.info(f"Downsampled relevant docs with N={len(relevant_docs_updated)}")
            with open(downsampled_qrels_path, 'w', encoding='utf-8') as f:
                json.dump(relevant_docs_updated, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved downsampled relevant docs to: {downsampled_qrels_path}")
        else:
            with open(downsampled_qrels_path, 'r', encoding='utf-8') as f:
                relevant_docs_updated = json.load(f)
            self.relevant_docs['text'] = relevant_docs_updated
            logger.info(f"Downsampled relevant docs loaded from: {downsampled_qrels_path} "
                        f"with N={len(relevant_docs_updated)}")
        return relevant_docs_updated

    def _load_downsampled_queries(self,
                                  relevant_docs_updated: defaultdict,
                                  downsampled_queries_path: Path) -> None:
        if not downsampled_queries_path.exists():
            downsampled_query_ids = list(relevant_docs_updated.keys())
            queries: Dataset = self.queries['test']
            logger.info(f"Loaded queries num_rows={queries.num_rows}")
            queries = queries.filter(lambda x: x['id'] in downsampled_query_ids)
            self.queries['test'] = queries
            logger.info(f"Downsampled queries to num_rows={queries.num_rows}")
            queries.save_to_disk(dataset_path=downsampled_queries_path)
            logger.info(f"Saved downsampled querues to: {downsampled_queries_path}")
        else:
            queries: Dataset = load_from_disk(dataset_path=downsampled_queries_path)
            self.queries['test'] = queries
            logger.info(f"Downsampled queries loaded from: {downsampled_queries_path} "
                        f"with num_rows={queries.num_rows}")

    def load_data(self, **kwargs) -> None:
        logger.info(f"Loading original {self.metadata.name} Dataset")
        super().load_data(**kwargs)

        downsampled_ids = self._load_downsampled_ids()

        downsampled_corpus_path = DATASET_SAVE_DIR_PATH / "downsampled_corpus.hf"
        downsampled_qrels_path = DATASET_SAVE_DIR_PATH / "downsampled_qrels.json"
        downsampled_queries_path = DATASET_SAVE_DIR_PATH / "downsampled_queries.hf"

        self._load_downsampled_corpus(downsampled_ids=downsampled_ids,
                                      downsampled_corpus_path=downsampled_corpus_path)
        relevant_docs_updated = self._load_downsampled_qrels(downsampled_ids=downsampled_ids,
                                                             downsampled_qrels_path=downsampled_qrels_path)
        self._load_downsampled_queries(relevant_docs_updated=relevant_docs_updated,
                                       downsampled_queries_path=downsampled_queries_path)
