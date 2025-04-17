import pdb
import bm25s
import Stemmer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)


logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]




class Llama4BM25Wrapper:
    def __init__(
            self,
            previous_results: str = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            **kwargs,
        ):
            super().__init__(
                model="BM25",
                batch_size=1,
                corpus_chunk_size=1,
                previous_results=previous_results,
                **kwargs,
            )

            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )
            slef.model =  bm25s.BM25()


    @classmethod
    def name(self):
        return "llama_bm25s"

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
        ):
        """Encode a list of sentences using the model.
        Args:
            sentences: The list of sentences to encode.
            task_name: The name of the task to use for building the encoding prompt.
            prompt_type: The prompt type (e.g. "query" | "passage") to use for building the encoding prompt.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        if not sentences:
            return None
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list of strings")
        if not all(isinstance(s, str) for s in sentences):
            raise TypeError("sentences must be a list of strings")
        print(prompt_type)
        pdb.set_trace()
        return [001, 002, 003]


llama4_bm25 = ModelMeta(
    loader=partial(
        Llama4BM25Wrapper,
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct-BM25",
    ),
    name="meta-llama/Llama-4-Scout-17B-16E-Instruct-BM25",
    languages=["eng-Latn"],
    revision="7717deedf0631e6f520b7c83c8f82dcbc2c4c21e",
    release_date="2024-10-08",
    modalities=["image", "text"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=131072,
    embed_dim=3072,
    license="apache-2.0",
    open_weights=True,
    public_training_code="",
    public_training_data="",
    framework=["PyTorch"],
    reference="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=vlm2vec_training_datasets,
)
