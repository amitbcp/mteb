
import sys
sys.path.append("/home/mattrowe/code/VLM2Vec")

from mteb.tasks.Image.Any2AnyRetrieval.eng.WebQAT2ITDownSampledRetrieval import WebQAT2ITDownsampledRetrieval


def main() -> None:
    retrieval_task = WebQAT2ITDownsampledRetrieval()
    retrieval_task.load_data()


if __name__ == "__main__":
    main()
