import os
import traceback
import pickle

import mteb
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")


def run_inference(model_name, task_name, batch_size, device):
    model = mteb.get_model(model_name)
    # specify what you want to evaluate it on
    # tasks = mteb.get_tasks(languages=["eng"],modalities=["text", "image"],task_types=task_type)

    # task_name = task.metadata.name
    try:
        evaluation = mteb.MTEB(tasks=[task_name])

        results = evaluation.run(
            model,
            save_corpus_embeddings=True,
            device=device,
            save_predictions=True,
            export_errors=True,
            verbosity=2,
            encode_kwargs={"batch_size": batch_size},
        )

        # tasks="_".join(task_type)
        results_path = os.path.join(f"./results/runs/{model_name}")
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
            print("Results Path Exists...")

        with open(f"{results_path}/{task_name}.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"SUCCESS !! Task : {task_name} ")
    except:
        print(f"*****************ERROR/FAIL !! Task : {task_name}")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark within OCI for Multimodal Retrieval."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base path to the dataset directory.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        nargs="+",
        default=["ROxfordEasyI2IMultiChoice"],
        help="Task to process",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model (e.g., 'cuda' or 'cpu').",
    )

    args = parser.parse_args()

    run_inference(args.model_name, args.task_name[0], args.batch_size, args.device)
