import numpy as np
import pandas as pd

import os
import sys
os.environ["HF_TOKEN"]="<ADD YOUR TOKEN>"
# Clone this repo - https://github.com/TIGER-AI-Lab/VLM2Vec and add to your path
sys.path.append("/Users/aamita/Oracle/oracle/devops/VLM2Vec/")
#install mteb locally or clone
os.environ["MTEB_CACHE"]="/Users/aamita/Oracle/oracle/devops/mteb/scripts/oci/" # the path should contain results/results/<model folders>
import mteb
from mteb.task_selection import results_to_dataframe


model_names = [
"google/siglip-so400m-patch14-384",
"google/siglip-large-patch16-384",
"laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
"facebook/dinov2-giant",
"facebook/dinov2-large",
"TIGER-Lab/VLM2Vec-Full",
"TIGER-Lab/VLM2Vec-Qwen2VL-2B",
"TIGER-Lab/VLM2Vec-Qwen2VL-7B",

"Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
"Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
"royokong/e5-v",

# internal
"OCI/VLM2Vec-Qwen2VL-2B-8K-all-data-full",
"OCI/VLM2Vec-Qwen2VL-2B-8K-all-data-lora",
"OCI/VLM2Vec-Qwen2VL-2B-8K-all-data-lora-multinode",
"llama4bm25",
"gpt4obm25",
"gpt4ominibm25"

]


models = [mteb.get_model_meta(name) for name in model_names]

tasks = mteb.get_tasks(languages=["eng"],modalities=["text", "image"],
                       task_types=[ "Any2AnyRetrieval","DocumentUnderstanding","VisionCentricQA"])

task_names = [ task.metadata.name for task in tasks]
results = mteb.load_results(models=models, tasks=tasks)

data = []
mteb_results = results
for model_res in mteb_results:
    for task_result in model_res.task_results:
        if not task_result.task_name in task_names :
            continue
        tasks = mteb.get_tasks(tasks=[task_result.task_name])

        # print(task_result)
        data.append(
                {
                    "Model": model_res.model_name,
                    "Revision": model_res.model_revision,
                    "task": task_result.task_name,
                    "task_type": tasks[0].metadata.type,
                    "ndcg_at_5": task_result.scores.get('test','default')[0].get("ndcg_at_5",np.nan),
                    "main_score":float(task_result.get_score()),
                }
        )


bench_df  = pd.DataFrame(data)
print(f"Size of Results before NAN Removal : {bench_df.shape}")
bench_df.dropna(subset=['ndcg_at_5'],inplace=True)
print(f"Size of Results before NAN Removal : {bench_df.shape}")

bench_df = bench_df.groupby(['Model','task_type'])['ndcg_at_5'].mean().reset_index()
bench_df_pivoted = bench_df.pivot(index='Model', columns='task_type', values='ndcg_at_5')
bench_df_pivoted['Avg'] = bench_df_pivoted.mean(axis=1, skipna=True)


print(bench_df_pivoted.round(3))
