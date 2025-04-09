import os
import sys
sys.path.append("/mnt/shared/aamita/project/image_retrieval/VLM2Vec/")

import warnings
warnings.filterwarnings('ignore')
import pdb
import mteb

model_name = 'TIGER-Lab/VLM2Vec-Qwen2VL-2B'

meta = mteb.get_model_meta(model_name)

print(f"Model Meta : {meta}")

model = mteb.get_model(model_name)


tasks = mteb.get_tasks(tasks=["ROxfordEasyI2IRetrieval"]) #VidoreTatdqaRetrieval ROxfordEasyI2IRetrieval
tasks = mteb.get_tasks(tasks=["VidoreTatdqaRetrieval"]) #VidoreTatdqaRetrieval ROxfordEasyI2IRetrieval
print(tasks)
# pdb.set_trace()
# run the evaluation
evaluation = mteb.MTEB(tasks=tasks)

results = evaluation.run(model,save_corpus_embeddings=True,device="auto",
                                 save_predictions=True, export_errors=True, verbosity= 3,
                                                          encode_kwargs={"batch_size": 1})


print(results[0].scores['test'][0]['ndcg_at_1'])
