import os

os.environ["HF_TOKEN"] = "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"

print("Loading MTEB......")
import mteb

print("MTEB loaded......")
from sentence_transformers import SentenceTransformer
import traceback


# # transformers==4.49 | Latest
"""

'BAAI/bge-m3',
'BAAI/bge-base-en-v1.5',
'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct',
'nyu-visionx/moco-v3-vit-b',
'nyu-visionx/moco-v3-vit-l',
'jinaai/jina-clip-v1',

"""

models = ["nyu-visionx/moco-v3-vit-l"]

vrd_tasks = {
    "ImageCoDeT2IMultiChoice": "it2i",
    "AROCocoOrder": "i2t",
    "AROFlickrOrder": "i2t",
    "AROVisualAttribution": "i2t",
    "AROVisualRelation": "i2t",
    "SugarCrepe": "i2t",
    "Winoground": "i2t",
}

vision_tasks = {"BLINKIT2IMultiChoice": "it2i"}


a2a_tasks = {
    "ROxfordEasyI2IMultiChoice": "i2i",
    "ROxfordMediumI2IMultiChoice": "i2i",
    "ROxfordHardI2IMultiChoice": "i2i",
    "RParisEasyI2IMultiChoice": "i2i",
    "RParisMediumI2IMultiChoice": "i2i",
    "RParisHardI2IMultiChoice": "i2i",
    "BLINKIT2IRetrieval": "it2i",
    "BLINKIT2TRetrieval": "it2t",
    "CIRRIT2IRetrieval": "it2i",
    "CUB200I2IRetrieval": "i2i",
    "EDIST2ITRetrieval": "t2it",
    "EncyclopediaVQAIT2ITRetrieval": "it2it",
    "Fashion200kI2TRetrieval": "i2t",
    "Fashion200kT2IRetrieval": "t2i",
    "FashionIQIT2IRetrieval": "it2i",
    "Flickr30kI2TRetrieval": "i2t",
    "Flickr30kT2IRetrieval": "t2i",
    "FORBI2IRetrieval": "i2i",
    "GLDv2I2IRetrieval": "i2i",
    "GLDv2I2TRetrieval": "i2t",
    "HatefulMemesI2TRetrieval": "i2t",
    "HatefulMemesT2IRetrieval": "t2i",
    "ImageCoDeT2IRetrieval": "t2i",
    "InfoSeekIT2ITRetrieval": "it2it",
    "InfoSeekIT2TRetrieval": "it2t",
    "LLaVAIT2TRetrieval": "it2t",
    "MemotionI2TRetrieval": "i2t",
    "MemotionT2IRetrieval": "t2i",
    "METI2IRetrieval": "i2i",
    "MSCOCOI2TRetrieval": "i2t",
    "MSCOCOT2IRetrieval": "t2i",
    "NIGHTSI2IRetrieval": "i2i",
    "OKVQAIT2TRetrieval": "it2t",
    "OVENIT2ITRetrieval": "it2it",
    "OVENIT2TRetrieval": "it2i",
    "ReMuQIT2TRetrieval": "it2t",
    "ROxfordEasyI2IRetrieval": "i2i",
    "ROxfordMediumI2IRetrieval": "i2i",
    "ROxfordHardI2IRetrieval": "i2i",
    "RP2kI2IRetrieval": "i2i",
    "RParisEasyI2IRetrieval": "i2i",
    "RParisMediumI2IRetrieval": "i2i",
    "RParisHardI2IRetrieval": "i2i",
    "SciMMIRI2TRetrieval": "i2t",
    "SciMMIRT2IRetrieval": "t2i",
    "SketchyI2IRetrieval": "i2i",
    "SOPI2IRetrieval": "i2i",
    "StanfordCarsI2IRetrieval": "i2i",
    "TUBerlinT2IRetrieval": "t2i",
    "VisualNewsI2TRetrieval": "i2t",
    "VisualNewsT2IRetrieval": "t2i",
    "VizWizIT2TRetrieval": "it2t",
    "VQA2IT2TRetrieval": "it2t",
    "WebQAT2ITRetrieval": "t2it",
    "WebQAT2TRetrieval": "t2t",
}

vidore_tasks = {
    "VidoreArxivQARetrieval": "t2i",
    "VidoreDocVQARetrieval": "t2i",
    "VidoreInfoVQARetrieval": "t2i",
    "VidoreTabfquadRetrieval": "t2i",
    "VidoreTatdqaRetrieval": "t2i",
    "VidoreShiftProjectRetrieval": "t2i",
    "VidoreSyntheticDocQAAIRetrieval": "t2i",
    "VidoreSyntheticDocQAEnergyRetrieval": "t2i",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval": "t2i",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval": "t2i",
}


# model_name = "BAAI/bge-m3"
# for model_name in models :
#     print(f"Model Name : {model_name} ")
#     try :
#         model = mteb.get_model(model_name)
#     except :
#         print("An error occurred:")
#         traceback.print_exc()

print("Loading Model .....")
model = mteb.get_model(models[0])
print("Model Loaded.....")


for key, value in vidore_tasks.items():
    print(f"*****************Visual Documents Dataset : {key}")
    try:
        tasks = mteb.get_tasks(tasks=[key])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            model,
            output_folder=f"results/{model}",
            save_corpus_embeddings=True,
            save_predictions=True,
            export_errors=True,
            verbosity=3,
            encode_kwargs={"batch_size": 2},
        )

    except:
        print(f"*****************Visual Documents : {key}")
        traceback.print_exc()


for key, value in a2a_tasks.items():
    print(f"*****************A2A Dataset : {key}")
    try:
        tasks = mteb.get_tasks(tasks=[key])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            model,
            output_folder=f"results/{model}",
            save_corpus_embeddings=True,
            save_predictions=True,
            export_errors=True,
            verbosity=3,
            encode_kwargs={"batch_size": 2},
        )

    except:
        print(f"*****************A2A_ERROR : {key}")
        traceback.print_exc()


# for key, value in vrd_tasks.items() :
#     print(f"*****************VRD Dataset : {key}")
#     try :

#         tasks = mteb.get_tasks(tasks=[key])
#         evaluation = mteb.MTEB(tasks=tasks)
#         results = evaluation.run(model, output_folder=f"results/{model}",
#                                         save_corpus_embeddings=True,
#                                         save_predictions=True, export_errors=True, verbosity= 3,
#   encode_kwargs={"batch_size": 2}
#                                         )

#     except :
#         print(f"*****************VRD_ERROR : {key}")
#         traceback.print_exc()
