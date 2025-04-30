import argparse
import os
import oci
os.environ["GENAI_REGION"]="us-chicago-1"
# os.environ["GENAI_STAGE"]="ppe"
os.environ["GENAI_PROFILE"]="BoatOc1"
# os.environ["GENAI_COMPARTMENT"]="ocid1.compartment.oc1..aaaaaaaabma2uwi3rcrlx5qxsihcr2k4ehf7jxer6p6c6ngga2zlhkgir3ka"

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_pil_image
import base64
import io


from oci._vendor.urllib3 import PoolManager



def make_security_token_signer(oci_config):
    pk = oci.signer.load_private_key_from_file(oci_config.get("key_file"), None)
    with open(oci_config.get("security_token_file")) as f:
        st_string = f.read()
    return oci.auth.signers.SecurityTokenSigner(st_string, pk)

def get_generative_ai_dp_client(endpoint, profile, use_session_token):
    config = oci.config.from_file('~/.oci/config', profile)
    if use_session_token:
        signer = make_security_token_signer(oci_config=config)
        return oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, signer=signer, service_endpoint=endpoint, timeout=(10,240))
    else:
        return oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, timeout=(10,240))

# def initArgs():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--st", action="store_true", help="use session token")
#     return parser.parse_args()

# regions:
# us-chicago
# eu-frankfurt-1
# ap-tokyo-1
# ap-osaka-1
# ca-montreal-1



def getEnvVariables():
    region = "us-chicago-1"
    profile = "DEFAULT"
    compartment = "<compartment_ocid>"
    stage = "prod"
    if os.getenv("GENAI_REGION") != None:
        region = os.getenv("GENAI_REGION")

    if os.getenv("GENAI_STAGE") != None:
        stage = os.getenv("GENAI_STAGE")

    if os.getenv("GENAI_PROFILE") != None:
        profile = os.getenv("GENAI_PROFILE")

    if os.getenv("GENAI_COMPARTMENT") != None:
        compartment = os.getenv("GENAI_COMPARTMENT")

    return (region, stage, profile, compartment)


def getEndpoint(region, stage):
    if stage == "prod":
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    elif stage == "dev":
        return f"https://dev.inference.generativeai.{region}.oci.oraclecloud.com"
    elif stage == "ppe":
        return f"https://ppe.inference.generativeai.{region}.oci.oraclecloud.com"
    else:
        print("Provide stage via env variable GENAI_STAGE: dev/ppe/prod")
        quit()

def checkCompartmentPresent(compartment_id):
    if "<compartment_ocid>" in compartment_id:
        print("ERROR:Please update your compartment id via env variable GENAI_COMPARMENT")
        quit()


region, stage, profile, compartment_id = getEnvVariables()
# Create a custom pool manager with larger pool
# custom_pool = PoolManager(
#     num_pools=20,  # Number of different connection pools (if calling many hosts)
#     maxsize=200     # Max connections per host (⬆️ this to fix your warning)
# )
# # Patch the default pool globally — hacky but works if SDK uses global PoolManager
# oci._vendor.urllib3.PoolManager = lambda *args, **kwargs: custom_pool



# region, stage, profile, compartment_id = getEnvVariables()
# generative_ai_inference_client = get_generative_ai_dp_client(
#     endpoint=getEndpoint(region, stage),
#     profile=profile,
#     use_session_token=True)


def get_inference(infer_image_url) :
    generative_ai_inference_client = get_generative_ai_dp_client(
    endpoint=getEndpoint(region, stage),
    profile=profile,
    use_session_token=True)

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    # Here to update to use desired model_id, more instruction in above TODO 2
    chat_detail.serving_mode = oci.generative_ai_inference.models.DedicatedServingMode(endpoint_id="ocid1.generativeaiendpoint.oc1.us-chicago-1.amaaaaaabgjpxjqa2n7d2sfaiyjvgws6tqbkzy3i754hipxf7cojrvk6aqaa")
    chat_detail.compartment_id = compartment_id

    content1 = oci.generative_ai_inference.models.TextContent()
    content1.text = """Carefully observe and analyse the image. Then extract the following from the image -
    (1) All the objects in the image \n
    (2) All the text in the image. If their is no text in the image, then skip this point and don't mention it. \n
    (3) Description of the image in detail"""
    content2 = oci.generative_ai_inference.models.ImageContent()
    image_url = oci.generative_ai_inference.models.ImageUrl()
    image_url.url = infer_image_url
    content2.image_url = image_url
    message = oci.generative_ai_inference.models.UserMessage()
    message.content = [content1,content2]


    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = [message]
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.num_generations = 1
    chat_request.max_tokens = 1024
    chat_request.is_stream = False
    chat_request.temperature = 0.00
    chat_request.top_p = 0.95
    chat_request.top_k = -1
    chat_request.frequency_penalty = 1.0

    chat_detail.chat_request = chat_request

    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.choices[0].message.content[0].text

    return response


def get_inference_gpt4(infer_image_url) :
    generative_ai_inference_client = get_generative_ai_dp_client(
    endpoint=getEndpoint(region, stage),
    profile=profile,
    use_session_token=True)

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    # Here to update to use desired model_id, more instruction in above TODO 2
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya663hlflxfx6kiwn7qjlnpye6n7caii5lnvcpjlwr2s2q")
    chat_detail.compartment_id = compartment_id

    content1 = oci.generative_ai_inference.models.TextContent()
    content1.text = """Carefully observe and analyse the image. Then extract the following from the image -
    (1) All the objects in the image \n
    (2) All the text in the image. If their is no text in the image, then skip this point and don't mention it. \n
    (3) Description of the image in detail"""
    content2 = oci.generative_ai_inference.models.ImageContent()
    image_url = oci.generative_ai_inference.models.ImageUrl()
    image_url.url = infer_image_url
    content2.image_url = image_url
    message = oci.generative_ai_inference.models.UserMessage()
    message.content = [content1,content2]


    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = [message]
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.num_generations = 1
    chat_request.max_tokens = 1024

    chat_request.temperature = 0
    chat_request.frequency_penalty = 1
    chat_request.presence_penalty = 0
    chat_request.top_p = 1
    chat_request.top_k = 0

    chat_detail.chat_request = chat_request

    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.choices[0].message.content[0].text

    return response

def get_inference_gpt4mini(infer_image_url) :
    generative_ai_inference_client = get_generative_ai_dp_client(
    endpoint=getEndpoint(region, stage),
    profile=profile,
    use_session_token=True)

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    # Here to update to use desired model_id, more instruction in above TODO 2
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyacg3e2j2kkruumwqpiewrrng2cakvncpmrlfoxsjemfqa")
    chat_detail.compartment_id = compartment_id

    content1 = oci.generative_ai_inference.models.TextContent()
    content1.text = """Carefully observe and analyse the image. Then extract the following from the image -
    (1) All the objects in the image \n
    (2) All the text in the image. If their is no text in the image, then skip this point and don't mention it. \n
    (3) Description of the image in detail"""
    content2 = oci.generative_ai_inference.models.ImageContent()
    image_url = oci.generative_ai_inference.models.ImageUrl()
    image_url.url = infer_image_url
    content2.image_url = image_url
    message = oci.generative_ai_inference.models.UserMessage()
    message.content = [content1,content2]


    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = [message]
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.num_generations = 1
    chat_request.max_tokens = 1024

    chat_request.temperature = 0
    chat_request.frequency_penalty = 1
    chat_request.presence_penalty = 0
    chat_request.top_p = 1
    chat_request.top_k = 0

    chat_detail.chat_request = chat_request

    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.choices[0].message.content[0].text

    return response


def get_inference_ocigenai(infer_image_url,model_id) :
    generative_ai_inference_client = get_generative_ai_dp_client(
    endpoint=getEndpoint(region, stage),
    profile=profile,
    use_session_token=True)

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    # Here to update to use desired model_id, more instruction in above TODO 2
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_id)
    chat_detail.compartment_id = compartment_id

    content1 = oci.generative_ai_inference.models.TextContent()
    content1.text = """Carefully observe and analyse the image. Then extract the following from the image -
    (1) All the objects in the image \n
    (2) All the text in the image. If their is no text in the image, then skip this point and don't mention it. \n
    (3) Description of the image in detail"""
    content2 = oci.generative_ai_inference.models.ImageContent()
    image_url = oci.generative_ai_inference.models.ImageUrl()
    image_url.url = infer_image_url
    content2.image_url = image_url
    message = oci.generative_ai_inference.models.UserMessage()
    message.content = [content1,content2]


    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = [message]
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.num_generations = 1
    chat_request.max_tokens = 1024

    chat_request.temperature = 0
    chat_request.frequency_penalty = 1
    chat_request.presence_penalty = 0
    chat_request.top_p = 1
    chat_request.top_k = 0

    chat_detail.chat_request = chat_request

    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.choices[0].message.content[0].text

    return response



def worker(index, image_url, model="llama4-17b-16e-scout"):
    if model=="gpt4o" :
        count = 5
        for i in range(count) :
            try :
                result = get_inference_gpt4(image_url)
                break
            except oci.exceptions.ServiceError as e:
                if e.status == 413:
                    print(f"Content too long... retrying... {i+1}")
    elif model=="gpt4omini" :
        # print("Using gpt4omini")
        count = 5
        for i in range(count) :
            try :
                result = get_inference_gpt4mini(image_url)
                break
            except oci.exceptions.ServiceError as e:
                if e.status == 413:
                    print(f"Content too long... retrying... {i+1}")
    elif model=="gpt41" :
        # print("Using gpt4omini")
        count = 5
        for i in range(count) :
            try :
                result = get_inference_ocigenai(image_url,model_id="ocid1.generativeaiendpoint.oc1.us-chicago-1.amaaaaaabgjpxjqa2n7d2sfaiyjvgws6tqbkzy3i754hipxf7cojrvk6aqaa")
                break
            except oci.exceptions.ServiceError as e:
                if e.status == 413:
                    print(f"Content too long... retrying... {i+1}")

    elif model=="gpt41mini" :
        # print("Using gpt4omini")
        count = 5
        for i in range(count) :
            try :
                result = get_inference_ocigenai(image_url,model_id="ocid1.generativeaiendpoint.oc1.us-chicago-1.amaaaaaabgjpxjqa2n7d2sfaiyjvgws6tqbkzy3i754hipxf7cojrvk6aqaa")
                break
            except oci.exceptions.ServiceError as e:
                if e.status == 413:
                    print(f"Content too long... retrying... {i+1}")
    else :
        result = get_inference(image_url)
    return index, result

def run_parallel(image_list: List[str], max_threads=20,model="llama4-17b-16e-scout"):
    results = [None] * len(image_list)  # Pre-allocate result list
    print("Running OCI Generative AI Inference in parallel...")
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(worker, idx, url, model) for idx, url in enumerate(image_list)]
        for future in tqdm(as_completed(futures),total=len(futures), desc="OCI Generative AI Inference"):
        # for future in as_completed(futures):
            index, result = future.result()
            results[index] = result  # Store in correct order
    return results

def run_parallel_progress(image_list: List[str], max_threads=20,model="llama4-17b-16e-scout"):
    results = [None] * len(image_list)  # Pre-allocate result list
    print("Running OCI Generative AI Inference in parallel...")
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(worker, idx, url, model) for idx, url in enumerate(image_list)]
        for future in tqdm(as_completed(futures),total=len(futures), desc="OCI Generative AI Inference last remaning data"):
        # for future in as_completed(futures):
            index, result = future.result()
            results[index] = result  # Store in correct order
    return results


def tensor_to_base64(image_tensor,pil_image=False):
    # Convert to PIL image
    if not pil_image:
        pil_image = to_pil_image(image_tensor)

    # Save the PIL image to a bytes buffer
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")  # You can also use "JPEG"
    img_bytes = buffered.getvalue()

    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Optionally create a full data URI
    img_data_uri = f"data:image/png;base64,{img_base64}"
    return img_data_uri
