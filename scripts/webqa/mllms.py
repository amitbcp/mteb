import json
import os
import oci
import base64

# from scripts.webqa.get_webqa_ds_image_types import _get_prompt_text


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Auth Config
# Please update config profile name and use the compartmentId that has policies grant permissions for using Generative AI Service
compartment_id = "ocid1.tenancy.oc1..aaaaaaaasz6cicsgfbqh6tj3xahi4ozoescfz36bjm3kucc7lotk2oqep47q"
CONFIG_PROFILE = "GENAIUSERS"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config,
                                                                                         service_endpoint=endpoint,
                                                                                         retry_strategy=oci.retry.NoneRetryStrategy(),
                                                                                         timeout=(10, 240))
chat_detail = oci.generative_ai_inference.models.ChatDetails()

# Model ids: obtained from OCI Generative AI service
MODEL_IDS = {
    "gpt-4o-mini": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyacg3e2j2kkruumwqpiewrrng2cakvncpmrlfoxsjemfqa",
    "gpt-4o": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya663hlflxfx6kiwn7qjlnpye6n7caii5lnvcpjlwr2s2q",
    "llama3.2": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya2xrydihzvu5pk6vlvfhtbnfapcvwhhugzo7jez4zcnaa",
    "llama4": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyarojgfh6msa452vziycwfymle5gxdvpwwxzara53topmq",
}


def get_images(root_path, extensions=(".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", '.tif')):
    image_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                image_files.append(os.path.join(dirpath, filename))
    return image_files


def ask_gpt(query=None, 
            image_path=None,
            img_data_uri=None,
            verbose=False, 
            model_name=""):
    # print(model_name)
    if img_data_uri is not None:
        image_url_obj = oci.generative_ai_inference.models.ImageUrl(url=img_data_uri)
        image_content = oci.generative_ai_inference.models.ImageContent()
        image_content.image_url = image_url_obj
    elif image_path is not None:
        base64_image = encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"
        image_url_obj = oci.generative_ai_inference.models.ImageUrl(url=data_url)
        image_content = oci.generative_ai_inference.models.ImageContent()
        image_content.image_url = image_url_obj
    else:
        image_content = None

    # set text content
    if query is None:
        query = "Describe the image in detail."
    content = oci.generative_ai_inference.models.TextContent()
    content.text = f"{query}"

    message = oci.generative_ai_inference.models.Message()
    message.role = "USER"
    if image_content is None:
        message.content = [content]
    else:
        message.content = [content, image_content]

    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.messages = [message]
    chat_request.max_tokens = 2048
    chat_request.temperature = 0.1
    chat_request.frequency_penalty = 0
    chat_request.presence_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = 10

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_IDS[model_name])
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id

    chat_response = generative_ai_inference_client.chat(chat_detail)
    response = chat_response.data.chat_response.choices[-1].message.content[0].text

    if verbose:
        print("**************************Chat Result**************************")
        # print(vars(chat_response))
        print(response)
        print("-" * 50)

    return response


if __name__ == "__main__":
    model_name = "llama4"
    # res = ask_gpt("tell me something interesting", 
    #               model_name=model_name)
    # print(res)

    # set image content
    imgs_dir = "/home/mattrowe/code/mteb/webqa/images"
    image_paths = get_images(imgs_dir)
    print(f"Number of images: {len(image_paths)}")
    image_types = {}
    for ii, image_path in enumerate(image_paths):
        img_name = os.path.basename(image_path)
        # print("-" * 50)
        # print(f"processing {ii}/{len(image_paths)}: {image_path}")

        # query = "What type of image of the following image?"
        # query = _get_prompt_text()
        # image_type = ask_gpt(query=query, 
        #                      image_path=image_path, 
        #                      model_name=model_name)
        # image_types[img_name] = image_type
        # print(img_name, image_type)
                             

        # path_parts = image_path.lstrip(os.sep).split(os.sep)
        # new_path = os.sep.join(path_parts[2:])
        # captions[new_path] = caption

        # # save results to json
        # json_path = f"./data/fusion_pymupdf_output_data/fusion_images_{model_name}_captions.json"
        # with open(json_path, 'w') as fp:
        #     json.dump(captions, fp)
