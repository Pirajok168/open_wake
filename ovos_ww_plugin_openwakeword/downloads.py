import os
import pathlib

import openwakeword
from openwakeword.utils import download_models, download_file

download_models()





CUSTOM_MODEL = {
    "hey_lada": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_lada.tflite"),
        "download_url": "https://github.com/Pirajok168/open_wake/blob/dev/ovos_ww_plugin_openwakeword/hey_lada.tflite"
    },
    "hey_jarvis": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_jarvis_v0.1.tflite"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite"
    }
}

def downloadTest(
        target_directory: str = os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models")
        ):



    # Always download melspectrogram and embedding models, if they don't already exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    for feature_model in openwakeword.FEATURE_MODELS.values():
        if not os.path.exists(os.path.join(target_directory, feature_model["download_url"].split("/")[-1])):
            download_file(feature_model["download_url"], target_directory)
            download_file(feature_model["download_url"].replace(".tflite", ".onnx"), target_directory)

    # Always download VAD models, if they don't already exist
    for vad_model in openwakeword.VAD_MODELS.values():
        if not os.path.exists(os.path.join(target_directory, vad_model["download_url"].split("/")[-1])):
            download_file(vad_model["download_url"], target_directory)

    # Get all model urls
    official_model_urls = [i["download_url"] for i in CUSTOM_MODEL.values()]


    for official_model_url in official_model_urls:
        if not os.path.exists(os.path.join(target_directory, official_model_url.split("/")[-1])):
            download_file(official_model_url, target_directory)
            download_file(official_model_url.replace(".tflite", ".onnx"), target_directory)



downloadTest()

def get_pretrained_model_paths_test(inference_framework="tflite"):
    if inference_framework == "tflite":
        return [CUSTOM_MODEL[i]["model_path"] for i in CUSTOM_MODEL.keys()]
    elif inference_framework == "onnx":
        return [CUSTOM_MODEL[i]["model_path"].replace(".tflite", ".onnx") for i in CUSTOM_MODEL.keys()]


print([i for i in get_pretrained_model_paths_test() if 'hey_jarvis' in i])

