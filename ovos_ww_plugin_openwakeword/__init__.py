# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Imports
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
import openwakeword
import numpy as np
from os.path import join, isfile, expanduser, dirname
import requests
from ovos_utils.xdg_utils import xdg_data_home
from os import makedirs


class OwwHotwordPlugin(HotWordEngine):
    """OpenWakeWord is an open-source wakeword or phrase engine that can be trained on 100% synthetic data.
    It can produce high-quality models for arbitrary words and phrases that perform well across
    a wide range of voices and acoustic environments.
    """

    def __init__(self, key_phrase="hey jarvis", config=None, lang="en-us"):
        super().__init__(key_phrase, config, lang)
        # Support for 0.6.0, which removes packaged defaults

        path = self.download_model(
            "https://github.com/Pirajok168/open_wake/blob/dev/ovos_ww_plugin_openwakeword/hey_lada.onnx"
        )

        path2 = self.download_model(
           "https://github.com/Pirajok168/open_wake/blob/dev/ovos_ww_plugin_openwakeword/hey_lada.tflite"
        )
        self.model = openwakeword.Model(
            wakeword_models=[path],
            custom_verifier_models=self.config.get('custom_verifier_models', {}),
            custom_verifier_threshold=self.config.get('custom_verifier_threshold', 0.1),
            inference_framework=self.config.get('inference_framework', 'tflite')
        )
        self.model_names = list(self.model.models.keys())

        # Define short buffer for audio to ensure correct chunk sizes
        self.audio_buffer = []
        self.has_found = False

    def update(self, chunk):
        """
        Predict on input audio using openWakeWord models.
        openWakeWord requires that audio be provided in chunks of 1280 samples,
        so a small buffer is used to ensure proper sizes.
        """
        audio_frame = np.frombuffer(chunk, dtype=np.int16).tolist()
        self.audio_buffer.extend(audio_frame)  # build up the buffer until it has enough samples

        if len(self.audio_buffer) >= 1280:
            if isinstance(self.audio_buffer, list):
                self.audio_buffer = np.asarray(self.audio_buffer)
            # Get prediction from openWakeWord
            prediction = self.model.predict(self.audio_buffer)

            # Clear the buffer after each prediction
            self.audio_buffer = []

            # Check for score above threshold
            for mdl_name in self.model_names:
                # https://github.com/dscripka/openwakeword#threshold-scores-for-activation
                if prediction[mdl_name] >= self.config.get("threshold", 0.5):
                    # Set flag indicating that a wakeword was detected
                    self.has_found = True

                    # Flush recent history of openWakeWord internal audio buffer to avoid re-activations
                    n_frames = self.model.model_inputs[mdl_name]
                    self.model.preprocessor.raw_data_buffer.extend([0.0]*n_frames*1280)
                    self.model.preprocessor.feature_buffer[-n_frames:, :] = np.zeros((n_frames, 96)).astype(np.float32)
                    self.model.preprocessor.melspectrogram_buffer[-250:, :] = np.zeros((250, 32)).astype(np.float32)

                    break

    @staticmethod
    def download_model(url):
        name = url.split("/")[-1].split(".")[0] + ".tflite"
        folder = join(xdg_data_home(), "precise-lite")
        model_path = join(folder, name)
        if not isfile(model_path):
            LOG.info("Downloading model for precise-lite:")
            LOG.info(url)
            content = requests.get(url).content
            makedirs(folder, exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(content)
            LOG.info(f"Model downloaded to {model_path}")

        return model_path

    def found_wake_word(self, frame_data):
        if self.has_found:
            self.has_found = False
            return True
        return False
