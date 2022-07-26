# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Optional

from nemo.core.classes import ModelPT
from nemo.utils import model_utils

__all__ = ["G2PModel"]


class G2PModel(ModelPT, ABC):
    @abstractmethod
    def convert_graphemes_to_phonemes(
        self,
        manifest_filepath: str,
        output_manifest_filepath: str,
        batch_size: int = 32,
        num_workers: int = 0,
        target_field: Optional[str] = None,
        verbose: Optional[bool] = True,
    ) -> List[str]:
        """
        Takes .json manifest and add G2P prediction to "pred_text" field
        Args:
            manifest_filepath: path to input manifest
            output_manifest_filepath: path to output manifest
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader

        Returns:
            G2P model predictions
        """
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        # recursively walk the subclasses to generate pretrained model info
        list_of_models = model_utils.resolve_subclass_pretrained_model_info(cls)
        return list_of_models
