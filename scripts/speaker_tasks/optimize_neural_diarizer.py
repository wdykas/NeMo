# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Optimize the Neural Diarizer hyper-parameters onto your dev set using Optuna."""

import argparse
import logging
import os
import tempfile

import numpy as np
import optuna
import wget
from joblib import parallel_backend
from omegaconf import OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.utils import logging as nemo_logger


def scale_weights(r, K):
    return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]


def objective(
    trial: optuna.Trial,
    manifest_path: str,
    model_config: str,
    pretrained_msdd_model: str,
    pretrained_speaker_model: str,
    pretrained_vad_model: str,
    temp_dir: str,
    collar: float,
    ignore_overlap: bool,
    oracle_vad: bool,
    optimize_segment_length: bool,
    optimize_clustering: bool,
):
    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=str(trial.number)) as output_dir:
        config = OmegaConf.load(model_config)
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        if not oracle_vad:
            config.diarizer.oracle_vad = False
            config.diarizer.clustering.parameters.oracle_num_speakers = False
            # Here, we use our in-house pretrained NeMo VAD model
            # todo: we should probably optimize these parameters or something.
            config.diarizer.vad.model_path = pretrained_vad_model
            config.diarizer.vad.parameters.onset = 0.8
            config.diarizer.vad.parameters.offset = 0.6
            config.diarizer.vad.parameters.pad_offset = -0.05

        config.diarizer.msdd_model.model_path = pretrained_msdd_model
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [trial.suggest_float("sigmoid_threshold", 0.7, 1.0)]
        config.diarizer.msdd_model.parameters.diar_eval_settings = [
            (collar, ignore_overlap),
        ]
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs
        # todo: without setting this to 1, process hangs
        config.num_workers = 1  # Workaround for multiprocessing hanging

        if optimize_segment_length:
            # Segmentation Optimization
            stt = trial.suggest_float("window_stt", 0.9, 2.5, step=0.05)
            end = trial.suggest_float("window_end", 0.25, 0.75, step=0.05)
            scale_n = trial.suggest_int("scale_n", 2, 11)
            _step = -1 * (stt - end) / (scale_n - 1)
            shift_ratio = 0.5
            window_mat = np.arange(stt, end - 0.01, _step).round(5)
            config.diarizer.speaker_embeddings.parameters.window_length_in_sec = window_mat.tolist()
            config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = (shift_ratio * window_mat / 2).tolist()
            config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1] * scale_n
        if optimize_clustering:
            # Clustering Optimization
            r_value = round(trial.suggest_float("r_value", 0.0, 2.25, step=0.01), 4)
            max_rp_threshold = round(trial.suggest_float("max_rp_threshold", 0.05, 0.35, step=0.01), 2)
            sparse_search_volume = trial.suggest_int("sparse_search_volume", 2, 100, log=True)
            max_num_speakers = trial.suggest_int("max_num_speakers", 8, 27, step=1)
            config.diarizer.clustering.parameters.max_num_speakers = max_num_speakers
            config.diarizer.clustering.parameters.sparse_search_volume = sparse_search_volume
            config.diarizer.clustering.parameters.max_rp_threshold = max_rp_threshold
            scale_n = len(config.diarizer.speaker_embeddings.parameters.multiscale_weights)
            config.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)

        system_vad_msdd_model = NeuralDiarizer(cfg=config)
        outputs = system_vad_msdd_model.diarize()
        # For each threshold, for each collar setting
        metric = outputs[0][0][0]
        DER = abs(metric)
        return DER


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", help="path to the manifest file", type=str)
    parser.add_argument(
        "--config_url",
        help="path to the config yaml file to use",
        type=str,
        default="https://raw.githubusercontent.com/NVIDIA/NeMo/msdd_scripts_docs/"
        "examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml",
    )
    parser.add_argument(
        "--pretrained_speaker_model", help="path to the speaker embedding model", type=str, default="titanet_large",
    )
    parser.add_argument(
        "--pretrained_vad_model", help="path to the VAD model", type=str, default="vad_multilingual_marblenet",
    )
    parser.add_argument(
        "--pretrained_msdd_model", help="path to the Neural Diarizer model", type=str, default="diar_msdd_telephonic",
    )
    parser.add_argument("--temp_dir", help="path to store temporary files", type=str, default="output/")
    parser.add_argument("--output_log", help="Where to store optuna output log", type=str, default="output.log")
    parser.add_argument(
        "--collar",
        help="collar used to be more forgiving at boundaries (usually set to 0 or 0.25)",
        type=float,
        default=0.0,
    )
    parser.add_argument("--ignore_overlap", help="ignore overlap when evaluating", action="store_true")
    parser.add_argument("--n_trials", help="Number of trials to run optuna", type=int, default=5)
    parser.add_argument("--n_jobs", help="Number of parallel jobs to run", type=int, default=5)
    parser.add_argument(
        "--oracle_vad", help="Enable oracle VAD (no need for VAD when enabled)", action="store_true",
    )
    parser.add_argument(
        "--optimize_clustering", help="Optimize clustering parameters", action="store_true",
    )
    parser.add_argument(
        "--optimize_segment_length", help="Optimize segment length parameters", action="store_true",
    )
    args = parser.parse_args()

    if not (args.optimize_clustering or args.optimize_segment_length):
        raise MisconfigurationException(
            "When using the script you must pass one or both flags: --optimize_clustering, --optimize_segment_length"
        )

    os.makedirs(args.temp_dir, exist_ok=True)

    nemo_logger.setLevel(logging.ERROR)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(args.output_log, mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    model_config = wget.download(args.config_url)

    func = lambda trial: objective(
        trial,
        manifest_path=args.manifest_path,
        model_config=model_config,
        pretrained_msdd_model=args.pretrained_msdd_model,
        pretrained_speaker_model=args.pretrained_speaker_model,
        pretrained_vad_model=args.pretrained_vad_model,
        temp_dir=args.temp_dir,
        collar=args.collar,
        ignore_overlap=args.ignore_overlap,
        oracle_vad=args.oracle_vad,
        optimize_clustering=args.optimize_clustering,
        optimize_segment_length=args.optimize_segment_length,
    )

    study = optuna.create_study(direction="minimize")
    with parallel_backend("multiprocessing"):
        study.optimize(func, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)
    print(f"Best DER {study.best_value}")
    print(f"Best Parameter Set: {study.best_params}")
