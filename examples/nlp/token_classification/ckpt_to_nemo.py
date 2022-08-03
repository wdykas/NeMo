import sys
sys.path = ["/home/apeganov/NeMo"] + sys.path

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from nemo.collections.nlp.models import MTEncDecModel, PunctuationCapitalizationModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.models.token_classification import PunctuationCapitalizationModelConfig
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils.config_utils import update_model_config
from nemo.utils.exp_manager import ExpManagerConfig


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", "-c", type=Path, help="Path to checkpoint to encapsulate.", required=True)
    parser.add_argument("--nemo", "-n", type=Path, help="Path to output nemo file", required=True)
    parser.add_argument("--cfg", "-f", type=Path, help="Path to config file", required=True)
    parser.add_argument(
        "--model_class",
        "-t",
        choices=["PunctuationCapitalizationModel", "MTEncDecModel"],
        default="PunctuationCapitalizationModel",
    )
    args = parser.parse_args()
    args.ckpt = args.ckpt.expanduser()
    args.nemo = args.nemo.expanduser()
    args.cfg = args.cfg.expanduser()
    return args


@dataclass
class MTEncDecConfig(NemoConfig):
    name: Optional[str] = 'MTEncDec'
    do_training: bool = True
    do_testing: bool = False
    nemo_file: Optional[str] = None
    model: MTEncDecModelConfig = MTEncDecModelConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@dataclass
class PunctuationCapitalizationConfig(NemoConfig):
    pretrained_model: Optional[str] = None
    name: Optional[str] = 'MTEncDec'
    do_training: bool = True
    do_testing: bool = False
    model: PunctuationCapitalizationModelConfig = PunctuationCapitalizationModelConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='Punctuation_and_Capitalization', files_to_copy=[])


def main():
    args = get_args()
    cfg = OmegaConf.load(args.cfg)
    if args.model_class == "MTEncDecModel":
        default_cfg = MTEncDecConfig()
        cfg = update_model_config(default_cfg, cfg)
        cls = MTEncDecModel
    else:
        cfg = OmegaConf.merge(OmegaConf.structured(PunctuationCapitalizationConfig()), cfg)
        cls = PunctuationCapitalizationModel
    model = cls(cfg.model)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    # model = cls.load_from_checkpoint(args.ckpt, cfg=cfg.model, strict=False)
    model.save_to(args.nemo)


if __name__ == "__main__":
    main()
