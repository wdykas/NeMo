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
#
# USAGE: python get_data.py --data-root=<where to put data> --data-set=<datasets_to_download> --num-workers=<number of parallel workers>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data-set=dev_clean,train_clean_100
import argparse
import fnmatch
import functools
import itertools
import json
import logging
import multiprocessing
import os
import subprocess
import tarfile
import urllib.request
import random
import pandas as pd
from bisect import bisect_left, bisect_right
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download LibriTTS and create manifests')
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--data-sets", default="dev_clean", type=str)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--train-min-size", default=None, type=int, help='duration size')
parser.add_argument("--val-size", default=512, type=int, help='utterance size')
parser.add_argument("--spk-size", default=None, type=int)
parser.add_argument("--balance-gender", action='store_true')
parser.add_argument("--separate-speaker", action='store_true')
parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()
random.seed(args.seed)

URLS = {
    'TRAIN_CLEAN_100': "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    'TRAIN_CLEAN_360': "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
    'TRAIN_OTHER_500': "https://www.openslr.org/resources/60/train-other-500.tar.gz",
    'DEV_CLEAN': "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    'DEV_OTHER': "https://www.openslr.org/resources/60/dev-other.tar.gz",
    'TEST_CLEAN': "https://www.openslr.org/resources/60/test-clean.tar.gz",
    'TEST_OTHER': "https://www.openslr.org/resources/60/test-other.tar.gz",
}


def __maybe_download_file(source_url, destination_path):
    if not destination_path.exists():
        logging.info(f"Download Dataset from {source_url}")
        tmp_file_path = destination_path.with_suffix('.tmp')
        urllib.request.urlretrieve(source_url, filename=str(tmp_file_path))
        tmp_file_path.rename(destination_path)


def __extract_file(filepath, data_dir):
    try:
        logging.info(f"Extract Dataset from {filepath}")
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        print(f"Error while extracting {filepath}. Already extracted?")


def __process_transcript(file_path: str):
    entries = []
    with open(file_path, encoding="utf-8") as fin:
        text = fin.readlines()[0].strip()

        # TODO(oktai15): add normalized text via Normalizer/NormalizerWithAudio
        wav_file = file_path.replace(".normalized.txt", ".wav")
        speaker_id = file_path.split('/')[-3]
        assert os.path.exists(wav_file), f"{wav_file} not found!"
        duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
        entry = {
            'audio_filepath': os.path.abspath(wav_file),
            'duration': float(duration),
            'text': text,
            'speaker': int(speaker_id),
        }

        entries.append(entry)

    return entries

def __process_speaker(dataset):
    original_speakers = set([d['speaker'] for d in dataset])
    target_speaker_size = args.spk_size if args.spk_size is not None else len(original_speakers)
    if len(original_speakers) < target_speaker_size:
        logging.warning(f"Dataset Total Speaker Size {len(original_speakers)} smaller than target size {target_speaker_size}")
    
    sp_durations = defaultdict(float)
    for d in dataset: sp_durations[d['speaker']] += d['duration']
    sp_durations = {k: v / 3600 for k, v in sp_durations.items()}
    sp_durations = dict(sorted(sp_durations.items(), key=lambda item: item[1], reverse=True))
    
    if args.balance_gender:
        df = pd.read_csv(str(args.data_root / 'LibriTTS'/ 'speakers.tsv'), sep='\t', index_col=False)
        M = set(df[df['GENDER'] == 'M'].READER.tolist())
        M_speakers = [s for s, d in sp_durations.items() if s in M]
        F = set(df[df['GENDER'] == 'F'].READER.tolist())
        F_speakers = [s for s, d in sp_durations.items() if s in F]
        
        if len(M_speakers) >= target_speaker_size // 2:
            M_speakers = M_speakers[:max(target_speaker_size // 2, target_speaker_size-len(F_speakers))]
            
        if len(F_speakers) >= target_speaker_size // 2:
            F_speakers = F_speakers[:max(target_speaker_size // 2, target_speaker_size-len(M_speakers))]
            
        logging.info(f"Dataset Male Size {len(M_speakers)} Female Size {len(F_speakers)}")
        target_spearkers = set(M_speakers + F_speakers)        
    else:
        target_spearkers = set([s for s, d in tuple(sp_durations.items())[:target_speaker_size]])
        
    total_durations = sum([sp_durations[s] for s in target_spearkers])    
    logging.info(f"Dataset Total Duration: {total_durations}")
    
    
    target_spearkers_ids = {s: _id if not args.separate_speaker else -1 for _id, s in enumerate(target_spearkers)}
    target_dataset = defaultdict(list)

    for d in dataset:
        if d['speaker'] not in target_spearkers:
            continue
            
        d['ori_speaker'] = d['speaker']
        d['speaker'] = target_spearkers_ids[d['speaker']]
        
        if args.separate_speaker:
            target_dataset[d['ori_speaker']].append(d)
        else:
            target_dataset[None].append(d)
           
        
    return target_dataset, target_spearkers_ids
    
    
def __process_dur(dataset):
    train = sorted(dataset, key=lambda x: x['duration'], reverse=True)
    train_duration = [t['duration'] for t in train]
    train_duration_accumulate = list(itertools.accumulate(train_duration))

    index = bisect_left(train_duration_accumulate, args.train_min_size * 60) + 1
    return train[:index].copy()       
    
def __process_data(data_folder, manifest_folder, num_workers):

    os.makedirs(manifest_folder, exist_ok=True)
    
    files = []
    entries = []    
    for root, dirnames, filenames in os.walk(data_folder):
        # we will use normalized text provided by the original dataset
        for filename in fnmatch.filter(filenames, '*.normalized.txt'):
            files.append(os.path.join(root, filename))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript)
        results = p.imap(processing_func, files)
        for result in tqdm(results, total=len(files)):
            entries.extend(result)
            
    datasets, speaker_ids = __process_speaker(entries)
    
    for speaker, dataset in datasets.items():
             
        random.shuffle(dataset)
        train = dataset[args.val_size:]
        valid = dataset[:args.val_size]
        
        if args.train_min_size: 
            train = __process_dur(train)
            
        manifest_speaker_folder = manifest_folder / str(speaker) if speaker != None else manifest_folder
        os.makedirs(manifest_speaker_folder, exist_ok=True)
        
        with open(str(manifest_speaker_folder / 'train_manifest.json'), 'w') as fout:
            for m in train:
                fout.write(json.dumps(m) + '\n')

        with open(str(manifest_speaker_folder / 'val_manifest.json'), 'w') as fout:
            for m in valid:
                fout.write(json.dumps(m) + '\n')
    
    if not args.separate_speaker:
        with open(str(manifest_folder / 'speaker.json'), 'w') as fout:
            json.dump(speaker_ids, fout)

def main():
    data_root = args.data_root
    data_sets = args.data_sets
    num_workers = args.num_workers
    
    if data_sets == "ALL":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other"
    if data_sets == "mini":
        data_sets = "dev_clean,train_clean_100"
    for data_set in data_sets.split(','):
        filepath = data_root / f"{data_set}.tar.gz"
        __maybe_download_file(URLS[data_set.upper()], filepath)
        __extract_file(str(filepath), str(data_root))
        __process_data(
            str(data_root / "LibriTTS" / data_set.replace("_", "-")),
            data_root / "LibriTTS" / (data_set.replace("_", "-") + "-manifest"),
            num_workers=num_workers,
        )


if __name__ == "__main__":
    main()
