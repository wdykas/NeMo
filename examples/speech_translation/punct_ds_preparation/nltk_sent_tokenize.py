import argparse
import os
from pathlib import Path
from subprocess import PIPE, run
from typing import Generator, Union

import nltk
from tqdm import tqdm

import prepare_small_data_for_punctuation_capitalization_task as small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=Path, required=True)
    parser.add_argument("--output_file", "-o", type=Path, required=True)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def get_num_lines(input_file: Union[str, os.PathLike]) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def sentence_generator(input_file: Path) -> Generator[str, None, None]:
    num_lines = get_num_lines(input_file)
    with input_file.open() as in_f:
        for line in tqdm(in_f, total=num_lines, unit="line", desc="Tokenizing into sentences"):
            for sent in nltk.sent_tokenize(line):
                if len(small.WORD.findall(sent)) > 0:
                    yield sent.rstrip()


def main() -> None:
    args = parse_args()
    with args.output_file.open('w') as out_f:
        for sent in sentence_generator(args.input_file):
            out_f.write(sent + '\n')


if __name__ == "__main__":
    main()
