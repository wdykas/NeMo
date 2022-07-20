import argparse
import itertools
import random
from pathlib import Path

from nltk_sent_tokenize import sentence_generator

import prepare_small_data_for_punctuation_capitalization_task as small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=Path, required=True)
    parser.add_argument("--output_file", "-o", type=Path, required=True)
    parser.add_argument("--num_sentences_in_segment", "-n", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--counts", "-r", nargs="+", type=int, default=[12, 6, 4, 3])
    parser.add_argument("--max_number_of_words_int_segment", "-m", type=int, default=128)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def main() -> None:
    args = parse_args()
    segment_num_sent_i = 0
    n_sent_in_current_segment = 0
    current_segment = ""
    n_words_in_current_segment = 0
    num_sent_in_segment = list(
        itertools.chain(*[[n_sent] * count for n_sent, count in zip(args.num_sentences_in_segment, args.counts)])
    )
    random.shuffle(num_sent_in_segment)
    with args.output_file.open('w') as f:
        for sent in sentence_generator(args.input_file):
            n_sent_in_current_segment += 1
            if n_words_in_current_segment + len(small.WORD.findall(sent)) >= args.max_number_of_words_int_segment:
                if n_sent_in_current_segment > 1:
                    f.write(current_segment.lstrip() + '\n')
                    segment_num_sent_i = (segment_num_sent_i + 1) % len(num_sent_in_segment)
                    if segment_num_sent_i == 0:
                        random.shuffle(num_sent_in_segment)
                current_segment = ""
                n_sent_in_current_segment = 0
                n_words_in_current_segment = 0
            elif n_sent_in_current_segment >= num_sent_in_segment[segment_num_sent_i]:
                f.write(current_segment.lstrip() + ' ' + sent + '\n')
                current_segment = ""
                n_sent_in_current_segment = 0
                n_words_in_current_segment = 0
                segment_num_sent_i = (segment_num_sent_i + 1) % len(num_sent_in_segment)
                if segment_num_sent_i == 0:
                    random.shuffle(num_sent_in_segment)
            else:
                n_words_in_current_segment += len(small.WORD.findall(sent))
                if n_sent_in_current_segment > 1:
                    current_segment += ' '
                current_segment += sent
        f.write(current_segment + '\n')


if __name__ == "__main__":
    main()
