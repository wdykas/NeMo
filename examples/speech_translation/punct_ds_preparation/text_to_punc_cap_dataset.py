import argparse
from pathlib import Path

from examples.speech_translation.punct_ds_preparation import \
    prepare_big_data_for_punctuation_capitalization_task_complex as big
from examples.speech_translation.punct_ds_preparation import \
    prepare_small_data_for_punctuation_capitalization_task as small


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_text", "-t", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument(
        "--only_first_punctuation_character_after_word_in_autoregressive",
        "-F",
        help="Add only first punctuation character after word to autoregressive labels.",
        action="store_true",
    )
    parser.add_argument(
        "--no_label_if_all_characters_are_upper_case",
        "-U",
        help="If this option is set all words capitalization are labelled as 'U' if the first character is in upper "
        "case. If this option is not set words which contain only uppercase letters (except one character words) "
        "are marked as 'U' and words which first character is in upper case but containing not lower case characters "
        "are marked as 'u'.",
        action="store_true",
    )
    parser.add_argument(
        "--create_model_input",
        "-i",
        help="Whether to write text without punctuation to output directory",
        action="store_true",
    )
    parser.add_argument("--bert_labels", "-b", help="Whether create BERT labels.", action="store_true")
    parser.add_argument(
        "--autoregressive_labels", "-a", help="Whether create autoregressive labels", action="store_true"
    )
    parser.add_argument(
        "--allowed_punctuation",
        "-p",
        help=f"A string containing punctuation marks on which training is performed."
        f"Do not include single quote and space into it. If single quotes are included they will be ignored. "
        f"BERT labels can include only {small.SUPPORTED_BERT_PUNCTUATION} punctuation characters.",
        type=set,
        default=set('"!(),-.:;?'),
    )
    parser.add_argument("--num_jobs", default=1, type=int)
    args = parser.parse_args()
    args.input_text = args.input_text.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def main():
    args = get_args()
    big.write_dataset_parallel(
        None,
        args.input_text,
        args.output_dir,
        args.create_model_input,
        args.bert_labels,
        args.autoregressive_labels,
        args.allowed_punctuation,
        args.only_first_punctuation_character_after_word_in_autoregressive,
        args.no_label_if_all_characters_are_upper_case,
        args.num_jobs,
    )


if __name__ == "__main__":
    main()
