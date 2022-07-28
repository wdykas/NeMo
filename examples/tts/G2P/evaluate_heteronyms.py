import argparse
import json
import os

from examples.text_processing.g2p.utils import clean
from nemo_text_processing.g2p.data.data_utils import get_wordid_to_nemo
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Calculate accuracy of heteronyms predictions")
parser.add_argument("--manifest", type=str, help="Path to manifest files with model predictions")
parser.add_argument(
    "--target",
    type=str,
    default=None,
    help="heteronym to use for accuracy calculation, the rest heteronyms will be ignored",
)
parser.add_argument(
    "--grapheme_field",
    default="text_graphemes",
    type=str,
    help="Name of the field in the manifest to load grapheme input from ",
)


def _process_wiki_eval(graphemes):
    graphemes = graphemes.replace("says sorry for calling guest 'retard'", "says sorry for calling guest retard")
    return graphemes


def eval_heteronyms(manifest, target_homograph=None):
    dict_target_homograph = {}

    num_skipped = 0
    correct = 0
    wordid_to_nemo_cmu = get_wordid_to_nemo()
    errors = []
    num_examples = 0
    with open(manifest, "r") as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)
            homograph_spans = line["homograph_span"]
            wordids = line["word_id"]
            start_end_indices = line["start_end"]
            if isinstance(homograph_spans, str):
                homograph_spans = [homograph_spans]
                wordids = [wordids]
                start_end_indices = [start_end_indices]

            for homograph, word_id, start_end in zip(homograph_spans, wordids, start_end_indices):
                num_examples += 1
                graphemes = clean(line[args.grapheme_field], do_lower=False)
                graphemes = _process_wiki_eval(graphemes)
                phonemes_preds = clean(line["pred_text"])
                phonemes_gt = clean(line["text"])

                homograph_ipa = wordid_to_nemo_cmu[word_id]
                homograph_count_in_grapheme = graphemes.lower().count(homograph.lower())
                is_correct = False

                if target_homograph is not None:
                    if homograph != target_homograph:
                        continue
                    else:
                        if homograph_ipa not in dict_target_homograph:
                            dict_target_homograph[homograph_ipa] = 0
                        dict_target_homograph[homograph_ipa] += 1
                        # print("=" * 40)
                        # print(graphemes)
                        # print("PRED:", phonemes_preds)
                        # print("GT  :", homograph_ipa)
                        # print("=" * 40)

                if homograph_count_in_grapheme == 1:
                    if homograph_ipa in phonemes_preds:
                        is_correct = True
                elif homograph_count_in_grapheme > 1 and graphemes.count(homograph.lower()) == phonemes_preds.count(
                    homograph_ipa
                ):
                    is_correct = True
                else:
                    if graphemes.count("-") == phonemes_gt.count("-") and graphemes.count("-") > 0:
                        graphemes = graphemes.replace("-", " - ")
                        phonemes_gt = phonemes_gt.replace("-", " - ")
                        phonemes_preds = phonemes_preds.replace("-", " - ")

                    graphemes = graphemes.split()
                    phonemes_preds = phonemes_preds.split()
                    phonemes_gt = phonemes_gt.split()

                    if graphemes.count(homograph) == 1:
                        if phonemes_preds.count(homograph_ipa) == 1:
                            is_correct = True
                    elif len(graphemes) != len(phonemes_preds) or len(phonemes_preds) != len(phonemes_gt):
                        print("SKIPPING:")
                        print(phonemes_preds)
                        print(phonemes_gt)
                        print(graphemes)
                        print(homograph)
                        # import pdb; pdb.set_trace()

                        num_skipped += 1
                    else:
                        idx = graphemes.index(homograph)
                        if phonemes_preds[idx] == phonemes_gt[idx]:
                            is_correct = True

                if is_correct:
                    correct += 1
                else:
                    entry = {k: v for k, v in line.items()}
                    entry["word_id"] = word_id
                    entry["start_end"] = start_end
                    entry["homograph_span"] = homograph
                    errors.append(entry)

    errors_file = os.path.basename(manifest).replace(".json", "_errors.json")
    with open(errors_file, "w") as f:
        for entry in errors:
            entry["duration"] = 0.01
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Errors saved in {errors_file}")

    print(
        f"{correct*100/num_examples:.2f}% -- {correct} out of {num_examples}, num_lines: {num_examples}, skipped: {num_skipped}"
    )
    if target_homograph is not None:
        total = sum(dict_target_homograph.values())
        print(target_homograph.upper(), dict_target_homograph)
        print(f"{correct * 100 / total:.2f}% -- {correct} out of {total}, skipped: {num_skipped}")


if __name__ == "__main__":
    args = parser.parse_args()
    # manifest = "/home/ebakhturina/NeMo/examples/tts/eval_wikihomograph_preds.json"
    #
    # manifest = "/home/ebakhturina/g2p_scripts/paper_experiments/eval_wikihomograph_g2pENG_preds.json"
    eval_heteronyms(args.manifest, args.target)
