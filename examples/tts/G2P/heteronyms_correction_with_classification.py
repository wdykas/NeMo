import json

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.torch.en_utils import english_word_tokenize
import torch
import pytorch_lightning as pl
import os
from nemo.collections.tts.models.G2P.g2p_classification import G2PClassificationModel
from nemo.collections.tts.torch.g2p_classification_data import read_wordids
from nemo.utils import logging
import sys # fmt: off
sys.path.append("/home/ebakhturina/NeMo/examples/tts/G2P/data") # fmt: off
from data_preparation_utils import remove_punctuation

def correct_heteronyms(pretrained_heteronyms_model, manifest_with_preds, batch_size: int = 32, num_workers: int = 0):
    # disambiguate heteronyms using IPA-classification model

    # setup GPU
    if torch.cuda.is_available():
        device = [0]  # use 0th CUDA device
        accelerator = 'gpu'
    else:
        device = 1
        accelerator = 'cpu'
    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    trainer = pl.Trainer(devices=device, accelerator=accelerator, logger=False, enable_checkpointing=False)

    if os.path.exists(pretrained_heteronyms_model):
        heteronyms_model = G2PClassificationModel.restore_from(
            pretrained_heteronyms_model, map_location=map_location
        )
    elif pretrained_heteronyms_model in G2PClassificationModel.get_available_model_names():
        heteronyms_model = G2PClassificationModel.from_pretrained(
            pretrained_heteronyms_model, map_location=map_location
        )
    else:
        raise ValueError(
            f'Invalid path to the pre-trained .nemo checkpoint or model name for G2PClassificationModel'
        )

    heteronyms_model.set_trainer(trainer)
    heteronyms_model = heteronyms_model.eval()

    # load heteronyms list
    heteronyms_path = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722"
    heteronyms = []
    with open(heteronyms_path, encoding="utf-8", mode="r") as f:
        for line in f:
            heteronyms.append(line.strip())

    target_ipa_id_to_label = {v: k for k, v in heteronyms_model.target_ipa_label_to_id.items()}
    # TODO: remove hardcoding, save to .nemo
    wiki_homograph_dict, target_ipa, target_ipa_label_to_id = read_wordids(
        "/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/wordids.tsv"
    )
    grapheme_unk_token = "҂"
    ipa_unk_token = "҂"
    (
        sentences,
        sentence_id_to_meta_info,
        start_end_indices_graphemes,
        homographs,
        heteronyms_ipa,
    ) = add_unk_token_to_manifest(
        manifest_with_preds,
        heteronyms,
        wiki_homograph_dict,
        grapheme_unk_token=grapheme_unk_token,
        ipa_unk_token=ipa_unk_token,
    )

    with torch.no_grad():
        heteronyms_preds = heteronyms_model.disambiguate(
            sentences=sentences,
            start_end_indices=start_end_indices_graphemes,
            homographs=homographs,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    manifest_corrected_heteronyms = _correct_heteronym_predictions(manifest_with_preds, sentence_id_to_meta_info, target_ipa_id_to_label, heteronyms_preds)
    get_metrics(manifest_corrected_heteronyms)

def get_wordid_to_nemo():
    wordid_to_nemo_cmu = {}
    with open("/home/ebakhturina/g2p_scripts/misc_data/wordid_to_nemo_cmu.tsv", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split("\t")
            wordid_to_nemo_cmu[line[0]] = line[1]
    return wordid_to_nemo_cmu


def _get_ipa_parts(words_per_segment, cur_ipa):
    try:
        current_ipa = []
        ipa_end_idx = 0
        add_previous = False
        for k, word in enumerate(words_per_segment):
            if word.isalpha():
                if add_previous:
                    import pdb

                    pdb.set_trace()
                    raise ValueError()
                if k == len(words_per_segment) - 1:
                    current_ipa.append(cur_ipa[ipa_end_idx:])
                else:
                    add_previous = True
            else:
                if add_previous:
                    new_ipa_end_idx = cur_ipa.index(word, ipa_end_idx)
                    current_ipa.append(cur_ipa[ipa_end_idx:new_ipa_end_idx])
                    ipa_end_idx = new_ipa_end_idx
                    add_previous = False

                word_len = len(word)
                current_ipa.append(cur_ipa[ipa_end_idx : ipa_end_idx + word_len])
                ipa_end_idx += word_len

        assert len(current_ipa) == len(words_per_segment)
    except Exception as e:
        return None
    return current_ipa


def add_unk_token_to_manifest(manifest, heteronyms, wiki_homograph_dict, grapheme_unk_token, ipa_unk_token):
    """ Replace heteronyms in graphemes ("text_grapheme") and their ipa prediction () with UNKNOWN token"""
    sentences = []
    # here we're going to store sentence id (in the originla manifest file) and homograph start indices present
    # in this sentences. These values willbe later use to replace unknown tokens with classification model predictions.
    sentence_id_to_meta_info = {}
    start_end_indices_graphemes = []
    homographs = []
    heteronyms_ipa = []
    len_mismatdh_skip = 0
    with open(manifest, "r", encoding="utf-8") as f_in:
        for sent_id, line in enumerate(f_in):
            line = json.loads(line)
            ipa = line["pred_text"].split()
            graphemes = line["text_graphemes"]
            line["text_graphemes_original"] = line["text_graphemes"]
            line["pred_text_default"] = line["pred_text"]

            if len(graphemes.split()) != len(ipa):
                logging.debug(f"len(graphemes+ != len(ipa), {line}, skipping...")
                len_mismatdh_skip += 1
                continue

            graphemes_with_unk = ""
            ipa_with_unk = ""
            heteronym_present = False
            for i, segment in enumerate(graphemes.split()):
                cur_graphemes_with_unk = []
                cur_ipa_with_unk = []
                words_per_segment = [w for w, _ in english_word_tokenize(segment, lower=False)]

                # get ipa segments that match words, i.e. split punctuation marks from words
                current_ipa = _get_ipa_parts(words_per_segment, ipa[i])

                # unmatched ipa and graphemes, e.g. incorrect punctuation mark predicted
                if current_ipa is None:
                    cur_graphemes_with_unk.append(segment)
                    cur_ipa_with_unk.append(ipa[i])
                else:
                    for j, word in enumerate(words_per_segment):
                        if word.lower() in heteronyms and word.lower() in wiki_homograph_dict:
                            cur_graphemes_with_unk.append(grapheme_unk_token)
                            cur_ipa_with_unk.append(ipa_unk_token)

                            # collect data for G2P classification
                            # start and end indices for heteronyms in grapheme form
                            start_idx_grapheme = line["text_graphemes"].index(
                                word, len(" ".join(graphemes.split()[:i]))
                            )
                            end_idx_grapheme = start_idx_grapheme + len(word)
                            start_end_indices_graphemes.append([start_idx_grapheme, end_idx_grapheme])

                            # start and end indices for heteronyms in ipa form ("pred_text" field to later replace
                            # unk token with classification model prediction)
                            start_idx_ipa = line["pred_text"].index(current_ipa[j], len(" ".join(ipa[:i])))
                            end_idx_ipa = start_idx_ipa + len(current_ipa[j])

                            if sent_id not in sentence_id_to_meta_info:
                                sentence_id_to_meta_info[sent_id] = {}
                                sentence_id_to_meta_info[sent_id]["unk_indices"] = []
                                sentence_id_to_meta_info[sent_id]["default_pred"] = []

                            sentence_id_to_meta_info[sent_id]["unk_indices"].append([start_idx_ipa, end_idx_ipa])
                            sentence_id_to_meta_info[sent_id]["default_pred"].append(current_ipa[j])

                            sentences.append(graphemes)
                            homographs.append(word.lower())
                            heteronym_present = True
                        else:
                            cur_graphemes_with_unk.append(word)
                            cur_ipa_with_unk.append(current_ipa[j])

                graphemes_with_unk += " " + "".join(cur_graphemes_with_unk)

                ipa_with_unk += " " + "".join(cur_ipa_with_unk)
                cur_ipa_with_unk = []
                cur_grapheme_unk_token = []

            if heteronym_present:
                line["text_graphemes"] = graphemes_with_unk.strip()

    logging.info(f"Skipped {len_mismatdh_skip} due to length mismatch")
    return (
        sentences,
        sentence_id_to_meta_info,
        start_end_indices_graphemes,
        homographs,
        heteronyms_ipa,
    )


def clean(text, do_lower=True):
    exclude_punct = "'ˈˌ-'"
    if do_lower:
        text = text.lower()
    # text = (
    #     text.replace("long-term", "longterm")
    #     .replace(" x-ray ", " xray ")
    #     .replace("t-shirts", "tshirts")
    #     .replace("twenty-five", "twentyfive")
    #     .replace("spider-man", "spiderman")
    #     .replace("full-time", "fulltime")
    #     .replace("three-year", "threeyear")
    #     .replace("one-year", "oneyear")
    #     .replace("five-year", "fiveyear")
    #     .replace("two-dimensional", "twodimensional")
    #     .replace("three-dimensional", "threedimensional")
    #     .replace("computer-generated", "computergenerated")
    #     .replace("long-range", "longrange")
    #     .replace("non-zero", "nonzero")
    #     .replace("air-conditioning", "airconditioning")
    #     .replace("pre-season", "preseason")
    #     .replace("build-up", "buildup")
    #     .replace("one-off", "oneoff")
    #     .replace("brother-in-law", "brotherinlaw")
    #     .replace("on-screen", "onscreen")
    #     .replace("non-verbal", "nonverbal")
    #     .replace("all-out", "allout")
    # )
    text = remove_punctuation(text, exclude=exclude_punct)
    text = text.replace("҂", "").replace("  ", " ").replace("  ", " ").strip()
    return text


def get_metrics(manifest: str):
    all_preds = []
    all_references = []
    all_graphemes = {}
    with open(manifest, "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            all_preds.append(line["pred_text"])
            all_references.append(line["text"])

            if line["text_graphemes"] not in all_graphemes:
                all_graphemes[line["text_graphemes"]] = []
            all_graphemes[line["text_graphemes"]].append(i)

    # collect all examples with multiple phoneme options and same grapheme form, choose the one with min PER
    all_graphemes = {k: v for k, v in all_graphemes.items() if len(v) > 1}
    lines_to_drop = []
    for phon_amb_indices  in all_graphemes.values():
        refs = all_references[phon_amb_indices[0]:phon_amb_indices[-1] + 1]
        preds = all_preds[phon_amb_indices[0]:phon_amb_indices[-1] + 1]
        pers = []
        for ref_, pred_ in zip(refs, preds):
            pers.append(word_error_rate(hypotheses=[pred_], references=[ref_], use_cer=True))

        min_idx = pers.index(min(pers))
        phon_amb_indices.pop(min_idx)
        lines_to_drop.extend(phon_amb_indices)

    # drop duplicated examples, only keep with min PER
    all_preds = [x for i, x in enumerate(all_preds) if i not in lines_to_drop]
    all_references = [x for i, x in enumerate(all_references) if i not in lines_to_drop]

    wer = word_error_rate(hypotheses=all_preds, references=all_references)
    per = word_error_rate(hypotheses=all_preds, references=all_references, use_cer=True)

    print("=" * 40)
    print(f"{manifest}: PER: {per * 100:.2f}%, WER: {wer * 100:.2f}%, lines: {len(all_references)}")
    print("=" * 40)
    return wer, per

def _correct_heteronym_predictions(manifest, sentence_id_to_meta_info, target_ipa_id_to_label, heteronyms_preds):
    """
    :param manifest: path to manifest with G2P predictions in "pred_text"
    :param sentence_id_to_meta_info:
    :param target_ipa_id_to_label:
    :param heteronyms_preds: preditcions of classification G2P model
    :return:
    """
    wordid_to_nemo_cmu = get_wordid_to_nemo()
    # replace unknown token
    heteronyms_sent_id = 0
    # TODO fall back if the UNK token is not predicted -> could lead to mismatch
    manifest_corrected_heteronyms = manifest.replace(".json", "_corrected_heteronyms.json")

    with open(manifest, "r", encoding="utf-8") as f_default, open(
            manifest_corrected_heteronyms, "w", encoding="utf-8"
    ) as f_corrected:
        for sent_id, line in enumerate(f_default):
            line = json.loads(line)

            corrected_pred_text = line["pred_text"]
            offset = 0
            if sent_id in sentence_id_to_meta_info:
                for start_end_idx in sentence_id_to_meta_info[sent_id]["unk_indices"]:
                    start_idx, end_idx = start_end_idx
                    word_id = target_ipa_id_to_label[heteronyms_preds[heteronyms_sent_id]]
                    ipa_pred = wordid_to_nemo_cmu[word_id]
                    if sent_id == 0 and start_idx == 0:
                        ipa_pred = ipa_pred[:-2]
                    heteronyms_sent_id += 1
                    start_idx += offset
                    end_idx += offset
                    corrected_pred_text = (
                            corrected_pred_text[:start_idx] + ipa_pred + corrected_pred_text[end_idx:]
                    )
                    # offset to correct indices, since ipa form chosen during classification could differ in length
                    # from the predicted one
                    offset += len(ipa_pred) - (end_idx - start_idx)

            line["pred_text"] = corrected_pred_text
            f_corrected.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Saved in {manifest_corrected_heteronyms}")
    return manifest_corrected_heteronyms
