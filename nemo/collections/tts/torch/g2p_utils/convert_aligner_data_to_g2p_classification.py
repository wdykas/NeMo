import json
import os
from nemo.collections.tts.torch.g2p_utils.data_utils import read_wordids
from tqdm import tqdm
from nemo.collections.tts.torch.g2p_utils.data_utils import get_wordid_to_nemo
import string

punct = string.punctuation

def is_valid(text, start_idx, end_idx):
	if start_idx > 0 and (text[start_idx - 1].isalpha() or text[start_idx - 1].isdigit()):
		return False
	if end_idx < len(text) and (text[end_idx].isalpha() or text[end_idx].isdigit()):
		return False
	return True

def convert_to_wiki_format(manifests):
	wordid_map = "/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/wordids.tsv"
	wiki_homograph_dict, target_ipa, target_ipa_label_to_id = read_wordids(wordid_map)

	heteronyms = wiki_homograph_dict.keys()
	wordid_to_nemo = get_wordid_to_nemo()

	sentences, start_end_indices, homographs, word_ids = [], [], [], []
	for manifest in manifests:
		with open(manifest, "r") as f_in:
			skipped_multiple_same_heter = 0
			total_lines = 0
			added_lines = 0
			for line in tqdm(f_in):
				line = json.loads(line)
				found_in_line = []
				text = line["text_graphemes"]
				text = text.lower()

				for heteronym in heteronyms:
					if heteronym in text:
						start_idx = text.find(heteronym)
						end_idx = start_idx + len(heteronym)
						if not is_valid(text, start_idx, end_idx):
							# print(f"skipping: {text} -- {heteronym}")
							# import pdb; pdb.set_trace()
							continue
						if text.count(heteronym) > 1:
							skipped_multiple_same_heter += 1
							# print("multiple heteronyms in 1 sentence")
							# import pdb; pdb.set_trace()
							# print()
						else:
							found_in_line.append(heteronym)
							wordid_to_wiki_forms_dict = wiki_homograph_dict[heteronym]
							for w_id in wordid_to_wiki_forms_dict.keys():
								if wordid_to_nemo[w_id] in line["text"]:
									sentences.append(line["text_graphemes"])
									homographs.append(heteronym)
									word_ids.append(w_id)

									start_end_indices.append((start_idx, end_idx))
				if len(found_in_line) > 0:
					added_lines += 1
				total_lines += 1

		print(f"Processed: {manifest}")
		print(f"skipped: {skipped_multiple_same_heter}")
		print(f"ok {added_lines} out of {total_lines}")
	return sentences, start_end_indices, homographs, word_ids

if __name__ == '__main__':
	lj_manifest = "/mnt/sdb_4/g2p/data_ipa/training_data_v8/raw_files/filtered_disamb_ljspeech_train_ipa.json"
	hifi_9017_manifest = "/mnt/sdb_4/g2p/data_ipa/training_data_v8/raw_files/disamb_9017_clean_train_heteronyms_filtered_ipa.json"

	sentences, start_end_indices, homographs, word_ids = convert_to_wiki_format([lj_manifest, hifi_9017_manifest])

