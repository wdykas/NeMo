from tqdm import tqdm
from nemo_text_processing.text_normalization.normalize import Normalizer
import os
from glob import glob
from typing import List


from nemo.collections.tts.torch.tts_tokenizers import EnglishCharsTokenizer
import pdb; pdb.set_trace()

eng_char_tok = EnglishCharsTokenizer()


def is_punct(str):
	# if str[0].isalpha():
	#     return False
	# punct_marks = [p for p in string.punctuation] + [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")]

	for s in str:
		if s.isalpha():  # s not in punct_marks:
			return False
	return True


# wiki_homograph_dict = get_wiki_homograph_data("cmu_ipa_dicts/wordids.tsv")


#
# wiki_homograph_dict = defaultdict(dict)
# with open("WikipediaHomographData-master/data/wordids.tsv", "r") as f_in:
#     for i, line in enumerate(f_in):
#         if i != 0:
#             line = line.replace('"', "").strip().split("\t")
#             grapheme = line[0].strip()[1:-1]
#             wordid = line[1].strip()[1:-1]
#             ipa_form = line[3].strip()[1:-1]
#             wiki_homograph_dict[grapheme][wordid] = ipa_form


def get_ipa_form(grapheme, wordid, wiki_homograph_dict):
	return wiki_homograph_dict[grapheme][wordid]


def get_heteronyms():
	heteronyms = []
	with open("misc_data/heteronyms.tsv", "r") as f:
		for line in f:
			heteronyms.append(line.strip())
	return heteronyms


def post_process_normalization(text):
	text = text.replace(" slash ", "/")
	return text


def normalize_wikihomograph_data(subset):
	output_dir = f"WikipediaHomographData-master/data/{subset}_normalized"
	normalizer = Normalizer(lang="en", input_case="cased", cache_dir="cache_dir")

	for file in tqdm(glob(f"WikipediaHomographData-master/data/{subset}/*.tsv")):
		file_name = os.path.basename(file)
		output_f = f"{output_dir}/{file_name}"
		if os.path.exists(output_f):
			continue
		with open(file, "r") as f_in, open(output_f, "w") as f_out:
			for i, line in tqdm(enumerate(f_in)):
				if i == 0:
					f_out.write(f'{line.strip()}\t{"normalized_sentence"}\n')
					continue

				line = line.strip().split("\t")
				homograph = file_name.replace(".tsv", "")
				skip_line = False

				# remove quotes
				graphemes, wordid, sentence, start, end = line
				sentence = line[2][1:-1]
				replace_token = "[REPLACE]"
				homograph_span = sentence[int(start) : int(end)]
				if homograph_span == homograph:
					sentence_to_normalize = (
						sentence[: int(start)] + replace_token + sentence[int(end) :]
					)
				else:
					if sentence.lower().count(homograph) == 1:
						sentence_to_normalize = sentence.replace(
							homograph, replace_token
						)
					else:
						skip_line = True

				if not skip_line:
					sentence_to_normalize = sentence_to_normalize.replace('""', '"')
					norm_text = normalizer.normalize(
						text=sentence_to_normalize,
						verbose=False,
						punct_post_process=True,
						punct_pre_process=True,
					)
					norm_text = post_process_normalization(norm_text)
					wordid = wordid[1:-1]
					norm_text = norm_text.replace(replace_token, wordid)
					f_out.write(
						f"{graphemes[1:-1]}\t{wordid}\t{sentence}\t{norm_text}\n"
					)
	print(f"Normalized data is saved at {output_dir}")


def get_words(text: str) -> List[str]:
	"""
	Returns:
		words:
	"""
	text = [w.lower() for w in text.split()]
	words = []
	for word in text:
		word_processed = False
		if "-" in word and word not in ipa_dict:
			if word.count("-") == 1:
				if (
					word.index("-") < (len(word) - 1)
					and word[word.index("-") + 1].isalpha()
				):
					l, r = word.split("-")
					words.append(l + "-")
					if len(r) > 0:
						words.append(r)
			else:
				parts = word.split("-")
				for i, p in enumerate(parts):
					if i < len(parts) - 1:
						p = p + "-"
					words.append(p)
			word_processed = True

		if "/" in word:
			parts = word.split("/")
			for i, p in enumerate(parts):
				if i < len(parts) - 1:
					p = p + "-"
				words.append(p)
			word_processed = True

		if not word_processed:
			words.append(word)
	return words


def process_line(line, add_unk_token=False):
	global ipa_dict
	global wordid_to_nemo_cmu
	global heteronyms
	global oov

	unk_token_grapheme = "<unk>" if add_unk_token else None
	unk_token_phoneme = "§" if add_unk_token else None

	line = (
		line.replace(":", ": ")
		.replace("  ", " ")
		.replace('Soccerbase"Willo', 'Soccerbase "Willo')
	)
	line_graphemes = ""
	line_ipa = ""
	skip = False
	words = get_words(line)
	for word in words:
		try:
			if "[" in word and "]" in word:
				line_ipa += " " + word
				line_graphemes += " " + word
				continue

			word = word.lower()
			word_ipa = ""
			word_start = ""
			word_end = ""

			if is_punct(word):
				word_ipa = word
				line_ipa += " " + word_ipa
				line_graphemes += " " + word
				continue

			# strip out punctuation attached to the word: , (, ), ...
			while not word[0].isalpha():
				word_start = word[0]
				word = word[1:]
			while not word[-1].isalpha():
				word_end = word[-1] + word_end
				word = word[:-1]

			if not word.isalpha() and word not in ipa_dict:
				# print(f"skipping: {word} in {line}")
				skip = True
				continue

			if word in ipa_dict:
				if word in heteronyms:
					heteronyms_to_disambiguate.append(word)
					skip = True
				else:
					word_ipa = ipa_dict[word][0]
			elif len(word) > 2 and word.endswith("'s") and word[:-2] in ipa_dict:
				word_ipa = ipa_dict[word[:-2]][0] + "z"
			else:
				oov.append(word)
				# print(f"OOV: {word}, skipping: {line}")

				if add_unk_token:
					word_ipa = unk_token_phoneme
					word = unk_token_grapheme
				else:
					skip = True
			line_ipa += " " + word_start + word_ipa + word_end
			line_graphemes += " " + word_start + word + word_end
		except Exception as e:
			print(e)
			print(f"SKIPPED: {word} in {line}")
			skip = True
		if skip:
			return None, None
	if skip:
		return None, None
	else:
		return line_ipa.strip(), line_graphemes.strip()


def process_file(file, add_unk_token):
	ipa_lines = []
	grapheme_lines = []
	with open(file, "r") as f:
		for i, line in enumerate(f):
			if i == 0:
				continue
			line = line.strip().split("\t")

			# remove quotes
			wordid, sentence = line[1], line[-1]
			# raw_line = sentence.replace(wordid, wordid.split("_")[0])
			sentence = sentence.replace(
				wordid, f"[{wordid}]"
			)  # f"[{wordid_to_nemo_cmu[wordid]}]")

			# sentence contains IPA form of the word in square brackets, i.e.
			try:
				ipa_line, grapheme_line = process_line(
					sentence, add_unk_token=add_unk_token
				)
			except:
				import pdb

				pdb.set_trace()
				print()

			"""
			both grapheme_line and ipa_line contain wordid in square brackets, we need to replace them with grapheme and correct ipa version
			grapheme_line:
			'higher density regions of the interstellar medium form clouds, or [diffuse_adj] <unk>, where star formation takes place.'
			ipa_line:
			'ˈhaɪɝ ˈdɛnsəti ˈɹidʒənz ˈəv ðə ˌɪntɝˈstɛɫɝ ˈmidiəm ˈfɔɹm ˈkɫaʊdz, ˈɔɹ [diffuse_adj] <unk>, ˈwɛɹ ˈstɑɹ fɔɹˈmeɪʃən ˈteɪks ˈpɫeɪs.'
			"""
			if grapheme_line is not None and ipa_line is not None:
				grapheme_line = grapheme_line.replace(
					f"[{wordid}]", wordid.split("_")[0]
				)
				ipa_line = ipa_line.replace(
					f"[{wordid}]", f"{wordid_to_nemo_cmu[wordid]}"
				)
			ipa_lines.append(ipa_line)
			grapheme_lines.append(grapheme_line)
	return (grapheme_lines, ipa_lines, wordid)


if __name__ == "__main__":
	ipa_dict = get_ipa_dict(
		"cmu_ipa_dicts/en_US-cmudict-0.7b_nv22.01.txt"
	)
	wordid_to_nemo_cmu = get_wordid_to_nemo_cmu()
	heteronyms = get_heteronyms()

	ADD_UNK_TOKEN = True
	punct_to_keep = "ˈˌ<>"
	wordid_stats = {}
	for subset in ["train", "eval"]:
		# normalized data and saves in "WikipediaHomographData-master/data/{subset}_normalized"
		# is output file is present, skips normalization
		normalize_wikihomograph_data(subset)
		lines_ipa = []
		heteronyms_to_disambiguate = []
		oov = []
		normalized_data = f"WikipediaHomographData-master/data/{subset}_normalized/"
		files = glob(f"{normalized_data}/*.tsv")
		processed_data = [
			process_file(file, add_unk_token=ADD_UNK_TOKEN) for file in tqdm(files)
		]

		valid_raw_lines = []
		valid_ipa_lines = []
		total = 0
		output_f = f"cmu_ipa_dicts/wiki_homograph_data_{subset}_manifest.json"
		if ADD_UNK_TOKEN:
			output_f = output_f.replace(".json", "_with_UNK.json")
		wordid_count = defaultdict(int)
		with open(output_f, "w", encoding="utf-8") as f_out:
			for entry in processed_data:
				raw_entries, ipa_entries, wordid = entry
				total += len(raw_entries)
				for i, ipa in enumerate(ipa_entries):
					if ipa is not None:
						valid_ipa_lines.append(ipa)
						valid_raw_lines.append(raw_entries[i])
						raw_ = remove_punctuation(
							raw_entries[i],
							remove_spaces=False,
							do_lower=True,
							exclude=punct_to_keep,
						)
						ipa_ = remove_punctuation(
							ipa,
							remove_spaces=False,
							do_lower=True,
							exclude=punct_to_keep,
						)
						entry = {
							"text": ipa_,
							"text_graphemes": raw_,
							"wordid": wordid,
							"duration": 0.001,
							"audio_filepath": "n/a",
						}
						wordid_count[wordid] += 1
						f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
		wordid_stats[subset] = wordid_count
		print(f"Normalized data converted to IPA saved at {output_f}")

		print(
			f"{len(valid_ipa_lines)} out of {total} ({round(len(valid_ipa_lines) * 100 / total, 2)}%)"
		)
		print(f"Num OOV words: {len(oov)}")
		oov_file = f"/tmp/oov_{subset}.txt"
		with open(oov_file, "w") as f_out:
			for w in oov:
				f_out.write(f"{w}\n")
		print(f"OOV words save in {oov_file}")
		print(f"Num heteronyms_to_disambiguate: {len(heteronyms_to_disambiguate)}")


all_wordids = []

for subset_wordids in wordid_stats.values():
	all_wordids.extend([id for id in subset_wordids.keys()])
	print(len(all_wordids))

all_wordids = list(set(all_wordids))
all_counts = {}

for subset, wordids_count in wordid_stats.items():
	all_counts[subset] = []
	for wordid in all_wordids:
		all_counts[subset].append(wordids_count[wordid])


df = DataFrame(
	{"train": all_counts["train"], "eval": all_counts["eval"], "wordids": all_wordids}
)
missing_eval_enries = df[(df["eval"] == 0) & (df["train"] > 0)]
print(missing_eval_enries.head())
print(f"{missing_eval_enries.shape[0]} homographs are missing from EVAL")
missing_train_enries = df[(df["eval"] > 0) & (df["train"] == 0)]
print(missing_eval_enries.head())
print(f"{missing_train_enries.shape[0]} homographs are missing from TRAIN")


# for subset in ["train", "eval"]:
#     with open(f"cmu_ipa_dicts/wiki_homograph_data_{subset}_manifest.json", "r") as f:
#         for line in f:
#             line = json.loads()


"""
for train: 6613 out of 14330 (46.15%)
for eval: 758 out of 1600 (47.38%)
"""
