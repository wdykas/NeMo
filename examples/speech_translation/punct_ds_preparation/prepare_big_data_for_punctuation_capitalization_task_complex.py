# TODO: exclude overlaps in in intact sentences segments
# TODO: parallelize cutting
# TODO: parallelize writing into file

import html
import logging
import multiprocessing as mp
import random
import re
import shutil
from itertools import accumulate
from math import ceil
from pathlib import Path
from queue import Empty
from subprocess import run
from time import sleep
from typing import Dict, List, Match, Pattern, Set, Tuple, Union

import nltk
import numpy as np
from tqdm import tqdm

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.modules import get_tokenizer

import prepare_small_data_for_punctuation_capitalization_task as small
from prepare_small_data_for_punctuation_capitalization_task import WC

logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')

random.seed(42)

SUPPORTED_CORPUS_TYPES = ["wikipedia"]


def create_triplet(tag):
    start = re.compile(f'<{tag}(?: [^>]*[^>/]>| ?>)', flags=re.I)
    end = re.compile(f'</{tag} ?>', flags=re.I)
    start_or_end = re.compile(start.pattern + '|' + end.pattern, flags=re.I)
    return start, end, start_or_end


PAGE_OPENING_NORMAL_TAG = re.compile(r'^ *<page>$', flags=re.I)
PAGE_CLOSING_NORMAL_TAG = re.compile(r'^ *</page>$', flags=re.I)
TITLE_OF_PAGE = re.compile(r'<title>(.+)</title>', flags=re.I)
COLON_TITLES = re.compile(f'[{WC}]+:[{WC}]')
TEXT_OF_PAGE = re.compile(r'<text[^>]*>(.+)</text>', flags=re.DOTALL | re.I)
QUOTES = re.compile('"\'')
REDIRECT = re.compile(r'^\s*#REDIRECT *\[\[[^]]*]]', flags=re.I)
MAY_REFER_TO = re.compile('^[^\n]+ may refer to:\n', flags=re.I)
DOUBLE_BRACES_WITH_CONTENT = re.compile(r'{{[^}{]*}}|\({{[^}{]*}}\)')
TABLE = re.compile('{|')
EQUALS_SIGN_HEADERS = re.compile('^[ \t]*={2,}[^\n=]+={2,}[ \t]*$', flags=re.MULTILINE)
SPECIAL_SQUARE_BRACKETS_START = re.compile(
    r'\[\[(?:[ :]{,2}File:|[ :]{,2}Image:|[ :]{,2}User:|[ :]{,2}User talk:|[ :]{,2}Special:| ?#footer| '
    r'?{{rdconfigarray)',
    flags=re.I
)
SPECIAL_SQUARE_BRACKETS_BORDER = re.compile(r'\[\[|]]')
# SINGLE_SQUARE_BRACKETS_WITH_CONTENT = re.compile(r'(?<!\[)\[([^][]*)](?!])')
DOUBLE_SQUARE_BRACKETS_WITH_CONTENT = re.compile(r'\[\[([^][]*)]]')
# DOUBLE_SQUARE_BRACKETS_WITH_CONTENT_SINGLE_SECTION = re.compile(r'\[\[([^][|]*)]]')
# DOUBLE_SQUARE_BRACKETS_WITH_CONTENT_TWO_SECTIONS = re.compile(r'\[\[[^][|]*\|([^][|]*)[^][]*]]')
# TRIPLE_QUOTES = re.compile(r"'''([^']+)'''")
END_SECTION = re.compile(
    r"^[ \t]*={2,}\s*(?:See also|References|Notes|Sources|Primary sources|Secondary sources|External links)\s*={2,}"
    r"[ \t]*$",
    flags=re.MULTILINE | re.I,
)
NORMALIZE_ENDING_PATTERN = re.compile(b'.*EOFEOFEOF', flags=re.DOTALL)
NEW_LINE_DUP = re.compile('\n{2,}')
DOC_HEAD = re.compile(
    '^<doc docid="([0-9]+)" source="([^"]+)" title="([^"]+)" start_line="([0-9]+)" end_line="([0-9]+)">$',
    flags=re.MULTILINE
)
DOC_HEAD_TMPL = '<doc docid="{}" source="{}" title="{}" start_line="{}" end_line="{}">'
DOC_END = '</doc>'
DOC_MARK_UP_LINES = re.compile('^</?doc.+\n', flags=re.MULTILINE)
DROP_TAGS = re.compile(
    r"</?(?:div|su[pb]|span|blockquote|em|big|small|s|br|nowiki|abbr|center|poem|i|u|font|kbd|mapframe|a|section|"
    r"onlyinclude|time|cite)(?: [^>]*>|/?>)|'{3}"
)
# REFERENCE = re.compile('<ref[^>]*>[^<]*</ref>')
REFERENCE_SHORT = re.compile('<ref[^>]*/>', flags=re.I)
REF_START, REF_END, REF_START_OR_END = create_triplet('ref')
MATH_START, MATH_END, MATH_START_OR_END = create_triplet('math')
TABLE_START = re.compile(':{,2}{\\|')
TABLE_END = re.compile('\n\\|}')
TABLE_START_OR_END = re.compile(TABLE_START.pattern + '|' + TABLE_END.pattern)
REMARK_START = re.compile('<!--')
REMARK_END = re.compile('-->')
REMARK_START_OR_END = re.compile(REMARK_START.pattern + '|' + REMARK_END.pattern)
GALLERY_START, GALLERY_END, GALLERY_START_OR_END = create_triplet('gallery')
IMAGEMAP_START, IMAGEMAP_END, IMAGEMAP_START_OR_END = create_triplet('imagemap')
SCORE_START, SCORE_END, SCORE_START_OR_END = create_triplet('score')
CODE_START, CODE_END, CODE_START_OR_END = create_triplet('code')
OL_START, OL_END, OL_START_OR_END = create_triplet('ol')
UL_START, UL_END, UL_START_OR_END = create_triplet('ul')
TIMELINE_START, TIMELINE_END, TIMELINE_START_OR_END = create_triplet('timeline')
NOINCLUDE_START, NOINCLUDE_END, NOINCLUDE_START_OR_END = create_triplet('noinclude')
HIERO_START, HIERO_END, HIERO_START_OR_END = create_triplet('hiero')
CHEM_START, CHEM_END, CHEM_START_OR_END = create_triplet('chem')
VAR_START, VAR_END, VAR_START_OR_END = create_triplet('var')
SYNTAXHIGHLIGHT_START, SYNTAXHIGHLIGHT_END, SYNTAXHIGHLIGHT_START_OR_END = create_triplet('syntaxhighlight')
PRE_START, PRE_END, PRE_START_OR_END = create_triplet('pre')
MAPFRAME_START, MAPFRAME_END, MAPFRAME_START_OR_END = create_triplet('mapframe')
EMPTY_PARENTHESES = re.compile(r' *\([ .,!;?|&#%^@$"\'<>{}/\\*~\][]*\) *')
DOUBLE_BRACES_START = re.compile('{{')
DOUBLE_BRACES_END = re.compile('}}')
DOUBLE_BRACES_START_OR_END = re.compile(DOUBLE_BRACES_START.pattern + '|' + DOUBLE_BRACES_END.pattern)
TAG = re.compile('<[a-z]+(?: [^>\n]+)?/?>')
XML_HEADER = re.compile('<\\?xml[^>\n]*\\?>', flags=re.I)
NEXT_LINE_TAG = re.compile(' *\n *<([a-zA-Z]+)(?: [^>\n]+)?>')
LIST_ELEMENT_START = re.compile('\n *(</?li(?: [^>]*>|/?>|>)|\\*|#|\\|)', flags=re.I)
GOOD_LINE_START = re.compile(f'[{WC}"]')
SUSPICIOUS_LINE = re.compile(
    f'^[^{WC}"]|https?:/|www.\\w|[,;:–‑—-] ?[,!;:?]|-–‑—— ?[!:?]|[{WC}]"[{WC}]|\\)[{WC}]|[{WC}]\\(|'
    f'[=*^\\\\~<>|{{}}]|[^{WC} \n`ː!@#$%&*()+≥\\\\{{}}\u2026"\'/?:§;´‘„“‚”»«’><.,'
    f'\u00a0\u1680\u1803\u202f\u205f\u3000\ufeff№[\\]‑–—-]|'
    f'\\([^"()]*"[^"()]*("[^"()]*"[^"()]*)*\\)',
    flags=re.MULTILINE
)
SUSPICIOUS_LINE_ENDING = re.compile('[^?!.\u2026)"]$', flags=re.MULTILINE)
SUSPICIOUS_PARENTHESES = re.compile(fr'\)[{WC}]|[{WC}]\(')
PARENTHESES = re.compile('[)(]')
LONG_HYPHEN = re.compile(r'—')
NOT_USUAL_HYPHENS = re.compile(r'[–‑—]')
SPACE_DUP = re.compile(' {2,}')
OPENING_PARENTHESES_WITH_SPACE = re.compile(r'\( +')
NO_SPACE_OPENING_PARENTHESES = re.compile(r'\b\(')
SPACE_CLOSING_PARENTHESES = re.compile(r' +\)')
CLOSING_PARENTHESES_NO_SPACE = re.compile(r'\)\b')
CLOSING_PARENTHESES_SPACE_PUNCTUATION_MARK = re.compile(r'\) ([.!:?;,…])')
PUNCTUATION_MARK_OPENING_PARENTHESES = re.compile(r'([.!:?;,…])\(')
SPACE_PUNCTUATION_MARK = re.compile(r' +([.!?:,;…])')
ELLIPSIS_WITHOUT_SPACE = re.compile(rf'\.\.([{WC}(])')
DIGIT_SPACE_PERCENT = re.compile(r'(\d) % *')
UNICODE_APOSTROPHE = re.compile(f'([{WC}])[‘’]([{WC}])')
TRAILING_PARENTHESES = re.compile(r' *\([^)]*\) *$', flags=re.MULTILINE)
BROKEN_PARENTHESES_WITH_CONTENT = re.compile(
    f'\\([^)(]*[^{WC}!?."\'] *\\)|'
    f'\\( *[^{WC}"][^)(]*\\)|'
    f'\\( *…? *\\)|'
    f'\\([^)]*(?:www\\.|https?:)[^)]*\\)|'
    f'\\([^)]{{,6}}\\)'
)
ALL_PARENTHESES = re.compile(r'\([^()]*\)')
ALL_PARENTHESES_WITH_PRECEDING_AND_FOLLOWING_SPACES = re.compile(' *' + ALL_PARENTHESES.pattern + f" *(?![{WC}'])")
# QUOTE_THEN_COMMA_OR_PERIOD = re.compile('"([,.])([^.])')
# COMMA_OR_PERIOD_THEN_QUOTE = re.compile('([^.])([,.])"')
SPACE_NEW_LINE = re.compile(' +\n *')
EMPTY_LINE = re.compile('^[() .,!;?|&#%^@$"\'<>{}/\\\\*~\\][]*$', flags=re.MULTILINE)
REPEATED_HYPHENS = re.compile('(-- )+')


MAX_NUM_CHARACTERS_IN_1_FILE = 10 ** 9
BUFFER_SIZE = 2 ** 24
POSSIBLE_LINE_ENDS = {'\n', '\r', '\v', '\f', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029'}
MAX_NUM_LINES_PER_PROCESS = 10 ** 6


def remove_tag_with_content(
    text: str, start_re: Pattern, end_re: Pattern, remove_whole_line: bool, pos_info: Tuple[Path, int, int]
) -> str:
    result = ""
    start_iter = start_re.finditer(text)
    end_iter = end_re.finditer(text)
    last_end = 0
    for start_m, end_m in zip(start_iter, end_iter):
        if start_m.span()[0] >= end_m.span()[0]:
            logging.warning(
                f"Encountered closing tag {repr(end_m.group(0))} in position {end_m.span()[0]} before or simultaneously "
                f"with opening tag {repr(start_m.group(0))} in position {start_m.span()[0]}. start_re={start_re}, "
                f"end_re={end_re}. Document is in file {pos_info[0]} lines between {pos_info[1]} and {pos_info[2]}. "
                f"Discarding the remainder of the document."
            )
            return result
        if start_m.span()[0] < last_end:
            if remove_whole_line:
                if end_m.span()[0] > last_end:
                    logging.warning(
                        f"Encountered closing tag {repr(end_m.group(0))} in position {end_m.span()[0]} in not parsed "
                        f"text (starting with position {last_end}) whereas no starting tag {start_re} was found in not "
                        f"parsed text. Probably tags {start_re} and {end_re} are multiline. Document is in lines "
                        f"between {pos_info[1]} and {pos_info[2]} in file {pos_info[0]}. Discarding the remainder of "
                        f"the document."
                    )
                    return result
                continue
            else:
                logging.warning(
                    f"Encountered 2 opening tags with regex '{start_re.pattern}' (the last match '{start_m.group(0)}' "
                    f"in position {start_m.span()[0]}) before closing tag with regex '{end_re.pattern}' in position "
                    f"{last_end}. Probably here nested tags are used. Document is in lines between {pos_info[1]} and "
                    f"{pos_info[2]} in file {pos_info[0]}. Discarding the remainder of the document."
                )
                return result
        if remove_whole_line:
            ind = text.rfind('\n', last_end, start_m.span()[0])
            if ind == -1:
                ind = last_end
            result += text[last_end: ind]
            last_end = text.find('\n', end_m.span()[1])
        else:
            result += text[last_end: start_m.span()[0]]
            last_end = end_m.span()[1]
    if last_end > 0:
        result += text[last_end:]
    return result


def remove_tag_with_content_nested(
    text: str,
    start_re: Pattern,
    end_re: Pattern,
    start_or_end_re: Pattern,
    remove_whole_line: bool,
    pos_info: Tuple[Path, int, int],
) -> str:
    result = ""
    num_opened = 0
    last_end = 0
    for m in start_or_end_re.finditer(text):
        if start_re.match(m.group(0)) is not None:
            if num_opened == 0:
                if last_end < m.span()[0]:
                    right = text.rfind('\n', last_end, m.span()[0]) if remove_whole_line else m.span()[0]
                    result += text[last_end: right]
            num_opened += 1
        else:
            assert end_re.match(m.group(0)) is not None
            if num_opened == 0:
                section_border = text.rfind('==\n', last_end, m.span()[0])
                last_end = m.span()[1]
                if section_border > 0:
                    result += text[last_end: section_border]
            else:
                num_opened -= 1
                if num_opened == 0:
                    cand = text.find('\n', m.span()[1])
                    cand = cand if cand > 0 else len(text)
                    last_end = cand if remove_whole_line else m.span()[1]
    if num_opened == 0:
        result += text[last_end:]
    return result


def remove_double_square_brackets_specials(text: str, pos_info: Tuple[Path, int, int]) -> str:
    result = ""
    last_end = 0
    for m in SPECIAL_SQUARE_BRACKETS_START.finditer(text):
        if m.span()[0] < last_end:
            continue
        start = m.span()[0]
        search_start = m.span()[1]
        result += text[last_end: start]
        num_openings = 1
        while num_openings > 0:
            mm = SPECIAL_SQUARE_BRACKETS_BORDER.search(text, search_start)
            if mm is None:
                return result
            if mm.group(0) == ']]':
                num_openings -= 1
            else:
                num_openings += 1
            if num_openings > 0:
                search_start = mm.span()[1]
        last_end = mm.span()[1]
    return result + text[last_end:]


def remove_lists(text: str) -> str:
    result = ""
    start_idx_of_clean_text = 0
    for m in LIST_ELEMENT_START.finditer(text):
        if m.span()[0] >= start_idx_of_clean_text:
            j = max(m.span()[0] - 1, 0)
            while j > start_idx_of_clean_text and text[j] in '\n ':
                j -= 1
            if text[j] == ':':
                right = text.rfind('\n', start_idx_of_clean_text, j)
                if right > 0:
                    result += text[start_idx_of_clean_text: text.rfind('\n', start_idx_of_clean_text, j)]
            else:
                if j - start_idx_of_clean_text > 500:
                    result += text[start_idx_of_clean_text: m.span()[0]]
            cand = text.find('\n', m.span()[1])
            start_idx_of_clean_text = cand if cand > 0 else len(text)
    result += text[start_idx_of_clean_text:]
    return result


def check_quotes_and_parentheses(line: str, do_no_allow_nested: bool = True) -> bool:
    opened = 0
    for m in PARENTHESES.finditer(line):
        if m.group(0) == '(':
            opened += 1
            if opened > 1 and do_no_allow_nested:
                return False
        else:
            opened -= 1
            if opened < 0:
                return False
    return opened == 0 and line.count('"') % 2 == 0


def normalize_quotes(text: str, skip_if_odd_number_of_quotes: bool) -> str:
    if skip_if_odd_number_of_quotes:
        if text.count('"') % 2 == 1:
            return text
    else:
        if text.count('"') % 2 == 1:
            raise ValueError("Odd number of quotes")
    result = ""
    already_checked = 0
    i = text.find('"')
    quote_count = 0
    while i >= 0:
        if quote_count % 2 == 0:
            if i == 0:
                result = '"'
            else:
                result += text[already_checked: i - (text[i - 1] == ' ')] + ' ' + '"'
            already_checked = i + 1 + (text[i + 1] == ' ')
        else:
            result += text[already_checked: i - (text[i - 1] == ' ')] + '"'
            if i < len(text) - 1:
                result += ' '
                already_checked = i + 1 + (text[i + 1] == ' ')
            else:
                already_checked = len(text)
        i = text.find('"', already_checked)
        quote_count += 1
    return result + text[already_checked:]


def remove_suspicious_lines_and_rearrange_quotes_and_spaces(
    original_text: str,
    normalize_and_check_quotes_and_parentheses: bool,
    check_suspicious_endings: bool,
    check_suspicious_parentheses: bool,
) -> Tuple[str, int]:
    text = UNICODE_APOSTROPHE.sub(r"\1'\2", original_text)
    text = text.replace('`', "'")
    text = text.replace('‘', "'")
    text = text.replace('‚', "'")
    text = text.replace('’', '"')
    text = text.replace('´', "'")
    text = text.replace("''", '"')
    text = text.replace('„', '"')
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('«', '"')
    text = text.replace('»', '"')
    if not text:
        return "", 0
    if normalize_and_check_quotes_and_parentheses:
        text = '\n'.join(
            [
                normalize_quotes(line, skip_if_odd_number_of_quotes=False) for line in text.split('\n')
                if check_quotes_and_parentheses(line) and '""' not in line
            ]
        )
        if not text:
            return text, original_text.count('\n') + (original_text[-1] != '\n')
    suspicious_regexps = [SUSPICIOUS_LINE]
    if check_suspicious_endings:
        suspicious_regexps.append(SUSPICIOUS_LINE_ENDING)
    if check_suspicious_parentheses:
        suspicious_regexps.append(SUSPICIOUS_PARENTHESES)
    num_removed_lines = (
            original_text.count('\n') - (original_text[-1] == '\n') - text.count('\n') + (text[-1] == '\n')
    )
    for suspicious_regexp in suspicious_regexps:
        if not text:
            break
        result = ""
        i = 0
        for m in suspicious_regexp.finditer(text, pos=text[0] == '\n', endpos=len(text) - (text[-1] == '\n')):
            if m.span()[0] >= i:
                right = text.rfind('\n', i, m.span()[0])
                if right > 0:
                    result += text[i: right]
                    num_removed_lines += 1
                cand = text.find('\n', m.span()[1])
                i = cand if cand > 0 else len(text)
        result += text[i:]
        text = result
    return text, num_removed_lines


def normalize_punctuation(text: str, lang: str) -> str:
    text = LONG_HYPHEN.sub(' - ', text)
    text = SPACE_DUP.sub(' ', text)
    text = NOT_USUAL_HYPHENS.sub('-', text)
    text = REPEATED_HYPHENS.sub('- ', text)
    text = OPENING_PARENTHESES_WITH_SPACE.sub('(', text)
    text = NO_SPACE_OPENING_PARENTHESES.sub(' (', text)
    text = SPACE_CLOSING_PARENTHESES.sub(')', text)
    text = CLOSING_PARENTHESES_NO_SPACE.sub(') ', text)
    text = CLOSING_PARENTHESES_SPACE_PUNCTUATION_MARK.sub(r')\1', text)
    text = PUNCTUATION_MARK_OPENING_PARENTHESES.sub(r'\1 (', text)
    text = DIGIT_SPACE_PERCENT.sub(r'\1% ', text)
    text = SPACE_PUNCTUATION_MARK.sub(r'\1', text)
    text = text.replace('…', '...')
    text = ELLIPSIS_WITHOUT_SPACE.sub(r'.. \1', text)
    text = text.replace('ː', ':')
    # if lang == 'en':
    #     # English "quotation"
    #     text = QUOTE_THEN_COMMA_OR_PERIOD.sub(r'\1"\2', text)
    # else:
    #     # French "quotation"
    #     text = COMMA_OR_PERIOD_THEN_QUOTE.sub(r'\1"\2', text)
    text = SPACE_NEW_LINE.sub('\n', text)
    return text


def get_wiki_text_lines(
    text: str,
    lang: str,
    tokenizer: TokenizerSpec,
    tok_chars: Set[str],
    untok_chars: Set[str],
    pos_info: Tuple[Path, int, int],
    nltk_tokenization: bool,
    remove_parentheses: bool,
) -> Tuple[List[str], Set[str], Set[str]]:
    text = html.unescape(html.unescape(text))
    text = small.SPACING_CHARACTERS_TO_REPLACE.sub(' ', text)
    text = REDIRECT.sub('', text)
    if MAY_REFER_TO.match(text) or text[-18:].strip() == '{{disambiguation}}':
        return [], tok_chars, untok_chars
    text = text.strip()
    if not text:
        return [], tok_chars, untok_chars
    end_section = END_SECTION.search(text, pos=text[0] == '\n')
    if end_section is not None:
        text = text[:end_section.span()[0]].strip()
    text = remove_tag_with_content_nested(
        text, SYNTAXHIGHLIGHT_START, SYNTAXHIGHLIGHT_END, SYNTAXHIGHLIGHT_START_OR_END, False, pos_info
    )
    text = remove_double_square_brackets_specials(text, pos_info)
    # text = TRIPLE_QUOTES.sub(r'\1', text)
    text = remove_tag_with_content_nested(text, REF_START, REF_END, REF_START_OR_END, False, pos_info)
    text = REFERENCE_SHORT.sub('', text)
    text = remove_tag_with_content_nested(text, MATH_START, MATH_END, MATH_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, CODE_START, CODE_END, CODE_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, HIERO_START, HIERO_END, HIERO_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, CHEM_START, CHEM_END, CHEM_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, VAR_START, VAR_END, VAR_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(
        text, DOUBLE_BRACES_START, DOUBLE_BRACES_END, DOUBLE_BRACES_START_OR_END, False, pos_info
    )
    text = remove_tag_with_content_nested(text, TABLE_START, TABLE_END, TABLE_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, REMARK_START, REMARK_END, REMARK_START_OR_END, False, pos_info)
    text = EMPTY_PARENTHESES.sub(' ', text)
    text = remove_tag_with_content_nested(text, GALLERY_START, GALLERY_END, GALLERY_START_OR_END, False, pos_info)
    text = remove_tag_with_content_nested(text, IMAGEMAP_START, IMAGEMAP_END, IMAGEMAP_START_OR_END, False, pos_info)
    text = remove_tag_with_content_nested(text, SCORE_START, SCORE_END, SCORE_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, TIMELINE_START, TIMELINE_END, TIMELINE_START_OR_END, False, pos_info)
    text = remove_tag_with_content_nested(text, OL_START, OL_END, OL_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, UL_START, UL_END, UL_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, MAPFRAME_START, MAPFRAME_END, MAPFRAME_START_OR_END, True, pos_info)
    text = remove_tag_with_content_nested(text, NOINCLUDE_START, NOINCLUDE_END, NOINCLUDE_START_OR_END, False, pos_info)
    text = remove_tag_with_content_nested(text, PRE_START, PRE_END, PRE_START_OR_END, False, pos_info)
    text = EQUALS_SIGN_HEADERS.sub('\n', text)

    def double_square_brackets_replacement(match: Match) -> str:
        match_text = match.group(1)
        match_text = match_text.split('|')
        if len(match_text) == 1:
            res = match_text[0]
        elif len(match_text) >= 2:
            res = match_text[1]
            # logging.warning(
            #     f"Found double square brackets with three sections {repr(match.group(0))} in document from lines "
            #     f"between {pos_info[1]} and {pos_info[2]}."
            # )
        else:
            res = ""
        if ':' in res:
            split_res = res.split(':')
            if split_res[1]:
                res = split_res[1]
            else:
                res = split_res[0]
        if (
                "#if:" in res
                or "&amp;" in res
                or '\n' in res
                or "#ifeq:" in res
                or '{}' in res
                or "{{" in res
                or "}}" in res
                or "lc:" in res

        ):
            res = ""
        return res

    text = DROP_TAGS.sub('', text)
    text = text.replace("''", '"')
    text = DOUBLE_SQUARE_BRACKETS_WITH_CONTENT.sub(double_square_brackets_replacement, text)
    text = NEW_LINE_DUP.sub('\n', text)
    if text:
        if text[0] == '\n':
            text = text[1:]
    else:
        return [], tok_chars, untok_chars
    # text = remove_lists(text)
    text = text.replace('[', '(')
    text = text.replace(']', ')')
    if text and text[-1] != '\n':
        text += '\n'
    if tokenizer is not None:
        text, tok_chars, untok_chars, _ = small.remove_untokenizable_characters_from_text(
            text, tokenizer, tok_chars, untok_chars, remove_entire_lines=True
        )
    text = ALL_PARENTHESES.sub(' ', text) if remove_parentheses else BROKEN_PARENTHESES_WITH_CONTENT.sub(' ', text)
    text = SPACE_DUP.sub(' ', text)
    after_suspicious_removal, _ = remove_suspicious_lines_and_rearrange_quotes_and_spaces(
        text,
        normalize_and_check_quotes_and_parentheses=True,
        check_suspicious_endings=True,
        check_suspicious_parentheses=True,
    )
    text = normalize_punctuation(after_suspicious_removal, lang)
    text = NEW_LINE_DUP.sub('\n', text)
    if nltk_tokenization:
        stripped = []
        for sent in nltk.sent_tokenize(text):
            sent = sent.lstrip()
            if not sent:
                continue
            if GOOD_LINE_START.match(sent[0]) is None:
                assert stripped, \
                    f"Text is supposed to be cleaned in a way that first character in every line is a word character." \
                    f" First 20 characters in text are: {repr(text[:20])}. Document is in file {pos_info[0]} between " \
                    f"lines {pos_info[1]} and {pos_info[2]}. Whole text after suspicious removal:\n" \
                    f"{after_suspicious_removal}"
                if sent[0] != ' ' and stripped[-1] != ' ':
                    stripped[-1] += ' '
                stripped[-1] += sent
            else:
                stripped.append(sent)
            stripped = [sent.rstrip() for sent in stripped]
    else:
        stripped = [sent.strip() for sent in text.split('\n')]
    return [sent for sent in stripped if sent], tok_chars, untok_chars


def count_in_blocks(files, size=BUFFER_SIZE, specific_to_count=None, num_characters=None):
    total_num_characters = 0
    finished = False
    while True:
        b = files.read(size)
        if not b:
            break
        if num_characters is not None:
            if total_num_characters + len(b) >= num_characters:
                b = b[:num_characters - total_num_characters]
                finished = True
        if specific_to_count is None:
            yield len(b)
        else:
            yield b.count(specific_to_count)
        if finished:
            break


def count_lines_in_file(file_path, start=0, num_characters=None):
    with file_path.open() as f:
        f.seek(start)
        count = sum(count_in_blocks(f, specific_to_count='\n', num_characters=num_characters))
    return count


def count_characters_in_file(file_path):
    with file_path.open() as f:
        count = sum(count_in_blocks(f))
    return count


def count_pages_in_file(file_path, start, num_characters):
    with file_path.open() as f:
        f.seek(start)
        count = sum(count_in_blocks(f, specific_to_count='<page', num_characters=num_characters))
    return count


def count_in_file_parts(file_path, part_num_characters, pattern):
    result = [0] * len(part_num_characters)
    num_preceding_characters_for_segment = list(accumulate(part_num_characters))
    current_segment_i = 0
    characters_read = 0
    buffer = 'filler'
    with file_path.open() as f:
        while buffer and num_preceding_characters_for_segment[current_segment_i] > characters_read:
            buffer = f.read(min(BUFFER_SIZE, num_preceding_characters_for_segment[current_segment_i] - characters_read))
            characters_read += len(buffer)
            result[current_segment_i] += buffer.count(pattern)
            if characters_read >= num_preceding_characters_for_segment[current_segment_i]:
                current_segment_i += 1
                if current_segment_i >= len(part_num_characters):
                    break
    return result


def move_by_n_characters_in_file(fd, n, buffer_size):
    characters_read = 0
    bytes_read = 0
    buffer = 'filler'
    while buffer and n > characters_read:
        buffer = fd.read(min(buffer_size, n - characters_read))
        characters_read += len(buffer)
        bytes_read += len(buffer.encode('utf-8'))
    return characters_read, bytes_read


def get_borders_with_documents_intact(file_path, num_parts):
    byte_borders = []
    num_characters_in_part = []
    length = count_characters_in_file(file_path)
    part_size = length // num_parts
    last_byte_border = 0
    total_characters_read = 0
    remainder = ""
    with file_path.open(buffering=BUFFER_SIZE) as f:
        for i in tqdm(range(num_parts), unit='part'):
            read_size = part_size * (i + 1) - total_characters_read
            characters_in_part, bytes_read = move_by_n_characters_in_file(f, read_size, BUFFER_SIZE)
            characters_in_part += len(remainder)
            bytes_read += len(remainder.encode('utf-8'))
            total_characters_read += characters_in_part
            if characters_in_part < read_size:
                byte_borders.append((last_byte_border, last_byte_border + bytes_read))
                num_characters_in_part.append(characters_in_part)
            else:
                line = f.readline()
                total_characters_read += len(line)
                success = False
                while line:
                    if '<page' in line:
                        new_page_start = line.index('<page')
                        remainder = line[new_page_start:]
                        line = line[:new_page_start]
                        characters_in_part += len(line)
                        bytes_read += len(line.encode('utf-8'))
                        new_byte_border = last_byte_border + bytes_read
                        byte_borders.append((last_byte_border, new_byte_border))
                        num_characters_in_part.append(characters_in_part)
                        last_byte_border = new_byte_border
                        success = True
                        break
                    characters_in_part += len(line)
                    bytes_read += len(line.encode('utf-8'))
                    line = f.readline()
                    total_characters_read += len(line)
                if not success:
                    byte_borders.append((last_byte_border, last_byte_border + bytes_read))
                    num_characters_in_part.append(characters_in_part)
    return byte_borders, num_characters_in_part


def show_prog(q, total_num_lines, name):
    prog = tqdm(total=total_num_lines, desc="Total", unit=name, unit_scale=True)
    while True:
        try:
            to_add = q.get(timeout=1)
            if to_add < 0:
                prog.close()
                return
            prog.n += to_add
            prog.update(0)
            if prog.n >= total_num_lines:
                break
        except Empty:
            continue
    prog.close()


def preprocess_wikipedia_parallel(
    num_jobs,
    file_path,
    output_dir,
    lang,
    tokenizer,
    sequence_length_range,
    start_doc_id=0,
    start_file_i=0,
    nltk_tokenization=True,
    remove_parentheses=False,
):
    logging.info("Calculating borders for multiprocessing...")
    byte_borders, num_characters_in_part = get_borders_with_documents_intact(file_path, num_jobs)
    logging.info(f"Found borders for multiprocessing: {byte_borders}")
    logging.info(f"Number of characters in parts: {num_characters_in_part}")
    num_output_files = [int(np.ceil(n / MAX_NUM_CHARACTERS_IN_1_FILE)) for n in num_characters_in_part]
    out_file_ids = list(accumulate(num_output_files, initial=start_file_i))
    logging.info(f"Calculating starting document ids for processes...")
    start_doc_ids = list(
        accumulate(
            # [count_pages_in_file(file_path, b[0], n) for b, n in zip(byte_borders, num_characters_in_part)],
            count_in_file_parts(file_path, num_characters_in_part, '<page'),
            initial=start_doc_id
        )
    )[:-1]
    logging.info(f"Starting document ids for processes are: {start_doc_ids}")
    logging.info(f"Calculating starting lines for processes...")
    start_line_ids = list(
        accumulate(
            # [count_lines_in_file(file_path, b[0], n) for b, n in zip(byte_borders, num_characters_in_part)],
            count_in_file_parts(file_path, num_characters_in_part, '\n'),
            initial=0
        )
    )[:-1]
    logging.info(f"Starting lines for processes are: {start_line_ids}")
    manager = mp.Manager()
    progress_queue = manager.Queue()
    logging.info("Creating progress process...")
    progress_process = mp.Process(target=show_prog, args=(progress_queue, count_lines_in_file(file_path), "Lines"))
    logging.info("Starting progress process...")
    progress_process.start()
    with mp.Pool(num_jobs) as pool:
        logging.info("Launching multiprocessing pool...")
        result = pool.map(
            preprocess_wikipedia,
            list(
                zip(
                    range(num_jobs),
                    [progress_queue] * num_jobs,
                    [file_path] * num_jobs,
                    byte_borders,
                    num_characters_in_part,
                    out_file_ids[:-1],
                    out_file_ids[1:],
                    num_output_files,
                    [output_dir] * num_jobs,
                    [lang] * num_jobs,
                    [tokenizer] * num_jobs,
                    [sequence_length_range] * num_jobs,
                    start_doc_ids,
                    start_line_ids,
                    [nltk_tokenization] * num_jobs,
                    [remove_parentheses] * num_jobs,
                    [5000] * num_jobs,
                )
            )
        )
    progress_queue.put(-1)
    progress_process.join()
    for i in range(1, len(result)):
        for k, v in result[i][0].items():
            result[0][0][k] += v
        result[0][1].update(result[i][1])
        result[0][2].update(result[i][2])
    return tuple(list(result[0]) + [start_file_i + sum(num_output_files)])


def preprocess_wikipedia(args):
    (
        rank,
        progress_queue,
        file_path,
        byte_borders,
        num_characters_in_part,
        start_out_file_i,
        first_forbidden_out_file_i,
        num_out_files,
        output_dir,
        lang,
        tokenizer,
        sequence_length_range,
        start_doc_id,
        start_line_id,
        nltk_tokenization,
        remove_parentheses,
        report_progress_every_n_lines
    ) = args
    sentences_by_number_of_words = {n: [] for n in range(sequence_length_range[0], sequence_length_range[1])}
    sentence_len_by_docs = {}
    doc_id_to_file_i = {}
    page = ""
    page_i = start_doc_id
    page_in_progress = False
    characters_for_1_file = num_characters_in_part // num_out_files
    total_number_of_characters_from_original_text_in_current_file = 0
    file_i = start_out_file_i
    doc_id = start_doc_id
    output_dir.mkdir(exist_ok=True, parents=True)
    current_file_path = output_dir / (str(file_i) + '.xml')
    tok_chars, untok_chars = {'\n', ' '}, set()
    num_lines_processed_when_progress_was_reported_last_time = start_line_id
    start_line, end_line = None, None
    file_text = ""
    with file_path.open(buffering=BUFFER_SIZE) as in_f:
        in_f.seek(byte_borders[0])
        num_read_characters = 0
        for i, line in enumerate(in_f, num_lines_processed_when_progress_was_reported_last_time):
            if len(line) > num_characters_in_part - num_read_characters:
                line = line[:num_characters_in_part - num_read_characters]
            num_read_characters += len(line)
            if i % report_progress_every_n_lines == 0:
                progress_queue.put(i - num_lines_processed_when_progress_was_reported_last_time)
                num_lines_processed_when_progress_was_reported_last_time = i
            total_number_of_characters_from_original_text_in_current_file += len(line)
            if '<page' in line:
                if PAGE_OPENING_NORMAL_TAG.match(line) is None:
                    logging.warning(
                        f'Encountered an unusual page opening tag in line {i} {repr(line)} in process {rank}'
                    )
                page_in_progress = True
                start_line = i
            if page_in_progress:
                page += line
            if '</page' in line:
                if page_in_progress:
                    if PAGE_CLOSING_NORMAL_TAG.match(line) is None:
                        logging.warning(
                            f'Encountered an unusual page opening tag in line {i} {repr(line)} in process {rank}.'
                        )
                    elif page.count('\n') == 1:
                        logging.warning(
                            f"Encountered a page which takes only one line. Line: {i}. Line {repr(line)} in process"
                            f"{rank}."
                        )
                    end_line = i
                    title = TITLE_OF_PAGE.search(page)
                    if title is None:
                        logging.warning(f"Title of page {page_i} from line {start_line} to {end_line} is not found.")
                        title = None
                    else:
                        title = title.group(1)
                    if COLON_TITLES.match(title) is None and '(disambiguation)' not in title:
                        text = TEXT_OF_PAGE.search(page)
                        if text is None:
                            logging.warning(
                                f"Text tag is not found on a page {page_i} from line {start_line} to {end_line} "
                                f"in process {rank} is not found. Skipping page.."
                            )
                        else:
                            pos_info = [file_path, start_line, end_line]
                            text, tok_chars, untok_chars = get_wiki_text_lines(
                                text.group(1),
                                lang,
                                tokenizer,
                                tok_chars,
                                untok_chars,
                                pos_info,
                                nltk_tokenization,
                                remove_parentheses,
                            )
                            if text:
                                file_text += doc_to_str(doc_id, file_path, title, start_line, end_line, '\n'.join(text))
                                arrangement, line_num_words = small.arrange_sentences_by_number_of_words_in_1_doc(
                                    text, sequence_length_range, [file_i, doc_id]
                                )
                                for k, v in arrangement.items():
                                    sentences_by_number_of_words[k] += v
                                sentence_len_by_docs[doc_id] = np.array(line_num_words)
                                doc_id_to_file_i[doc_id] = file_i
                                doc_id += 1
                                if total_number_of_characters_from_original_text_in_current_file > characters_for_1_file:
                                    assert file_i < first_forbidden_out_file_i, f"File you are going to write into " \
                                        f"is probably filled in other process. There is an error in distribution of " \
                                        f"data between processes."
                                    with current_file_path.open('w') as out_f:
                                        out_f.write(file_text)
                                    file_text = ""
                                    file_i += 1
                                    current_file_path = output_dir / (str(file_i) + '.xml')
                                    total_number_of_characters_from_original_text_in_current_file = 0
                else:
                    logging.warning(
                        f'Encountered closing page tag without opening tag. Line number: {i}. Line {repr(line)} in '
                        f'process {rank}.'
                    )
                page = ""
                page_i += 1
                start_line = None
                end_line = None
                page_in_progress = False
            if num_read_characters >= num_characters_in_part:
                break
        if len(page) != 0:
            logging.warning(
                f"The page {page_i} with title {title} in file {file_path} between lines {start_line} and {end_line} "
                f"is not finished in process {rank}."
            )
    progress_queue.put(i + 1 - num_lines_processed_when_progress_was_reported_last_time)
    if total_number_of_characters_from_original_text_in_current_file:
        assert file_i < first_forbidden_out_file_i, f"File you are going to write into is probably filled in other " \
            f"process. There is an error in distribution of data between processes."
        with current_file_path.open('w') as out_f:
            out_f.write(file_text)
    return sentences_by_number_of_words, sentence_len_by_docs, doc_id_to_file_i


def prepend_file_i(not_whole_segments, doc_id_to_file_i):
    print("not_whole_segments.shape:", not_whole_segments.shape)
    return np.concatenate(
        [
            np.expand_dims(
                np.vectorize(doc_id_to_file_i.get)(not_whole_segments[:, 0]),
                1
            ),
            not_whole_segments
        ],
        1
    )


def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    return True


def move_to_line(fd, line_i, read_size=65536):
    new_line_count = 0
    block = 'FILLER'
    num_blocks = -1
    last_block_count = 0
    pos_before_last_block = fd.tell()
    while block and new_line_count < line_i:
        pos_before_last_block = fd.tell()
        block = fd.read(read_size)
        last_block_count = block.count('\n')
        new_line_count += last_block_count
        num_blocks += 1
    if new_line_count < line_i:
        return False
    i = 0
    j = 0
    while i < line_i - new_line_count + last_block_count:
        j = block.index('\n', j) + 1
        i += 1
    fd.seek(pos_before_last_block + len(block[:j].encode('utf-8')))
    return True


def get_capitalization_label(word, no_label_if_all_characters_are_upper_case):
    if no_label_if_all_characters_are_upper_case:
        if word[0].isupper():
            return 'U'
    else:
        if len(word) > 1 and word.isupper():
            return 'U'
        if word[0].isupper():
            return 'u'
    return 'O'


def write_dataset_sub(
    borders,
    orig_file,
    output_dir,
    create_model_input,
    bert_labels,
    autoregressive_labels,
    allowed_punctuation,
    only_first_punctuation_character_after_word_in_autoregressive,
    no_label_if_all_characters_are_upper_case,
):
    extended_punctuation = allowed_punctuation | {' ', '\n'}
    output_dir.mkdir(parents=True, exist_ok=True)
    text_fn, input_fn = output_dir / Path('text.txt'), output_dir / Path('input.txt')
    bert_fn, ar_fn = output_dir / Path('bert_labels.txt'), output_dir / Path('autoregressive_labels.txt')
    original_text = ""
    with orig_file.open(buffering=BUFFER_SIZE) as in_f:
        move_to_line(in_f, borders[0])
        for l_i in range(borders[1] - borders[0]):
            original_text += in_f.readline()
    with text_fn.open('w', buffering=BUFFER_SIZE) as tf:
        tf.write(original_text)

    def autoregressive_repl3(match):
        w = match.group(1)
        p = match.group(2)
        plbl = ''
        if p:
            for c in p:
                if c in extended_punctuation:
                    plbl += c
        return 'U' if len(w) > 1 and w.isupper() else ('u' if w[0].isupper() else 'O') + plbl

    def autoregressive_repl4(match):
        p = match.group(2)
        plbl = ''
        if p:
            for c in p:
                if c in extended_punctuation:
                    plbl += c
        return 'U' if match.group(1)[0].isupper() else 'O' + plbl

    def bert_repl1(match):
        w = match.group(1)
        p = match.group(2)
        if p:
            c_i = 0
            while c_i < len(p) and p[c_i] not in allowed_punctuation:
                c_i += 1
        return (p[c_i] if p and c_i < len(p) else 'O') \
            + ('U' if len(w) > 1 and w.isupper() else ('u' if w[0].isupper() else 'O')) \
            + ('\n' if '\n' in p else ' ')

    def bert_repl2(match):
        p = match.group(0)
        if p:
            c_i = 0
            while c_i < len(p) and p[c_i] not in allowed_punctuation:
                c_i += 1
        return (p[c_i] if p and c_i < len(p) else 'O') \
            + ('U' if match.group(1)[0].isupper() else 'O') \
            + ('\n' if '\n' in p else ' ')

    def autoregressive_repl1(match):
        w = match.group(1)
        p = match.group(2)
        if p:
            c_i = 0
            while c_i < len(p) and p[c_i] not in allowed_punctuation:
                c_i += 1
        return ('U' if len(w) > 1 and w.isupper() else ('u' if w[0].isupper() else 'O')) \
            + (p[c_i] if p and c_i < len(p) else '') \
            + ('\n' if '\n' in p else ' ') if p else ' '

    def autoregressive_repl2(match):
        p = match.group(2)
        if p:
            c_i = 0
            while c_i < len(p) and p[c_i] not in allowed_punctuation:
                c_i += 1
        return ('U' if match.group(1)[0].isupper() else 'O') \
            + (p[c_i] if p and c_i < len(p) else '') \
            + ('\n' if '\n' in p else ' ') if p else ' '

    def model_input_repl(match):
        return match.group(1).lower() + ('\n' if '\n' in match.group(2) else ' ')

    if create_model_input:
        logging.info("    Creating model input...")
        with input_fn.open('w') as inp_f:
            inp_f.write(small.WORD_WITH_FOLLOWING_PUNCTUATION.sub(model_input_repl, original_text))
    wrong_characters = re.compile('[^' + ''.join(allowed_punctuation | set(' \n/.,UOu+-')) + ']+')
    if bert_labels:
        logging.info("    Creating Evelina labels...")
        with bert_fn.open('w') as bf:
            repl = bert_repl2 if no_label_if_all_characters_are_upper_case else bert_repl1
            bf.write(wrong_characters.sub('', small.WORD_WITH_FOLLOWING_PUNCTUATION.sub(repl, original_text)))
    if autoregressive_labels:
        logging.info("    Creating autoregressive labels...")
        with ar_fn.open('w') as af:
            if only_first_punctuation_character_after_word_in_autoregressive:
                repl = autoregressive_repl2 if no_label_if_all_characters_are_upper_case else autoregressive_repl1
                af.write(wrong_characters.sub('', small.WORD_WITH_FOLLOWING_PUNCTUATION.sub(repl, original_text)))
            else:
                repl = autoregressive_repl4 if no_label_if_all_characters_are_upper_case else autoregressive_repl3
                af.write(wrong_characters.sub('', small.WORD_WITH_FOLLOWING_PUNCTUATION.sub(repl, original_text)))


def write_dataset_parallel(
    borders,
    input_file,
    output_dir,
    create_model_input,
    bert_labels,
    autoregressive_labels,
    allowed_punctuation,
    only_first_punctuation_character_after_word_in_autoregressive,
    no_label_if_all_characters_are_upper_case,
    num_jobs,
):
    num_jobs = min(num_jobs, borders[1] - borders[0])
    num_parts = max(ceil((borders[1] - borders[0]) / MAX_NUM_LINES_PER_PROCESS), num_jobs)
    num_lines_in_part = (borders[1] - borders[0]) // num_parts
    job_borders = [
        (borders[0] + i * num_lines_in_part, borders[0] + (i + 1) * num_lines_in_part) for i in range(num_parts - 1)
    ] + [(borders[0] + (num_parts - 1) * num_lines_in_part, borders[1])]
    text_fn, input_fn = output_dir / 'text.txt', output_dir / 'input.txt'
    bert_fn, ar_fn = output_dir / 'bert_labels.txt', output_dir / 'autoregressive_labels.txt'
    tmp_dir = output_dir / 'tmp'
    output_dirs = [tmp_dir / str(i) for i in range(num_parts)]
    manager = mp.Manager()
    progress_queue = manager.Queue()
    progress_process = mp.Process(target=show_prog, args=(progress_queue, borders[1] - borders[0], "line"))
    progress_process.start()
    with mp.Pool(num_jobs) as pool:
        pool.starmap(
            write_dataset_fast,
            list(
                zip(
                    job_borders,
                    [input_file] * num_parts,
                    output_dirs,
                    [create_model_input] * num_parts,
                    [bert_labels] * num_parts,
                    [autoregressive_labels] * num_parts,
                    [allowed_punctuation] * num_parts,
                    [only_first_punctuation_character_after_word_in_autoregressive] * num_parts,
                    [no_label_if_all_characters_are_upper_case] * num_parts,
                    range(num_parts),
                    [progress_queue] * num_parts,
                )
            )
        )
    progress_queue.put(-1)
    progress_process.join()
    for joined_fn in [text_fn, input_fn, bert_fn, ar_fn]:
        with joined_fn.open('w') as f:
            run(['cat'] + [str(d / f'{joined_fn.stem}.txt') for d in output_dirs], stdout=f)
    shutil.rmtree(str(tmp_dir))


def write_dataset_fast(
    borders,
    input_file,
    output_dir,
    create_model_input,
    bert_labels,
    autoregressive_labels,
    allowed_punctuation,
    only_first_punctuation_character_after_word_in_autoregressive,
    no_label_if_all_characters_are_upper_case,
    part_number=None,
    progress_queue=None,
):
    extended_punctuation = allowed_punctuation | {' ', '\n'}
    output_dir.mkdir(parents=True, exist_ok=True)
    text_fn, input_fn = output_dir / Path('text.txt'), output_dir / Path('input.txt')
    bert_fn, ar_fn = output_dir / Path('bert_labels.txt'), output_dir / Path('autoregressive_labels.txt')
    autoregressive_text = ""
    input_text = ""
    with input_file.open(buffering=BUFFER_SIZE) as in_f:
        move_to_line(in_f, borders[0])
        for l_i in range(borders[1] - borders[0]):
            input_text += in_f.readline()
    if part_number is None:
        prog = tqdm(total=len(input_text), desc="Total", unit='char', unit_scale=True)
    with text_fn.open('w', buffering=BUFFER_SIZE) as tf, \
            input_fn.open('w', buffering=BUFFER_SIZE) as inp_f, \
            bert_fn.open('w', buffering=BUFFER_SIZE) as bf:
        line_progress = 0
        for m in small.WORD_WITH_FOLLOWING_PUNCTUATION.finditer(input_text):
            all_text = m.group(0)
            tf.write(all_text)
            if part_number is None:
                prog.n += len(all_text)
                if prog.n % 10000 == 0:
                    prog.update()
            else:
                line_progress += all_text.count('\n')
                if line_progress > 5000:
                    progress_queue.put(line_progress)
                    line_progress = 0
            word, punctuation = m.group(1), m.group(2)
            punctuation = m.group(2)
            if create_model_input:
                inp_f.write(word.lower() + ('\n' if '\n' in punctuation else ' '))
            if bert_labels:
                if punctuation:
                    c_i = 0
                    while c_i < len(punctuation) and punctuation[c_i] not in allowed_punctuation:
                        c_i += 1
                    if c_i < len(punctuation):
                        lbl = punctuation[c_i]
                    else:
                        lbl = 'O'
                else:
                    lbl = 'O'
                lbl += get_capitalization_label(word, no_label_if_all_characters_are_upper_case)
                lbl += '\n' if '\n' in punctuation else ' '
                bf.write(lbl)
            if autoregressive_labels:
                autoregressive_text += get_capitalization_label(word, no_label_if_all_characters_are_upper_case)
                if only_first_punctuation_character_after_word_in_autoregressive:
                    if punctuation:
                        c_i = 0
                        while c_i < len(punctuation) and punctuation[c_i] not in allowed_punctuation:
                            c_i += 1
                        if c_i < len(punctuation):
                            autoregressive_text += punctuation[c_i]
                        autoregressive_text += '\n' if '\n' in punctuation else ' '
                    else:
                        autoregressive_text += ' '
                else:
                    for c in punctuation:
                        if c in extended_punctuation:
                            autoregressive_text += c
    autoregressive_text = autoregressive_text.rstrip(' ')
    if not only_first_punctuation_character_after_word_in_autoregressive:
        allowed = allowed_punctuation | set(' \nUOu/.,+-')
        allowed.remove('-')
        wrong_characters = re.compile('[^' + ''.join(allowed) + '-]+')
        autoregressive_text = wrong_characters.sub('', autoregressive_text)
    with ar_fn.open('w') as af:
        af.write(autoregressive_text)


def write_dataset(
    borders,
    input_file,
    output_dir,
    create_model_input,
    bert_labels,
    autoregressive_labels,
    allowed_punctuation,
    only_first_punctuation_character_after_word_in_autoregressive,
    no_label_if_all_characters_are_upper_case,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    text_fn, input_fn = output_dir / Path('text.txt'), output_dir / Path('input.txt')
    bert_fn, ar_fn = output_dir / Path('bert_labels.txt'), output_dir / Path('autoregressive_labels.txt')
    with input_file.open(buffering=BUFFER_SIZE) as in_f, \
            text_fn.open('w', buffering=BUFFER_SIZE) as tf, \
            input_fn.open('w', buffering=BUFFER_SIZE) as inp_f, \
            bert_fn.open('w', buffering=BUFFER_SIZE) as bf, \
            ar_fn.open('w', buffering=BUFFER_SIZE) as af:
        move_to_line(in_f, borders[0])
        for l_i in tqdm(range(borders[1] - borders[0])):
            line = in_f.readline().strip()
            if not line:
                raise ValueError(
                    f"Line number {l_i} in file {input_file} is empty, whereas all lines in file for cutting has "
                    f"to be not empty. You have to either check second element in `borders` parameter or check "
                    f"creation of {input_file}."
                )
            tf.write(line + '\n')
            line = [s for s in small.WORD.split(line) if s]
            if create_model_input:
                inp_f.write(' '.join([s.lower() for s in line if small.WORD.match(s)]) + '\n')
            if bert_labels:
                bf.write(
                    small.create_bert_labels(
                        line, allowed_punctuation, no_label_if_all_characters_are_upper_case
                    ) + '\n'
                )
            if autoregressive_labels:
                af.write(
                    small.create_autoregressive_labels(
                        line,
                        allowed_punctuation,
                        only_first_punctuation_character_after_word_in_autoregressive,
                        no_label_if_all_characters_are_upper_case,
                    ) + '\n'
                )


def read_doc(fd):
    text = []
    line = fd.readline()
    while line and not line.startswith('</doc>'):
        text.append(line.strip())
        line = fd.readline()
    return text


def cut_and_save(segments, doc_dir, output_file):
    current_pos = 0
    current_file_i = -1
    current_fd = None
    current_doc = None
    current_doc_id = -1
    line_i = 0
    with output_file.open('w') as f:
        for s_i, segment in tqdm(enumerate(segments), total=len(segments)):
            file_i = segment[0]
            doc_id = segment[1]
            if current_doc_id > doc_id:
                logging.warning(f"Documents are not in order: current_doc_id={current_doc_id}, next doc_id={doc_id}")
            if current_file_i != file_i:
                current_file_i = file_i
                if current_fd is not None:
                    current_fd.close()
                current_fd = (doc_dir / (str(current_file_i) + '.xml')).open()
                current_doc_id = -1
                line_i = 0
            if current_doc_id != doc_id:
                line = 'FILLER'
                count = 0
                while line and not line.startswith(f'<doc docid="{doc_id}"'):
                    line = current_fd.readline()
                    count += 1
                    line_i += 1
                if count != 1:
                    logging.warning(
                        f"The next document is supposed to start right after previous document: "
                        f"file_i={file_i} current_doc_id={current_doc_id} doc_id={doc_id} count={count} line_i={line_i}"
                    )
                current_doc = read_doc(current_fd)
                current_doc_id = doc_id
                line_i += len(current_doc)
            text_seg = small.cut_words(' '.join(current_doc[segment[2] : segment[3]]), segment[4], segment[5]) + '\n'
            f.write(text_seg)
            current_pos += len(text_seg)


def read_docs_from_file(file_path: Path) -> Tuple[Dict[int, Dict[str, Union[Path, int, str]]], List[int]]:
    current_doc = ""
    curr_doc_id = None
    docs = {}
    num_raw_characters_by_doc = []
    with file_path.open(buffering=BUFFER_SIZE) as f:
        for i, line in enumerate(f):
            start = DOC_HEAD.match(line)
            if start is not None:
                if curr_doc_id is not None:
                    raise ValueError(
                        f"Encountered start of document number {start.group(1)} on line {i} in file {file_path} while "
                        f"document number {curr_doc_id} is still in progress."
                    )
                num_raw_characters_by_doc.append(0)
                curr_source, curr_title = start.group(2), start.group(3)
                curr_doc_id, curr_start_line, curr_end_line = [int(start.group(i)) for i in [1, 4, 5]]
            if line.startswith("</doc>"):
                if curr_doc_id is None:
                    raise ValueError(
                        f"Encountered end of document on line {i} in file {file_path} while there is no document in "
                        f"progress."
                    )
                docs[curr_doc_id] = {
                    "source": curr_source,
                    "title": curr_title,
                    "start_line": curr_start_line,
                    "end_line": curr_end_line,
                    "text": current_doc
                }
                current_doc = ""
                curr_doc_id = None
            if curr_doc_id is not None and start is None:
                current_doc += line
                num_raw_characters_by_doc[-1] += len(line)
    return docs, num_raw_characters_by_doc


def doc_to_str(docid, source, title, start_line, end_line, text):
    res = DOC_HEAD_TMPL.format(docid, source, title, start_line, end_line) + '\n' + text
    if text[-1] != '\n':
        res += '\n'
    return res + DOC_END + '\n'


def write_docs_to_file(docs, file_path, append=False):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('a' if append else 'w') as f:
        for k, v in docs.items():
            f.write(doc_to_str(k, v['source'], v["title"], v["start_line"], v["end_line"], v["text"]))


def shuffle_file_lines(input_file, output_file):
    with output_file.open('w') as f:
        run(['shuf', str(input_file)], stdout=f)


def collect_info_about_preprocessed_data(args):
    (rank, progresss_queue, files, sequence_length_range) = args
    sentences_by_number_of_words = {
        n: [] for n in range(sequence_length_range[0], sequence_length_range[1])
    }
    sentence_len_by_docs, doc_id_to_file_i = {}, {}
    for p in files:
        if is_int(p.stem) and p.suffixes == ['.xml']:
            file_i = int(p.stem)
            docs, _ = read_docs_from_file(p)
            for doc_id, doc in docs.items():
                doc_id_to_file_i[doc_id] = file_i
                arrangement, line_num_words = small.arrange_sentences_by_number_of_words_in_1_doc(
                    doc['text'].splitlines(), sequence_length_range, [file_i, doc_id]
                )
                sentence_len_by_docs[doc_id] = np.array(line_num_words)
                for k, v in arrangement.items():
                    sentences_by_number_of_words[k] += v
        progresss_queue.put(1)
    return sentences_by_number_of_words, sentence_len_by_docs, doc_id_to_file_i


def collect_info_about_preprocessed_data_parallel(document_dir, sequence_length_range, num_jobs):
    sentences_by_number_of_words = {
        n: [] for n in range(sequence_length_range[0], sequence_length_range[1])
    }
    sentence_len_by_docs, doc_id_to_file_i = {}, {}
    files = [f for f in document_dir.iterdir() if is_int(f.stem) and f.suffixes == ['.xml']]
    num_jobs = max(num_jobs, len(files))
    num_files_per_job = len(files) // num_jobs
    distributed_files = (
        [files[i * num_files_per_job: (i + 1) * num_files_per_job] for i in range(num_jobs - 1)]
        + [files[(num_jobs - 1) * num_files_per_job:]]
    )
    manager = mp.Manager()
    progress_queue = manager.Queue()
    progress_process = mp.Process(target=show_prog, args=(progress_queue, len(files), "Files"))
    progress_process.start()
    with mp.Pool(num_jobs) as pool:
        result = pool.map(
            collect_info_about_preprocessed_data,
            list(
                zip(
                    range(num_jobs),
                    [progress_queue] * num_jobs,
                    distributed_files,
                    [sequence_length_range] * num_jobs,
                )
            )
        )
        logging.info("Stopping tqdm process...")
        progress_queue.put(-1)
        progress_process.join()
    found_documents = set()
    for r in result:
        for k, v in r[0].items():
            sentences_by_number_of_words[k] += v
        new_doc_ids = set(r[1])
        if found_documents & new_doc_ids:
            raise ValueError(f"Found duplicate documents with ids {found_documents & new_doc_ids}")
        sentence_len_by_docs.update(r[1])
        doc_id_to_file_i.update(r[2])
    return sentences_by_number_of_words, sentence_len_by_docs, doc_id_to_file_i


def join_sentence_len(di_ss_se, sentence_len_by_docs):
    return sum(sentence_len_by_docs[di_ss_se[0]][di_ss_se[1]: di_ss_se[2]])


def main():
    args = small.get_args(
        SUPPORTED_CORPUS_TYPES, add_nltk_tokenization_parameter=True, add_resume_argument=True, add_num_jobs=True)
    document_dir = args.output_dir / Path("documents")
    if args.resume_from is None:
        tokenizer = get_tokenizer(args.tokenizer)
        sentences_by_number_of_words = {
            n: [] for n in range(args.sequence_length_range[0], args.sequence_length_range[1])
        }
        sentence_len_by_docs = {}
        doc_id_to_file_i = {}
        num_docs, num_files = 0, 0
        for corpus_type, file_path in zip(args.corpus_types, args.input_files):
            if corpus_type == SUPPORTED_CORPUS_TYPES[0]:
                logging.info(f"Preprocessing wikipedia file {file_path}...")
                res = preprocess_wikipedia_parallel(
                    args.num_jobs,
                    file_path,
                    document_dir,
                    args.input_language,
                    tokenizer,
                    args.sequence_length_range,
                    num_docs,
                    num_files,
                    args.nltk_tokenization,
                    '(' not in args.allowed_punctuation or ')' not in args.allowed_punctuation,
                )
                (corpus_sentences_by_number_of_words, corpus_sentence_len_by_docs, corpus_doc_id_to_file_i, num_files) \
                    = res
                for k, v in corpus_sentences_by_number_of_words.items():
                    sentences_by_number_of_words[k] += v
                sentence_len_by_docs.update(corpus_sentence_len_by_docs)
                doc_id_to_file_i.update(corpus_doc_id_to_file_i)
            else:
                raise ValueError(
                    f"Unsupported corpus type '{corpus_type}. Supported corpus types are {SUPPORTED_CORPUS_TYPES}"
                )
            number_of_corpus_sentences = sum([len(v) for v in corpus_sentences_by_number_of_words.values()])
            logging.info(
                f"Finished preprocessing corpus {file_path}. Number of sentences the corpus: "
                f"{number_of_corpus_sentences}, number of documents in the corpus: {len(corpus_sentence_len_by_docs)}"
            )
    else:
        logging.info(f"Loading stats and info about dataset in directory '{document_dir}'...")
        sentences_by_number_of_words, sentence_len_by_docs, doc_id_to_file_i = \
            collect_info_about_preprocessed_data_parallel(document_dir, args.sequence_length_range, args.num_jobs)
    for k, v in sentences_by_number_of_words.items():
        for e in v:
            if e[0] == 0 and e[1] == 325:
                print(f"length={k} segment={e}")
    number_of_sentences_in_input = sum([len(e) for e in sentence_len_by_docs.values()])
    if args.size is None:
        args.size = number_of_sentences_in_input
        if args.dev_size > args.size:
            raise ValueError(f"Parameter `--dev_size={args.dev_size}` is greater than size of all dataset {args.size}")
    sorted_text_file = args.output_dir / 'sorted_text.txt'
    if args.resume_from is None or args.resume_from in ["normalization", "cutting"]:
        if (
            sum([len(x) for x in sentences_by_number_of_words.values()])
            < args.size * args.percentage_segments_with_intact_sentences / 100
        ):
            raise ValueError(
                f"Cannot find enough segments consisting of whole sentences to build dataset with {args.size} segments "
                f"and at least {args.percentage_segments_with_intact_sentences}% segments consisting of whole "
                f"sentences. Try to reduce dataset size of parameter `--percentage_segments_with_intact_sentences"
            )
        logging.info(f"Selecting segments with intact sentences...")
        result, number_of_words_stats, remaining_by_docs = small.select_close_to_uniform_distribution(
            sentences_by_number_of_words,
            args.size,
            args.percentage_segments_with_intact_sentences,
            {k: len(v) for k, v in sentence_len_by_docs.items()},
            1,
        )
        result = np.array(result)
        result = np.concatenate(
            [
                result,
                np.zeros([result.shape[0], 1], dtype=result.dtype),
                np.expand_dims(
                    np.vectorize(
                        join_sentence_len, otypes=[result.dtype], signature='(n),()->()',
                    )(result[:, 1:], sentence_len_by_docs),
                    1,
                ),
            ],
            1,
        )
        logging.info("Selecting segments with not intact sentences...")
        result = np.concatenate(
            [
                result,
                prepend_file_i(
                    np.array(
                        small.create_not_whole_sentence_segments(
                            sentence_len_by_docs,
                            remaining_by_docs,
                            number_of_words_stats,
                            args.size,
                            args.percentage_segments_with_intact_sentences,
                        ),
                    ),
                    doc_id_to_file_i
                )
            ]
        )
        result = result[np.argsort(result[:, 0])]  # sort by file index
        result = result[np.argsort(result[:, 1], kind='stable')]  # sort by document index
        logging.info("Cutting segments...")
        cut_and_save(result, document_dir, sorted_text_file)
    shuffled_text_file = args.output_dir / 'shuffled_text.txt'
    if args.resume_from is None or args.resume_from in ["normalization", "cutting", "shuffling"]:
        logging.info("shuffling segments...")
        shuffle_file_lines(sorted_text_file, shuffled_text_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.test_size > 0:
        logging.info("Writing test dataset...")
        write_dataset(
            [0, args.test_size],
            shuffled_text_file,
            args.output_dir / Path("test"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    if args.dev_size > 0:
        logging.info("Writing dev dataset...")
        write_dataset(
            [args.test_size, args.test_size + args.dev_size],
            shuffled_text_file,
            args.output_dir / Path("dev"),
            args.create_model_input,
            args.bert_labels,
            args.autoregressive_labels,
            args.allowed_punctuation,
            args.only_first_punctuation_character_after_word_in_autoregressive,
            args.no_label_if_all_characters_are_upper_case,
        )
    logging.info("Writing train dataset...")
    write_dataset(
        [args.test_size + args.dev_size, args.size],
        shuffled_text_file,
        args.output_dir / Path("train"),
        args.create_model_input,
        args.bert_labels,
        args.autoregressive_labels,
        args.allowed_punctuation,
        args.only_first_punctuation_character_after_word_in_autoregressive,
        args.no_label_if_all_characters_are_upper_case,
    )


if __name__ == "__main__":
    main()
