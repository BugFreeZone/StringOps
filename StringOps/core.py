import re
import string
import random
import unicodedata
from collections import Counter

# ===========================
# Базовые и расширенные функции
# ===========================

def is_palindrome(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def count_vowels(s: str) -> int:
    vowels = 'aeiouаеёиоуыэюя'
    return sum(1 for c in s.lower() if c in vowels)

def count_consonants(s: str) -> int:
    consonants = 'bcdfghjklmnpqrstvwxyzбвгджзйклмнпрстфхцчшщ'
    return sum(1 for c in s.lower() if c in consonants)

def reverse_words(s: str) -> str:
    return ' '.join(s.split()[::-1])

def reverse_string(s: str) -> str:
    return s[::-1]

def to_camel_case(s: str) -> str:
    words = re.split(r'[\s_-]+', s)
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def to_snake_case(s: str) -> str:
    words = re.split(r'[\s-]+', s)
    return '_'.join(word.lower() for word in words)

def to_kebab_case(s: str) -> str:
    words = re.split(r'[\s_]+', s)
    return '-'.join(word.lower() for word in words)

def remove_punctuation(s: str) -> str:
    return ''.join(c for c in s if c not in string.punctuation)

def unique_characters(s: str) -> set:
    return set(s)

def most_common_char(s: str) -> str:
    if not s:
        return ''
    return Counter(s).most_common(1)[0][0]

def count_words(s: str) -> int:
    return len(s.split())

def capitalize_words(s: str) -> str:
    return s.title()

def swap_case(s: str) -> str:
    return s.swapcase()

def remove_digits(s: str) -> str:
    return ''.join(c for c in s if not c.isdigit())

def only_digits(s: str) -> str:
    return ''.join(c for c in s if c.isdigit())

def is_anagram(s1: str, s2: str) -> bool:
    s1_clean = ''.join(sorted(c.lower() for c in s1 if c.isalnum()))
    s2_clean = ''.join(sorted(c.lower() for c in s2 if c.isalnum()))
    return s1_clean == s2_clean

def remove_whitespace(s: str) -> str:
    return ''.join(s.split())

def normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def find_substrings(s: str, sub: str) -> list:
    indices = []
    i = s.find(sub)
    while i != -1:
        indices.append(i)
        i = s.find(sub, i + 1)
    return indices

def starts_with(s: str, prefix: str) -> bool:
    return s.startswith(prefix)

def ends_with(s: str, suffix: str) -> bool:
    return s.endswith(suffix)

def remove_duplicates(s: str) -> str:
    seen = set()
    return ''.join(seen.add(c) or c for c in s if c not in seen)

def rot13(s: str) -> str:
    return s.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"
    ))

def extract_emails(s: str) -> list:
    return re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', s)

def extract_urls(s: str) -> list:
    return re.findall(r'https?://[^\s]+', s)

def is_numeric(s: str) -> bool:
    return s.isdigit()

def is_alpha(s: str) -> bool:
    return s.isalpha()

def is_alnum(s: str) -> bool:
    return s.isalnum()

def remove_non_ascii(s: str) -> str:
    return ''.join(c for c in s if ord(c) < 128)

def extract_hashtags(s: str) -> list:
    return re.findall(r'#\w+', s)

def extract_mentions(s: str) -> list:
    return re.findall(r'@\w+', s)

def count_substring(s: str, sub: str) -> int:
    return s.count(sub)

def truncate(s: str, length: int, suffix: str = '...') -> str:
    return s if len(s) <= length else s[:length - len(suffix)] + suffix

def pad_left(s: str, width: int, char: str = ' ') -> str:
    return s.rjust(width, char)

def pad_right(s: str, width: int, char: str = ' ') -> str:
    return s.ljust(width, char)

def pad_center(s: str, width: int, char: str = ' ') -> str:
    return s.center(width, char)

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def get_initials(s: str) -> str:
    return ''.join(word[0].upper() for word in s.split() if word)

def replace_multiple(s: str, replacements: dict) -> str:
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    return pattern.sub(lambda m: replacements[m.group(0)], s)

def split_by_length(s: str, length: int) -> list:
    return [s[i:i+length] for i in range(0, len(s), length)]

def find_longest_word(s: str) -> str:
    words = re.findall(r'\w+', s)
    return max(words, key=len) if words else ''

def find_shortest_word(s: str) -> str:
    words = re.findall(r'\w+', s)
    return min(words, key=len) if words else ''

def is_title_case(s: str) -> bool:
    return s == s.title()

def is_uppercase(s: str) -> bool:
    return s.isupper()

def is_lowercase(s: str) -> bool:
    return s.islower()

def remove_html_tags(s: str) -> str:
    return re.sub(r'<[^>]+>', '', s)

def extract_numbers(s: str) -> list:
    return re.findall(r'\d+(?:\.\d+)?', s)

def mask_string(s: str, start: int = 0, end: int = None, mask_char: str = '*') -> str:
    if end is None or end > len(s):
        end = len(s)
    return s[:start] + mask_char * (end - start) + s[end:]

def repeat_string(s: str, times: int) -> str:
    return s * times

def remove_empty_lines(s: str) -> str:
    return '\n'.join(line for line in s.splitlines() if line.strip())

def get_lines(s: str) -> list:
    return s.splitlines()

def join_lines(lines: list, sep: str = '\n') -> str:
    return sep.join(lines)

def get_word_frequencies(s: str) -> dict:
    words = re.findall(r'\w+', s.lower())
    return dict(Counter(words))

def get_char_frequencies(s: str) -> dict:
    return dict(Counter(s))

def replace_first(s: str, old: str, new: str) -> str:
    return s.replace(old, new, 1)

def replace_last(s: str, old: str, new: str) -> str:
    pos = s.rfind(old)
    if pos == -1:
        return s
    return s[:pos] + new + s[pos+len(old):]

def count_lines(s: str) -> int:
    return len(s.splitlines())

def count_unique_words(s: str) -> int:
    words = re.findall(r'\w+', s.lower())
    return len(set(words))

def remove_non_letters(s: str) -> str:
    return ''.join(c for c in s if c.isalpha())

def remove_non_numbers(s: str) -> str:
    return ''.join(c for c in s if c.isdigit())

def find_all_capitalized_words(s: str) -> list:
    return re.findall(r'\b[A-ZА-ЯЁ][a-zа-яё]*', s)

def find_all_lowercase_words(s: str) -> list:
    return re.findall(r'\b[a-zа-яё]+\b', s)

def is_valid_email(s: str) -> bool:
    return bool(re.fullmatch(r'[\w\.-]+@[\w\.-]+\.\w+', s))

def is_valid_url(s: str) -> bool:
    return bool(re.fullmatch(r'https?://[^\s]+', s))

def remove_control_chars(s: str) -> str:
    return ''.join(c for c in s if c.isprintable())

def get_unicode_names(s: str) -> list:
    return [unicodedata.name(c, f'U+{ord(c):04X}') for c in s]

def escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))

def unescape_html(s: str) -> str:
    import html
    return html.unescape(s)

def is_ascii(s: str) -> bool:
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def is_printable(s: str) -> bool:
    return all(c.isprintable() for c in s)

def extract_dates(s: str) -> list:
    pattern = r'\b(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b'
    return re.findall(pattern, s)

def extract_phone_numbers(s: str) -> list:
    pattern = r'\+7\d{10}|\b8\d{10}\b'
    return re.findall(pattern, s)

def get_ngrams(s: str, n: int) -> list:
    words = s.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def find_all_numbers(s: str) -> list:
    return re.findall(r'-?\d+\.?\d*', s)

def extract_words(s: str) -> list:
    return re.findall(r'\w+', s)

# ===========================
# Ещё больше функций
# ===========================

def random_string(length: int, chars: str = string.ascii_letters + string.digits) -> str:
    return ''.join(random.choice(chars) for _ in range(length))

def shuffle_string(s: str) -> str:
    chars = list(s)
    random.shuffle(chars)
    return ''.join(chars)

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Строки должны быть одинаковой длины")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def jaccard_similarity(s1: str, s2: str) -> float:
    set1, set2 = set(s1), set(s2)
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 1.0

def soundex(s: str) -> str:
    s = s.upper()
    replacements = (
        ('BFPV', '1'), ('CGJKQSXZ', '2'), ('DT', '3'),
        ('L', '4'), ('MN', '5'), ('R', '6')
    )
    result = s[0]
    for char in s[1:]:
        for group, code in replacements:
            if char in group:
                if code != result[-1]:
                    result += code
                break
        else:
            if result[-1] != '0':
                result += '0'
    result = re.sub(r'0', '', result)
    return (result + '000')[:4]

def caesar_cipher(s: str, shift: int) -> str:
    def shift_char(c):
        if c.isupper():
            return chr((ord(c) - 65 + shift) % 26 + 65)
        elif c.islower():
            return chr((ord(c) - 97 + shift) % 26 + 97)
        else:
            return c
    return ''.join(shift_char(c) for c in s)

def count_sentences(s: str) -> int:
    return len(re.findall(r'[.!?]+', s))

def split_sentences(s: str) -> list:
    return re.split(r'(?<=[.!?])\s+', s.strip())

def find_palindromic_words(s: str) -> list:
    words = re.findall(r'\w+', s.lower())
    return [word for word in words if word == word[::-1] and len(word) > 1]

def find_repeated_words(s: str) -> set:
    words = re.findall(r'\w+', s.lower())
    return set([word for word, count in Counter(words).items() if count > 1])

def compress_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def to_binary(s: str) -> str:
    return ' '.join(format(ord(c), '08b') for c in s)

def from_binary(b: str) -> str:
    return ''.join([chr(int(x, 2)) for x in b.split()])

def to_hex(s: str) -> str:
    return s.encode('utf-8').hex()

def from_hex(h: str) -> str:
    return bytes.fromhex(h).decode('utf-8')

def to_base64(s: str) -> str:
    import base64
    return base64.b64encode(s.encode()).decode()

def from_base64(b64: str) -> str:
    import base64
    return base64.b64decode(b64.encode()).decode()

def find_longest_common_substring(s1: str, s2: str) -> str:
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

def find_all_substrings(s: str, min_length: int = 2) -> set:
    substrings = set()
    for i in range(len(s)):
        for j in range(i + min_length, len(s) + 1):
            substrings.add(s[i:j])
    return substrings

def is_subsequence(s: str, sub: str) -> bool:
    it = iter(s)
    return all(c in it for c in sub)

def find_ngrams_char(s: str, n: int) -> list:
    return [s[i:i+n] for i in range(len(s)-n+1)]

def count_uppercase(s: str) -> int:
    return sum(1 for c in s if c.isupper())

def count_lowercase(s: str) -> int:
    return sum(1 for c in s if c.islower())

def find_first_digit(s: str) -> int:
    m = re.search(r'\d', s)
    return m.start() if m else -1

def find_first_letter(s: str) -> int:
    m = re.search(r'[a-zA-Zа-яА-ЯёЁ]', s)
    return m.start() if m else -1

def split_by_punctuation(s: str) -> list:
    return re.split(r'[{}]+'.format(re.escape(string.punctuation)), s)

def remove_duplicate_words(s: str) -> str:
    seen = set()
    result = []
    for word in s.split():
        w = word.lower()
        if w not in seen:
            seen.add(w)
            result.append(word)
    return ' '.join(result)

def is_pangram(s: str, alphabet: str = string.ascii_lowercase) -> bool:
    return set(alphabet).issubset(set(s.lower()))

def find_missing_letters(s: str, alphabet: str = string.ascii_lowercase) -> set:
    return set(alphabet) - set(s.lower())

def is_isogram(s: str) -> bool:
    s = ''.join(c.lower() for c in s if c.isalpha())
    return len(set(s)) == len(s)

def to_title_case(s: str) -> str:
    return s.title()

def to_sentence_case(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:].lower() if s else s

def to_alternating_case(s: str) -> str:
    return ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(s))

def extract_words_of_length(s: str, length: int) -> list:
    return [word for word in re.findall(r'\w+', s) if len(word) == length]

def remove_brackets_content(s: str) -> str:
    return re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', s)

def find_words_with_prefix(s: str, prefix: str) -> list:
    return [word for word in re.findall(r'\w+', s) if word.startswith(prefix)]

def find_words_with_suffix(s: str, suffix: str) -> list:
    return [word for word in re.findall(r'\w+', s) if word.endswith(suffix)]

def enumerate_words(s: str) -> list:
    return list(enumerate(re.findall(r'\w+', s)))

def get_middle_char(s: str) -> str:
    l = len(s)
    if l == 0:
        return ''
    mid = l // 2
    return s[mid] if l % 2 else s[mid-1:mid+1]

# ===========================
# Функции для кодировок и BOM
# ===========================

BOM_SIGNATURES = {
    'utf-8-sig': b'\xef\xbb\xbf',
    'utf-16-le': b'\xff\xfe',
    'utf-16-be': b'\xfe\xff',
    'utf-32-le': b'\xff\xfe\x00\x00',
    'utf-32-be': b'\x00\x00\xfe\xff',
}

def encode_string(s: str, encoding: str = 'utf-8', errors: str = 'strict') -> bytes:
    return s.encode(encoding, errors=errors)

def decode_bytes(b: bytes, encoding: str = 'utf-8', errors: str = 'strict') -> str:
    return b.decode(encoding, errors=errors)

def try_decode_bytes(b: bytes, encodings: list = None) -> tuple:
    if encodings is None:
        encodings = [
            'utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be',
            'cp1251', 'cp1252', 'cp866', 'koi8-r', 'koi8-u', 'iso8859-1', 'iso8859-5', 'latin1', 'mac_cyrillic'
        ]
    for enc in encodings:
        try:
            return b.decode(enc), enc
        except Exception:
            continue
    return None, None

def detect_encoding(b: bytes, default: str = 'utf-8'):
    try:
        import chardet
    except ImportError:
        raise ImportError("Для функции detect_encoding требуется пакет chardet")
    result = chardet.detect(b)
    encoding = result['encoding'] or default
    return encoding, result['confidence']

def is_valid_encoding(b: bytes, encoding: str) -> bool:
    try:
        b.decode(encoding)
        return True
    except Exception:
        return False

def recode_string(s: str, from_enc: str, to_enc: str, errors: str = 'strict') -> str:
    return s.encode(from_enc).decode(to_enc, errors=errors)

def fix_mojibake(s: str, wrong_enc: str, right_enc: str) -> str:
    return s.encode(wrong_enc).decode(right_enc)

def get_string_encoding_info(s: str) -> dict:
    encodings = [
        'utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be',
        'cp1251', 'cp1252', 'cp866', 'koi8-r', 'koi8-u', 'iso8859-1', 'iso8859-5', 'latin1', 'mac_cyrillic'
    ]
    info = {}
    for enc in encodings:
        try:
            b = s.encode(enc)
            info[enc] = {'bytes': b, 'hex': b.hex()}
        except Exception as e:
            info[enc] = {'error': str(e)}
    return info

def print_encodings_table(s: str):
    info = get_string_encoding_info(s)
    print(f"{'Encoding':<12} {'Bytes':<30} {'Hex'}")
    for enc, val in info.items():
        if 'bytes' in val:
            print(f"{enc:<12} {str(val['bytes']):<30} {val['hex']}")
        else:
            print(f"{enc:<12} {'ERROR':<30} {val['error']}")

def has_bom(b: bytes) -> str:
    for enc, sig in BOM_SIGNATURES.items():
        if b.startswith(sig):
            return enc
    return None

def strip_bom(b: bytes) -> bytes:
    for sig in BOM_SIGNATURES.values():
        if b.startswith(sig):
            return b[len(sig):]
    return b

def add_bom(b: bytes, encoding: str) -> bytes:
    sig = BOM_SIGNATURES.get(encoding)
    if sig and not b.startswith(sig):
        return sig + b
    return b

COMMON_ENCODINGS = [
    'utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be',
    'utf-32', 'utf-32-le', 'utf-32-be', 'ascii',
    'cp1251', 'cp1252', 'cp866', 'koi8-r', 'koi8-u', 'iso8859-1', 'iso8859-5',
    'latin1', 'mac_cyrillic', 'mac_roman', 'big5', 'gb2312', 'shift_jis', 'euc-jp'
]

def batch_recode_files(
    file_paths: list,
    src_encoding: str,
    dst_encoding: str,
    remove_bom: bool = False,
    add_bom_to: str = None,
    errors: str = 'strict'
) -> dict:
    results = {}
    for path in file_paths:
        try:
            with open(path, 'rb') as f:
                data = f.read()
            if remove_bom:
                data = strip_bom(data)
            text = data.decode(src_encoding, errors=errors)
            out_bytes = text.encode(dst_encoding, errors=errors)
            if add_bom_to:
                out_bytes = add_bom(out_bytes, add_bom_to)
            with open(path, 'wb') as f:
                f.write(out_bytes)
            results[path] = (True, "OK")
        except Exception as e:
            results[path] = (False, str(e))
    return results

def universal_recode(
    b: bytes,
    dst_encoding: str,
    encodings_to_try: list = None,
    remove_bom: bool = True,
    add_bom_to: str = None,
    errors: str = 'strict'
) -> bytes:
    if encodings_to_try is None:
        encodings_to_try = COMMON_ENCODINGS
    if remove_bom:
        b = strip_bom(b)
    try:
        enc, conf = detect_encoding(b)
        text = b.decode(enc, errors=errors)
    except Exception:
        for enc in encodings_to_try:
            try:
                text = b.decode(enc, errors=errors)
                break
            except Exception:
                continue
        else:
            raise UnicodeDecodeError("Не удалось определить кодировку")
    out_bytes = text.encode(dst_encoding, errors=errors)
    if add_bom_to:
        out_bytes = add_bom(out_bytes, add_bom_to)
    return out_bytes

# ===========================
# Ещё больше функций (разное)
# ===========================

def remove_html_entities(s: str) -> str:
    import html
    return html.unescape(s)

def count_paragraphs(s: str) -> int:
    return len([p for p in s.split('\n\n') if p.strip()])

def get_paragraphs(s: str) -> list:
    return [p.strip() for p in s.split('\n\n') if p.strip()]

def remove_extra_newlines(s: str) -> str:
    return re.sub(r'\n{2,}', '\n\n', s.strip())

def only_ascii_letters(s: str) -> str:
    return ''.join(c for c in s if c in string.ascii_letters)

def only_cyrillic_letters(s: str) -> str:
    return ''.join(c for c in s if re.match(r'[а-яА-ЯёЁ]', c))

def is_palindrome_sentence(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def reverse_sentences(s: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', s.strip())
    return ' '.join(sentences[::-1])

def find_words_with_digits(s: str) -> list:
    return [word for word in re.findall(r'\w+', s) if any(c.isdigit() for c in word)]

def find_words_without_digits(s: str) -> list:
    return [word for word in re.findall(r'\w+', s) if not any(c.isdigit() for c in word)]

def find_words_with_hyphen(s: str) -> list:
    return re.findall(r'\w+-\w+', s)

def extract_parentheses_content(s: str) -> list:
    return re.findall(r'\((.*?)\)', s)

def extract_square_brackets_content(s: str) -> list:
    return re.findall(r'\[(.*?)\]', s)

def extract_curly_braces_content(s: str) -> list:
    return re.findall(r'\{(.*?)\}', s)

def remove_non_bmp(s: str) -> str:
    return ''.join(c for c in s if ord(c) <= 0xFFFF)

def contains_emoji(s: str) -> bool:
    return bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', s))

def extract_emoji(s: str) -> list:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]", flags=re.UNICODE)
    return emoji_pattern.findall(s)

def remove_emoji(s: str) -> str:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', s)

def is_valid_ipv4(s: str) -> bool:
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, s):
        return False
    parts = s.split('.')
    return all(0 <= int(part) <= 255 for part in parts)

def extract_ipv4(s: str) -> list:
    pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    return [ip for ip in re.findall(pattern, s) if is_valid_ipv4(ip)]

def is_valid_hex_color(s: str) -> bool:
    return bool(re.fullmatch(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})', s))

def extract_hex_colors(s: str) -> list:
    return re.findall(r'#[A-Fa-f0-9]{3,6}\b', s)

def is_valid_uuid(s: str) -> bool:
    return bool(re.fullmatch(r'[a-fA-F0-9]{8}-([a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}', s))

def extract_uuids(s: str) -> list:
    return re.findall(r'[a-fA-F0-9]{8}-([a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}', s)

def remove_tabs(s: str) -> str:
    return s.replace('\t', '')

def tabs_to_spaces(s: str, spaces: int = 4) -> str:
    return s.replace('\t', ' ' * spaces)

def spaces_to_tabs(s: str, spaces: int = 4) -> str:
    return re.sub(' ' * spaces, '\t', s)

def is_mixed_case(s: str) -> bool:
    return any(c.islower() for c in s) and any(c.isupper() for c in s)

def count_punctuation(s: str) -> int:
    return sum(1 for c in s if c in string.punctuation)

def remove_quotes(s: str) -> str:
    return s.replace('"', '').replace("'", '')

def add_quotes(s: str, quote_char: str = '"') -> str:
    return f"{quote_char}{s}{quote_char}"

def reverse_each_word(s: str) -> str:
    return ' '.join(word[::-1] for word in s.split())

def is_blank(s: str) -> bool:
    return not s.strip()
