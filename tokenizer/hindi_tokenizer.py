import string
import unicodedata
import re


HINDI_SPLIT_PATTERN = (
    r'[' + re.escape(string.punctuation) + r'\u0964\u0965\uAAF1\uAAF0\uABEB\uABEC\uABED\uABEE\uABEF\u1C7E\u1C7F]+'  # Punctuation and specific Unicode points
    r'|[\u0900-\u097F\u200C\u200D]+'  # Hindi script characters including matras and combining characters
    r'|\s+'  # Whitespace
    r'|\d+'  # Digits
    r'|[a-zA-Z]+'  # English alphabets
)

def normalize_text(text):
   # combine क and ि into कि rather than separate entities.
   return unicodedata.normalize('NFC', text)