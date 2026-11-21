# Concept: encode text as its UTF-8 byte sequence rather than character code points.
# Goal: implement a reversible byte-level tokenizer.
# UTF-8 represents non-ASCII characters using multiple byte IDs.
# This prepares the byte sequences that BPE can later merge.
# A byte is an 8-bit value from 0 to 255; unlike a Unicode code point, it is always in that range.
# Example: encode the ASCII character 'A' as UTF-8.
# str.encode() is a Python string method: input str -> output bytes; ASCII 'A' needs one byte.
encoded = 'A'.encode("utf-8")
print(encoded)        # b'A'
# bytes is iterable, yielding integer byte values: 65 is 0b01000001, the byte for ASCII A.
print(list(encoded))  # [65]

# Example: one Japanese hiragana character needs three UTF-8 bytes; the sample data stays unchanged.
encoded = 'あ'.encode("utf-8")
print(encoded)        # b'\xe3\x81\x82'
print(list(encoded))  # [227, 129, 130]

# The bytes() built-in converts [65] to b'A'; bytes.decode("utf-8") returns a Python str.
ids = [65]
decoded = bytes(ids).decode("utf-8")
print(decoded)   # 'A'


# ByteTokenizer uses raw UTF-8 bytes as token IDs, giving a compact universal base vocabulary.
class ByteTokenizer:
    def encode(self, text):
        # Input str -> UTF-8 bytes via str.encode() -> list[int] via the bytes iterator.
        return list(text.encode("utf-8"))

    def decode(self, ids):
        # bytes() converts integer values 0..255 to bytes; bytes.decode() returns a Python str.
        # The whole sequence must be valid UTF-8 for the default strict decoder to succeed.
        return bytes(ids).decode("utf-8")


# Example usage: instantiate the tokenizer and round-trip a string.
tokenizer = ByteTokenizer()
text = "hello世界😁"

# Encode converts the string into a list of UTF-8 byte values.
ids = tokenizer.encode(text)
print(ids)  # [104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]

# Decode rebuilds the UTF-8 byte sequence and converts it back into text.
decoded = tokenizer.decode(ids)
print(decoded)  # Restores "hello" plus the Japanese word for "world" and the emoji.

# ===== CONCEPT NOTES =====
# CONCEPT: ByteTokenizer converts Unicode text to UTF-8 and uses each byte value (0–255) as a token ID.
# GOAL: Show the byte-level alternative to the prior character tokenizer; later BPE lessons learn to merge frequent neighboring byte IDs.
# INPUT: Examples encode A, one Japanese hiragana character, and text containing the Japanese word for "world" plus an emoji. decode takes byte IDs.
# OUTPUT: A becomes b'A' and [65]; the hiragana example becomes b'\xe3\x81\x82' and [227, 129, 130]. The longer example prints [104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129] and decodes to its original text.
# HOW INPUT BECOMES OUTPUT:
# 1. encode calls text.encode('utf-8'), turning each character into one or more bytes.
# 2. list(...) exposes each byte as an integer; the hiragana example gives 227, 129, 130.
# 3. decode builds bytes(ids) from those integers and decodes the combined byte sequence as UTF-8.
# NOTES: Every byte ID is in 0..255, but decoding requires a complete valid UTF-8 sequence; decoding only part of a multibyte character can raise UnicodeDecodeError. The script writes no files.
