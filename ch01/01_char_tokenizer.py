# Concept: represent each Unicode character by its integer code point.
# Goal: show a minimal reversible character-level tokenizer.
# The examples include ASCII, Japanese characters, and an emoji.
# This is a simple baseline before moving to byte-level tokenization.
# Python str iterates over Unicode code points, not UTF-8 bytes; an emoji is one
# character here even though its UTF-8 representation occupies several bytes.
text = "hello世界😁"
# list() is a Python built-in that turns an iterable str into a list of characters.
# list(text) keeps each Unicode character as one list item, including multibyte characters.
print(list(text))  # h, e, l, l, o, the two Japanese characters for "world", and one emoji.

# ord() is a Python built-in: one-character str -> integer Unicode code point.
# For example, 'h' -> 104; the emoji's code point 128513 is larger than one byte can hold.
print(ord('h'))  # 104
print(ord('😁'))  # 128513

# chr() is the inverse built-in: integer Unicode code point -> one-character str.
# For example, 104 -> 'h'; 128513 -> the same emoji.
print(chr(104))    # 'h'
print(chr(128513)) # '😁'

ids = [ord(char) for char in list(text)]
print(ids)  # [104, 101, 108, 108, 111, 19990, 30028, 128513]

# A tokenizer converts readable text to integer token IDs for a model, then reverses it.
# This simple tokenizer uses each Python character's Unicode code point as its ID.
class CharTokenizer:
    def encode(self, text):
        # Input: Python str; output: list[int], one code point per character.
        # The list comprehension applies ord() to each character without changing its order.
        return [ord(char) for char in text]

    def decode(self, ids):
        # Input: an iterable of integer code points; output: one Python str.
        # chr() restores each character; str.join combines them with no separator.
        return ''.join([chr(i) for i in ids])

tokenizer = CharTokenizer()
text = "hello世界😁"

# Encode means convert the input string into integer character IDs.
ids = tokenizer.encode(text)
print(ids)  # [104, 101, 108, 108, 111, 19990, 30028, 128513]

# Decode means turn those IDs back into a readable string.
decoded = tokenizer.decode(ids)
print(decoded)  # Restores "hello" plus the Japanese word for "world" and the emoji.

# ===== CONCEPT NOTES =====
# CONCEPT: Character-level tokenization maps each Python character to its Unicode code point with ord(), and restores it with chr().
# GOAL: Demonstrate a reversible tokenizer for arbitrary Unicode text; the next lesson changes the unit from characters to UTF-8 bytes.
# INPUT: The example is "hello" plus the Japanese word for "world" and a grinning emoji. encode takes str; decode takes integer IDs.
# OUTPUT: Code points are [104, 101, 108, 108, 111, 19990, 30028, 128513]; decode reconstructs the original text. ord('h') is 104 and ord('😁') is 128513.
# HOW INPUT BECOMES OUTPUT:
# 1. list(text) produces eight characters: h, e, l, l, o, two characters spelling "world" in Japanese, and the emoji.
# 2. encode walks the text and applies ord() to each character, creating the integer IDs.
# 3. decode applies chr() to each ID and joins the characters into one string.
# NOTES: These IDs are Unicode code points, not UTF-8 byte values; one character may occupy multiple UTF-8 bytes. The script only prints examples and writes no files.
