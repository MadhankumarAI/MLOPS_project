
import re

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)

def extract_entities_robust(tokens, tags, text=None):
    entities = []
    current_entity = None
    last_found_pos = 0

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if text:
            start = text.find(token, last_found_pos)
            if start == -1:
                start = last_found_pos
            end = start + len(token)
            last_found_pos = end
        else:
            start = last_found_pos
            end = start + len(token)
            last_found_pos = end + 1

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": token, "start": start, "end": end}
        elif tag.startswith("I-") and current_entity:
            if text:
                gap = text[current_entity["end"]:start]
                current_entity["text"] += gap + token
            else:
                current_entity["text"] += " " + token
            current_entity["end"] = end
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    return entities

import sys
# Set stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

text = "2022 में, Sundar Pichai"
# Index of 'S' in "2022 में, Sundar Pichai"
# 2022 (4) + ' ' (1) + में (3) + ',' (1) + ' ' (1) = 10
# Wait, let's check exact string index
s_index = text.find('Sundar')
print(f"Index of 'S': {s_index}")

tokens = _tokenize(text)
print(f"Tokens: {tokens}")
# Assume Sundar Pichai is identified
tags = ["O", "O", "O", "O", "B-PER", "I-PER"]
# Wait, let's adjust tags based on actual tokens
# 2022 (0), मे (1), ं (2), , (3), Sundar (4), Pichai (5)
if len(tokens) > 5:
    tags = ["O"] * (len(tokens) - 2) + ["B-PER", "I-PER"]
else:
    tags = ["O", "O", "O", "B-PER", "I-PER"]

entities = extract_entities_robust(tokens, tags, text=text)
print("\n--- Testing clean_hindi_artifacts ---")
import re

def clean_hindi_artifacts_refined(text: str) -> str:
    if not re.search(r'[\u0900-\u097F]', text):
        return text
    def replace_word(match):
        word = match.group(0)
        if re.search(r'[\u0900-\u097F]', word):
            cleaned = re.sub(r'[A-Za-z]', '', word)
            return cleaned if cleaned else word
        return word
    return re.sub(r'\S+', replace_word, text)

# Test with various artifacts
test_cases = [
    ("Suसुन्दर पिचै", "सुन्दर पिचै"),
    ("Caचलिफ़ोर्निअo", "चलिफ़ोर्निअ"),
    ("Beबेन्गलुरुo", "बेन्गलुरु"),
    ("Normal Hindi", "Normal Hindi"), # Should not touch "Normal" or "Hindi"
    ("Mixing 2022 में", "Mixing 2022 में"),
    ("Suसुन्दर  Pichai", "सुन्दर  Pichai") # Preserve double spaces
]

for inp, expected in test_cases:
    result = clean_hindi_artifacts_refined(inp)
    status = "PASS" if result == expected else f"FAIL (got '{result}')"
    print(f"'{inp}' -> '{result}' : {status}")
