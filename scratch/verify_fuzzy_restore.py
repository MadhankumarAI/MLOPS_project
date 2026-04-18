import sys
sys.stdout.reconfigure(encoding='utf-8')
import re

# Mock transliterate_text
def transliterate_text(text, lang):
    return text.upper() # Mock: uppercase as Hindi

def _restore_entities(text: str, placeholders: dict, target_lang: str, transliterate: bool) -> str:
    def replace_match(match):
        # Extract the ID from the mangled placeholder (e.g., "__ ENT 0 __" -> 0)
        id_match = re.search(r"\d+", match.group(0))
        if not id_match:
            return match.group(0)
            
        ent_id = id_match.group()
        placeholder_key = f"__ENT{ent_id}__"
        
        original = placeholders.get(placeholder_key)
        if not original:
            return match.group(0)
            
        replacement = original
        if transliterate:
            replacement = transliterate_text(original, target_lang)
        return replacement

    # Regex to find markers like __ENT0__, __ ENT 0 __, __ent 0__, etc.
    pattern = r"__\s*ENT\s*\d+\s*__"
    return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

# Test data
placeholders = {
    "__ENT0__": "Sundar Pichai",
    "__ENT1__": "California",
    "__ENT9__": "Turing Tag"
}

mangled_responses = [
    "2022 में, __ENT0__ का __ENT1__ ...", # Clean
    "2022 में, __ ENT0 __ का __ ENT1 __ ...", # Spaces
    "2022 में, __ ENT 9 __ ने घोषणा की", # Space inside number? No, my regex is digit+, but let's see.
    "__ent0__ is here", # Lowercase
]

for resp in mangled_responses:
    restored = _restore_entities(resp, placeholders, "hi", True)
    print(f"Original: {resp}")
    print(f"Restored: {restored}")
    print("-" * 20)
