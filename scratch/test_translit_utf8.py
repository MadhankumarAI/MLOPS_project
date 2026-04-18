import json
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from api.services.translation.transliterate_util import transliterate_text

words = ['Mumtaz', 'Mahal', 'Taj', 'Shah', 'Jahan', 'm', 'u', 'M', 'umtaz']
results = {w: transliterate_text(w, 'hi') for w in words}

with open('scratch/translit_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results written to scratch/translit_results.json")
