from api.services.ner_service import NERService
from api.services.translate_service import TranslateService

# Initialize services
ner_svc = NERService()
translate_svc = TranslateService(ner=ner_svc)

text = "In 2022, Sundar Pichai of California returned to Bengaluru."
target_lang = "hi"

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Specifically test the cleanup function directly
from api.services.translation.transliterate_util import clean_hindi_artifacts
dirty_text = "2022 में, Cचलिफ़ोर्निअ के Sसुन्दर् पिचै Bबेन्गलुरु में लौटे MIमिच्रोसोफ़्त् Saसत्य"
cleaned = clean_hindi_artifacts(dirty_text)
print(f"Dirty: {dirty_text}")
print(f"Cleaned: {cleaned}")

# Check if it handles the user's specific case
case = "Miमिच्रोसोफ़्त्"
print(f"Testing '{case}' -> '{clean_hindi_artifacts(case)}'")
