import sys
sys.stdout.reconfigure(encoding='utf-8')
from api.services.ner_service import NERService
from api.services.translate_service import TranslateService

ner = NERService()
t = TranslateService(ner)
result = t.translate('Google and Microsoft are investing in India.', 'hi')
print(result['translated_text'])
