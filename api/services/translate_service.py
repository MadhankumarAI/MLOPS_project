from api.services.ner_service import NERService
from api.services.translation.factory import get_translator


class TranslateService:
    def __init__(self, ner: NERService, backend: str = "google"):
        self.ner = ner
        self.translator = get_translator(backend)

    def translate(self, text: str, target_lang: str) -> dict:
        tokens, tags = self.ner.predict(text)
        entities = self.ner.extract_entities(tokens, tags)

        masked_text, placeholders = self._mask_entities(text, entities)
        translated = self.translator.translate(masked_text, target_lang)
        final_text = self._restore_entities(translated, placeholders)

        return {
            "source_text": text,
            "translated_text": final_text,
            "entities": entities,
            "target_lang": target_lang,
        }

    def _mask_entities(self, text: str, entities: list[dict]) -> tuple[str, dict]:
        placeholders = {}
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        for ent in sorted_entities:
            placeholder = f"__ENT{len(placeholders)}__"
            placeholders[placeholder] = ent["text"]
            text = text[:ent["start"]] + placeholder + text[ent["end"]:]

        return text, placeholders

    def _restore_entities(self, text: str, placeholders: dict) -> str:
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)
        return text
