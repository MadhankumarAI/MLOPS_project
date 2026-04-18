from deep_translator import MyMemoryTranslator
import traceback

try:
    text = "Hello world"
    source_lang = "english"
    target_lang = "hindi"
    print(f"Translating '{text}' from '{source_lang}' to '{target_lang}' using MyMemory...")
    translated = MyMemoryTranslator(source=source_lang, target=target_lang).translate(text)
    print(f"Result: {translated}")
except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()
