from api.services.translation.transliterate_util import transliterate_text

words = ["Mumtaz", "Mahal", "Taj", "Shah", "Jahan"]
for w in words:
    print(f"{w} -> {transliterate_text(w, 'hi')}")
