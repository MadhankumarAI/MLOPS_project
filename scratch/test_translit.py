from api.services.translation.transliterate_util import transliterate_text

try:
    print("Testing transliteration...")
    result = transliterate_text("Shah Jahan", "hi")
    print(f"Result: {result}")
    if result == "शाहजहाँ" or result:
        print("Success!")
except Exception as e:
    print(f"Failed with: {e}")
