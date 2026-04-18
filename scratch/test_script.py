from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

text1 = "Sundar Pichai"
text2 = "California"
text3 = "Microsoft"

for text in [text1, text2, text3]:
    print(f"--- {text} ---")
    print("ITRANS:", transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI))
    print("ITRANS lower:", transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI))
    print("HK lower:", transliterate(text.lower(), sanscript.HK, sanscript.DEVANAGARI))
    print("OPTITRANS lower:", transliterate(text.lower(), sanscript.OPTITRANS, sanscript.DEVANAGARI))
