"""The Arabic finger alphabet (unified Arabic Sign Language) and how each dataset names its classes.

The mappings were checked on 2026-11-15: ArSL2018 and ASLAD-190K name the hand shapes the same way;
ArSL21L swaps ta/taa and ya/yaa (confirmed by training on one dataset and predicting the other, and
against the KArSL letter videos). ArSL21L's waw is the KArSL form of the letter, ArSL2018/ASLAD
show a variant: both are kept as the same letter.
"""

ARASL2018_ASLAD = {
    "aleff": "ا", "bb": "ب", "taa": "ت", "thaa": "ث", "jeem": "ج", "haa": "ح", "khaa": "خ",
    "dal": "د", "thal": "ذ", "ra": "ر", "zay": "ز", "seen": "س", "sheen": "ش", "saad": "ص",
    "dhad": "ض", "ta": "ط", "dha": "ظ", "ain": "ع", "ghain": "غ", "fa": "ف", "gaaf": "ق",
    "kaaf": "ك", "laam": "ل", "meem": "م", "nun": "ن", "ha": "ه", "waw": "و", "ya": "ي",
    "toot": "ة", "yaa": "ئ", "la": "لا", "al": "ال",
}

ARSL21L = {**ARASL2018_ASLAD, "ta": "ت", "taa": "ط", "ya": "ئ", "yaa": "ي"}

# KArSL sign ids 32-70 (1-31 are numbers)
KARSL_SIGNS = dict(zip(range(32, 71), "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي ة أ ؤ ئ ئـ ء إ آ ى لا ال".split()))

# The 32 hand shapes shared by all datasets: what the recognizer learns.
STATIC_LETTERS = sorted(set(ARASL2018_ASLAD.values()))

# KArSL frames not used for training: its ة is the ت hand shape plus a movement, and its ئ signs
# were ambiguous in the check. The photo datasets cover both letters.
KARSL_EXCLUDED = {"ة", "ئ"}
