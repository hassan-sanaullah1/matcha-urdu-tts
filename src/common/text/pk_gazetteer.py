"""Pakistani names, places, institutions and companies written in Latin -> Urdu script.

Why: inside English sentences the urdu-phones front end routes a Latin token through the English
dictionary first, so "Lahore" was read by espeak as English, and unknown names went to a rule
reader that mis-handles "qu" (Quetta -> kʋeːttaː, Siddiqui -> sɪddɪkʋiː). A Pakistani bilingual
speaker says these as Urdu words. This table is applied BEFORE any English routing; the model then
reads them with its trained Urdu grapheme rows.

Entries that are also ordinary English words used as English (Jazz, Lucky, Fast, Pearl, Continental)
are deliberately absent; brand names spoken as English stay English.

prepare_pk(text) = gazetteer substitution + frontend.prepare (frontend.py, from urdu-phones).
Usage: python src/pk_gazetteer.py --test      (exact-output tests)
       python src/pk_gazetteer.py "The flight from Peshawar to Quetta"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import functools
import re
import unicodedata

# multi-word and hyphenated forms first (longest match wins)
PHRASES = {
    "quaid-i-azam": "قائداعظم", "quaid-e-azam": "قائداعظم", "gilgit-baltistan": "گلگت بلتستان",
    "khyber pakhtunkhwa": "خیبر پختونخوا", "azad kashmir": "آزاد کشمیر", "azad jammu and kashmir": "آزاد جموں و کشمیر",
    "data ganj bakhsh": "داتا گنج بخش", "shaukat khanum": "شوکت خانم", "allama iqbal": "علامہ اقبال",
    "liaquat ali khan": "لیاقت علی خان", "dera ghazi khan": "ڈیرہ غازی خان", "dera ismail khan": "ڈیرہ اسماعیل خان",
    "rahim yar khan": "رحیم یار خان", "mandi bahauddin": "منڈی بہاؤالدین", "toba tek singh": "ٹوبہ ٹیک سنگھ",
    "aga khan": "آغا خان", "sui northern": "سوئی ناردرن", "sui southern": "سوئی سدرن",
    "zarghun south": "زرغون ساؤتھ", "mari energies": "ماڑی انرجیز",
    "noor-e-sehar": "نورِ سحر", "reko diq": "ریکوڈک", "al baraka": "البرکہ", "sehat umeed": "صحت امید",
    "roshan mustaqbil": "روشن مستقبل", "kissan dost": "کسان دوست", "rani kot": "رانی کوٹ",
}

WORDS = {
    # languages and demonyms (were read by English rules: Pakistani -> "pack-i-stani", Urdu -> "or-doo")
    "pakistani": "پاکستانی", "pakistanis": "پاکستانی", "urdu": "اردو", "punjabi": "پنجابی", "sindhi": "سندھی",
    "baloch": "بلوچ", "balochi": "بلوچی", "pashto": "پشتو", "pushto": "پشتو", "pashtun": "پشتون",
    "pakhtun": "پختون", "saraiki": "سرائیکی", "seraiki": "سرائیکی", "kashmiri": "کشمیری", "hindko": "ہندکو",
    "brahui": "براہوی", "lahori": "لاہوری", "karachiite": "کراچی والے", "karachiites": "کراچی والے",
    # provinces, regions
    "pakistan": "پاکستان", "punjab": "پنجاب", "sindh": "سندھ", "balochistan": "بلوچستان",
    "baluchistan": "بلوچستان", "pakhtunkhwa": "پختونخوا", "khyber": "خیبر", "kashmir": "کشمیر",
    "waziristan": "وزیرستان", "baltistan": "بلتستان", "gilgit": "گلگت", "hunza": "ہنزہ", "swat": "سوات",
    "chitral": "چترال", "skardu": "سکردو", "thar": "تھر", "cholistan": "چولستان", "potohar": "پوٹھوہار",
    "fata": "فاٹا", "kpk": "کے پی کے",
    # cities, towns
    "karachi": "کراچی", "lahore": "لاہور", "islamabad": "اسلام آباد", "rawalpindi": "راولپنڈی",
    "pindi": "پنڈی", "peshawar": "پشاور", "quetta": "کوئٹہ", "multan": "ملتان", "faisalabad": "فیصل آباد",
    "hyderabad": "حیدرآباد", "sialkot": "سیالکوٹ", "gujranwala": "گوجرانوالہ", "gujrat": "گجرات",
    "sukkur": "سکھر", "larkana": "لاڑکانہ", "abbottabad": "ایبٹ آباد", "murree": "مری", "mardan": "مردان",
    "bahawalpur": "بہاولپور", "sargodha": "سرگودھا", "sahiwal": "ساہیوال", "jhelum": "جہلم",
    "mirpur": "میرپور", "muzaffarabad": "مظفرآباد", "gwadar": "گوادر", "turbat": "تربت", "kohat": "کوہاٹ",
    "nowshera": "نوشہرہ", "okara": "اوکاڑہ", "kasur": "قصور", "sheikhupura": "شیخوپورہ",
    "chakwal": "چکوال", "attock": "اٹک", "nawabshah": "نوابشاہ", "thatta": "ٹھٹھہ", "badin": "بدین",
    "ghotki": "گھوٹکی", "daharki": "ڈہرکی", "zhob": "ژوب", "sibi": "سبی",
    "khuzdar": "خضدار", "karak": "کرک", "hangu": "ہنگو", "dasu": "داسو", "tarbela": "تربیلا",
    "mangla": "منگلا", "karakoram": "قراقرم", "gulberg": "گلبرگ",
    # people: common Pakistani given names and surnames
    "muhammad": "محمد", "mohammad": "محمد", "mohammed": "محمد", "ahmed": "احمد", "ahmad": "احمد",
    "ali": "علی", "hassan": "حسن", "hasan": "حسن", "hussain": "حسین", "husain": "حسین",
    "usman": "عثمان", "umar": "عمر", "omar": "عمر", "bilal": "بلال", "hamza": "حمزہ", "zain": "زین",
    "imran": "عمران", "kamran": "کامران", "farhan": "فرحان", "adnan": "عدنان", "asad": "اسد",
    "tariq": "طارق", "tahir": "طاہر", "shahid": "شاہد", "khalid": "خالد", "waqas": "وقاص",
    "fawad": "فواد", "junaid": "جنید", "saad": "سعد", "saqib": "ثاقب", "shoaib": "شعیب",
    "fatima": "فاطمہ", "ayesha": "عائشہ", "aisha": "عائشہ", "zainab": "زینب", "maryam": "مریم",
    "sana": "ثنا", "saima": "صائمہ", "rabia": "رابعہ", "nadia": "نادیہ", "hina": "حنا",
    "bushra": "بشریٰ", "mehwish": "مہوش", "sadia": "سعدیہ", "amna": "آمنہ", "sara": "سارہ",
    "mahnoor": "ماہ نور", "iqra": "اقرا", "kausar": "کوثر", "humaira": "حمیرا", "tayyaba": "طیبہ",
    "qudsia": "قدسیہ", "talat": "طلعت", "surriya": "ثریا", "anam": "انعم", "mashal": "مشعل",
    "siddiqui": "صدیقی", "qureshi": "قریشی", "chaudhry": "چودھری", "chaudhary": "چودھری",
    "malik": "ملک", "butt": "بٹ", "awan": "اعوان", "rana": "رانا", "raja": "راجہ", "mirza": "مرزا",
    "hashmi": "ہاشمی", "rizvi": "رضوی", "naqvi": "نقوی", "zaidi": "زیدی", "jafri": "جعفری",
    "abbasi": "عباسی", "bhatti": "بھٹی", "gondal": "گوندل", "baloch": "بلوچ", "khattak": "خٹک",
    "yousafzai": "یوسفزئی", "afridi": "آفریدی", "durrani": "درانی", "memon": "میمن", "soomro": "سومرو",
    "bhutto": "بھٹو", "zardari": "زرداری", "sharif": "شریف", "jinnah": "جناح", "iqbal": "اقبال",
    "liaquat": "لیاقت", "edhi": "ایدھی", "hayat": "حیات", "tamim": "تمیم", "shahed": "شاہد",
    "afza": "افزا", "suhail": "سہیل", "asghar": "اصغر", "kalsoom": "کلثوم", "qadir": "قادر",
    "khan": "خان", "syed": "سید", "sheikh": "شیخ", "hazrat": "حضرت", "allama": "علامہ",
    # knowledge-base people and places the English rules got wrong (Nabeel -> "neh-beel", Ashraf -> "esh-ref")
    "nabeel": "نبیل", "rasheed": "رشید", "ashraf": "اشرف", "hyder": "حیدر", "abid": "عابد",
    "bakhtiyar": "بختیار", "seema": "سیما", "adil": "عادل", "aamir": "عامر", "shahzad": "شہزاد",
    "nabi": "نبی", "lak": "لک", "cantt": "کینٹ",
    "mehmood": "محمود", "mushtaq": "مشتاق", "imtiaz": "امتیاز", "janjua": "جنجوعہ", "ghulam": "غلام",
    "hamid": "حامد", "asif": "آصف", "aligarh": "علی گڑھ", "mach": "مچھ", "kehkashan": "کہکشاں",
    "makran": "مکران", "maiwand": "میوند", "hilal": "ہلال", "shawal": "شوال", "kawagarh": "کواگڑھ",
    "dubai": "دبئی", "gharonda": "گھروندا", "dastarkhawn": "دسترخوان", "dastarkhwan": "دسترخوان",
    "azadi": "آزادی", "ezshifa": "ایزی شفا", "sehat": "صحت", "umeed": "امید",
    "quaid": "قائد", "azam": "اعظم", "sahib": "صاحب", "begum": "بیگم", "bibi": "بی بی",
    # institutions and companies spoken as Urdu/local words
    "habib": "حبیب", "meezan": "میزان", "faysal": "فیصل", "alfalah": "الفلاح", "askari": "عسکری",
    "engro": "اینگرو", "mari": "ماڑی", "sui": "سوئی", "zarghun": "زرغون",
    "shewa": "شیوا", "wapda": "واپڈا", "nadra": "نادرا", "nepra": "نیپرا", "ogra": "اوگرا",
    "parco": "پارکو", "suparco": "سپارکو", "lums": "لمز", "nust": "نسٹ", "comsats": "کامسیٹس",
    "khanum": "خانم", "taleemabad": "تعلیم آباد", "maqsad": "مقصد", "sabaq": "سبق", "daraz": "دراز",
    "careem": "کریم", "easypaisa": "ایزی پیسہ", "jazzcash": "جیز کیش", "ufone": "یوفون",
    "zong": "زونگ", "ptcl": "پی ٹی سی ایل", "ptv": "پی ٹی وی", "isi": "آئی ایس آئی",
    "inshallah": "ان شاء اللہ", "insha'allah": "ان شاء اللہ", "mashallah": "ماشاءاللہ",
    "alhamdulillah": "الحمدللہ", "assalam": "السلام", "salaam": "سلام", "shukriya": "شکریہ",
    # Islamic and everyday Pakistani terms an LLM writes in Latin
    "eid": "عید", "eid ul fitr": "عید الفطر", "eid ul adha": "عید الاضحیٰ", "ramzan": "رمضان",
    "ramadan": "رمضان", "zakat": "زکوٰۃ", "iftar": "افطار", "sehri": "سحری", "suhoor": "سحری",
    "jummah": "جمعہ", "hajj": "حج", "umrah": "عمرہ", "halal": "حلال", "haram": "حرام",
    "jazzcash": "جیز کیش", "jazz cash": "جیز کیش", "easypaisa": "ایزی پیسہ", "raast": "راست",
    "sadapay": "سادہ پے", "nayapay": "نیا پے",
}

TABLE = {**PHRASES, **WORDS}
_RX = re.compile(r"(?<![A-Za-z'])(" + "|".join(re.escape(k) for k in sorted(TABLE, key=len, reverse=True))
                 + r")(?![A-Za-z])", re.I)


_ORDINAL = re.compile(r"\b(\d{1,4})(st|nd|rd|th)\b", re.I)


# The mirror of TABLE: an Urdu spelling whose sounds are hidden -> the Latin key the English
# router already says correctly. Urdu does not write the short vowel of "-man", so چیئرمین was
# read "chairmeen"; Latin "Chairman" already gives ʧˈeːrmən. Same mechanism as text_norm.PK_PHONES
# (سجاول -> "Sajawal"), and the place to add any other word that is heard wrong.
# A borrowed English suffix is written joined to the stem, so the plain-word rule below could not
# see it: چیئرمینسی was left whole and its سی came out as a stray "see" before the next word.
_UR_EXACT_TO = {"چیئرمین": "Chairman", "چیئر مین": "Chairman", "چیئرپرسن": "Chairperson",
                "چیئرمینسی": "chairmancy", "چیئرمین سی": "chairmancy",
                "چیئرمینشپ": "chairmanship", "چیئرمین شپ": "chairmanship",
                "چیئرمینز": "Chairmans", "چیئرپرسنز": "Chairpersons"}
# Bounded by non-Urdu on both sides, so a match inside a longer Urdu word is not rewritten.
_UR_EXACT = re.compile(r"(?<![؀-ۿ])(" + "|".join(
    sorted((re.escape(k) for k in _UR_EXACT_TO), key=len, reverse=True)) + r")(?![؀-ۿ])")


def apply(text):
    """Latin Pakistani names -> Urdu script. Possessive 's is kept as English ("Bakhsh's" stays).
    Also English ordinals: "3rd March" -> "third March" (NeMo read it as "three rd"), and the few
    Urdu spellings that hide their vowels (_UR_EXACT_TO) back to Latin."""
    from num2words import num2words
    text = unicodedata.normalize("NFC", str(text))
    text = _ORDINAL.sub(lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = _UR_EXACT.sub(lambda m: _UR_EXACT_TO[m.group(0)], text)
    return _RX.sub(lambda m: TABLE[m.group(0).lower()], text)


@functools.lru_cache(maxsize=1)
def _frontend():
    import text_norm                                # corrects the front end's acronym word list, once
    return text_norm.frontend()


def _lot_fix(shaped, us_line):
    """Give LOT-set words the British vowel, which is the Pakistani one.

    espeak en-us merges LOT into PALM ("dollar", "bond", "job" all get ɑː, the vowel of "father"), and EN_MAP
    turns that into aː. Pakistani English keeps them apart and says ɔ, as British English does. Measured on
    the project's own clips (src/5_evaluate/phone_check.py), this was the single biggest label-vs-audio mismatch:
    55 of 937 English word tokens. en-gb marks exactly these words with ɒ, so it decides, word by word, which ɑː to move.
    """
    ph = _phones()
    gb_line = ph.espeak([shaped], "en-gb")[0]
    us, gb = us_line.split(), gb_line.split()
    if len(us) != len(gb):
        return us_line
    return " ".join(w.replace("ɑː", "ɒ").replace("ɑ", "ɒ") if "ɒ" in g else w for w, g in zip(us, gb))


# Which English sound rules every word goes through (one switch, so training labels and speaking always agree):
#   "A"  American espeak + Pakistani table + LOT fix (Round 3)
#   "B"  British (RP) espeak vowels (new -> nyoo, mobile -> mobaail, student -> styoodent, your -> yor,
#        LOT -> o, BATH pass/class/after -> paas; TRAP cat/bank stays short), keeping the rolled r and the
#        Pakistani table. Plain en-gb merges BATH into TRAP (pass -> pess); en-gb-x-rp keeps them apart. Scored against the project's audio the two tie (0.201 vs 0.207),
#        so the recipe uses B (chosen by ear).
EN_RULES = os.environ.get("PK_EN_RULES", "B")
GB_VOICE = "en-gb-x-rp"
_RHOTIC = ("ɹ", "ɚ")
_VOWEL_CH = set("aeiouɑɒɔəɛɪʊʌɐɜæᵻɵ")
# British tokens renamed to the American-espeak token EN_MAP already maps the Pakistani way (EN_MAP was written
# for en-us, and a token it does not know is dropped): TRAP a -> æ (-> ɛ), GOAT əʊ -> oʊ (-> oː), SQUARE eə -> ɛə
_GB_FIX = {"a": "æ", "əʊ": "oʊ", "eə": "ɛə"}


def _gb_word(us_word, gb_word):
    """British tokens with the r put back wherever American espeak has one, token by token."""
    import difflib
    U = [t for t in us_word.split("|") if t]
    B = [t for t in gb_word.split("|") if t]

    def key(T):
        return ["V" if set(t.replace("ˈ", "").replace("ˌ", "")) & _VOWEL_CH else t.replace("ˈ", "") for t in T]

    def fix(t):
        stress = "".join(c for c in t if c in "ˈˌ")
        core = t.replace("ˈ", "").replace("ˌ", "")
        return stress + _GB_FIX.get(core, core)

    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=key(U), b=key(B), autojunk=False).get_opcodes():
        if op == "equal":
            for u, b in zip(U[i1:i2], B[j1:j2]):
                out.append(fix(b))
                if any(r in u for r in _RHOTIC) and "ɹ" not in b:
                    out.append("ɹ")
        else:
            out += [fix(b) for b in B[j1:j2]]
            if any(r in u for u in U[i1:i2] for r in _RHOTIC) and not any("ɹ" in b for b in B[j1:j2]):
                out.append("ɹ")
    return "|".join(out)


def _english_span_pk(span, epenthesis=False, unknown=None):
    """phones.english_span with rule variant EN_RULES; installed over it so training and speaking share it."""
    ph = _phones()
    import en_g2p                                    # common/text/en_g2p.py
    unknown = {} if unknown is None else unknown
    shaped = en_g2p.shape(span).strip()
    if not shaped:
        return ""
    o = re.sub(r"\([a-z\-]+\)", "", ph.espeak([shaped], "en-us")[0])
    if EN_RULES == "B":
        gb = re.sub(r"\([a-z\-]+\)", "", ph.espeak([shaped], GB_VOICE)[0])
        us_w, gb_w = o.split(), gb.split()
        tokens = [_gb_word(u, b) for u, b in zip(us_w, gb_w)] if len(us_w) == len(gb_w) else _lot_fix(shaped, o).split()
    else:
        tokens = _lot_fix(shaped, o).split()
    words = []
    for w in tokens:
        p = ph._map_word(w, ph.EN_MAP, unknown)
        # espeak's v+w (reservoir "rɛzəvwɑː") both become ʋ; English never has two in a row, so keep one
        p = [x for i, x in enumerate(p) if not (x == "ʋ" and i and p[i - 1] == "ʋ")]
        if epenthesis:
            core = [x for x in p if x != ph.STRESS]
            if len(core) > 1 and core[0] == "s" and core[1] not in set(ph.VOWELS):
                p = ["ɪ"] + p
        if p:
            words.append(ph.render(p))
    return " ".join(words)


@functools.lru_cache(maxsize=1)
def _phones():
    import phones
    phones.english_span = _english_span_pk          # one place, so every caller gets the same phones
    return phones


def prepare_pk(text, letters="ur"):
    """Any text an LLM produces -> the exact symbol string the model reads (training and speaking alike).

    text_norm first (markup, web addresses, money, units, IDs, acronym shapes), then the gazetteer, then the
    urdu-phones front end. letters="ur": acronym letters are Urdu letter names (ای ٹی ایف), measured 9/10
    heard right on the Round 3 pilot against 6/10 as English sound symbols.
    """
    import text_norm
    _phones()
    # The Urdu verdict is taken here, on the text as it arrived: apply() below rewrites Pakistani
    # names into Urdu script, which would make every English sentence look like an Urdu one.
    urdu = text_norm.is_urdu(text)
    return _frontend().prepare(apply(text_norm.normalize(text)), letters=letters, urdu=urdu)


# (input, substring that MUST appear in the gazetteer output, substring that must NOT appear)
TESTS = [
    ("Our head office is in Islamabad.", "اسلام آباد", "Islamabad"),
    ("The flight from Peshawar to Quetta is late.", "کوئٹہ", "Quetta"),
    ("Dr. Ayesha Siddiqui will join.", "صدیقی", "Siddiqui"),
    ("Contact Mr. Muhammad Usman Qureshi.", "محمد عثمان قریشی", "Qureshi"),
    ("offices in Karachi and Lahore", "لاہور", "Lahore"),
    ("Quaid-i-Azam University", "قائداعظم", "Quaid"),
    ("Gilgit-Baltistan is beautiful", "گلگت بلتستان", "Gilgit"),
    ("parts of Sindh and southern Punjab", "سندھ", "Sindh"),
    ("Mari Energies operates near Daharki", "ماڑی انرجیز", "Mari"),
    ("Sui Northern Gas Pipelines", "سوئی ناردرن", "Sui"),
    ("Habib Bank Limited", "حبیب", "Habib"),
    ("Shaukat Khanum Memorial Cancer Hospital", "شوکت خانم", "Shaukat"),
    ("Allama Iqbal International Airport", "علامہ اقبال", "Allama"),
    ("Hazrat Data Ganj Bakhsh's shrine", "داتا گنج بخش", "Ganj"),
    ("the Zarghun South field", "زرغون ساؤتھ", "Zarghun"),
    ("Waziristan and Sujawal blocks", "وزیرستان", None),     # Sujawal keeps Latin here; its exact sounds come later
    ("in Gujranwala and Sialkot", "گوجرانوالہ", "Gujranwala"),
    ("Engro Fertilizers", "اینگرو", "Engro"),
    ("The results come next week, inshallah.", "ان شاء اللہ", "inshallah"),
    ("Mehwish Hayat and Fawad Khan", "مہوش حیات", "Mehwish"),
    ("students from LUMS and NUST", "لمز", "LUMS"),
    ("Dera Ghazi Khan district", "ڈیرہ غازی خان", "Dera"),
    ("Aga Khan University", "آغا خان", "Aga"),
    ("Pakistan and England", "پاکستان", "Pakistan"),
    ("The meeting is on 3rd March.", "third March", "3rd"),
    # must stay English
    ("I love jazz music.", "jazz", None),
    ("Lucky draw winners", "Lucky", None),
    ("The pearl necklace", "pearl", None),
    ("She is a fast learner", "fast", None),
    ("Khanate history", "Khanate", "خان"),
    ("Alibaba and Aliexpress", "Alibaba", "علی"),
]


# (input, substring that MUST appear in the final symbols, substring that must NOT appear): prepare_pk level
PHONE_TESTS = [
    ("reservoir", "rˈɛzərʋaːr", "ʋʋ"),                 # espeak v+w -> one ʋ
    ("cooled", "kˈuːlɖ", "oː"),                        # keeps the pool/pole vowel contrast in the input
    ("cold", "kˈoːlɖ", "uː"),
    ("please", "plˈiːz", None),
    ("Sky47", "skˈaːɪ", None),
    ("The NASA mission", "nˈɛsə", "این"),            # word acronyms are said, not spelled
    ("FIFA world cup", "fˈiːfə", "ایف"),
    ("SENIOR MANAGEMENT ONLY", "ˈoːnlɪ", "وائی"),       # capitalised headings stay words
    ("EBITDA rose", "ˈɛbɪʈɖ", "بی آئی"),
    ("CFO Nabeel Rasheed", "نبیل رشید", "nˈɛb"),              # Pakistani names in Urdu script, not English rules
    ("KP office in Bannu Cantt", "bənnˈuː کینٹ", "سی اے این"),
    ("Fauji Foundation", "fˈɔʤiː", "فوجی"),                 # Urdu script lost the "au" (heard as "Fuji")
    ("Sujawal Block", "sʊʤˈaːʋəl", "سجاول"),
    ("Kehkashan, Clifton", "klˈɪfʈən", "کلفٹن"),
    ("short term A1", "اے", "ə ʋɔn"),
    ("Aligarh", "علی گڑھ", "اخبار"),                        # the Roman-Urdu matcher read it as "newspaper"
    ("Lt Gen Hamid Niaz", "lɛfʈˈɛnənʈ ʤˈɛnrəl", "ایل ٹی"),
    ("Maj Gen Mehmood Aslam Hayat", "mˈeːʤər ʤˈɛnrəl محمود", "mˈəʤ"),
    ("Sehat Umeed Referral Program", "صحت امید", "jˈuːmiːɖ"),
    ("Gen Z is online", "ʤˈɛn", "ʤˈɛnrəl"),                # a bare "Gen" before Z is a generation, not a rank
    ("Bhitai-6 well، Mari D&PL میں ہے", "bʰɪʈˈaːiː sˈɪks", "منفی"),  # well names: English number, no "minus"
    # Digit by digit, in Urdu, and the "+" is silent (it used to be "plus", then "جمع").
    ("ہم سے (+92) 51-111 410 410 پر رابطہ کریں", "nˈɔ dˈoː pˈaːnʧ ˈeːk", "plˈəs"),
    ("PSX and IMF", "پی ایس ایکس", None),               # real initialisms still spelled
    ("She has a PhD", "پی ایچ ڈی", None),
    ("IFRS 16", "آئی ایف آر ایس", None),
    # The Urdu spelling hides the vowel of "-man", so this was read "chairmeen".
    ("چیئرمین انور علی حیدر ہیں۔", "ʧˈeːrmən", "چیئرمین"),
    ("ESG کمیٹی کی چیئرپرسن سیما عادل ہیں۔", "ʧˈeːrpərsən", "چیئرپرسن"),
    # A joined English suffix: the سی of چیئرمینسی was heard as a stray "see" before the name.
    ("Audit Committee کی چیئرمینسی Abid Niaz Hasan کرتے ہیں۔", "ʧˈeːrmənsiː", "چیئرمین"),
    # ── numbers are read in Urdu ──────────────────────────────────────────
    ("اس سال 30 فیصد اضافہ ہوا۔", "tˈiːs", "thirty"),
    ("کمپنی کا revenue PKR 65bn تھا۔", "pɛnsˈəʈʰ ˈərəb", "bˈɪljən"),
    ("خالص منافع 65.14 بلین روپے رہا۔", "ɛʃˈaːrɪjaː", "pˈɔɪnʈ"),
    ("گیس فیلڈ 1954 میں دریافت ہوا۔", "ʊnnˈiːs sˈɔ ʧəʋʋˈən", "həzˈaːr"),
    ("ملازمین کی تعداد 1,760 ہے۔", "ˈeːk həzˈaːr sˈaːt sˈɔ sˈaːʈʰ", "sˈɛʋən"),
    # A hyphen between two numbers is a range, not منفی and not the English "to" (heard as "two").
    ("gas 30-36 MMSCFD رہی", "tˈiːs سے ʧʰəttˈiːs", "mˈənfiː"),
    ("2 + 2 برابر 4 ہے", "dˈoː dˈoː", "جمع"),
    # identifiers: digit by digit
    ("ہمارا NTN 1414673-8 ہے۔", "ˈeːk ʧˈaːr ˈeːk ʧˈaːr", "lˈaːkʰ"),
    ("اسلام آباد – 44000 میں دفتر ہے۔", "ʧˈaːr ʧˈaːr sˈɪfər sˈɪfər sˈɪfər", "həzˈaːr"),
    # ...but a bare five-digit quantity is still a quantity
    ("پیداوار 44000 بیرل رہی۔", "ʧəʋaːlˈiːs həzˈaːr", "sˈɪfər sˈɪfər"),
    # a letter+number code keeps its English reading
    ("Bhitai-6 well، Mari G-10/4 میں ہے", "bʰɪʈˈaːiː sˈɪks", "ʧʰˈeː"),
]


def run_tests():
    fails = 0
    for inp, must, must_not in PHONE_TESTS:
        out = prepare_pk(inp)
        ok = must in out and (must_not is None or must_not not in out)
        fails += not ok
        print(("PASS" if ok else "FAIL"), inp, "->", out)
    for inp, must, must_not in TESTS:
        out = apply(inp)
        ok = must in out and (must_not is None or must_not not in out)
        fails += not ok
        print(("PASS" if ok else "FAIL"), inp, "->", out)
    print(f"{len(TESTS) + len(PHONE_TESTS) - fails}/{len(TESTS) + len(PHONE_TESTS)} passed")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--prepare", action="store_true", help="also run the full front end (prepare_pk: what the model reads)")
    a = ap.parse_args()
    if a.test:
        sys.exit(1 if run_tests() else 0)
    s = " ".join(a.text)
    print(apply(s))
    if a.prepare:
        print(prepare_pk(s))


if __name__ == "__main__":
    main()
