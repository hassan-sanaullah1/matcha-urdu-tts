"""Open-vocabulary evaluation suite -> data/4_eval/open_suite_rebuild.json (frozen copy: open_suite.json).

The kiosk is an LLM: it can say any English word, name, acronym or number, in English, Urdu or both. So the
suite is mostly general text, weighted about 40% toward MARI (its knowledge base is the main use), and it
includes words and acronyms that appear neither in the knowledge base nor in the target voice's recordings, so general
rules are tested on text they cannot have been tuned to.

Categories (id prefix): gen_en, gen_rare, gen_ur, gen_cs, mari_en, mari_cs, acr_kb, acr_gen, hard.
pron_eval.py --rebuild also imports this file for its word lists (GEN_EN, GEN_CS, MARI_EN, MARI_CS, ACR_GEN) and
kb_acronyms(), every acronym in the MARI knowledge base.

Input:  data/4_eval/knowledge_base/mari_energies_knowledge_base.md, assets/dict/american-english,
        data/2_clean/voice_clean/manifest.jsonl (to flag words the voice never said; skipped if absent)
Usage (from the kiosk-tts folder):  python src/5_evaluate/make_open_suite.py
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

KB = os.path.join(paths.EVAL, "knowledge_base", "mari_energies_knowledge_base.md")

GEN_EN = [
    "Good morning! How can I help you today?",
    "Please wait a moment while I look that up for you.",
    "The weather in Islamabad will be sunny with a light breeze this afternoon.",
    "You can reset your password from the settings page on your phone.",
    "Our office is closed on public holidays, but the helpline is open around the clock.",
    "Drinking enough water every day is one of the simplest ways to stay healthy.",
    "The train to Lahore leaves at nine in the morning and takes about four hours.",
    "If the screen freezes, hold the power button for ten seconds and restart the device.",
    "Thank you for your patience. Your request has been submitted successfully.",
    "The museum has a new exhibition on the history of the Indus Valley civilisation.",
    "I'm sorry, I didn't quite catch that. Could you please repeat your question?",
    "Electric vehicles are becoming more popular because fuel prices keep rising.",
    "The doctor recommended a balanced diet, regular exercise and at least seven hours of sleep.",
    "Cricket fans across the country celebrated after the team won the final.",
    "Artificial intelligence can help companies analyse large amounts of data quickly.",
    "Please keep your ticket with you until you leave the building.",
    "Solar panels convert sunlight directly into electricity without any moving parts.",
    "The library allows members to borrow up to five books at a time.",
    "Heavy rain is expected in the northern areas, so travellers should check road conditions.",
    "Would you like me to send these details to your email address?",
    "Remote work has changed how many people think about their careers.",
    "The conference will focus on climate change, water security and sustainable agriculture.",
    "Your appointment has been confirmed for Thursday afternoon.",
    "Recycling paper, plastic and glass helps reduce waste in our cities.",
    "The new smartphone has a larger battery and a much better camera.",
    "Children should be encouraged to read books outside the school curriculum.",
    "This building has wheelchair access and lifts on every floor.",
    "The company announced record profits despite a difficult economic year.",
    "Please switch off your mobile phone during the presentation.",
    "Is there anything else I can help you with before you go?",
]
GEN_UR = [
    "السلام علیکم! میں آپ کی کیا مدد کر سکتی ہوں؟",
    "براہ کرم تھوڑا انتظار کریں، میں آپ کے لیے معلومات تلاش کر رہی ہوں۔",
    "آج اسلام آباد میں موسم خوشگوار رہے گا اور ہلکی ہوا چلے گی۔",
    "ہمارا دفتر صبح نو بجے سے شام پانچ بجے تک کھلا رہتا ہے۔",
    "آپ کی درخواست کامیابی سے جمع ہو گئی ہے، شکریہ۔",
    "بچوں کو روزانہ کتابیں پڑھنے کی عادت ڈالنی چاہیے۔",
    "پانی کی بچت ہم سب کی ذمہ داری ہے۔",
    "لاہور سے کراچی جانے والی ٹرین صبح روانہ ہوتی ہے۔",
    "معاف کیجیے، میں آپ کی بات ٹھیک سے سن نہیں سکی، کیا آپ دوبارہ بتا سکتے ہیں؟",
    "کیا میں آپ کی کسی اور چیز میں مدد کر سکتی ہوں؟",
]
GEN_CS = [
    "آپ اپنا password settings میں جا کر reset کر سکتے ہیں۔",
    "ہماری helpline چوبیس گھنٹے available ہے، آپ کسی بھی وقت call کر سکتے ہیں۔",
    "اگر آپ کا internet slow ہے تو پہلے router کو restart کریں۔",
    "آپ کی appointment Thursday کو confirm ہو گئی ہے۔",
    "یہ building wheelchair friendly ہے اور ہر floor پر lift موجود ہے۔",
    "آج کل electric vehicles کافی popular ہو رہی ہیں کیونکہ petrol مہنگا ہو گیا ہے۔",
    "آپ کی request successfully submit ہو گئی ہے، ہماری team جلد رابطہ کرے گی۔",
    "Doctor نے balanced diet اور regular exercise کا مشورہ دیا ہے۔",
    "Solar panels سورج کی روشنی کو directly بجلی میں convert کرتے ہیں۔",
    "کیا میں یہ details آپ کے email پر بھیج دوں؟",
    "Presentation کے دوران اپنا mobile phone silent پر رکھیں۔",
    "Artificial intelligence سے companies بہت تیزی سے data analyse کر سکتی ہیں۔",
    "آپ کا complaint number ہماری system میں register ہو گیا ہے۔",
    "Museum میں Indus Valley civilisation پر ایک نئی exhibition لگی ہے۔",
    "Heavy rain کی وجہ سے northern areas کی roads بند ہو سکتی ہیں۔",
    "آپ online form fill کر کے اپنی application track کر سکتے ہیں۔",
    "یہ offer صرف اس مہینے کے آخر تک valid ہے۔",
    "آپ کا parcel dispatch ہو چکا ہے اور دو دن میں deliver ہو جائے گا۔",
    "Meeting کا agenda آپ کو email کر دیا گیا ہے۔",
    "شکریہ! آپ کا feedback ہمارے لیے بہت important ہے۔",
]
MARI_EN = [
    "Mari Energies is an integrated Pakistani energy company and the largest company in Pakistan by market capitalization.",
    "The company supplies feed gas to all major fertilizer plants in the country.",
    "Mari Energies was formerly known as Mari Petroleum Company Limited, or MPCL.",
    "The company is listed on the Pakistan Stock Exchange under the symbol MARI.",
    "Its long term credit rating is triple A.",
    "The Mari gas field is located in Daharki, in the Ghotki district of Sindh.",
    "Mari Energies is diversifying into critical minerals and technology.",
    "For a live share price, please check the PSX website.",
    "The company follows strict health, safety and environment standards.",
    "Our corporate social responsibility programmes focus on education and healthcare.",
    "The head office is in Islamabad.",
    "Mari Technologies is a wholly owned subsidiary focused on large scale data and technology projects.",
    "The company has exploration interests across onshore and offshore blocks.",
    "Graduate trainee programmes are announced on the careers page.",
    "Mari Energies works with several banks, including Meezan Bank and HBL.",
    "The company reports its results for each financial year.",
    "Gas production is measured in million standard cubic feet per day.",
    "Suppliers must comply with the company's anti-corruption policy.",
    "Tree plantation drives are part of the company's environmental commitments.",
    "You can contact the company through its official website.",
]
MARI_CS = [
    "Mari Energies پاکستان کی سب سے بڑی کمپنی ہے market capitalization کے لحاظ سے۔",
    "یہ کمپنی ملک کے تمام بڑے fertilizer plants کو gas فراہم کرتی ہے۔",
    "Mari Energies کا پرانا نام Mari Petroleum Company Limited تھا۔",
    "کمپنی Pakistan Stock Exchange پر MARI کے symbol سے listed ہے۔",
    "ماڑی gas field ڈہرکی، ضلع گھوٹکی، سندھ میں واقع ہے۔",
    "کمپنی اب critical minerals اور technology میں بھی کام کر رہی ہے۔",
    "Live share price کے لیے براہ کرم PSX کی website دیکھیں۔",
    "کمپنی کا head office اسلام آباد میں ہے۔",
    "Mari Technologies کمپنی کا ایک مکمل ملکیتی subsidiary ہے۔",
    "کمپنی education اور healthcare کے لیے CSR programmes چلاتی ہے۔",
    "Graduate trainee programme کا اعلان careers page پر کیا جاتا ہے۔",
    "کمپنی کی long term credit rating triple A ہے۔",
    "Suppliers کو کمپنی کی anti-corruption policy پر عمل کرنا ہوتا ہے۔",
    "کمپنی کی ESG reporting ہر سال شائع ہوتی ہے۔",
    "آپ کمپنی سے official website کے ذریعے رابطہ کر سکتے ہیں۔",
    "ماڑی انرجیز کے پاس onshore اور offshore دونوں blocks میں exploration interests ہیں۔",
    "Gas production کو MMSCFD میں ناپا جاتا ہے۔",
    "کمپنی HSE standards پر سختی سے عمل کرتی ہے۔",
    "Tree plantation drives کمپنی کے environmental commitments کا حصہ ہیں۔",
    "کمپنی ہر financial year کے نتائج شائع کرتی ہے۔",
]
ACR_GEN = ["NASA", "UNICEF", "GPS", "USB", "HTML", "PDF", "ATM", "SMS", "CNIC", "NTN", "GST", "VAT", "CEO", "CFO",
           "HR", "IT", "AI", "UN", "WHO", "IMF", "GDP", "KYC", "OTP", "PIN", "SIM", "Wi-Fi", "LED", "LCD", "DNA", "BBC",
           "CNN", "FIFA", "ICC", "PCB", "PTA", "NEPRA", "SECP", "HEC", "FBR", "NADRA", "WAPDA", "PIA", "PSO", "SBP",
           "LNG", "CNG", "LPG", "EV", "MBA", "PhD"]
HARD = [
    "The share price closed at PKR 651.65, down 0.42% for the day.",
    "Please call +92 51 111 410 410 or email info@marienergies.com.pk.",
    "The meeting is at 10:30 am on 28 July 2026.",
    "Revenue grew by 12.5% to Rs. 383 billion in FY2024-25.",
    "Phase III of the project covers 155,756 sq km.",
    "The National Tax Number is 1414673-8.",
    "آج کی meeting دوپہر 2:30 پر ہے۔",
    "اس کی قیمت PKR 5,500 ہے۔",
    "Visit www.psx.com.pk for the latest KSE-100 index.",
    "The plant produces 36 MMSCFD of gas and 7 BOPD of condensate.",
    "Our 3rd quarterly report will be published on 15th October.",
    "The E&P sector added 2.81x more value than expected.",
]


def kb_acronyms():
    t = open(KB, encoding="utf-8").read()
    toks = set(re.findall(r"\b[A-Z][A-Z&]{1,9}s?\b", t))
    drop = {"I", "II", "III", "IV", "PM", "AM"}
    return sorted(a for a in toks if a not in drop and not a.endswith("&"))


def main():
    rnd = random.Random(0)
    words = [w.strip() for w in open(paths.DICT_US, encoding="utf-8", errors="ignore")
             if w.strip().isalpha() and w.strip().islower() and 8 <= len(w.strip()) <= 14]
    seen_words = set()
    manifest = os.path.join(paths.VOICE_CLEAN, "manifest.jsonl")
    if os.path.exists(manifest):
        for l in open(manifest, encoding="utf-8"):
            seen_words |= {w.lower() for w in re.findall(r"[A-Za-z]+", json.loads(l)["text"])}
    rare = [w for w in rnd.sample(words, 400) if w not in seen_words][:30]
    items = []
    add = lambda cat, i, text, lang: items.append(dict(id=f"{cat}_{i:03d}", cat=cat, lang=lang, text=text))
    for i, t in enumerate(GEN_EN):
        add("gen_en", i, t, "en")
    for i, w in enumerate(rare):
        add("gen_rare", i, f"The word of the day is {w}.", "en")
    for i, t in enumerate(GEN_UR):
        add("gen_ur", i, t, "ur")
    for i, t in enumerate(GEN_CS):
        add("gen_cs", i, t, "cs")
    for i, t in enumerate(MARI_EN):
        add("mari_en", i, t, "en")
    for i, t in enumerate(MARI_CS):
        add("mari_cs", i, t, "cs")
    for i, a in enumerate(kb_acronyms()):
        add("acr_kb", i, f"Please check the {a} section for details.", "en")
    for i, a in enumerate(ACR_GEN):
        add("acr_gen", i, f"Please check the {a} section for details.", "en")
    for i, t in enumerate(HARD):
        add("hard", i, t, "cs" if re.search(r"[؀-ۿ]", t) else "en")
    for it in items:
        it["unseen"] = [w for w in re.findall(r"[A-Za-z]+", it["text"]) if w.lower() not in seen_words]
    json.dump(items, open(os.path.join(paths.EVAL, "open_suite_rebuild.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    by = {}
    for it in items:
        by[it["cat"]] = by.get(it["cat"], 0) + 1
    mari = sum(v for k, v in by.items() if k.startswith(("mari", "acr_kb")))
    print(f"{len(items)} items, {by}, MARI share {mari / len(items):.0%}")


if __name__ == "__main__":
    main()
