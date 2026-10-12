import re
import nltk
from nltk.corpus import cmudict

# Ensure CMUDict is available
try:
    CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    CMU_DICT = cmudict.dict()

print("Initializing phoneme and rhyme indices...")

# Base ARPABET to Custom Respelling Mapping
ARPABET_MAP = {
    'IY': 'ee', 'IH': 'i', 'EH': 'e', 'AE': 'a',
    'AA': 'aa', 'AO': 'aw', 'UH': 'uu',
    'UW': 'oo', 'EY': 'ay', 'AY': 'igh',
    'OY': 'oy', 'AW': 'ow', 'OW': 'oh',

    'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'dh', 'F': 'f',
    'G': 'g', 'HH': 'h', 'JH': 'j', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ng', 'P': 'p', 'R': 'r',
    'S': 's', 'SH': 'sh', 'T': 't', 'TH': 'th', 'V': 'v',
    'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
}

# ARPABET to IPA Mapping (No length marks, using ɹ everywhere)
ARPABET_TO_IPA = {
    'IY': 'i', 'IH': 'ɪ', 'EH': 'ɛ', 'AE': 'æ', 'AH': 'ʌ', 'AA': 'ɑ', 'AO': 'ɔ',
    'UH': 'ʊ', 'UW': 'u', 'ER': 'ɜɹ', 'EY': 'eɪ', 'AY': 'aɪ', 'OY': 'ɔɪ', 'AW': 'aʊ', 'OW': 'oʊ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'ɹ', 'S': 's',
    'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

VOWEL_SET = {'IY', 'IH', 'EH', 'AE', 'AH', 'AA', 'AO', 'UH', 'UW', 'ER', 'EY', 'AY', 'OY', 'AW', 'OW'}

# Comprehensive Valid Onset Clusters (including 2 and 3-consonant blends & stop+glide)
VALID_ONSETS = {
    ('B', 'R'), ('B', 'L'), ('D', 'R'), ('F', 'R'), ('F', 'L'),
    ('G', 'R'), ('G', 'L'), ('K', 'R'), ('K', 'L'), ('P', 'R'),
    ('P', 'L'), ('T', 'R'), ('S', 'T'), ('S', 'P'), ('S', 'K'),
    ('S', 'M'), ('S', 'N'), ('S', 'L'), ('SH', 'R'), ('TH', 'R'),
    ('TH', 'W'), ('M', 'Y'), ('N', 'Y'), ('L', 'Y'),
    ('S', 'P', 'R'), ('S', 'P', 'L'), ('S', 'T', 'R'), ('S', 'K', 'R'),
    ('S', 'K', 'W'), ('K', 'W'), ('G', 'W'),
    # Stop + Glide clusters
    ('K', 'Y'), ('P', 'Y'), ('B', 'Y'), ('F', 'Y'), ('V', 'Y'), ('H', 'Y'),
    ('D', 'Y'), ('T', 'Y'), ('S', 'Y'), ('Z', 'Y')
}


def get_all_phonemes(word):
    word = word.lower()
    if word not in CMU_DICT:
        return None
    return CMU_DICT[word]


def get_ipa_transcription(phones):
    ipa_tokens = []
    for ph in phones:
        stress = ph[-1] if ph[-1].isdigit() else ''
        base = ph[:-1] if stress else ph

        if base == 'IY' and stress == '0':
            symbol = 'i'  # Happy vowel IPA
        elif base == 'UW' and stress == '0':
            symbol = 'u'  # Weak goose / influence vowel IPA
        elif base == 'AH':
            symbol = 'ʌ' if stress in {'1', '2'} else 'ə'  # Strut vs Schwa
        elif base == 'ER':
            symbol = 'ɜɹ' if stress in {'1', '2'} else 'əɹ'  # Nurse vs Letter
        else:
            symbol = ARPABET_TO_IPA.get(base, base.lower())

        if stress == '1' and base in VOWEL_SET:
            symbol = symbol[0] + '\u0301' + symbol[1:]

        ipa_tokens.append(symbol)
    return "/" + "".join(ipa_tokens) + "/"


def get_ipa_tail_from_tuple(tail_phones):
    ipa_tokens = []
    for ph in tail_phones:
        stress = ph[-1] if ph[-1].isdigit() else ''
        base = ph[:-1] if stress else ph
        symbol = ARPABET_TO_IPA.get(base, base.lower())
        ipa_tokens.append(symbol)
    return "-" + "".join(ipa_tokens)


# Build a pre-indexed rhyme dictionary grouped strictly by IPA tail representation
RHYME_INDEX = {}
for w, variants in CMU_DICT.items():
    for var in variants:
        primary_idx = -1
        for idx, ph in enumerate(var):
            if ph.endswith('1'):
                primary_idx = idx
                break
        if primary_idx != -1:
            tail = tuple(var[primary_idx:])
            ipa_tail = get_ipa_tail_from_tuple(tail)
            if ipa_tail not in RHYME_INDEX:
                RHYME_INDEX[ipa_tail] = set()
            RHYME_INDEX[ipa_tail].add(w)

print("Ready!")


def get_phoneme_breakdown(phones):
    breakdowns = []
    for ph in phones:
        stress = ph[-1] if ph[-1].isdigit() else ''
        base = ph[:-1] if stress else ph
        ipa = ARPABET_TO_IPA.get(base, base.lower())
        desc = f"{base} (IPA: {ipa}"
        if stress:
            desc += f", stress: {stress}"
        desc += ")"
        breakdowns.append(desc)
    return breakdowns


def find_rhymes(word, target_phones, limit=5):
    primary_idx = -1
    for idx, ph in enumerate(target_phones):
        if ph.endswith('1'):
            primary_idx = idx
            break
    if primary_idx == -1:
        return []

    tail = tuple(target_phones[primary_idx:])
    ipa_tail = get_ipa_tail_from_tuple(tail)
    rhymes = list(RHYME_INDEX.get(ipa_tail, set()))
    filtered = [w for w in rhymes if w != word.lower()]
    if limit:
        return filtered[:limit]
    return filtered


def apply_custom_rules(phonemes_with_stress):
    translated_tokens = []
    i = 0
    n = len(phonemes_with_stress)

    while i < n:
        ph_raw = phonemes_with_stress[i]
        stress = ph_raw[-1] if ph_raw[-1].isdigit() else None
        base_ph = ph_raw[:-1] if stress else ph_raw

        next_ph = phonemes_with_stress[i + 1] if i + 1 < n else None
        next_base = next_ph[:-1] if next_ph and next_ph[-1].isdigit() else next_ph

        # --- Strict Vowel + R Rule ---
        if next_base == 'R':
            rhotic_map = {
                'EH': 'air',
                'IH': 'eer',
                'UH': 'oor',
                'AW': 'our',
                'AY': 'ire'
            }
            if base_ph in rhotic_map:
                translated_tokens.append({
                    'text': rhotic_map[base_ph],
                    'is_vowel': True,
                    'stress': stress,
                    'base_ph': base_ph + '+R'
                })
                i += 2
                continue

        # --- Specific Vowel Rules ---
        if base_ph == 'AH':
            respell = 'u' if stress in {'1', '2'} else 'uh'
        elif base_ph == 'ER':
            respell = 'ur' if stress in {'1', '2'} else 'er'
        elif base_ph == 'IY' and stress == '0':
            respell = 'ee'
        elif base_ph == 'UW' and stress == '0':
            respell = 'oo'
        else:
            respell = ARPABET_MAP.get(base_ph, base_ph.lower())

        # Rule: goose -> ooh after t
        if base_ph == 'UW' and stress != '0' and i > 0:
            prev_raw = phonemes_with_stress[i - 1]
            prev_base = prev_raw[:-1] if prev_raw[-1].isdigit() else prev_raw
            if prev_base == 'T':
                respell = 'ooh'

        is_vowel = base_ph in VOWEL_SET

        translated_tokens.append({
            'text': respell,
            'is_vowel': is_vowel,
            'stress': stress if is_vowel else None,
            'base_ph': base_ph
        })
        i += 1

    return translated_tokens


def syllabify_and_format(tokens):
    syllables = []
    current_syl = []

    for token in tokens:
        current_syl.append(token)
        if token['is_vowel']:
            syllables.append(current_syl)
            current_syl = []

    if current_syl and syllables:
        syllables[-1].extend(current_syl)
    elif current_syl:
        syllables.append(current_syl)

    # Apply Maximal Onset Principle (MOP) Syllabification Rules
    for i in range(len(syllables) - 1):
        curr_syl = syllables[i]
        next_syl = syllables[i + 1]

        curr_vowel_idx = next((idx for idx, t in enumerate(curr_syl) if t['is_vowel']), -1)
        next_vowel_idx = next((idx for idx, t in enumerate(next_syl) if t['is_vowel']), -1)

        if curr_vowel_idx == -1 or next_vowel_idx == -1:
            continue

        coda_consonants = curr_syl[curr_vowel_idx + 1:]
        onset_consonants = next_syl[:next_vowel_idx]
        total_intervocalic = coda_consonants + onset_consonants
        N = len(total_intervocalic)

        split_handled = False

        # --- RULE 1: Doubled Consonant Check ---
        if N >= 2:
            for c_idx in range(N - 1):
                if total_intervocalic[c_idx]['text'] == total_intervocalic[c_idx + 1]['text']:
                    split_idx = c_idx + 1
                    syllables[i] = curr_syl[:curr_vowel_idx + 1] + total_intervocalic[:split_idx]
                    syllables[i + 1] = total_intervocalic[split_idx:] + next_syl[next_vowel_idx:]
                    split_handled = True
                    break

        # --- RULE 2: Maximal Onset Principle (MOP) with Valid Onsets ---
        if not split_handled:
            if N == 0:
                continue

            best_onset_len = 1
            for length in range(min(N, 3), 0, -1):
                cluster_tuple = tuple(t['base_ph'] for t in total_intervocalic[-length:])
                if length == 1 or cluster_tuple in VALID_ONSETS:
                    best_onset_len = length
                    break

            curr_token = curr_syl[curr_vowel_idx]
            is_stressed_lax = (
                        curr_token['stress'] in {'1', '2'} and curr_token['base_ph'] in {'IH', 'EH', 'AE', 'AH', 'UH',
                                                                                         'AO'})

            if is_stressed_lax and N == best_onset_len and N > 1:
                for length in range(best_onset_len - 1, 0, -1):
                    cluster_tuple = tuple(t['base_ph'] for t in total_intervocalic[-length:])
                    if length == 1 or cluster_tuple in VALID_ONSETS:
                        best_onset_len = length
                        break

            split_idx = N - best_onset_len
            syllables[i] = curr_syl[:curr_vowel_idx + 1] + total_intervocalic[:split_idx]
            syllables[i + 1] = total_intervocalic[split_idx:] + next_syl[next_vowel_idx:]

    formatted_syllables = []
    for idx, syl in enumerate(syllables):
        if not syl:
            continue

        has_primary_stress = any(t['stress'] == '1' for t in syl)
        syl_text = "".join([t['text'] for t in syl])

        if has_primary_stress:
            syl_text = syl_text.upper()

        formatted_syllables.append(syl_text)

    return "-".join(formatted_syllables)


def respell_word(word):
    all_phones = get_all_phonemes(word)
    if not all_phones:
        return [(None, None, None, [], f"'{word}' not found in dictionary.")]

    results = []
    for phones in all_phones:
        ipa = get_ipa_transcription(phones)
        breakdown = get_phoneme_breakdown(phones)
        tokens = apply_custom_rules(phones)
        respelling = syllabify_and_format(tokens)
        rhymes = find_rhymes(word, phones)
        results.append((phones, ipa, breakdown, rhymes, respelling))
    return results


def print_guide():
    print("\n==================================================================")
    print(" COMPLETE PHONEMIC RESPELLING & IPA REFERENCE GUIDE (GUIDEP)")
    print("==================================================================")
    print(" VOWELS:")
    print("   - IY  (Fleece)   : 'ee'  (IPA: i)")
    print("   - IY0 (Happy)    : 'ee'  (IPA: i) [Unstressed]")
    print("   - IH  (Kit)      : 'i'   (IPA: ɪ)")
    print("   - EH  (Dress)    : 'e'   (IPA: ɛ)")
    print("   - AE  (Trap)     : 'a'   (IPA: æ)")
    print("   - AH1/2 (Strut)  : 'u'   (IPA: ʌ)")
    print("   - AH0 (Schwa)    : 'uh'  (IPA: ə)")
    print("   - AA  (Father)   : 'aa'  (IPA: ɑ)")
    print("   - AO  (Thought)  : 'aw'  (IPA: ɔ)")
    print("   - UH  (Foot)     : 'uu'  (IPA: ʊ)")
    print("   - UW  (Goose)    : 'oo'  (IPA: u)")
    print("   - UW0 (Influence): 'oo'  (IPA: u) [Unstressed]")
    print("   - ER1/2 (Nurse)  : 'ur'  (IPA: ɜɹ)")
    print("   - ER0 (Letter)   : 'er'  (IPA: əɹ)")
    print("   - EY  (Face)     : 'ay'  (IPA: eɪ)")
    print("   - AY  (Price)    : 'igh' (IPA: aɪ)")
    print("   - OY  (Choice)   : 'oy'  (IPA: ɔɪ)")
    print("   - AW  (Mouth)    : 'ow'  (IPA: aʊ)")
    print("   - OW  (Go)       : 'oh'  (IPA: oʊ)")
    print("\n CONSONANTS:")
    print("   - B   (Bed)      : 'b'   (IPA: b)")
    print("   - CH  (Cheese)   : 'ch'  (IPA: tʃ)")
    print("   - D   (Dog)      : 'd'   (IPA: d)")
    print("   - DH  (The)      : 'dh'  (IPA: ð)")
    print("   - F   (Fish)     : 'f'   (IPA: f)")
    print("   - G   (Go)       : 'g'   (IPA: ɡ)")
    print("   - HH  (Hat)      : 'h'   (IPA: h)")
    print("   - JH  (Jump)     : 'j'   (IPA: dʒ)")
    print("   - K   (Cat)      : 'k'   (IPA: k)")
    print("   - L   (Left)     : 'l'   (IPA: l)")
    print("   - M   (Man)      : 'm'   (IPA: m)")
    print("   - N   (Net)      : 'n'   (IPA: n)")
    print("   - NG  (Sing)     : 'ng'  (IPA: ŋ)")
    print("   - P   (Pen)      : 'p'   (IPA: p)")
    print("   - R   (Red)      : 'ɹ'   (IPA: r)")
    print("   - S   (Sun)      : 's'   (IPA: s)")
    print("   - SH  (Ship)     : 'sh'  (IPA: ʃ)")
    print("   - T   (Top)      : 't'   (IPA: t)")
    print("   - TH  (Thing)    : 'th'  (IPA: θ)")
    print("   - V   (Voice)    : 'v'   (IPA: v)")
    print("   - W   (Water)    : 'w'   (IPA: w)")
    print("   - Y   (Yes)      : 'y'   (IPA: j)")
    print("   - Z   (Zoo)      : 'z'   (IPA: z)")
    print("   - ZH  (Measure)  : 'zh'  (IPA: ʒ)")
    print("==================================================================\n")


def export_all(mode):
    if mode == 'RESRHM':
        filename = "all_resrhm.txt"
        print(f"Exporting and sorting unified rhyme categories to {filename}...")

        sorted_tails = sorted(RHYME_INDEX.keys())
        total_tails = len(sorted_tails)
        cat_count = 0

        with open(filename, 'w', encoding='utf-8') as f:
            for ipa_tail in sorted_tails:
                cat_count += 1
                if cat_count % 200 == 0 or cat_count == total_tails:
                    pct = (cat_count / total_tails) * 100
                    print(f"Progress: {cat_count}/{total_tails} rhyme categories processed ({pct:.1f}%)", end='\r',
                          flush=True)

                words = sorted(list(RHYME_INDEX[ipa_tail]))

                f.write(f"==================================================\n")
                f.write(f"Rhyme Category: {ipa_tail} (Total Words: {len(words)})\n")
                f.write(f"==================================================\n")

                for word in words:
                    variants = CMU_DICT[word]
                    respellings = []
                    for var in variants:
                        p_idx = -1
                        for idx, ph in enumerate(var):
                            if ph.endswith('1'):
                                p_idx = idx
                                break
                        if p_idx != -1:
                            var_tail = tuple(var[p_idx:])
                            if get_ipa_tail_from_tuple(var_tail) == ipa_tail:
                                tokens = apply_custom_rules(var)
                                res = syllabify_and_format(tokens)
                                if res not in respellings:
                                    respellings.append(res)
                    resp_str = ", ".join(respellings)
                    f.write(f"  {word} : {resp_str}\n")
                f.write("\n")
        print(f"\nExport complete! Saved to {filename}")
    else:
        filename = f"all_{mode.lower()}.txt"
        total_words = len(CMU_DICT)
        print(f"Exporting all CMUDict words to {filename}...")

        count = 0
        with open(filename, 'w', encoding='utf-8') as f:
            for word, variants in CMU_DICT.items():
                count += 1
                if count % 2000 == 0 or count == total_words:
                    pct = (count / total_words) * 100
                    print(f"Progress: {count}/{total_words} words processed ({pct:.1f}%)", end='\r', flush=True)

                for phones in variants:
                    if mode == 'RESPELL':
                        tokens = apply_custom_rules(phones)
                        res = syllabify_and_format(tokens)
                        f.write(f"{word}: {res}\n")
                    elif mode == 'IPA':
                        ipa = get_ipa_transcription(phones)
                        f.write(f"{word}: {ipa}\n")
                    elif mode == 'NLTK':
                        f.write(f"{word}: {phones}\n")
        print(f"\nExport complete! Saved to {filename}")


# --- Interactive User Input Loop ---
if __name__ == "__main__":
    print("========================================")
    print(" Phonemic Respelling & IPA Generator")
    print(" Type 'GUIDEP' for the full phonetics guide.")
    print(" Type 'ALLRESPELL', 'ALLIPA', 'ALLNLTK', 'ALLRESRHM' to export.")
    print(" Type 'quit' or 'exit' to stop.")
    print("========================================")

    while True:
        try:
            user_input = input("\nEnter a word or command: ").strip()
            if not user_input:
                continue

            cmd = user_input.upper()
            if cmd in {'QUIT', 'EXIT'}:
                print("Goodbye!")
                break
            elif cmd == 'GUIDEP':
                print_guide()
                continue
            elif cmd in {'ALLRESPELL', 'ALLIPA', 'ALLNLTK', 'ALLRESRHM'}:
                mode_map = {
                    'ALLRESPELL': 'RESPELL',
                    'ALLIPA': 'IPA',
                    'ALLNLTK': 'NLTK',
                    'ALLRESRHM': 'RESRHM'
                }
                export_all(mode_map[cmd])
                continue

            variants = respell_word(user_input)
            print(f"\nResult for '{user_input}':")
            for idx, (nltk_phones, ipa, breakdown, rhymes, respelling) in enumerate(variants, 1):
                rhyme_output = rhymes if rhymes else ['None (fallback: check suffix similarity)']
                print(f"  Variant {idx}:")
                print(f"    NLTK Phonemes  : {nltk_phones}")
                print(f"    IPA (w/ stress): {ipa}")
                print(f"    Phoneme Guide  : {breakdown}")
                print(f"    Rhymes         : {rhyme_output}")
                print(f"    Respelling     : {respelling}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break