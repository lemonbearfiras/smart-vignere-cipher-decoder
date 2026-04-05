from collections import Counter
from functools import lru_cache
import argparse
import csv
import heapq
import json
import os
import string


ALPHABET = string.ascii_uppercase
DEFAULT_DICTIONARY_FILENAME = "words.txt"
DEFAULT_BIGRAMS_FILENAME = "bigrams.txt"
DEFAULT_TRIGRAMS_FILENAME = "trigrams.txt"
ENGLISH_FREQ = {
    "A": 0.08167,
    "B": 0.01492,
    "C": 0.02782,
    "D": 0.04253,
    "E": 0.12702,
    "F": 0.02228,
    "G": 0.02015,
    "H": 0.06094,
    "I": 0.06966,
    "J": 0.00153,
    "K": 0.00772,
    "L": 0.04025,
    "M": 0.02406,
    "N": 0.06749,
    "O": 0.07507,
    "P": 0.01929,
    "Q": 0.00095,
    "R": 0.05987,
    "S": 0.06327,
    "T": 0.09056,
    "U": 0.02758,
    "V": 0.00978,
    "W": 0.02360,
    "X": 0.00150,
    "Y": 0.01974,
    "Z": 0.00074,
}
COMMON_WORDS = {
    "THE", "BE", "TO", "OF", "AND", "IN", "THAT", "HAVE", "FOR", "NOT",
    "WITH", "YOU", "THIS", "FROM", "ARE", "WAS", "WERE", "THERE", "WHEN",
    "WHERE", "WHAT", "WHICH", "HELLO", "SECRET", "MESSAGE", "WORLD",
    "ATTACK", "DAWN", "OVER", "QUICK", "BROWN", "FOX", "JUMPS", "LAZY", "DOG",
}
COMMON_SHORT_WORDS = {
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO",
    "UP", "US", "WE", "YOU",
}
COMMON_ENDINGS = ("ING", "ED", "ER", "ES", "LY", "TION", "MENT", "NESS", "FUL", "ABLE")
BAD_CLUSTERS = ("QJ", "JQ", "ZX", "QZ", "VV", "JJ", "QQ", "WX", "XQ")
FALLBACK_BIGRAMS = {
    ("I", "LOVE"), ("LOVE", "YOU"), ("SO", "MUCH"), ("OVER", "THE"),
    ("THIS", "IS"), ("ARE", "YOU"),
}
FALLBACK_TRIGRAMS = {
    ("I", "LOVE", "YOU"), ("THE", "QUICK", "BROWN"), ("JUMPS", "OVER", "THE"),
}
KNOWN_WORD_BONUSES = {
    "ATTACK": 120,
    "DAWN": 120,
    "HELLO": 60,
    "WORLD": 60,
    "SECRET": 60,
    "MESSAGE": 60,
}
SHORT_CIPHERTEXT_THRESHOLD = 24
COMMON_START_WORDS = {"THE", "THIS", "THAT", "I", "YOU", "WE", "HE", "SHE", "IT", "ATTACK", "HELLO"}
COMMON_STOP_WORDS = {"THE", "A", "AN", "OF", "TO", "IN", "ON", "AT", "FOR", "WITH", "FROM"}


def sanitize(text):
    return "".join(char for char in text.upper() if char in ALPHABET)


def tokenize_words(text):
    current = []
    words = []
    for char in text.upper():
        if char in ALPHABET:
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def restore_format(original_text, cleaned_plaintext):
    restored = []
    plain_index = 0
    for char in original_text:
        if char.upper() in ALPHABET:
            plain_char = cleaned_plaintext[plain_index]
            restored.append(plain_char.lower() if char.islower() else plain_char)
            plain_index += 1
        else:
            restored.append(char)
    return "".join(restored)


def vigenere_decrypt(ciphertext, key):
    key_indexes = [ord(char) - 65 for char in key]
    return "".join(
        chr((ord(char) - 65 - key_indexes[index % len(key_indexes)]) % 26 + 65)
        for index, char in enumerate(ciphertext)
    )


def chi_squared_score(text):
    n = len(text)
    if n == 0:
        return float("inf")
    freqs = Counter(text)
    score = 0.0
    for letter in ALPHABET:
        observed = freqs.get(letter, 0)
        expected = ENGLISH_FREQ[letter] * n
        score += ((observed - expected) ** 2) / expected
    return score


def caesar_decrypt(text, shift):
    return "".join(chr((ord(char) - 65 - shift) % 26 + 65) for char in text)


def split_by_key_length(text, key_length):
    return [text[index::key_length] for index in range(key_length)]


def ranked_caesar_shifts(segment, limit=3):
    scored = []
    for shift in range(26):
        decrypted = caesar_decrypt(segment, shift)
        scored.append((shift, chi_squared_score(decrypted)))
    scored.sort(key=lambda item: item[1])
    return scored[:limit]


def load_dictionary_path(dictionary_path=None):
    if dictionary_path and os.path.exists(dictionary_path):
        return dictionary_path
    project_dictionary = os.path.join(os.path.dirname(__file__), DEFAULT_DICTIONARY_FILENAME)
    if os.path.exists(project_dictionary):
        return project_dictionary
    if dictionary_path:
        raise FileNotFoundError(
            f"Dictionary file not found: {dictionary_path}. Also could not find fallback dictionary: {project_dictionary}"
        )
    return None


@lru_cache(maxsize=None)
def load_phrase_file(filename, expected_size):
    path = os.path.join(os.path.dirname(__file__), filename)
    phrases = set()
    if not os.path.exists(path):
        return frozenset()
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = tuple(part for part in (sanitize(piece) for piece in line.strip().split()) if part)
            if len(parts) == expected_size:
                phrases.add(parts)
    return frozenset(phrases)


def load_common_bigrams():
    return load_phrase_file(DEFAULT_BIGRAMS_FILENAME, 2) or frozenset(FALLBACK_BIGRAMS)


def load_common_trigrams():
    return load_phrase_file(DEFAULT_TRIGRAMS_FILENAME, 3) or frozenset(FALLBACK_TRIGRAMS)


@lru_cache(maxsize=8)
def load_dictionary_words_cached(resolved_path):
    words = set()
    if resolved_path:
        with open(resolved_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                cleaned = sanitize(line.strip())
                if cleaned:
                    words.add(cleaned)
    else:
        words.update(COMMON_WORDS)
    return tuple(sorted(words))


def build_dictionary_index(words):
    index = {}
    for word in words:
        index.setdefault(len(word), set()).add(word)
    return index


def build_dictionary_set(dictionary_index):
    dictionary_words = set()
    for words in dictionary_index.values():
        dictionary_words.update(words)
    return dictionary_words


def exact_word_matches(tokens, dictionary_words):
    matches = []
    score = 0
    for token in tokens:
        if len(token) >= 2 and token in dictionary_words:
            matches.append(token)
            score += len(token) * len(token) * 4
    return score, sorted(set(matches), key=lambda item: (-len(item), item))


def fast_word_score(tokens, dictionary_words):
    exact_score, exact_matches = exact_word_matches(tokens, dictionary_words)
    unknown_count = sum(1 for token in tokens if len(token) >= 2 and token not in dictionary_words)
    short_bonus = sum(8 for token in tokens if token in COMMON_SHORT_WORDS)
    return exact_score + short_bonus - (unknown_count * 10), exact_matches


def grammar_score_from_tokens(tokens):
    score = 0.0
    joined = "".join(tokens)
    vowels = sum(joined.count(vowel) for vowel in "AEIOU")
    vowel_ratio = vowels / max(len(joined), 1)
    score -= abs(vowel_ratio - 0.38) * 30

    for pair in BAD_CLUSTERS:
        score -= joined.count(pair) * 10

    for token in tokens:
        if token in COMMON_SHORT_WORDS:
            score += 10
        if token in COMMON_WORDS:
            score += 4
        if len(token) >= 4 and token[-3:] in COMMON_ENDINGS:
            score += 5
        if len(token) >= 5 and any(token.endswith(ending) for ending in COMMON_ENDINGS):
            score += 6
        if len(token) >= 3:
            token_vowels = sum(token.count(vowel) for vowel in "AEIOU")
            if token_vowels == 0:
                score -= 18
            elif token_vowels / len(token) < 0.20:
                score -= 6
        consonant_run = 0
        max_run = 0
        for char in token:
            if char in "AEIOU":
                consonant_run = 0
            else:
                consonant_run += 1
                max_run = max(max_run, consonant_run)
        if max_run >= 5:
            score -= max_run * 5

    if len(tokens) >= 2:
        score += min(len(tokens), 6) * 2
        if tokens[0] in COMMON_START_WORDS:
            score += 12
        if tokens[-1] not in COMMON_STOP_WORDS and len(tokens[-1]) >= 3:
            score += 6

    if len(tokens) >= 2:
        for left, right in zip(tokens, tokens[1:]):
            if left in COMMON_STOP_WORDS and right in COMMON_WORDS:
                score += 6
            if left in COMMON_WORDS and right in COMMON_WORDS:
                score += 2

    return score


def segment_joined_text(cleaned_text, dictionary_words, bigrams, trigrams, max_word_length=16):
    if not cleaned_text:
        return [cleaned_text]

    n = len(cleaned_text)
    best_scores = [-10**9] * (n + 1)
    previous = [(-1, "")] * (n + 1)
    best_scores[0] = 0

    for index in range(n):
        if best_scores[index] <= -10**8:
            continue
        for end in range(index + 1, min(n, index + max_word_length) + 1):
            fragment = cleaned_text[index:end]
            if fragment in dictionary_words:
                candidate_score = best_scores[index] + (len(fragment) * len(fragment))
            else:
                candidate_score = best_scores[index] - max(6, len(fragment) * 3)

            prev_index, prev_word = previous[index]
            if prev_word and (prev_word, fragment) in bigrams:
                candidate_score += 20
            if prev_index >= 0:
                prev_prev_index, prev_prev_word = previous[prev_index]
                if prev_prev_word and (prev_prev_word, prev_word, fragment) in trigrams:
                    candidate_score += 45

            if candidate_score > best_scores[end]:
                best_scores[end] = candidate_score
                previous[end] = (index, fragment)

    if previous[n][0] == -1:
        return [cleaned_text]

    segments = []
    cursor = n
    while cursor > 0:
        start, word = previous[cursor]
        if start < 0:
            return [cleaned_text]
        segments.append(word)
        cursor = start
    segments.reverse()
    return segments


def word_pair_score(tokens, dictionary_words, bigrams):
    score = 0
    hits = []
    bad_pairs = 0
    for index in range(len(tokens) - 1):
        pair = (tokens[index], tokens[index + 1])
        if pair in bigrams:
            score += 35
            hits.append(" ".join(pair))
        elif tokens[index] in dictionary_words and tokens[index + 1] in dictionary_words:
            bad_pairs += 1
    return score, hits, bad_pairs


def word_trigram_score(tokens, dictionary_words, trigrams):
    score = 0
    hits = []
    bad_trigrams = 0
    for index in range(len(tokens) - 2):
        trigram = (tokens[index], tokens[index + 1], tokens[index + 2])
        if trigram in trigrams:
            score += 70
            hits.append(" ".join(trigram))
        elif all(token in dictionary_words for token in trigram):
            bad_trigrams += 1
    return score, hits, bad_trigrams


def choose_tokens(formatted_plaintext, dictionary_words, bigrams, trigrams):
    original_tokens = tokenize_words(formatted_plaintext)
    if len(original_tokens) <= 1:
        segmented = segment_joined_text(sanitize(formatted_plaintext), dictionary_words, bigrams, trigrams)
        return segmented
    return original_tokens


def fast_prefilter_score(formatted_plaintext, dictionary_words):
    tokens = tokenize_words(formatted_plaintext)
    fast_score, exact_hits = fast_word_score(tokens, dictionary_words)
    cleaned = sanitize(formatted_plaintext)
    known_word_bonus = 0
    for word, bonus in KNOWN_WORD_BONUSES.items():
        if word in cleaned:
            known_word_bonus += bonus
    chi_penalty = chi_squared_score(cleaned)
    return fast_score + known_word_bonus - chi_penalty, tokens, exact_hits


def full_candidate_score(formatted_plaintext, dictionary_words, bigrams, trigrams):
    tokens = choose_tokens(formatted_plaintext, dictionary_words, bigrams, trigrams)
    exact_score, exact_hits = exact_word_matches(tokens, dictionary_words)
    grammar = grammar_score_from_tokens(tokens)
    pair_score, pair_hits, bad_pairs = word_pair_score(tokens, dictionary_words, bigrams)
    trigram_score, trigram_hits, bad_trigrams = word_trigram_score(tokens, dictionary_words, trigrams)
    chi_penalty = chi_squared_score("".join(tokens))
    unknown_penalty = sum(max(12, len(token) * 4) for token in tokens if len(token) >= 2 and token not in dictionary_words)
    prune_penalty = 0
    if len(tokens) >= 6 and bad_pairs > max(4, (len(tokens) * 3) // 4):
        prune_penalty += 250
    if len(tokens) >= 7 and bad_trigrams > max(2, (len(tokens) * 2) // 3):
        prune_penalty += 250

    known_word_bonus = 0
    joined = "".join(tokens)
    for word, bonus in KNOWN_WORD_BONUSES.items():
        if word in tokens or word in joined:
            known_word_bonus += bonus

    total = (
        exact_score
        + grammar
        + pair_score
        + trigram_score
        + known_word_bonus
        - chi_penalty
        - unknown_penalty
        - prune_penalty
    )
    hits = exact_hits + [item for item in pair_hits if item not in exact_hits]
    hits += [item for item in trigram_hits if item not in hits]
    details = {
        "exact_word_score": exact_score,
        "grammar_score": grammar,
        "bigram_score": pair_score,
        "trigram_score": trigram_score,
        "known_word_bonus": known_word_bonus,
        "unknown_penalty": unknown_penalty,
        "prune_penalty": prune_penalty,
        "chi_penalty": chi_penalty,
        "exact_word_hits": exact_hits,
        "bigram_hits": pair_hits,
        "trigram_hits": trigram_hits,
        "bad_pair_count": bad_pairs,
        "bad_trigram_count": bad_trigrams,
    }
    return total, tokens, hits, grammar, chi_penalty, details


def evaluate_candidates(ciphertext_raw, candidate_words, dictionary_index, prefilter_limit, priority_keys=None, source_map=None):
    ciphertext_clean = sanitize(ciphertext_raw)
    dictionary_words = build_dictionary_set(dictionary_index)
    bigrams = load_common_bigrams()
    trigrams = load_common_trigrams()
    priority_keys = priority_keys or set()
    source_map = source_map or {}

    shortlist = []
    for key in candidate_words:
        plaintext = vigenere_decrypt(ciphertext_clean, key)
        formatted_plaintext = restore_format(ciphertext_raw, plaintext)
        prefilter_score, _, exact_hits = fast_prefilter_score(formatted_plaintext, dictionary_words)
        if key in priority_keys:
            prefilter_score += 10000
        shortlist.append((prefilter_score, key, plaintext, formatted_plaintext, exact_hits))

    top_prefilter = heapq.nlargest(min(prefilter_limit, len(shortlist)), shortlist, key=lambda item: item[0])

    results = []
    for _, key, plaintext, formatted_plaintext, _ in top_prefilter:
        total, tokens, hits, grammar, chi_penalty, details = full_candidate_score(
            formatted_plaintext,
            dictionary_words,
            bigrams,
            trigrams,
        )
        results.append(
            {
                "key": key,
                "source": source_map.get(key, "dictionary"),
                "key_length": len(key),
                "plaintext": plaintext,
                "formatted_plaintext": " ".join(tokens) if len(tokens) > 1 else formatted_plaintext,
                "score": total,
                "dictionary_hits": hits,
                "tokens": tokens,
                "grammar_score": grammar,
                "chi_penalty": chi_penalty,
                "score_details": details,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    for result in results:
        result["rank_reason"] = build_rank_reason(result)
    return results


def generate_frequency_keys(ciphertext_clean, key_length, shift_limit=3, keep=120):
    segments = split_by_key_length(ciphertext_clean, key_length)
    shift_options = [ranked_caesar_shifts(segment, limit=shift_limit) for segment in segments]

    partial = [("", 0.0)]
    for options in shift_options:
        next_partial = []
        for key_prefix, score_prefix in partial:
            for shift, shift_score in options:
                next_partial.append((key_prefix + chr(65 + shift), score_prefix + shift_score))
        next_partial.sort(key=lambda item: item[1])
        partial = next_partial[:keep]

    return [key for key, _ in partial]


def merge_unique_candidates(dictionary_candidates, frequency_candidates):
    seen = set()
    merged = []
    for key in dictionary_candidates + frequency_candidates:
        if key not in seen:
            merged.append(key)
            seen.add(key)
    return merged


def limit_candidate_words(candidate_words, max_words=None):
    if max_words is None or max_words <= 0:
        return candidate_words
    return candidate_words[:max_words]


def dictionary_attack_with_key_length(
    ciphertext,
    key_length,
    dictionary_path=None,
    prefilter_limit=250,
    max_words=None,
    short_cipher_mode=True,
):
    cleaned = sanitize(ciphertext)
    if not cleaned:
        raise ValueError("Ciphertext must contain at least one A-Z letter.")
    if key_length < 1:
        raise ValueError("Key length must be at least 1.")

    resolved_dictionary_path = load_dictionary_path(dictionary_path)
    all_words = list(load_dictionary_words_cached(resolved_dictionary_path))
    dictionary_index = build_dictionary_index(all_words)
    candidate_words = sorted(dictionary_index.get(key_length, set()))
    if not candidate_words:
        raise ValueError(f"No dictionary words of length {key_length} were found in the dictionary.")
    candidate_words = limit_candidate_words(candidate_words, max_words)
    source_map = {key: "dictionary" for key in candidate_words}
    frequency_candidates = []
    if short_cipher_mode and len(cleaned) <= SHORT_CIPHERTEXT_THRESHOLD:
        frequency_candidates = generate_frequency_keys(cleaned, key_length)
        for key in frequency_candidates:
            source_map[key] = "frequency"
    candidate_words = merge_unique_candidates(candidate_words, frequency_candidates)

    results = evaluate_candidates(
        ciphertext,
        candidate_words,
        dictionary_index,
        prefilter_limit,
        priority_keys=set(frequency_candidates),
        source_map=source_map,
    )
    return results, len(candidate_words), resolved_dictionary_path


def dictionary_attack_unknown_key_length(
    ciphertext,
    dictionary_path=None,
    min_key_length=None,
    max_key_length=None,
    prefilter_limit=250,
    max_words=None,
    short_cipher_mode=True,
):
    cleaned = sanitize(ciphertext)
    if not cleaned:
        raise ValueError("Ciphertext must contain at least one A-Z letter.")

    resolved_dictionary_path = load_dictionary_path(dictionary_path)
    all_words = list(load_dictionary_words_cached(resolved_dictionary_path))
    dictionary_index = build_dictionary_index(all_words)

    candidate_words = []
    for word_length in sorted(dictionary_index):
        if min_key_length is not None and word_length < min_key_length:
            continue
        if max_key_length is not None and word_length > max_key_length:
            continue
        candidate_words.extend(sorted(dictionary_index[word_length]))

    if not candidate_words:
        raise ValueError("No usable dictionary words were found.")
    candidate_words = limit_candidate_words(candidate_words, max_words)
    source_map = {key: "dictionary" for key in candidate_words}
    frequency_candidates = []
    if short_cipher_mode and len(cleaned) <= SHORT_CIPHERTEXT_THRESHOLD:
        lower = min_key_length or 1
        upper = max_key_length or min(12, max(1, len(cleaned)))
        for key_length in range(lower, upper + 1):
            frequency_candidates.extend(generate_frequency_keys(cleaned, key_length, keep=40))
        for key in frequency_candidates:
            source_map[key] = "frequency"
        candidate_words = merge_unique_candidates(candidate_words, frequency_candidates)

    results = evaluate_candidates(
        ciphertext,
        candidate_words,
        dictionary_index,
        prefilter_limit,
        priority_keys=set(frequency_candidates) if short_cipher_mode and len(cleaned) <= SHORT_CIPHERTEXT_THRESHOLD else None,
        source_map=source_map,
    )
    return results, len(candidate_words), resolved_dictionary_path


def parse_key_length(value):
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    parsed = int(stripped)
    if parsed < 1:
        raise argparse.ArgumentTypeError("key length must be at least 1")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Try dictionary words as Vigenere keys and rank the best decryptions."
    )
    parser.add_argument("ciphertext", nargs="?", default="", help="Ciphertext to analyze")
    parser.add_argument(
        "--key-length",
        type=parse_key_length,
        default=None,
        help='Exact key length to search in the dictionary. Use "" or omit for unknown length.',
    )
    parser.add_argument(
        "--dictionary",
        help="Path to a dictionary file; defaults to project words.txt if present",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top results to show (default: 10)",
    )
    parser.add_argument(
        "--min-key-length",
        type=int,
        help="Minimum key length to consider when key length is unknown",
    )
    parser.add_argument(
        "--max-key-length",
        type=int,
        help="Maximum key length to consider when key length is unknown",
    )
    parser.add_argument(
        "--prefilter-limit",
        type=int,
        default=250,
        help="How many candidates survive the cheap first pass for full scoring (default: 250)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        help="Stop after this many dictionary words from the filtered candidate list",
    )
    parser.add_argument(
        "--output",
        help="Write the full result summary to a text file",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for inputs instead of requiring flags",
    )
    parser.add_argument(
        "--short-cipher-mode",
        choices=["on", "off"],
        default="on",
        help="Enable or disable the short-cipher frequency fallback (default: on)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text output",
    )
    parser.add_argument(
        "--best-only-json",
        action="store_true",
        help="Emit only the best result as compact JSON",
    )
    parser.add_argument(
        "--json-pretty",
        choices=["on", "off"],
        default="on",
        help="Pretty-print JSON output or emit compact JSON (default: on)",
    )
    parser.add_argument(
        "--csv",
        help="Write top candidates to a CSV file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the best plaintext",
    )
    parser.add_argument(
        "--explain-top",
        type=int,
        default=0,
        help="Show a longer explanation for the top N candidates",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the brief best-result summary",
    )
    parser.add_argument(
        "--explain-all",
        action="store_true",
        help="Show a longer explanation for all displayed candidates",
    )
    return parser.parse_args()


def prompt_int(prompt_text, allow_blank=False):
    while True:
        raw = input(prompt_text).strip()
        if raw == "" and allow_blank:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter a number.")


def run_interactive_prompt(args):
    if not args.ciphertext:
        args.ciphertext = input("Ciphertext: ").strip()
    if args.key_length is None:
        raw = input('Key length (blank for unknown): ').strip()
        args.key_length = None if raw == "" else int(raw)
    if args.key_length is None:
        if args.min_key_length is None:
            args.min_key_length = prompt_int("Minimum key length (blank for none): ", allow_blank=True)
        if args.max_key_length is None:
            args.max_key_length = prompt_int("Maximum key length (blank for none): ", allow_blank=True)
    if args.prefilter_limit == 250:
        entered = prompt_int("Prefilter limit [250]: ", allow_blank=True)
        if entered is not None:
            args.prefilter_limit = entered
    if args.max_words is None:
        args.max_words = prompt_int("Max dictionary words (blank for all): ", allow_blank=True)
    if args.top == 10:
        entered = prompt_int("Top results to show [10]: ", allow_blank=True)
        if entered is not None:
            args.top = entered
    if not args.output:
        output_path = input("Save results to file (blank to skip): ").strip()
        args.output = output_path or None
    return args


def build_output_text(mode_label, dictionary_path, key_length_label, words_checked, shortlist_size, best, results, top_n):
    lines = [
        f"Mode: {mode_label}",
        f"Dictionary file: {dictionary_path or 'built-in words'}",
        f"Requested key length: {key_length_label}",
        f"Dictionary words checked: {words_checked}",
        f"Full-score shortlist: {shortlist_size}",
        "",
        f"Best key: {best['key']}",
        f"Best source: {best['source']}",
        f"Decrypted text: {best['formatted_plaintext']}",
        f"Dictionary hits: {', '.join(best['dictionary_hits'][:8]) if best['dictionary_hits'] else 'none'}",
        f"Grammar score: {best['grammar_score']:.2f}",
        f"Chi penalty: {best['chi_penalty']:.2f}",
        f"Total score: {best['score']:.2f}",
        f"Rank reason: {best['rank_reason']}",
        f"Known-word bonus: {best['score_details']['known_word_bonus']:.2f}",
        f"Bigram hits: {', '.join(best['score_details']['bigram_hits'][:8]) if best['score_details']['bigram_hits'] else 'none'}",
        f"Trigram hits: {', '.join(best['score_details']['trigram_hits'][:8]) if best['score_details']['trigram_hits'] else 'none'}",
        "",
        "Top candidates:",
    ]
    for result in results[: max(1, top_n)]:
        hits = ",".join(result["dictionary_hits"][:5]) if result["dictionary_hits"] else "none"
        lines.append(
            f"  key={result['key']:<16} source={result['source']:<10} score={result['score']:.2f} "
            f"grammar={result['grammar_score']:.2f} known={result['score_details']['known_word_bonus']:.2f} hits={hits}"
        )
    return "\n".join(lines)


def build_summary_text(mode_label, key_length_label, best):
    return "\n".join(
        [
            f"Mode: {mode_label}",
            f"Requested key length: {key_length_label}",
            f"Best key: {best['key']}",
            f"Best source: {best['source']}",
            f"Decrypted text: {best['formatted_plaintext']}",
            f"Rank reason: {best['rank_reason']}",
        ]
    )


def build_explanations_text(results, explain_top):
    if explain_top <= 0:
        return ""

    lines = ["", "Detailed explanations:"]
    for index, result in enumerate(results[: max(1, explain_top)], start=1):
        details = result["score_details"]
        lines.extend(
            [
                f"  {index}. key={result['key']} source={result['source']}",
                f"     reason: {result['rank_reason']}",
                f"     plaintext: {result['formatted_plaintext']}",
                f"     exact_word_score={details['exact_word_score']:.2f}",
                f"     grammar_score={details['grammar_score']:.2f}",
                f"     bigram_score={details['bigram_score']:.2f}",
                f"     trigram_score={details['trigram_score']:.2f}",
                f"     known_word_bonus={details['known_word_bonus']:.2f}",
                f"     unknown_penalty={details['unknown_penalty']:.2f}",
                f"     prune_penalty={details['prune_penalty']:.2f}",
                f"     chi_penalty={details['chi_penalty']:.2f}",
                f"     exact_word_hits={', '.join(details['exact_word_hits']) if details['exact_word_hits'] else 'none'}",
                f"     bigram_hits={', '.join(details['bigram_hits']) if details['bigram_hits'] else 'none'}",
                f"     trigram_hits={', '.join(details['trigram_hits']) if details['trigram_hits'] else 'none'}",
            ]
        )
    return "\n".join(lines)


def build_rank_reason(candidate):
    reasons = []
    if candidate["score_details"]["known_word_bonus"] > 0:
        known_hits = [hit for hit in candidate["score_details"]["exact_word_hits"] if hit in KNOWN_WORD_BONUSES]
        if known_hits:
            reasons.append(f"known words matched: {', '.join(known_hits[:3])}")
    if candidate["score_details"]["trigram_hits"]:
        reasons.append(f"trigram hits: {', '.join(candidate['score_details']['trigram_hits'][:2])}")
    if candidate["score_details"]["bigram_hits"]:
        reasons.append(f"bigram hits: {', '.join(candidate['score_details']['bigram_hits'][:2])}")
    if candidate["score_details"]["exact_word_hits"]:
        reasons.append(f"exact words: {', '.join(candidate['score_details']['exact_word_hits'][:3])}")
    if not reasons:
        reasons.append("won on overall grammar and frequency score")
    return "; ".join(reasons)


def write_csv_output(path, results, top_n):
    fieldnames = [
        "rank",
        "key",
        "source",
        "key_length",
        "score",
        "formatted_plaintext",
        "rank_reason",
        "grammar_score",
        "chi_penalty",
        "known_word_bonus",
        "bigram_hits",
        "trigram_hits",
        "dictionary_hits",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results[: max(1, top_n)], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "key": result["key"],
                    "source": result["source"],
                    "key_length": result["key_length"],
                    "score": result["score"],
                    "formatted_plaintext": result["formatted_plaintext"],
                    "rank_reason": result["rank_reason"],
                    "grammar_score": result["grammar_score"],
                    "chi_penalty": result["chi_penalty"],
                    "known_word_bonus": result["score_details"]["known_word_bonus"],
                    "bigram_hits": " | ".join(result["score_details"]["bigram_hits"]),
                    "trigram_hits": " | ".join(result["score_details"]["trigram_hits"]),
                    "dictionary_hits": " | ".join(result["dictionary_hits"]),
                }
            )


def build_output_payload(mode_label, dictionary_path, key_length_label, words_checked, shortlist_size, best, results, top_n):
    return {
        "mode": mode_label,
        "dictionary_file": dictionary_path or "built-in words",
        "requested_key_length": key_length_label,
        "dictionary_words_checked": words_checked,
        "full_score_shortlist": shortlist_size,
        "best": best,
        "top_candidates": results[: max(1, top_n)],
    }


def main():
    args = parse_args()

    if args.interactive:
        args = run_interactive_prompt(args)

    if (
        args.key_length is None
        and args.min_key_length is not None
        and args.max_key_length is not None
        and args.min_key_length > args.max_key_length
    ):
        raise ValueError("min-key-length cannot be greater than max-key-length")

    if args.key_length is None:
        results, words_checked, dictionary_path = dictionary_attack_unknown_key_length(
            args.ciphertext,
            dictionary_path=args.dictionary,
            min_key_length=args.min_key_length,
            max_key_length=args.max_key_length,
            prefilter_limit=args.prefilter_limit,
            max_words=args.max_words,
            short_cipher_mode=(args.short_cipher_mode == "on"),
        )
        mode_label = "dictionary key search (unknown key length)"
        key_length_label = "unknown"
    else:
        results, words_checked, dictionary_path = dictionary_attack_with_key_length(
            args.ciphertext,
            args.key_length,
            dictionary_path=args.dictionary,
            prefilter_limit=args.prefilter_limit,
            max_words=args.max_words,
            short_cipher_mode=(args.short_cipher_mode == "on"),
        )
        mode_label = "dictionary key search"
        key_length_label = str(args.key_length)

    best = results[0]
    output_text = build_output_text(
        mode_label,
        dictionary_path,
        key_length_label,
        words_checked,
        min(args.prefilter_limit, words_checked),
        best,
        results,
        args.top,
    )
    summary_text = build_summary_text(mode_label, key_length_label, best)
    explain_count = args.top if args.explain_all else args.explain_top
    explain_text = build_explanations_text(results, explain_count)
    output_payload = build_output_payload(
        mode_label,
        dictionary_path,
        key_length_label,
        words_checked,
        min(args.prefilter_limit, words_checked),
        best,
        results,
        args.top,
    )
    best_only_payload = {
        "key": best["key"],
        "source": best["source"],
        "key_length": best["key_length"],
        "plaintext": best["formatted_plaintext"],
        "score": best["score"],
        "rank_reason": best["rank_reason"],
    }
    if args.quiet:
        final_output = best["formatted_plaintext"]
    elif args.best_only_json:
        final_output = json.dumps(best_only_payload, separators=(",", ":"))
    elif args.json:
        final_output = (
            json.dumps(output_payload, indent=2)
            if args.json_pretty == "on"
            else json.dumps(output_payload, separators=(",", ":"))
        )
    else:
        final_output = summary_text if args.summary_only else output_text
        if explain_text:
            final_output = final_output + explain_text
    print(final_output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(final_output + "\n")
        print()
        print(f"Saved results to: {args.output}")
    if args.csv:
        write_csv_output(args.csv, results, args.top)
        print()
        print(f"Saved CSV to: {args.csv}")


if __name__ == "__main__":
    main()
