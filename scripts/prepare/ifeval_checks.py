"""IFEval-style constraint checks used by prepare_ifeval.py.

Maps the 25 IFEval constraint ids to natural-language criteria (``constraint_to_criterion``)
and verifies a response against a constraint programmatically (``verify_constraint``),
following the logic of the IFEval reference implementation (google-research/
instruction_following_eval, Apache-2.0). Requires ``nltk`` (punkt sentence tokenizer,
downloaded on first use) and ``langdetect`` (for the response-language constraint).
"""
from __future__ import annotations

import json
import re
from collections import Counter

import nltk
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.tokenize import RegexpTokenizer

DetectorFactory.seed = 0


# ── Templates ────────────────────────────────────────────────────────────────

CONSTRAINT_TEMPLATES = {
    # Keywords
    "keywords:existence": "Does the response include the keyword '{keyword}'?",
    "keywords:frequency": "Does the response mention the word '{keyword}' at least {frequency} times?",
    "keywords:forbidden_words": "Does the response avoid using any of these words: {forbidden_words}?",
    "keywords:letter_frequency": "Does the response contain the letter '{letter}' at least {let_frequency} times?",

    # Length
    "length_constraints:number_words": "Does the response contain {relation} {num_words} words?",
    "length_constraints:number_sentences": "Does the response contain {relation} {num_sentences} sentences?",
    "length_constraints:number_paragraphs": "Does the response contain {relation} {num_paragraphs} paragraphs?",
    "length_constraints:nth_paragraph_first_word": "Does the {num_paragraphs}th paragraph begin with the word '{first_word}'?",

    # Format
    "detectable_format:number_highlighted_sections": "Does the response contain {num_highlights} highlighted sections (wrapped in asterisks)?",
    "detectable_format:number_bullet_lists": "Does the response contain {num_bullets} bullet-point lists?",
    "detectable_format:json_format": "Is the response formatted as valid JSON?",
    "detectable_format:title": "Does the response include a title wrapped in double angular brackets?",
    "detectable_format:multiple_sections": "Does the response contain the specified sections with appropriate headings?",
    "detectable_format:constrained_response": "Is the response exactly one of the allowed options?",

    # Content
    "detectable_content:number_placeholders": "Does the response contain {num_placeholders} placeholder brackets like [address] or [name]?",
    "detectable_content:postscript": "Does the response end with a postscript section (P.S.)?",

    # Punctuation
    "punctuation:no_comma": "Does the response avoid using any commas?",

    # Case
    "change_case:capital_word_frequency": "Does the response contain {relation} {capital_frequency} fully capitalized words?",
    "change_case:english_capital": "Is the entire response written in uppercase?",
    "change_case:english_lowercase": "Is the entire response written in lowercase?",

    # Start/End
    "startend:end_checker": "Does the response end with the phrase '{end_phrase}'?",
    "startend:quotation": "Is the entire response wrapped in quotation marks?",

    # Combination
    "combination:two_responses": "Does the response contain two distinct responses separated by asterisks?",
    "combination:repeat_prompt": "Does the response begin by repeating the original prompt?",

    # Language
    "language:response_language": "Is the response written in {language}?",
}


# ── Template instantiation ───────────────────────────────────────────────────

def _parse_kwargs(kwargs):
    """Parse kwargs from various input formats."""
    if isinstance(kwargs, str):
        try:
            return json.loads(kwargs)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(kwargs, dict):
        return kwargs
    return {}


def _format_list(items):
    """Format a list as a comma-separated string with quotes."""
    if isinstance(items, str):
        # Already a string (sometimes happens in the data)
        return items
    if isinstance(items, list):
        return ", ".join(f"'{w}'" for w in items)
    return str(items)


def constraint_to_criterion(constraint_id: str, kwargs) -> str:
    """Map a constraint ID + kwargs to a natural-language VFM criterion.

    Args:
        constraint_id: e.g. "length_constraints:number_words"
        kwargs: dict or JSON string with parameters for the template

    Returns:
        Natural-language criterion, e.g.
        "Does the response contain at least 400 words?"
    """
    template = CONSTRAINT_TEMPLATES.get(constraint_id)
    if template is None:
        return f"Does the response satisfy the constraint '{constraint_id}'?"

    kw = _parse_kwargs(kwargs)

    # Handle special cases
    if constraint_id == "keywords:forbidden_words" and "forbidden_words" in kw:
        kw["forbidden_words"] = _format_list(kw["forbidden_words"])

    if constraint_id == "detectable_format:multiple_sections":
        # kwargs may have section_spliter (sic) and num_sections
        section_splitter = kw.get("section_spliter") or kw.get("section_splitter", "")
        num_sections = kw.get("num_sections", "")
        if section_splitter and num_sections:
            return (f"Does the response contain {num_sections} sections "
                    f"separated by '{section_splitter}'?")
        # Fall through to generic template

    try:
        return template.format(**kw)
    except (KeyError, IndexError):
        # Graceful fallback if kwargs don't match template
        return f"Does the response satisfy the constraint '{constraint_id}'?"


# ── Utility functions (matching IFEval instructions_util.py) ─────────────────

_word_tokenizer = RegexpTokenizer(r"\w+")


def count_words(text):
    """Count words using NLTK regex tokenizer (matches IFEval canonical)."""
    return len(_word_tokenizer.tokenize(text))


def count_sentences(text):
    """Count sentences using NLTK punkt tokenizer."""
    from nltk.tokenize import sent_tokenize
    try:
        return len(sent_tokenize(text))
    except LookupError:
        for res in ("punkt", "punkt_tab"):
            nltk.download(res, quiet=True)
        return len(sent_tokenize(text))


def _check_relation(value, threshold, relation):
    """Check a numeric relation: 'at least' or 'at most' / 'less than'."""
    relation = relation.lower().strip()
    if relation == "at least":
        return value >= threshold
    elif relation in ("at most", "less than"):
        return value <= threshold
    return value >= threshold  # default to at least


# ── Verification functions ───────────────────────────────────────────────────

def verify_keywords_existence(response, keyword=None, **kw):
    """Check if keyword appears in response (case-insensitive)."""
    if keyword is None:
        return True
    return re.search(re.escape(keyword), response, re.IGNORECASE) is not None


def verify_keywords_frequency(response, keyword=None, frequency=None,
                              relation="at least", **kw):
    """Check keyword frequency meets threshold."""
    if keyword is None or frequency is None:
        return True
    count = len(re.findall(re.escape(keyword), response, re.IGNORECASE))
    return _check_relation(count, int(frequency), relation)


def verify_keywords_forbidden_words(response, forbidden_words=None, **kw):
    """Check that no forbidden words appear in response."""
    if not forbidden_words:
        return True
    if isinstance(forbidden_words, str):
        forbidden_words = [w.strip() for w in forbidden_words.split(",")]
    for word in forbidden_words:
        if re.search(r"\b" + re.escape(word.strip()) + r"\b", response,
                     re.IGNORECASE):
            return False
    return True


def verify_keywords_letter_frequency(response, letter=None, let_frequency=None,
                                     relation="at least", **kw):
    """Check letter frequency in response."""
    if letter is None or let_frequency is None:
        return True
    count = Counter(response.lower())[letter.lower()]
    return _check_relation(count, int(let_frequency), relation)


def verify_length_number_words(response, num_words=None, relation="at least", **kw):
    """Check word count relation."""
    if num_words is None:
        return True
    return _check_relation(count_words(response), int(num_words), relation)


def verify_length_number_sentences(response, num_sentences=None,
                                   relation="at least", **kw):
    """Check sentence count relation."""
    if num_sentences is None:
        return True
    return _check_relation(count_sentences(response), int(num_sentences), relation)


def verify_length_number_paragraphs(response, num_paragraphs=None,
                                    relation="at least", **kw):
    """Check paragraph count. Paragraphs separated by double newlines."""
    if num_paragraphs is None:
        return True
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    return _check_relation(len(paragraphs), int(num_paragraphs), relation)


def verify_length_nth_paragraph_first_word(response, num_paragraphs=None,
                                           first_word=None, **kw):
    """Check first word of nth paragraph."""
    if num_paragraphs is None or first_word is None:
        return True
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    n = int(num_paragraphs)
    if n > len(paragraphs) or n < 1:
        return False
    paragraph = paragraphs[n - 1]
    words = paragraph.split()
    if not words:
        return False
    # Strip punctuation, compare case-insensitive
    actual = re.sub(r"[^\w]", "", words[0]).lower()
    expected = re.sub(r"[^\w]", "", first_word).lower()
    return actual == expected


def verify_format_highlighted_sections(response, num_highlights=None, **kw):
    """Check number of highlighted sections (wrapped in asterisks)."""
    if num_highlights is None:
        return True
    # Match *text* and **text** patterns
    highlights = re.findall(r"\*{1,2}[^\n\*]+\*{1,2}", response)
    # Filter out empty after stripping asterisks
    highlights = [h.strip("* ") for h in highlights if h.strip("* ")]
    return len(highlights) >= int(num_highlights)


def verify_format_bullet_lists(response, num_bullets=None, **kw):
    """Check number of bullet points."""
    if num_bullets is None:
        return True
    # Match lines starting with * or -
    star_bullets = re.findall(r"^\s*\*[^\*].*$", response, re.MULTILINE)
    dash_bullets = re.findall(r"^\s*-.*$", response, re.MULTILINE)
    total = len(star_bullets) + len(dash_bullets)
    return total >= int(num_bullets)


def verify_format_json(response, **kw):
    """Check if response is valid JSON."""
    # Strip markdown code block markers
    text = response.strip()
    for prefix in ["```json", "```Json", "```JSON", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def verify_format_title(response, **kw):
    """Check for title wrapped in double angular brackets <<title>>."""
    return bool(re.search(r"<<[^\n]+>>", response))


def verify_format_multiple_sections(response, section_spliter=None,
                                    num_sections=None, **kw):
    """Check for sections separated by section_spliter."""
    if section_spliter is None:
        section_spliter = "SECTION"
    if num_sections is None:
        return True
    pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
    parts = re.split(pattern, response)
    # Filter out empty parts
    parts = [p for p in parts if p.strip()]
    return len(parts) >= int(num_sections)


def verify_format_constrained_response(response, **kw):
    """Check if response is exactly one of the constrained options."""
    text = response.strip().lower()
    return any(opt in text for opt in [
        "my answer is yes.", "my answer is no.", "my answer is maybe."
    ])


def verify_content_number_placeholders(response, num_placeholders=None, **kw):
    """Check number of placeholder brackets like [address]."""
    if num_placeholders is None:
        return True
    placeholders = re.findall(r"\[.*?\]", response)
    return len(placeholders) >= int(num_placeholders)


def verify_content_postscript(response, **kw):
    """Check for postscript section (P.S.)."""
    text = response.lower()
    return bool(re.search(r"p\.?\s?s\.?", text, re.MULTILINE))


def verify_punctuation_no_comma(response, **kw):
    """Check that response contains no commas."""
    return "," not in response


def verify_case_capital_word_frequency(response, capital_frequency=None,
                                      relation="at least", **kw):
    """Check number of fully capitalized words."""
    if capital_frequency is None:
        return True
    words = _word_tokenizer.tokenize(response)
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return _check_relation(caps, int(capital_frequency), relation)


def verify_case_english_capital(response, **kw):
    """Check if entire response is uppercase."""
    return response.upper() == response


def verify_case_english_lowercase(response, **kw):
    """Check if entire response is lowercase."""
    return response.lower() == response


def verify_startend_end_checker(response, end_phrase=None, **kw):
    """Check if response ends with the specified phrase."""
    if end_phrase is None:
        return True
    text = response.strip().strip('"').lower()
    return text.endswith(end_phrase.lower())


def verify_startend_quotation(response, **kw):
    """Check if response is wrapped in quotation marks."""
    text = response.strip()
    return (text.startswith('"') and text.endswith('"') and len(text) >= 2)


def verify_combination_two_responses(response, **kw):
    """Check for two distinct responses separated by asterisks (******)."""
    parts = re.split(r"\*{6,}", response)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return False
    return parts[0] != parts[1]


def verify_combination_repeat_prompt(response, prompt=None, **kw):
    """Check if response begins by repeating the original prompt."""
    if prompt is None:
        return True
    return response.strip().lower().startswith(prompt.strip().lower())


def verify_language_response_language(response, language=None, **kw):
    """Check if response is written in the specified language."""
    if language is None:
        return True
    try:
        detected = detect(response)
    except LangDetectException:
        return True  # can't detect → assume pass (matches canonical)
    # Map common language names to ISO codes
    lang_map = {
        "en": "en", "english": "en",
        "es": "es", "spanish": "es",
        "fr": "fr", "french": "fr",
        "de": "de", "german": "de",
        "it": "it", "italian": "it",
        "pt": "pt", "portuguese": "pt",
        "zh": "zh-cn", "chinese": "zh-cn",
        "ja": "ja", "japanese": "ja",
        "ko": "ko", "korean": "ko",
        "ru": "ru", "russian": "ru",
        "ar": "ar", "arabic": "ar",
        "hi": "hi", "hindi": "hi",
        "nl": "nl", "dutch": "nl",
        "sv": "sv", "swedish": "sv",
        "pl": "pl", "polish": "pl",
        "tr": "tr", "turkish": "tr",
        "vi": "vi", "vietnamese": "vi",
        "th": "th", "thai": "th",
        "uk": "uk", "ukrainian": "uk",
        "cs": "cs", "czech": "cs",
        "ro": "ro", "romanian": "ro",
        "da": "da", "danish": "da",
        "fi": "fi", "finnish": "fi",
        "hu": "hu", "hungarian": "hu",
        "no": "no", "norwegian": "no",
        "id": "id", "indonesian": "id",
        "ms": "ms", "malay": "ms",
        "bg": "bg", "bulgarian": "bg",
        "hr": "hr", "croatian": "hr",
        "sk": "sk", "slovak": "sk",
        "sl": "sl", "slovenian": "sl",
        "lt": "lt", "lithuanian": "lt",
        "lv": "lv", "latvian": "lv",
        "et": "et", "estonian": "et",
        "he": "he", "hebrew": "he",
        "fa": "fa", "persian": "fa", "farsi": "fa",
        "ur": "ur", "urdu": "ur",
        "bn": "bn", "bengali": "bn",
        "ta": "ta", "tamil": "ta",
        "te": "te", "telugu": "te",
        "ml": "ml", "malayalam": "ml",
        "sw": "sw", "swahili": "sw",
    }
    expected = lang_map.get(language.lower(), language.lower())
    return detected == expected


# ── Registry ─────────────────────────────────────────────────────────────────

VERIFIERS = {
    "keywords:existence": verify_keywords_existence,
    "keywords:frequency": verify_keywords_frequency,
    "keywords:forbidden_words": verify_keywords_forbidden_words,
    "keywords:letter_frequency": verify_keywords_letter_frequency,
    "length_constraints:number_words": verify_length_number_words,
    "length_constraints:number_sentences": verify_length_number_sentences,
    "length_constraints:number_paragraphs": verify_length_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": verify_length_nth_paragraph_first_word,
    "detectable_format:number_highlighted_sections": verify_format_highlighted_sections,
    "detectable_format:number_bullet_lists": verify_format_bullet_lists,
    "detectable_format:json_format": verify_format_json,
    "detectable_format:title": verify_format_title,
    "detectable_format:multiple_sections": verify_format_multiple_sections,
    "detectable_format:constrained_response": verify_format_constrained_response,
    "detectable_content:number_placeholders": verify_content_number_placeholders,
    "detectable_content:postscript": verify_content_postscript,
    "punctuation:no_comma": verify_punctuation_no_comma,
    "change_case:capital_word_frequency": verify_case_capital_word_frequency,
    "change_case:english_capital": verify_case_english_capital,
    "change_case:english_lowercase": verify_case_english_lowercase,
    "startend:end_checker": verify_startend_end_checker,
    "startend:quotation": verify_startend_quotation,
    "combination:two_responses": verify_combination_two_responses,
    "combination:repeat_prompt": verify_combination_repeat_prompt,
    "language:response_language": verify_language_response_language,
}


def verify_constraint(constraint_id, response, prompt=None, **kwargs):
    """Verify a single constraint against a response.

    Args:
        constraint_id: e.g. "length_constraints:number_words"
        response: The text to verify
        prompt: Original prompt (needed for combination:repeat_prompt)
        **kwargs: Constraint parameters

    Returns:
        True if constraint is satisfied, False otherwise
    """
    verifier = VERIFIERS.get(constraint_id)
    if verifier is None:
        return None  # Unknown constraint type
    return verifier(response, prompt=prompt, **kwargs)
