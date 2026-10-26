"""
PetPulse food checker: rule-based allergy matching for food labels.

Compares ingredient text with the pet's owner-recorded allergies (and "without
X" / "no X" restrictions in its diet) using a small, explicit list of
ingredient names per allergen. Matching uses whole words, so "chickpea" never
matches "chicken". The result is explainable and testable; it is NOT a safety
check: finding no match never means a food is safe.
"""

import re

POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"  # a recorded allergen appears in the ingredients
NO_MATCH_DETECTED = "NO_MATCH_DETECTED"    # nothing matched, but safety is not established
UNABLE_TO_VERIFY = "UNABLE_TO_VERIFY"      # the text or the records aren't good enough to check

# Ingredient names that count as a match for each allergen group. Kept short and explicit.
ALIASES = {
    "chicken": ["chicken", "chicken meal", "chicken fat", "chicken broth", "chicken liver", "chicken by-product meal",
                "chicken by-products", "hydrolyzed chicken", "dehydrated chicken", "poultry", "poultry meal",
                "poultry fat", "poultry by-product meal"],
    "beef": ["beef", "beef meal", "beef broth", "beef liver", "beef fat", "beef tallow", "bovine"],
    "fish": ["fish", "fish meal", "fish oil", "salmon", "tuna", "whitefish", "cod", "anchovy", "sardine", "menhaden"],
    "dairy": ["milk", "dairy", "cheese", "whey", "casein", "lactose", "yogurt", "butter"],
    "egg": ["egg", "egg product", "dried egg"],
    "wheat": ["wheat", "wheat flour", "wheat gluten", "whole wheat"],
    "soy": ["soy", "soya", "soybean", "soybean meal", "soy protein"],
    "corn": ["corn", "maize", "corn meal", "corn gluten meal"],
    "pork": ["pork", "pork meal", "pork fat", "bacon", "ham", "lard"],
    "lamb": ["lamb", "lamb meal"],
    "turkey": ["turkey", "turkey meal"],
    "peanut": ["peanut", "peanut butter"],
}
# Other ways an owner might write the allergy itself.
GROUP_NAMES = {"milk": "dairy", "lactose": "dairy", "eggs": "egg", "poultry": "chicken", "gluten": "wheat",
               "soybean": "soy", "peanuts": "peanut"}
NO_ALLERGY = {"none", "no", "n/a", "na", "unknown", "nothing", "not recorded", "none known", "no known allergies",
              "no allergies", "not recorded by the owner"}
MIN_WORDS = 4  # less readable text than this can't be checked


def _normalize(text):
    text = (text or "").lower()
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # rejoin words split across lines: "chick-\nen"
    return re.sub(r"\s+", " ", text).strip()


def _group_for(term):
    term = GROUP_NAMES.get(term, term)
    if term in ALIASES:
        return term
    for group, aliases in ALIASES.items():
        if term in aliases:
            return group
    for group in ALIASES:  # e.g. "chicken (severe)"
        if re.search(rf"\b{group}\b", term):
            return group
    return None


def _clean_term(term):
    term = re.sub(r"\(.*?\)", " ", term.lower())
    term = re.sub(r"\b(allergic to|allergy to|allergy|allergies|intolerance|sensitive to|severe|mild)\b", " ", term)
    return re.sub(r"[^a-z\- ]", " ", term).strip(" -")


def parse_restrictions(allergies, diet=None):
    """
    Turn the owner's free-text allergies (e.g. "Chicken, beef") and diet
    restrictions ("Dry food without chicken") into a list of things to look for.
    """
    found = []
    parts = re.split(r"[,;/\n]|\band\b|&|\bor\b", allergies or "", flags=re.IGNORECASE)
    for part in parts:
        found.append((part, "Owner-recorded allergy"))
    for match in re.finditer(r"\b(?:without|no|avoid|avoids|free of)\s+([a-z][a-z \-]*?)(?=[,.;:)]|\band\b|$)",
                             (diet or "").lower()):
        found.append((match.group(1), "Diet restriction"))
    for match in re.finditer(r"\b([a-z]+)-free\b", (diet or "").lower()):
        found.append((match.group(1), "Diet restriction"))

    restrictions, seen = [], set()
    for raw, source in found:
        term = _clean_term(raw)
        if not term or term in NO_ALLERGY or len(term) < 3:
            continue
        group = _group_for(term)
        key = group or term
        if key in seen:
            continue
        seen.add(key)
        restrictions.append({"term": key, "source": source, "aliases": ALIASES.get(group, [term])})
    return restrictions


def _alias_pattern(alias):
    words = [re.escape(word) for word in re.split(r"[\s\-]+", alias)]
    # Whole words only; spaces or hyphens between words; simple plurals ("eggs", "by-products").
    return re.compile(r"(?<![a-z])" + r"[\s\-]+".join(words) + r"(?:e?s)?(?![a-z])")


def _evidence(text, start, end):
    """The ingredient item around a match (between commas, semicolons, or brackets)."""
    left = max(text.rfind(mark, 0, start) for mark in ",;:()[]") + 1
    rights = [position for position in (text.find(mark, end) for mark in ",;()[].") if position != -1]
    right = min(rights) if rights else len(text)
    return text[left:right].strip()


def check_allergens(ingredients_text, allergies, diet=None, reliable=True):
    """
    Compare ingredient text with a pet's recorded allergies and diet restrictions.

    reliable=False means the photo or OCR quality is poor; then "no match" isn't
    reported, only UNABLE_TO_VERIFY (a found conflict is always reported).

    Returns {"status", "restrictions", "matches", "reasons"}; each match has
    "allergy", "source", "phrase" (as written on the label), and "evidence".
    """
    text = _normalize(ingredients_text)
    restrictions = parse_restrictions(allergies, diet)

    matches = []
    for restriction in restrictions:
        covered = []
        for alias in sorted(restriction["aliases"], key=len, reverse=True):  # "chicken meal" before "chicken"
            for match in _alias_pattern(alias).finditer(text):
                if any(start <= match.start() < end for start, end in covered):
                    continue
                covered.append(match.span())
                matches.append((match.start(), {
                    "allergy": restriction["term"],
                    "source": restriction["source"],
                    "phrase": match.group(0),
                    "evidence": _evidence(text, *match.span()),
                }))
    matches = [match for _, match in sorted(matches, key=lambda pair: pair[0])]  # in label order

    reasons = []
    readable_words = len(re.findall(r"[a-z]{2,}", text))
    if matches:
        status = POTENTIAL_CONFLICT
        reasons.append("A recorded allergy or diet restriction appears to match the ingredient text.")
    elif not restrictions:
        status = UNABLE_TO_VERIFY
        reasons.append("No allergies or diet restrictions are recorded for this pet, so there is nothing to check against.")
    elif readable_words < MIN_WORDS:
        status = UNABLE_TO_VERIFY
        reasons.append("Too little readable ingredient text to check.")
    elif not reliable:
        status = UNABLE_TO_VERIFY
        reasons.append("The photo or text recognition quality is poor, so ingredients may be missing.")
    else:
        status = NO_MATCH_DETECTED
        reasons.append("None of the recorded allergies matched the text. This does not mean the food is safe.")
    if text and "ingredient" not in text:
        reasons.append("The word 'ingredients' wasn't found, so the full ingredient list may not be shown.")

    return {"status": status, "restrictions": restrictions, "matches": matches, "reasons": reasons}
