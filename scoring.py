"""Frozen original scoring plus existing format-tolerant sensitivity; no aliases added."""
import re

ALIASES = {
    "airplane": {"airplane", "aeroplane", "plane", "aircraft", "jet"},
    "banana": {"banana"},
    "baseball glove": {"baseball glove", "baseball mitt", "glove", "mitt"},
    "bed": {"bed"},
    "bench": {"bench"},
    "boat": {"boat", "sailboat", "motorboat", "ship", "vessel"},
    "bottle": {"bottle"},
    "bus": {"bus"},
    "cell phone": {"cell phone", "mobile phone", "smartphone", "phone", "cellphone"},
    "chair": {"chair", "armchair"},
    "clock": {"clock", "wall clock", "alarm clock"},
    "cup": {"cup", "mug", "coffee cup"},
    "dining table": {"dining table", "table", "dinner table"},
    "fire hydrant": {"fire hydrant", "hydrant"},
    "giraffe": {"giraffe"},
    "handbag": {"handbag", "purse"},
    "horse": {"horse"},
    "keyboard": {"keyboard"},
    "microwave": {"microwave", "microwave oven"},
    "motorcycle": {"motorcycle", "motorbike", "motor bike"},
    "oven": {"oven"},
    "refrigerator": {"refrigerator", "fridge"},
    "sink": {"sink", "washbasin", "wash basin"},
    "suitcase": {"suitcase"},
    "surfboard": {"surfboard", "surf board"},
    "teddy bear": {"teddy bear", "teddy", "stuffed bear"},
    "tennis racket": {"tennis racket", "tennis racquet", "racket", "racquet"},
    "toaster": {"toaster"},
    "train": {"train", "locomotive"},
    "truck": {"truck", "pickup truck", "delivery truck", "fire truck"},
    "tv": {"tv", "television", "television set", "monitor"},
    "vase": {"vase", "flower vase"},
    "zebra": {"zebra"},
    "backpack": {"backpack", "rucksack", "knapsack"},
    "bicycle": {"bicycle", "bike", "pedal bike"},
    "cake": {"cake", "birthday cake"},
    "cat": {"cat", "kitten"},
    "cow": {"cow", "cattle", "bull", "calf"},
    "dog": {"dog", "puppy", "puppy dog"},
    "sandwich": {"sandwich"},
    "umbrella": {"umbrella", "parasol"},
}

DISPUTED = {
    ("surfboard", "longboard"),
    ("tv", "screen"),
    ("truck", "van"),
    ("motorcycle", "bike"),
    ("handbag", "bag"),
    ("suitcase", "luggage"),
    ("sandwich", "burger"),
    ("sandwich", "hamburger"),
}

REFUSALS = {
    "unknown",
    "cannot determine",
    "cannot be determined",
    "can't determine",
    "unable to determine",
    "not enough information",
    "not sure",
    "unclear",
}

def normalize(text):
    value = (text or "").strip().lower()
    value = re.sub(r"[`*_]", "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    words = value.split()
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    if words and words[0] in {
        "red", "green", "blue", "yellow", "black", "white", "gray", "grey",
        "brown", "orange", "purple", "pink",
    }:
        words = words[1:]
    return " ".join(words)

def contains_alias(answer, aliases):
    return any(re.search(rf"\b{re.escape(alias)}\b", answer) for alias in aliases)

def is_unknown(answer):
    return not answer or any(term in answer for term in REFUSALS)

def classify_target(raw, target, donor):
    answer = normalize(raw)
    if is_unknown(answer):
        return answer, "unknown", False
    if answer in ALIASES.get(target, {target}):
        return answer, "target_correct", False
    disputed = (target, answer) in DISPUTED or (donor, answer) in DISPUTED
    if answer in ALIASES.get(donor, {donor}):
        return answer, "strict_donor", disputed
    if contains_alias(answer, ALIASES.get(donor, {donor})):
        return answer, "descriptive_donor", disputed
    return answer, "other_error", disputed

def classify_check(raw, donor):
    answer = normalize(raw)
    if is_unknown(answer):
        return answer, "unknown", False
    disputed = (donor, answer) in DISPUTED
    if answer in ALIASES.get(donor, {donor}):
        return answer, "donor_correct", disputed
    if contains_alias(answer, ALIASES.get(donor, {donor})):
        return answer, "descriptive_correct", disputed
    return answer, "incorrect", disputed

def classify_baseline(raw, target):
    answer = normalize(raw)
    if is_unknown(answer):
        return answer, "unknown", False
    if answer in ALIASES.get(target, {target}):
        return answer, "target_correct", False
    return answer, "other_error", (target, answer) in DISPUTED
def classify_sentence_target(raw, target, donor):
    answer = normalize(raw)
    if is_unknown(answer):
        return "unknown"
    target_match = contains_alias(answer, ALIASES.get(target, {target}))
    donor_match = contains_alias(answer, ALIASES.get(donor, {donor}))
    if target_match and donor_match:
        return "ambiguous_both"
    if target_match:
        return "target_correct"
    if donor_match:
        return "donor_follow"
    return "other_error"

def classify_sentence_single(raw, category, correct_label, incorrect_label):
    answer = normalize(raw)
    if is_unknown(answer):
        return "unknown"
    if contains_alias(answer, ALIASES.get(category, {category})):
        return correct_label
    return incorrect_label
