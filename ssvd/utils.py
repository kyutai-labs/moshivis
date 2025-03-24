"""Extra utils for annotations scripts, main for post-processing"""

import re
from functools import lru_cache
from typing import Dict, List, Pattern, Sequence, Tuple

PIXELPROSE_TRIM_CANDIDATES = (
    "The image is",
    "This image is",
    "The background is",
    "The text is in",
    "The font is",
    "The style of the image is",
    "This is a photograph",
)


def preprocess_pixelprose_captions(caption: str) -> Dict[str, str]:
    """Preprocess PixelProse captions"""
    caption = caption.strip()
    if caption.startswith("This image displays"):
        caption = caption[len("This image displays:") :].strip()

    caption = caption[0].upper() + caption[1:]
    sentences = [s.strip().replace("\n", " ") for s in caption.split(".")]
    sentences = [x for x in sentences if len(x) > 0]
    if len(sentences) > 0:
        for idx, sentence in enumerate(sentences[2:], 2):
            if any(sentence.startswith(c) for c in PIXELPROSE_TRIM_CANDIDATES):
                sentences = sentences[:idx]
                break
        if not sentences[-1].endswith("."):
            sentences[-1] += "."

    return {"caption": ". ".join(sentences)}


def maybe_shorten_caption(caption: str, max_cap_len: int = 1500) -> str:
    """Postprocess a caption to shorten it to a max number of characters (avoid OOM)"""
    if len(caption) < max_cap_len:
        shortened_cap = caption
    else:
        shortened_cap = ""
        for sentence in caption.split("."):
            if len(shortened_cap) + len(sentence) < max_cap_len:
                shortened_cap += sentence + "."
            else:
                break
            if shortened_cap[-2:] == "..":
                shortened_cap = shortened_cap[:-1]
        if not shortened_cap:
            shortened_cap = caption[:max_cap_len]
    return shortened_cap


@lru_cache
def compile_pattern(s: str) -> Pattern:
    """cached compile"""
    return re.compile(s)


@lru_cache
def get_replace_pattern() -> Pattern:
    """Light postprocessing of the LLM output"""
    left_right_replace = r'([*\s"]?)+'
    speaker_string = r"(Speaker [1-2]|Me|Question|Answer):(\s[(].+[)])?"
    pattern = re.compile(
        f'({left_right_replace + speaker_string + left_right_replace}|"$)'
    )
    return pattern


def get_strings_for_logging(
    s: List[Dict], length_q: int = 40, length_a: int = 160
) -> Tuple[str, str]:
    """Postprocess for logging"""
    q, a = "None", "None"

    if not s:
        return q, a

    if isinstance(s[0], dict):
        if "question" in s[0]:
            q, a = s[0]["question"], s[0]["answer"]
        elif "caption" in s[0]:
            q, a = s[0]["caption"], s[1]["caption"]
        elif "text" in s[0]:
            q, a = s[0]["text"], s[1]["text"]
        else:
            q, a = "None", "None"

    if isinstance(s[0], str):
        q, a = s[0], s[1]

    def __extend_string__(s: str, length: int) -> str:
        if len(s) < length:
            return s + " " * (length - len(s))
        return s[: length - 3] + "..."

    return __extend_string__(q, length=length_q), __extend_string__(a, length=length_a)


def sanitize_line(s: str) -> str:
    """Some extra post-processing on the lines"""
    if not isinstance(s, str):
        raise ValueError
    s = s.replace("*", "").strip()
    if s[0] in ['"', "'"]:
        s = s[1:]
    if s[-1] in ['"', "'"]:
        s = s[:-1]
    return s.strip()


def postprocess_synth_annot(
    uid: str,
    res: Dict[str, str] | List[str],
    idx: Dict[str, int],
    min_num_turns: int = 3,
    trim_first_question: bool = False,
) -> Sequence:
    """Postprocess synthetic annotations (jsonl contents) to the format
    which will be ultimately stored in the annotations database
    (one row per speaker turn)
    """
    rows = []
    try:
        speaker = 1  # MTCs always start with OTHER

        for turn, it in enumerate(res):
            speaker = int((turn % 2) == 0)

            if turn == 0 and speaker == 1 and trim_first_question:
                it = it[: it.find("?") + 1]

            # remove potentially repeated question in the answer given by SPEAKER_MAIN
            if speaker == 0:
                pos = it.find("?")
                if 0 <= pos < len(it) - 10:
                    it = it[pos + 1 :]

            # stop at very short answers
            if len(it) < 2:
                break

            # Otherwise, add the row
            rows.append((uid, idx[uid], turn, speaker, sanitize_line(it)))

        # Skip if dialogue is too short
        if len(rows) < min_num_turns:
            raise KeyError

        # Update for future generation
        idx[uid] += 1
    # if any error occured, skip this dialog entirely
    except (KeyError, ValueError):
        return []

    # Rows to write in the database
    return rows
