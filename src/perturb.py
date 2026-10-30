"""Meaning-preserving, (near-)invisible formatting perturbations.

Every perturbation is a function str -> str.  For code prompts, "prose-only" perturbations are applied only to
natural-language regions (instruction text, docstrings, comments) so the code stays syntactically valid; layout
perturbations (trailing spaces, CRLF, tabs, trailing newlines) are applied to the whole message.
"""
import io
import re
import tokenize

# ---------------------------------------------------------------- whitespace
def trailing_space(s):
    """Add a single space at the end of every line."""
    return "\n".join(l + " " for l in s.split("\n"))

def crlf(s):
    """Windows line endings."""
    return s.replace("\r\n", "\n").replace("\n", "\r\n")

def double_space(s):
    """Every single space between two non-space characters becomes two spaces."""
    return re.sub(r"(?<=\S) (?=\S)", "  ", s)

def trailing_newlines(s):
    """Two extra newlines after the message."""
    return s + "\n\n"

def nbsp(s):
    """Inter-word spaces become NO-BREAK SPACE (U+00A0) - visually identical."""
    return re.sub(r"(?<=\S) (?=\S)", "\u00a0", s)

def tabs(s):
    """Leading 4-space indentation groups become tab characters (code only)."""
    def fix(line):
        m = re.match(r"^((?:    )+)", line)
        return "\t" * (len(m.group(1)) // 4) + line[m.end():] if m else line
    return "\n".join(fix(l) for l in s.split("\n"))

# ---------------------------------------------------------------- punctuation
def curly_quotes(s):
    """Straight double quotes -> typographic “ ” (opening after start/space/bracket)."""
    s = re.sub(r'(^|[\s(\[{])"', "\\1\u201c", s)
    return s.replace('"', "\u201d")

def curly_apostrophe(s):
    """Straight single quotes / apostrophes -> ‘ ’ (most become the right single quote U+2019)."""
    s = re.sub(r"(^|[\s(\[{])'(?!(?:s|t|re|ve|ll|d|m)\b)(?=\w)", "\\1\u2018", s)   # not before 's 't 're ...
    return s.replace("'", "\u2019")

def unicode_dash(s):
    """'--' -> em dash, spaced ' - ' -> ' – ' (en dash), digit-digit ranges -> en dash."""
    s = s.replace("--", "\u2014")
    s = s.replace(" - ", " \u2013 ")
    return re.sub(r"(?<=\d)-(?=\d)", "\u2013", s)

def ellipsis_char(s):
    """'...' -> single ellipsis character U+2026."""
    return s.replace("...", "\u2026")

def punct_all(s):
    return ellipsis_char(unicode_dash(curly_apostrophe(curly_quotes(s))))

WS = ["trailing_space", "crlf", "double_space", "trailing_newlines", "nbsp"]
WS_CODE = ["trailing_space", "crlf", "double_space", "trailing_newlines", "tabs"]
PUNCT = ["curly_quotes", "curly_apostrophe", "unicode_dash", "ellipsis_char", "punct_all"]
FUNCS = {k: globals()[k] for k in WS + ["tabs"] + PUNCT}
FAMILY = {**{k: "whitespace" for k in WS + ["tabs"]}, **{k: "punctuation" for k in PUNCT},
          "canonical": "canonical", "control": "control"}
# perturbations that must only touch natural-language regions inside code
PROSE_ONLY = {"double_space", "nbsp"} | set(PUNCT)


def prose_spans(code):
    """(start, end) character spans of docstrings and comments in `code` (python source), skipping doctest lines."""
    spans = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return spans
    starts = [0]
    for line in code.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    off = lambda rc: starts[rc[0] - 1] + rc[1]
    for t in toks:
        if t.type == tokenize.COMMENT:
            spans.append((off(t.start), off(t.end)))
        elif t.type == tokenize.STRING and t.string.lstrip("rbuRBU").startswith(('"""', "'''")):
            a, b = off(t.start) + 3, off(t.end) - 3            # inside the triple quotes
            # split into lines, skip doctest example lines (">>>") and their expected-output line
            pos, skip_next = a, False
            for line in code[a:b].splitlines(keepends=True):
                st = line.strip()
                if st.startswith(">>>") or (skip_next and st):
                    skip_next = st.startswith(">>>")
                else:
                    skip_next = False
                    spans.append((pos, pos + len(line)))
                pos += len(line)
    return spans


def apply_to_code(name, code):
    """Apply perturbation `name` to python source; prose-only perturbations touch docstrings/comments only."""
    f = FUNCS[name]
    if name not in PROSE_ONLY:
        return f(code)
    out, last = [], 0
    for a, b in sorted(prose_spans(code)):
        out.append(code[last:a]); out.append(f(code[a:b])); last = b
    out.append(code[last:])
    return "".join(out)
