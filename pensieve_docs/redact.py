"""Secrets and personal data are blanked out right after parsing, before anything is cached, indexed or
written to graph.json. Your files on disk are never changed; only pensieve-docs' copy of the text is.

redact(text) -> (text, Counter of kinds). The replacement has no letters, so it can never become a concept.
"""
import re
from collections import Counter

MASK = "▒▒▒"


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(map(int, reversed(digits))):
        total += d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2)
    return total % 10 == 0


def _card(m):
    d = re.sub(r"\D", "", m.group(0))
    isbn = len(d) == 13 and d.startswith(("978", "979"))  # ISBN-13s often pass Luhn too
    return 13 <= len(d) <= 19 and len(set(d)) > 1 and not isbn and _luhn(d)


def _phone(m):
    s = m.group(0)
    d = re.sub(r"\D", "", s)
    if not 8 <= len(d) <= 15 or re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):  # not an ISO date
        return False
    # international (+61 …), bracketed area code ((08) …), trunk prefix (04xx …), or North American 3-3-4
    return s[0] in "+(0" or bool(re.fullmatch(r"\d{3}[ .-]\d{3}[ .-]\d{4}", s))


# (kind, pattern, group to blank or 0 for the whole match, extra check). Most specific first.
RULES = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), 0, None),
    ("api key", re.compile(r"\b(?:sk-ant-[\w-]{20,}|sk-[\w-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}"
                           r"|AKIA[0-9A-Z]{16}|xox[abposr]-[\w-]{10,}|AIza[\w-]{35}|glpat-[\w-]{20,})"), 0, None),
    ("token", re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}"), 0, None),  # JWT
    ("password", re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|passphrase)\b\s*[:=]\s*(\S+)"), 1, None),
    ("token", re.compile(r"(?i)\b(?:secret|api[_ -]?key|access[_ -]?token|auth[_ -]?token|token|client[_ -]?secret)"
                         r"\b\s*[:=]\s*[\"']?([\w+/=.-]{16,})"), 1, None),  # long values only: "token: a unit of text" stays
    ("password", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:([^\s@/]+)@"), 1, None),  # user:password@host
    ("card number", re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"), 0, _card),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), 0, None),
    ("phone", re.compile(r"(?<![\w/.-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}(?![\w/-]|\.\d)"),
     0, _phone),
]


def redact(text: str) -> tuple[str, Counter]:
    found = Counter()
    for kind, rx, group, check in RULES:
        def sub(m):
            if check and not check(m):
                return m.group(0)
            found[kind] += 1
            if not group:
                return MASK
            a, b = m.span(group)
            return m.group(0)[: a - m.start()] + MASK + m.group(0)[b - m.start():]

        text = rx.sub(sub, text)
    return text, found


if __name__ == "__main__":
    sample = """password: hunter2   api_key = sk-ant-api03-abcdefghijklmnopqrstuvwxyz
    Call Jane on +61 412 345 678 or (08) 8313 4455 or 555-123-4567; card 4111 1111 1111 1111.
    jane.doe@example.com, postgres://admin:s3cret@db.local/app, token: a unit of text.
    Built in 2024-06-12 with 400 500 600 homes, DOI 10.5555/grid.solar.2024, p. 12-14, 3.14159, ISBN 978-0-395-95920-6."""
    out, kinds = redact(sample)
    print(out, dict(kinds), sep="\n")
    for secret in ("hunter2", "sk-ant", "412 345", "8313", "555-123", "4111", "jane.doe", "s3cret"):
        assert secret not in out, secret
    for keep in ("a unit of text", "2024-06-12", "400 500 600", "10.5555/grid.solar.2024", "12-14", "3.14159", "978-0-395-95920-6", "Jane"):
        assert keep in out, keep
    print("ok")
