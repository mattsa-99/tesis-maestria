"""Genera docs/capitulos/08 - referencias.md (APA 7) con las obras del .bib citadas en los capítulos.

Una obra se considera citada si el apellido de su primer autor aparece en el texto a menos de
60 caracteres de su año de publicación. Uso: python docs/herramientas/generar_referencias.py
"""
import re
from pathlib import Path

BIB = Path("referencias/bibliografia.bib")
CAP = Path("docs/capitulos")
OUT = CAP / "08 - referencias.md"
ACCENTS = {r"\'": "́", r'\"': "̈", r"\v": "̌", r"\~": "̃", r"\`": "̀"}


def latex_to_text(s: str) -> str:
    s = s.replace(r"{\o}", "ø").replace(r"\&", "&").replace("--", "–")
    for cmd, mark in ACCENTS.items():
        s = re.sub(r"\{?" + re.escape(cmd) + r"\{?([A-Za-z])\}?\}?", lambda m: m.group(1) + mark, s)
    import unicodedata
    s = unicodedata.normalize("NFC", s)
    s = s.replace("`", "\u2018").replace("'", "\u2019")
    return re.sub(r"[{}]", "", s).strip()


def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {"_type": m.group(1).lower(), "_key": m.group(2).strip()}
        for f in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", m.group(3), re.M | re.S):
            fields[f.group(1).lower()] = f.group(2)
        entries.append(fields)
    return entries


def split_authors(raw: str) -> list[str]:
    return [a.strip() for a in re.split(r"\s+and\s+", raw.replace("\n", " "))]


def surname(author: str) -> str:
    author = latex_to_text(author)
    return author.split(",")[0].strip() if "," in author else author.split()[-1]


def apa_author(author: str) -> str:
    if author.strip() == "others":
        return "et al."
    if author.strip().startswith("{"):          # autor corporativo
        return latex_to_text(author)
    a = latex_to_text(author)
    last, first = (a.split(",", 1) + [""])[:2] if "," in a else (a.split()[-1], " ".join(a.split()[:-1]))
    initials = " ".join(p[0] + "." for p in re.split(r"[\s-]+", first.strip()) if p)
    return f"{last.strip()}, {initials}".strip(", ")


def apa_authors(raw: str) -> str:
    names = [apa_author(a) for a in split_authors(raw)]
    if names[-1] == "et al.":
        return ", ".join(names[:-1]) + ", et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " y " + names[-1]


def fmt(e: dict) -> str:
    t = e["_type"]
    auth = apa_authors(e.get("author", ""))
    year = e.get("year", "s. f.")
    title = latex_to_text(e.get("title", ""))
    tail = []
    if t == "article":
        vol = e.get("volume", "")
        num = f"({e['number']})" if "number" in e else ""
        src = f"*{latex_to_text(e.get('journal', ''))}*" + (f", *{vol}*{num}" if vol else "")
        if "pages" in e:
            src += f", {latex_to_text(e['pages'])}"
        ref = f"{auth} ({year}). {title}. {src}."
    elif t == "book":
        ed = f" ({e['edition']}.ª ed.)" if "edition" in e else ""
        ref = f"{auth} ({year}). *{title}*{ed}. {latex_to_text(e.get('publisher', ''))}."
    elif t == "inproceedings":
        pages = f" (pp. {latex_to_text(e['pages'])})" if "pages" in e else ""
        series = f" ({latex_to_text(e['series'])}, vol. {e['volume']})" if "series" in e else ""
        pub = f" {latex_to_text(e['publisher'])}." if "publisher" in e else ""
        ref = f"{auth} ({year}). {title}. En *{latex_to_text(e.get('booktitle', ''))}*{series}{pages}.{pub}"
    elif t == "techreport":
        ref = f"{auth} ({year}). *{title}* [Informe técnico]. {latex_to_text(e.get('institution', ''))}."
    else:
        how = latex_to_text(e.get("howpublished", ""))
        ref = f"{auth} ({year}). *{title}*." + (f" {how}." if how else "")
    if "doi" in e:
        tail.append(f"https://doi.org/{e['doi']}")
    elif "url" in e:
        tail.append(e["url"])
    elif "note" in e:
        tail.append(latex_to_text(e["note"]) + ".")
    return (ref + (" " + " ".join(tail) if tail else "")).replace("..", ".")


def main() -> None:
    text = "\n".join(p.read_text() for p in sorted(CAP.glob("0[0-7]*.md")))
    text = text.split("# Abstract")[0] + text.split("# Abstract")[1].split("# 1. Introducción", 1)[-1]
    cited, missing = [], []
    for e in parse_bib(BIB.read_text()):
        first = split_authors(e.get("author", ""))[0]
        name = "RAE" if "Real Academia" in first else surname(first)
        pat = re.escape(name) + r"[^\n]{0,60}?" + re.escape(e.get("year", "")) if name != "RAE" else r"RAE y ASALE, 2010"
        (cited if re.search(pat, text) else missing).append(e)
    cited.sort(key=lambda e: (surname(split_authors(e["author"])[0]).lower(), len(split_authors(e["author"])), e.get("year", "")))
    lines = ["# Referencias", "",
             "Lista generada a partir de `referencias/bibliografia.bib` con las obras citadas en el texto "
             "(formato APA 7). Las entradas marcadas en el .bib como no verificadas deben contrastarse con la fuente.", ""]
    lines += [fmt(e) for e in cited]
    OUT.write_text(lines[0] + "\n\n" + lines[2] + "\n\n" + "\n\n".join(lines[4:]) + "\n")
    print(f"citadas: {len(cited)} | en .bib sin citar: {[e['_key'] for e in missing]}")


if __name__ == "__main__":
    main()
