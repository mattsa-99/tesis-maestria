"""Convierte las fórmulas de los capítulos a la sintaxis que GitHub renderiza sin errores (y de vuelta).

Por qué hace falta: GitHub procesa el Markdown (cursivas, escapes con barra invertida) ANTES de reconocer
las fórmulas escritas con `$...$`. Eso rompe `\\{`, `\\,`, los pares de `_` o `*` y las fórmulas dentro de cursivas.
GitHub documenta una sintaxis que evita ese procesamiento:

    en línea:      $`x_{i}`$          (en lugar de  $x_{i}$)
    destacadas:    ```math ... ```    (en lugar de  $$ ... $$)

Uso:
    python docs/herramientas/compatibilizar_github.py             # informa, no modifica
    python docs/herramientas/compatibilizar_github.py --aplicar   # estándar $...$  ->  sintaxis de GitHub
    python docs/herramientas/compatibilizar_github.py --revertir  # sintaxis de GitHub -> estándar $...$ y $$...$$
                                                                   # (para VS Code, pandoc sin tex_math_gfm, etc.)

Las fórmulas que quedaban dentro de cursivas (`*...$k$...*`) se sacan de ellas, porque GitHub no las renderiza.
En las fórmulas en línea, `<` y `>` pasan a `\\lt` y `\\gt` (GitHub los escapa dos veces con la sintaxis de comillas).
Ambas conversiones son idempotentes.
"""
import re
import sys
from pathlib import Path

FILES = sorted(Path("docs/capitulos").glob("0*.md"))
# Solo se protegen bloques de código, código en línea y destinos de enlaces. NO las etiquetas HTML:
# el `<` y el `>` de dos fórmulas distintas se confundirían con una etiqueta.
PROTECT = re.compile(r"(```(?!math).*?```|`[^`\n]*`|\]\([^)\n]*\))", re.S)
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![$`])\$(?![\s$`])([^$\n]+?)(?<![\s$`])\$(?![$])")
ITALIC = re.compile(r"(?<![\*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\*\w])")
PH = re.compile(r"(\x00[MD]\d+\x00)")


def _split_italics(line: str) -> str:
    def repl(m: re.Match) -> str:
        if "\x00" not in m.group(1):
            return m.group(0)
        out = []
        for seg in PH.split(m.group(1)):
            if PH.fullmatch(seg) or not seg.strip():
                out.append(seg)
            else:
                lead, trail = seg[: len(seg) - len(seg.lstrip())], seg[len(seg.rstrip()):]
                out.append(f"{lead}*{seg.strip()}*{trail}")
        return "".join(out)
    return ITALIC.sub(repl, line)


def a_github(text: str) -> str:
    text = a_estandar(text)          # parte siempre de la forma estándar: aplicar dos veces no cambia nada
    prot, math = [], []
    text = PROTECT.sub(lambda m: (prot.append(m.group(0)), f"\x01P{len(prot) - 1}\x01")[1], text)
    text = DISPLAY.sub(lambda m: (math.append(("D", m.group(1).strip())), f"\x00D{len(math) - 1}\x00")[1], text)
    text = INLINE.sub(lambda m: (math.append(("M", m.group(1))), f"\x00M{len(math) - 1}\x00")[1], text)
    text = "\n".join(_split_italics(l) for l in text.split("\n"))

    def back(m: re.Match) -> str:
        kind, body = math[int(m.group(2))]
        if kind == "D":
            return f"\n\n```math\n{body}\n```\n\n"
        # GitHub escapa dos veces < y > dentro de $`...`$ y se ven como "&lt;"; \lt y \gt se dibujan igual.
        body = body.replace("<", r"\lt ").replace(">", r"\gt ")
        return f"$`{body}`$"
    text = re.sub(r"\x00([MD])(\d+)\x00", back, text)
    text = re.sub(r"[ \t]*\n*(```math\n.*?\n```)[ \t]*\n*", r"\n\n\1\n\n", text, flags=re.S)
    return re.sub(r"\x01P(\d+)\x01", lambda m: prot[int(m.group(1))], text)


def a_estandar(text: str) -> str:
    text = re.sub(r"```math\n(.*?)\n```", lambda m: "$$" + m.group(1) + "$$", text, flags=re.S)
    return re.sub(r"\$`([^`\n]+)`\$", lambda m: "$" + m.group(1) + "$", text)


def convertir(name: str, txt: str, f) -> str:
    if name.startswith("00 - resumen"):
        es, en = txt.split("# Abstract", 1)
        return f(es) + "# Abstract" + f(en)
    return f(txt)


if __name__ == "__main__":
    modo = "aplicar" if "--aplicar" in sys.argv else "revertir" if "--revertir" in sys.argv else None
    fn = a_estandar if modo == "revertir" else a_github
    total = 0
    for p in FILES:
        cur = p.read_text()
        new = convertir(p.name, cur, fn)
        assert convertir(p.name, new, fn) == new, f"no idempotente: {p.name}"
        n = len(re.findall(r"\$`|```math|\$\$|(?<![`$])\$(?![`$])", new))
        print(f"{p.name}: {'modificado' if new != cur else 'sin cambios'}")
        total += new != cur
        if modo and new != cur:
            p.write_text(new)
    print("archivos modificados:" if modo else "archivos que se modificarían:", total)
