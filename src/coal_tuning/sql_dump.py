from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InsertBlock:
    table: str
    columns: list[str]
    rows: list[list[object | None]]


def parse_mysql_dump(path: str | Path) -> dict[str, list[dict[str, object | None]]]:
    text = Path(path).read_text(encoding="utf-8")
    tables: dict[str, list[dict[str, object | None]]] = {}
    pos = 0
    marker = "INSERT INTO `"
    while True:
        start = text.find(marker, pos)
        if start < 0:
            break
        stmt_end = _find_statement_end(text, start)
        if stmt_end < 0:
            break
        stmt = text[start : stmt_end + 1]
        block = parse_insert(stmt)
        if block:
            rows = tables.setdefault(block.table, [])
            rows.extend(dict(zip(block.columns, row)) for row in block.rows)
        pos = stmt_end + 1
    return tables


def parse_insert(stmt: str) -> InsertBlock | None:
    prefix = "INSERT INTO `"
    if not stmt.startswith(prefix):
        return None
    table_end = stmt.find("`", len(prefix))
    table = stmt[len(prefix) : table_end]

    col_start = stmt.find("(", table_end)
    col_end = stmt.find(")", col_start)
    columns = [c.strip().strip("`") for c in stmt[col_start + 1 : col_end].split(",")]

    values_kw = "\nVALUES\n"
    values_start = stmt.find(values_kw, col_end)
    if values_start < 0:
        values_kw = " VALUES "
        values_start = stmt.find(values_kw, col_end)
    if values_start < 0:
        return None
    values_text = stmt[values_start + len(values_kw) :].rstrip().rstrip(";")
    rows = _parse_values(values_text)
    return InsertBlock(table=table, columns=columns, rows=rows)


def _find_statement_end(text: str, start: int) -> int:
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_string = False
        else:
            if ch == "'":
                in_string = True
            elif ch == ";":
                return i
    return -1


def _parse_values(values_text: str) -> list[list[object | None]]:
    rows: list[list[object | None]] = []
    i = 0
    n = len(values_text)
    while i < n:
        while i < n and values_text[i] in " \n\r\t,":
            i += 1
        if i >= n:
            break
        if values_text[i] != "(":
            i += 1
            continue
        row, i = _parse_row(values_text, i + 1)
        rows.append(row)
    return rows


def _parse_row(text: str, i: int) -> tuple[list[object | None], int]:
    row: list[object | None] = []
    token: list[str] = []
    in_string = False
    escaped = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if escaped:
                token.append(_unescape_char(ch))
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_string = False
            else:
                token.append(ch)
            i += 1
            continue

        if ch == "'":
            in_string = True
            i += 1
            continue
        if ch == ",":
            row.append(_convert_token("".join(token)))
            token = []
            i += 1
            continue
        if ch == ")":
            row.append(_convert_token("".join(token)))
            return row, i + 1
        token.append(ch)
        i += 1
    return row, i


def _unescape_char(ch: str) -> str:
    return {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "0": "\0",
        "\\": "\\",
        "'": "'",
        '"': '"',
    }.get(ch, ch)


def _convert_token(raw: str) -> object | None:
    token = raw.strip()
    if token.upper() == "NULL":
        return None
    if token == "":
        return ""
    return token

