# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal markdown renderer for the visualizer.

The visualizer only needs a predictable subset of Markdown for agent prompts
and trajectory messages. Rendering server-side avoids depending on browser
CDN availability for core readability.
"""

from __future__ import annotations

import re
from html import escape
from urllib.parse import urlsplit

from markupsafe import Markup

_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)\s*([A-Za-z0-9_+.-]*)\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_UL_RE = re.compile(r"^(\s*)[-+*]\s+(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.DOTALL)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?:[^()\s]+|\([^()]*\))+)(?:\s+\"[^\"]*\")?\)")
_STRONG_RE = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_EM_STAR_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_EM_UNDER_RE = re.compile(r"(?<!_)_([^_\n]+)_(?!_)")
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)\|?\s*$")


def render_markdown(text: str) -> Markup:
    """Render a constrained Markdown subset into safe HTML."""
    if not text or not text.strip():
        return Markup("")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    return Markup(_render_blocks(lines))


def _render_blocks(lines: list[str]) -> str:
    html_parts: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            block_html, i = _consume_fence(lines, i, fence)
            html_parts.append(block_html)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            html_parts.append(f"<h{level}>{_render_inline(heading.group(2).strip())}</h{level}>")
            i += 1
            continue

        if _is_table(lines, i):
            block_html, i = _consume_table(lines, i)
            html_parts.append(block_html)
            continue

        if _HR_RE.match(line):
            html_parts.append("<hr>")
            i += 1
            continue

        if _BLOCKQUOTE_RE.match(line):
            block_html, i = _consume_blockquote(lines, i)
            html_parts.append(block_html)
            continue

        if _UL_RE.match(line) or _OL_RE.match(line):
            block_html, i = _consume_list(lines, i)
            html_parts.append(block_html)
            continue

        paragraph_lines: list[str] = []
        while i < len(lines):
            current = lines[i]
            if not current.strip():
                break
            if paragraph_lines and _starts_special_block(lines, i):
                break
            paragraph_lines.append(current.strip())
            i += 1
        html_parts.append("<p>{}</p>".format(_render_inline("\n".join(paragraph_lines))))

    return "".join(html_parts)


def _consume_fence(lines: list[str], start: int, opening: re.Match[str]) -> tuple[str, int]:
    fence_marker = opening.group(1)
    fence_char = fence_marker[0]
    fence_len = len(fence_marker)
    lang = opening.group(2).strip()

    code_lines: list[str] = []
    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(fence_char * fence_len):
            suffix = stripped[fence_len:]
            if not suffix or suffix.startswith(fence_char):
                i += 1
                break
        code_lines.append(line)
        i += 1

    language_attr = ""
    if lang:
        language_attr = f' class="language-{escape(lang, quote=True)}"'
    code_html = escape("\n".join(code_lines))
    return f"<pre><code{language_attr}>{code_html}</code></pre>", i


def _consume_blockquote(lines: list[str], start: int) -> tuple[str, int]:
    quote_lines: list[str] = []
    i = start
    while i < len(lines):
        current = lines[i]
        if not current.strip():
            quote_lines.append("")
            i += 1
            continue
        match = _BLOCKQUOTE_RE.match(current)
        if not match:
            break
        quote_lines.append(match.group(1))
        i += 1
    return f"<blockquote>{_render_blocks(quote_lines)}</blockquote>", i


def _consume_list(lines: list[str], start: int) -> tuple[str, int]:
    ordered = _OL_RE.match(lines[start]) is not None
    pattern = _OL_RE if ordered else _UL_RE
    other_pattern = _UL_RE if ordered else _OL_RE
    tag = "ol" if ordered else "ul"

    items: list[str] = []
    i = start
    while i < len(lines):
        match = pattern.match(lines[i])
        if not match:
            break

        indent = len(match.group(1))
        first_line = match.group(3) if ordered else match.group(2)
        item_lines = [first_line]
        i += 1

        while i < len(lines):
            current = lines[i]
            if not current.strip():
                break

            next_same = pattern.match(current)
            if next_same and len(next_same.group(1)) == indent:
                break

            next_other = other_pattern.match(current)
            if next_other and len(next_other.group(1)) == indent:
                break

            current_indent = len(current) - len(current.lstrip(" "))
            if current_indent > indent:
                item_lines.append(current[indent + 1 :].lstrip())
                i += 1
                continue
            break

        if len(item_lines) == 1:
            items.append(f"<li>{_render_inline(item_lines[0].strip())}</li>")
        else:
            nested_lines = [item_lines[0]] + [line.lstrip() for line in item_lines[1:]]
            items.append(f"<li>{_render_blocks(nested_lines)}</li>")

        while i < len(lines) and not lines[i].strip():
            i += 1
            if i < len(lines) and pattern.match(lines[i]):
                break

    return f"<{tag}>{''.join(items)}</{tag}>", i


def _consume_table(lines: list[str], start: int) -> tuple[str, int]:
    header_cells = _split_table_row(lines[start])
    i = start + 2
    body_rows: list[list[str]] = []
    while i < len(lines):
        current = lines[i]
        if not current.strip() or "|" not in current or _starts_special_block(lines, i):
            break
        row = _split_table_row(current)
        if len(row) != len(header_cells):
            break
        body_rows.append(row)
        i += 1

    thead = "".join(f"<th>{_render_inline(cell.strip())}</th>" for cell in header_cells)
    tbody = "".join(
        "<tr>"
        + "".join(f"<td>{_render_inline(cell.strip())}</td>" for cell in row)
        + "</tr>"
        for row in body_rows
    )
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>", i


def _starts_special_block(lines: list[str], index: int) -> bool:
    line = lines[index]
    return bool(
        _FENCE_RE.match(line)
        or _HEADING_RE.match(line)
        or _HR_RE.match(line)
        or _BLOCKQUOTE_RE.match(line)
        or _UL_RE.match(line)
        or _OL_RE.match(line)
        or _is_table(lines, index)
    )


def _is_table(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index]
    separator = lines[index + 1]
    if "|" not in header:
        return False
    return _TABLE_SEPARATOR_RE.match(separator) is not None


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _render_inline(text: str) -> str:
    parts: list[str] = []
    last = 0
    for match in _CODE_SPAN_RE.finditer(text):
        if match.start() > last:
            parts.append(_render_text_segment(text[last : match.start()]))
        parts.append(f"<code>{escape(match.group(2).strip())}</code>")
        last = match.end()
    if last < len(text):
        parts.append(_render_text_segment(text[last:]))
    return "".join(parts)


def _render_text_segment(text: str) -> str:
    escaped = escape(text)
    placeholders: list[str] = []

    def store(html: str) -> str:
        token = f"\x00{len(placeholders)}\x00"
        placeholders.append(html)
        return token

    def link_repl(match: re.Match[str]) -> str:
        label = escape(match.group(1))
        href = match.group(2)
        if not _is_safe_href(href):
            return label
        href_attr = escape(href, quote=True)
        return store(f'<a href="{href_attr}" target="_blank" rel="noreferrer">{label}</a>')

    escaped = _LINK_RE.sub(link_repl, escaped)
    escaped = _STRIKE_RE.sub(r"<del>\1</del>", escaped)
    escaped = _STRONG_RE.sub(r"<strong>\2</strong>", escaped)
    escaped = _EM_STAR_RE.sub(r"<em>\1</em>", escaped)
    escaped = _EM_UNDER_RE.sub(r"<em>\1</em>", escaped)
    escaped = escaped.replace("\n", "<br>\n")

    for index, html in enumerate(placeholders):
        escaped = escaped.replace(f"\x00{index}\x00", html)
    return escaped


def _is_safe_href(href: str) -> bool:
    stripped = href.strip()
    lowered = stripped.lower()
    if lowered.startswith(("javascript:", "vbscript:", "data:")):
        return False
    parts = urlsplit(stripped)
    return parts.scheme in {"", "http", "https", "mailto"}
