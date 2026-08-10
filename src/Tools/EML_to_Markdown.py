"""Convert .eml email files to individual Markdown documents.

Each email is stored as a separate Markdown file. Reply emails containing
embedded conversation chains (quoted previous messages) are split so that
each conversation turn is stored individually. Deduplication ensures that
the same message appearing at multiple positions in an email chain is
recorded only once.  Deduplication also works across runs: existing
Markdown files in the output folder are scanned on startup so that
repeated invocations with overlapping email batches will not produce
duplicate output files.

Output filename format: YYYYMMDD_HHMMSS_nnnn_SenderName_SubjectSlug.md
Heading format: # Date - Sender Name - Subject

Usage:
    python EML_to_Markdown.py --input-folder /path/to/eml --output-folder /path/to/output
    python EML_to_Markdown.py  # interactive mode
"""
import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from Python_Lib.My_Lib_Stock import get_input_with_while_cycle


input_folder = r"E:\0 Obsidian_Notes_Private\0 RoboMat\ChemSpeed\Emails\eml"
output_folder = r"E:\0 Obsidian_Notes_Private\0 RoboMat\ChemSpeed\Emails\markdown"


REPLY_PREFIX_PATTERN = re.compile(r"^(?:(?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
MESSAGE_ID_PATTERN = re.compile(r"<[^>]+>")
WINDOWS_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*]')
WHITESPACE_PATTERN = re.compile(r"\s+")
QUOTE_LINE_PATTERN = re.compile(r"^\s*>")
REPLY_HISTORY_PATTERNS = [
	re.compile(r"^On .+wrote:$", re.IGNORECASE),
	re.compile(r"^.+<.+>\s+wrote:$", re.IGNORECASE),
	re.compile(r"^From:\s+.+$", re.IGNORECASE),
	re.compile(r"^Sent:\s+.+$", re.IGNORECASE),
	re.compile(r"^To:\s+.+$", re.IGNORECASE),
	re.compile(r"^Cc:\s+.+$", re.IGNORECASE),
	re.compile(r"^Subject:\s+.+$", re.IGNORECASE),
	re.compile(r"^-{2,}\s*Original Message\s*-{2,}$", re.IGNORECASE),
	re.compile(r"^_{4,}$"),
	re.compile(r"^在 .+写道[:：]$"),
	re.compile(r"^发件人[:：].+$"),
	re.compile(r"^发送时间[:：].+$"),
	re.compile(r"^收件人[:：].+$"),
	re.compile(r"^主题[:：].+$"),
]

CHAIN_ON_WROTE_EN = re.compile(r"^On\s+.+\s+wrote:\s*$", re.IGNORECASE)
CHAIN_ON_WROTE_ZH = re.compile(r"^在\s+.+\s*写道[:：]\s*$")
CHAIN_SEPARATOR = re.compile(
	r"^(?:-{2,}\s*Original Message\s*-{2,}|_{4,})$", re.IGNORECASE
)
CHAIN_HDR_FROM = re.compile(r"^(?:From|发件人)\s*[:：]\s*(.+)$", re.IGNORECASE)
CHAIN_HDR_SENT = re.compile(r"^(?:Sent|Date|日期|发送时间)\s*[:：]\s*(.+)$", re.IGNORECASE)
CHAIN_HDR_TO = re.compile(r"^(?:To|收件人)\s*[:：]\s*(.+)$", re.IGNORECASE)
CHAIN_HDR_CC = re.compile(r"^(?:Cc|抄送)\s*[:：]\s*(.+)$", re.IGNORECASE)
CHAIN_HDR_SUBJECT = re.compile(r"^(?:Subject|主题)\s*[:：]\s*(.+)$", re.IGNORECASE)


def _clean_line_for_match(line: str) -> str:
	"""Strip leading '>' quote markers and bold formatting for boundary matching."""
	result = re.sub(r'^(\s*>)+\s?', '', line)
	result = re.sub(r'\*([^*]+)\*', r'\1', result)
	return result.strip()


@dataclass
class AttachmentInfo:
	"""Metadata for a saved email attachment."""

	original_name: str
	saved_path: Path
	markdown_path: str
	content_type: str


@dataclass
class ChainSegment:
	"""A single message segment extracted from an embedded email reply chain."""

	sender: str
	date_str: str
	subject: str
	body: str


@dataclass
class EmailRecord:
	source_path: Path
	message_id: str
	subject: str
	normalized_subject: str
	sender: str
	to_line: str
	cc_line: str
	date_header: str
	timestamp: datetime
	in_reply_to: list[str]
	references: list[str]
	body_text: str
	attachments_payload: list[tuple[str, bytes, str]]
	source_index: int
	thread_index: int = -1
	attachments: list[AttachmentInfo] | None = None


class DisjointSet:
	"""Union-Find data structure for grouping emails into conversation threads."""

	def __init__(self, size: int):
		self.parent = list(range(size))
		self.rank = [0] * size

	def find(self, item: int) -> int:
		if self.parent[item] != item:
			self.parent[item] = self.find(self.parent[item])
		return self.parent[item]

	def union(self, left: int, right: int) -> None:
		left_root = self.find(left)
		right_root = self.find(right)
		if left_root == right_root:
			return
		if self.rank[left_root] < self.rank[right_root]:
			left_root, right_root = right_root, left_root
		self.parent[right_root] = left_root
		if self.rank[left_root] == self.rank[right_root]:
			self.rank[left_root] += 1


def _strip_wrapping_quotes(text: str) -> str:
	"""Remove surrounding single or double quotes from *text*."""
	return text.strip().strip('"').strip("'")


def _sanitize_filename(name: str, fallback: str = "item") -> str:
	"""Replace characters invalid in Windows filenames and collapse whitespace."""
	sanitized = WINDOWS_FILENAME_PATTERN.sub("_", name)
	sanitized = WHITESPACE_PATTERN.sub(" ", sanitized).strip(" .")
	return sanitized or fallback


def _slugify(text: str, fallback: str = "conversation") -> str:
	"""Convert *text* to an ASCII-safe slug suitable for filenames."""
	sanitized = _sanitize_filename(text, fallback=fallback)
	slug = re.sub(r"[^A-Za-z0-9._ -]", "_", sanitized)
	slug = slug.replace(" ", "_")
	slug = re.sub(r"_+", "_", slug).strip("._")
	return slug[:120] or fallback


def _extract_sender_display_name(sender: str) -> str:
	"""Extract the display name from a formatted sender header.

	For 'John Smith <john@example.com>', returns 'John Smith'.
	For 'Last, First <first.last@example.com>' (Outlook format), returns 'First'.
	For plain text 'Last, First', also returns 'First'.
	For '<john@example.com>' or 'john@example.com', returns 'john@example.com'.
	Returns 'unknown' when sender is empty.
	"""
	if not sender:
		return "unknown"
	if "<" not in sender and "," in sender:
		return _extract_first_name(sender)
	parsed = getaddresses([sender])
	# Prefer an entry that has a real email address (contains '@') — this
	# correctly handles "Last, First <addr>" which getaddresses splits into a
	# bogus ('', 'LastName') and a real ('FirstName', 'addr@host') entry.
	real_entries = [(name, addr) for name, addr in parsed if "@" in addr]
	candidates = real_entries if real_entries else parsed
	for name, addr in candidates:
		display = WHITESPACE_PATTERN.sub(" ", name).strip()
		if display:
			if "," in display:
				return _extract_first_name(display)
			return display
		if addr:
			return addr
	return sender.strip() or "unknown"


def _extract_first_name(display_name: str) -> str:
	"""Extract the first name from a display name for use in filenames.

	Handles two formats:
	- 'Last, First ...' (comma-separated, common in Outlook): takes the part
	  after the first comma, strips it, and returns the first word.
	- 'First Last ...' (space-separated): returns the first word.

	Examples:
	  'Goncalves, Liliana' -> 'Liliana'
	  'Wonjune Lee'        -> 'Wonjune'
	  'john@example.com'   -> 'john@example.com' (no change for email addresses)
	"""
	name = display_name.strip()
	if not name or name == "unknown":
		return name
	if "," in name:
		after_comma = name.split(",", 1)[1].strip()
		first_word = after_comma.split()[0] if after_comma.split() else name
		return first_word
	return name.split()[0] if name.split() else name


def _content_fingerprint(text: str) -> str:
	"""Compute a SHA-1 hex digest of normalized text for deduplication."""
	normalized = _normalize_for_match(text)
	return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _extract_message_ids(raw_value: str | None) -> list[str]:
	"""Extract all angle-bracket Message-ID values from a header string."""
	if not raw_value:
		return []
	ids = []
	for match in MESSAGE_ID_PATTERN.findall(raw_value):
		normalized = match.strip().lower()
		if normalized not in ids:
			ids.append(normalized)
	return ids


def _normalize_subject(subject: str) -> str:
	"""Strip reply/forward prefixes and normalize whitespace in a subject line."""
	cleaned = subject or "(no subject)"
	while True:
		updated = REPLY_PREFIX_PATTERN.sub("", cleaned).strip()
		if updated == cleaned:
			break
		cleaned = updated
	cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
	return cleaned or "(no subject)"


def _normalize_for_match(text: str) -> str:
	"""Collapse whitespace and lowercase *text* for content comparison."""
	return WHITESPACE_PATTERN.sub(" ", text.strip()).lower()


def _html_to_text(html_text: str) -> str:
	"""Convert HTML content to plain text, preserving basic structure."""
	text = html_text
	replacements = [
		(r"(?i)<br\s*/?>", "\n"),
		(r"(?i)</p>", "\n\n"),
		(r"(?i)</div>", "\n"),
		(r"(?i)</h[1-6]>", "\n\n"),
		(r"(?i)<li[^>]*>", "\n- "),
		(r"(?i)</li>", ""),
		(r"(?is)<script.*?</script>", ""),
		(r"(?is)<style.*?</style>", ""),
	]
	for pattern, replacement in replacements:
		text = re.sub(pattern, replacement, text)
	text = re.sub(r"(?is)<[^>]+>", "", text)
	text = re.sub(r"\r\n?", "\n", text)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()


def _extract_body(message) -> str:
	"""Extract the plain-text body from an email message, falling back to HTML."""
	plain_parts: list[str] = []
	html_parts: list[str] = []

	for part in message.walk():
		if part.is_multipart():
			continue
		disposition = (part.get_content_disposition() or "").lower()
		if disposition == "attachment":
			continue
		content_type = part.get_content_type().lower()
		try:
			payload = part.get_content()
		except Exception:
			payload_bytes = part.get_payload(decode=True) or b""
			charset = part.get_content_charset() or "utf-8"
			payload = payload_bytes.decode(charset, errors="replace")
		if not isinstance(payload, str):
			continue
		if content_type == "text/plain":
			plain_parts.append(payload)
		elif content_type == "text/html":
			html_parts.append(payload)

	if plain_parts:
		return "\n\n".join(part.strip() for part in plain_parts if part.strip()).strip()
	if html_parts:
		return "\n\n".join(_html_to_text(part) for part in html_parts if part.strip()).strip()
	return ""


def _remove_reply_markers(text: str) -> str:
	"""Remove quoted reply content and header-style markers from email body text."""
	lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
	kept_lines: list[str] = []

	for index, line in enumerate(lines):
		stripped = line.strip()
		if QUOTE_LINE_PATTERN.match(stripped):
			continue
		cleaned = _clean_line_for_match(line)
		if any(pattern.match(cleaned) for pattern in REPLY_HISTORY_PATTERNS):
			has_header_context = any(
				idx < len(lines) and lines[idx].strip() and ":" in lines[idx]
				for idx in range(index + 1, min(index + 5, len(lines)))
			)
			if has_header_context or stripped.lower().startswith("on "):
				break
		kept_lines.append(line)

	cleaned = "\n".join(kept_lines)
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	return cleaned.strip()


def _remove_duplicate_suffix(text: str, previous_bodies: Iterable[str]) -> str:
	"""Trim trailing content that duplicates a previously seen email body."""
	lines = text.splitlines()
	if len(lines) < 3:
		return text.strip()

	normalized_previous = []
	for previous in previous_bodies:
		normalized = [_normalize_for_match(line) for line in previous.splitlines() if _normalize_for_match(line)]
		if normalized:
			normalized_previous.append("\n".join(normalized))

	if not normalized_previous:
		return text.strip()

	normalized_lines = [_normalize_for_match(line) for line in lines]
	best_cut = None

	for start_index in range(1, len(lines)):
		candidate_lines = [line for line in normalized_lines[start_index:] if line]
		if len(candidate_lines) < 4:
			continue
		candidate_block = "\n".join(candidate_lines)
		if len(candidate_block) < 120:
			continue
		if any(candidate_block in previous for previous in normalized_previous):
			best_cut = start_index
			break

	if best_cut is None:
		return text.strip()
	trimmed = "\n".join(lines[:best_cut]).strip()
	return trimmed or text.strip()


def _cleanup_body(text: str, previous_bodies: Iterable[str] | None = None) -> str:
	"""Clean up email body text by removing reply markers and duplicate suffixes."""
	cleaned = text.replace("\ufeff", "").replace("\xa0", " ")
	cleaned = _remove_reply_markers(cleaned)
	if previous_bodies:
		cleaned = _remove_duplicate_suffix(cleaned, previous_bodies)
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	return cleaned.strip()


def _try_parse_embedded_headers(lines: list[str], start: int) -> dict[str, str] | None:
	"""Try to parse a block of email headers starting at line index *start*.

	Returns a dict with keys 'from', 'sent', 'to', 'cc', 'subject' (as found),
	plus '_body_start' indicating where the message body begins.
	Returns None when no valid header block is detected.
	"""
	header_patterns = [
		("from", CHAIN_HDR_FROM),
		("sent", CHAIN_HDR_SENT),
		("to", CHAIN_HDR_TO),
		("cc", CHAIN_HDR_CC),
		("subject", CHAIN_HDR_SUBJECT),
	]
	headers: dict[str, str] = {}
	i = start
	while i < len(lines) and not _clean_line_for_match(lines[i]):
		i += 1

	while i < len(lines):
		cleaned = _clean_line_for_match(lines[i])
		if not cleaned:
			i += 1
			if headers:
				break
			continue

		matched = False
		for key, pattern in header_patterns:
			m = pattern.match(cleaned)
			if m:
				headers[key] = m.group(1).strip()
				matched = True
				break

		if not matched:
			if headers:
				break
			return None
		i += 1

	if "from" not in headers:
		return None
	headers["_body_start"] = str(i)
	return headers


def _parse_on_wrote_line(line: str) -> dict[str, str]:
	"""Parse an 'On ... wrote:' or '在 ... 写道：' line.

	Returns a dict with 'from' and 'sent' keys extracted from the line.
	"""
	m = re.match(r"^On\s+(.+?)\s+wrote:\s*$", line, re.IGNORECASE)
	if m:
		full = m.group(1)
		email_match = re.search(r"(\S+\s+)?<([^>]+)>", full)
		if email_match:
			date_part = full[: email_match.start()].strip().rstrip(",")
			sender_part = email_match.group(0).strip()
			return {"from": sender_part, "sent": date_part}
		return {"from": "", "sent": full}

	m = re.match(r"^在\s+(.+?)\s*[,，]\s*(.+?)\s*写道[:：]\s*$", line)
	if m:
		return {"from": m.group(2).strip(), "sent": m.group(1).strip()}

	m = re.match(r"^在\s+(.+?)\s+写道[:：]\s*$", line)
	if m:
		return {"from": "", "sent": m.group(1).strip()}

	return {"from": "", "sent": ""}


def _split_email_chain(body_text: str) -> list[ChainSegment]:
	"""Split an email body into message segments from the reply chain.

	Detects embedded quoted emails (Outlook-style header blocks, Gmail-style
	'On ... wrote:' markers, and Chinese-style '在 ... 写道：') and extracts
	each previous message as a separate ChainSegment.

	Returns segments for embedded messages only (not the primary/top message).
	"""
	text = body_text.replace("\r\n", "\n").replace("\r", "\n")
	lines = text.split("\n")
	boundaries: list[tuple[int, dict[str, str]]] = []

	i = 0
	while i < len(lines):
		cleaned = _clean_line_for_match(lines[i])

		if CHAIN_SEPARATOR.match(cleaned):
			headers = _try_parse_embedded_headers(lines, i + 1)
			if headers:
				boundaries.append((i, headers))
				i = int(headers["_body_start"])
				continue

		if CHAIN_ON_WROTE_EN.match(cleaned):
			info = _parse_on_wrote_line(cleaned)
			info["_body_start"] = str(i + 1)
			boundaries.append((i, info))
			i += 1
			continue

		# Multi-line "On ... wrote:" — email clients often wrap long attribution lines
		if re.match(r'^On\s+', cleaned, re.IGNORECASE) and not CHAIN_ON_WROTE_EN.match(cleaned):
			for j in range(i + 1, min(i + 4, len(lines))):
				combined_parts = [cleaned]
				for k in range(i + 1, j + 1):
					combined_parts.append(_clean_line_for_match(lines[k]))
				combined = ' '.join(combined_parts)
				if CHAIN_ON_WROTE_EN.match(combined):
					info = _parse_on_wrote_line(combined)
					info["_body_start"] = str(j + 1)
					boundaries.append((i, info))
					i = j + 1
					break
			else:
				i += 1
			continue

		if CHAIN_ON_WROTE_ZH.match(cleaned):
			info = _parse_on_wrote_line(cleaned)
			info["_body_start"] = str(i + 1)
			boundaries.append((i, info))
			i += 1
			continue

		if CHAIN_HDR_FROM.match(cleaned):
			headers = _try_parse_embedded_headers(lines, i)
			if headers and sum(1 for k in headers if not k.startswith("_")) >= 3:
				boundaries.append((i, headers))
				i = int(headers["_body_start"])
				continue

		i += 1

	if not boundaries:
		return []

	segments: list[ChainSegment] = []
	for idx, (boundary_line, headers) in enumerate(boundaries):
		body_start = int(headers.get("_body_start", str(boundary_line + 1)))
		body_end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
		body_lines: list[str] = []
		for line in lines[body_start:body_end]:
			body_lines.append(re.sub(r"^>\s?", "", line))
		body = re.sub(r"\n{3,}", "\n\n", "\n".join(body_lines)).strip()
		body = _remove_reply_markers(body)

		segments.append(
			ChainSegment(
				sender=headers.get("from", ""),
				date_str=headers.get("sent", ""),
				subject=headers.get("subject", ""),
				body=body,
			)
		)

	return segments


def _format_address_header(header_value: str | None) -> str:
	"""Format an email address header into a semicolon-separated display string."""
	if not header_value:
		return ""
	parts = []
	for name, address in getaddresses([header_value]):
		display_name = WHITESPACE_PATTERN.sub(" ", name).strip()
		if display_name and address:
			parts.append(f"{display_name} <{address}>")
		elif address:
			parts.append(address)
		elif display_name:
			parts.append(display_name)
	return "; ".join(parts)


def _parse_timestamp(date_header: str | None, source_path: Path) -> datetime:
	"""Parse a Date header into a UTC datetime, falling back to file mtime."""
	if date_header:
		try:
			parsed = parsedate_to_datetime(date_header)
			if parsed.tzinfo is None:
				return parsed.replace(tzinfo=timezone.utc)
			return parsed.astimezone(timezone.utc)
		except Exception:
			pass
	modified_time = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
	return modified_time


def _extract_attachments(message) -> list[tuple[str, bytes, str]]:
	"""Extract all attachment payloads from an email message."""
	attachments: list[tuple[str, bytes, str]] = []
	for part in message.walk():
		if part.is_multipart():
			continue
		filename = part.get_filename()
		disposition = (part.get_content_disposition() or "").lower()
		if disposition != "attachment" and not filename:
			continue
		payload = part.get_payload(decode=True)
		if payload is None:
			continue
		content_type = part.get_content_type().lower()
		attachments.append((filename or "attachment", payload, content_type))
	return attachments


def _load_email_record(eml_path: Path, source_index: int) -> EmailRecord:
	"""Parse a single .eml file into an EmailRecord."""
	with eml_path.open("rb") as handle:
		message = BytesParser(policy=policy.default).parse(handle)

	raw_message_id = (message.get("Message-ID") or "").strip().lower()
	if not raw_message_id:
		hash_value = hashlib.sha1(str(eml_path).encode("utf-8")).hexdigest()[:16]
		raw_message_id = f"<synthetic-{hash_value}>"

	subject = WHITESPACE_PATTERN.sub(" ", str(message.get("Subject") or "(no subject)")).strip()
	return EmailRecord(
		source_path=eml_path,
		message_id=raw_message_id,
		subject=subject or "(no subject)",
		normalized_subject=_normalize_subject(subject),
		sender=_format_address_header(message.get("From")),
		to_line=_format_address_header(message.get("To")),
		cc_line=_format_address_header(message.get("Cc")),
		date_header=str(message.get("Date") or "").strip(),
		timestamp=_parse_timestamp(message.get("Date"), eml_path),
		in_reply_to=_extract_message_ids(message.get("In-Reply-To")),
		references=_extract_message_ids(message.get("References")),
		body_text=_extract_body(message),
		attachments_payload=_extract_attachments(message),
		source_index=source_index,
	)


def _group_threads(records: list[EmailRecord]) -> list[list[EmailRecord]]:
	"""Group EmailRecords into conversation threads by Message-ID references and subject."""
	if not records:
		return []

	dsu = DisjointSet(len(records))
	id_to_index = {record.message_id: index for index, record in enumerate(records)}

	for index, record in enumerate(records):
		for message_id in record.in_reply_to + record.references:
			linked_index = id_to_index.get(message_id)
			if linked_index is not None:
				dsu.union(index, linked_index)

	subject_buckets: dict[str, list[int]] = {}
	for index, record in enumerate(records):
		if record.in_reply_to or record.references:
			continue
		subject_buckets.setdefault(record.normalized_subject, []).append(index)

	for indexes in subject_buckets.values():
		if len(indexes) < 2:
			continue
		anchor = indexes[0]
		for other in indexes[1:]:
			dsu.union(anchor, other)

	components: dict[int, list[EmailRecord]] = {}
	for index, record in enumerate(records):
		components.setdefault(dsu.find(index), []).append(record)

	threads = []
	for thread_index, thread in enumerate(components.values()):
		ordered = sorted(thread, key=lambda item: (item.timestamp, item.source_index, item.source_path.name.lower()))
		for record in ordered:
			record.thread_index = thread_index
		threads.append(ordered)

	threads.sort(key=lambda items: (items[0].timestamp, items[0].normalized_subject))
	return threads


def _build_thread_slug(thread: list[EmailRecord]) -> str:
	"""Build a filesystem-safe slug identifying a conversation thread."""
	first_record = thread[0]
	prefix = first_record.timestamp.strftime("%Y%m%d")
	subject_slug = _slugify(first_record.normalized_subject)
	return f"{prefix}_{subject_slug}"


def _save_thread_attachments(thread: list[EmailRecord], output_path: Path, thread_slug: str) -> None:
	"""Save all attachments from a thread to disk and populate AttachmentInfo."""
	attachments_dir = output_path / "attachments" / thread_slug
	counter = 1
	for record in thread:
		saved_attachments: list[AttachmentInfo] = []
		timestamp_prefix = record.timestamp.strftime("%Y%m%d_%H%M%S")
		for original_name, payload, content_type in record.attachments_payload:
			safe_name = _sanitize_filename(original_name, fallback=f"attachment_{counter}")
			target_name = f"{timestamp_prefix}_{counter:02d}_{safe_name}"
			target_path = attachments_dir / target_name
			target_path.parent.mkdir(parents=True, exist_ok=True)
			while target_path.exists():
				counter += 1
				target_name = f"{timestamp_prefix}_{counter:02d}_{safe_name}"
				target_path = attachments_dir / target_name
			target_path.write_bytes(payload)
			relative_path = target_path.relative_to(output_path).as_posix()
			saved_attachments.append(
				AttachmentInfo(
					original_name=original_name,
					saved_path=target_path,
					markdown_path=quote(relative_path, safe="/._-"),
					content_type=content_type,
				)
			)
			counter += 1
		record.attachments = saved_attachments


def _format_timestamp(timestamp: datetime) -> str:
	"""Format a datetime as 'YYYY-MM-DD HH:MM:SS UTC'."""
	return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_single_email_file(
	output_dir: Path,
	file_counter: int,
	timestamp: datetime | None,
	date_str_fallback: str,
	sender_display: str,
	subject: str,
	sender_full: str,
	to_line: str,
	cc_line: str,
	message_id: str,
	source_path: str,
	body: str,
	attachments: list[AttachmentInfo] | None,
) -> Path:
	"""Write a single email as an individual Markdown file.

	Filename format: YYYYMMDD_HHMMSS_nnnn_SenderName_SubjectSlug.md
	Heading format: # Date - Sender Name - Subject
	"""
	if timestamp:
		date_prefix = timestamp.strftime("%Y%m%d_%H%M%S")
		display_time = _format_timestamp(timestamp)
	else:
		parsed_ts = None
		if date_str_fallback:
			try:
				parsed_ts = parsedate_to_datetime(date_str_fallback)
			except Exception:
				pass
		if parsed_ts:
			date_prefix = parsed_ts.strftime("%Y%m%d_%H%M%S")
			display_time = parsed_ts.strftime("%Y-%m-%d %H:%M:%S")
		else:
			date_prefix = "00000000_000000"
			display_time = date_str_fallback or "unknown date"

	safe_sender = _slugify(_extract_first_name(sender_display), fallback="unknown")[:30]
	subject_slug = _slugify(_normalize_subject(subject))[:60]
	filename = f"{date_prefix}_{file_counter:04d}_{safe_sender}_{subject_slug}.md"
	filepath = output_dir / filename

	lines = [
		f"# {display_time} - {sender_display} - {subject}",
		"",
		f"- From: {sender_full or sender_display}",
		f"- To: {to_line or 'unknown'}",
	]
	if cc_line:
		lines.append(f"- Cc: {cc_line}")
	lines.append(f"- Subject: {subject}")
	if message_id:
		lines.append(f"- Message-ID: {message_id}")
	if source_path:
		lines.append(f"- Source: {source_path}")
	if attachments:
		lines.append("- Attachments:")
		for att in attachments:
			lines.append(
				f"  - [{att.original_name}]({att.markdown_path}) ({att.content_type})"
			)
	lines.extend(["", body or "_No textual body extracted._", ""])

	filepath.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
	return filepath


def _load_existing_fingerprints(output_dir: Path) -> set[str]:
	"""Scan existing Markdown files in *output_dir* and return body fingerprints.

	The body is extracted by skipping the heading line (``# ...``) and all
	subsequent metadata lines (``- Key: ...``) plus surrounding blank lines.
	"""
	fingerprints: set[str] = set()
	for md_path in output_dir.glob("*.md"):
		try:
			text = md_path.read_text(encoding="utf-8")
		except Exception:
			continue
		lines = text.split("\n")
		# Skip heading, blank lines, and metadata (lines starting with "- ")
		body_start = 0
		in_metadata = False
		for idx, line in enumerate(lines):
			stripped = line.strip()
			if idx == 0 and stripped.startswith("# "):
				continue
			if stripped.startswith("- ") or stripped.startswith("  - "):
				in_metadata = True
				continue
			if in_metadata and not stripped:
				body_start = idx + 1
				break
			if not stripped:
				continue
			# Reached body without metadata block
			body_start = idx
			break
		body = "\n".join(lines[body_start:]).strip()
		if len(body) >= 10:
			fingerprints.add(_content_fingerprint(body))
	return fingerprints


def convert_eml_folder_to_markdown(input_folder_path: str, output_folder_path: str) -> list[dict[str, str | int]]:
	"""Convert all EML files in a folder to individual Markdown documents.

	Each email is written as a separate Markdown file.  Embedded conversation
	turns found in reply chains are also extracted and stored individually.

	Deduplication is applied both within and across runs: before writing,
	existing Markdown files in the output folder are scanned and their body
	content fingerprinted so that identical messages already present on disk
	are never written again.  This makes the function safe for incremental
	use — call it repeatedly with new or partially overlapping batches of
	EML files and only genuinely new content will be added.

	Returns a list of dicts with 'markdown_path', 'subject', and 'source_info' keys.
	"""
	source_dir = Path(os.path.abspath(_strip_wrapping_quotes(input_folder_path)))
	output_dir = Path(os.path.abspath(_strip_wrapping_quotes(output_folder_path)))

	if not source_dir.exists():
		raise FileNotFoundError(f"Input folder does not exist: {source_dir}")
	if not source_dir.is_dir():
		raise NotADirectoryError(f"Input path is not a folder: {source_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	eml_paths = sorted(source_dir.rglob("*.eml"), key=lambda path: path.as_posix().lower())
	if not eml_paths:
		return []

	records = [_load_email_record(eml_path, source_index=index) for index, eml_path in enumerate(eml_paths)]
	threads = _group_threads(records)

	seen_fingerprints: set[str] = _load_existing_fingerprints(output_dir)
	existing_md_count = sum(1 for _ in output_dir.glob("*.md"))
	file_counter = existing_md_count
	results: list[dict[str, str | int]] = []

	for thread in threads:
		thread_slug = _build_thread_slug(thread)
		_save_thread_attachments(thread, output_dir, thread_slug)
		previous_bodies: list[str] = []

		for record in thread:
			primary_body = _cleanup_body(record.body_text, previous_bodies=previous_bodies)
			if primary_body:
				previous_bodies.append(primary_body)

			fp = _content_fingerprint(primary_body) if primary_body else ""
			if fp and fp not in seen_fingerprints and len(primary_body) >= 10:
				seen_fingerprints.add(fp)
				file_counter += 1
				sender_name = _extract_sender_display_name(record.sender)
				filepath = _write_single_email_file(
					output_dir=output_dir,
					file_counter=file_counter,
					timestamp=record.timestamp,
					date_str_fallback="",
					sender_display=sender_name,
					subject=record.subject,
					sender_full=record.sender,
					to_line=record.to_line,
					cc_line=record.cc_line,
					message_id=record.message_id,
					source_path=record.source_path.as_posix(),
					body=primary_body,
					attachments=record.attachments,
				)
				results.append(
					{
						"markdown_path": str(filepath),
						"subject": record.subject,
						"source_info": f"primary from {record.source_path.name}",
					}
				)

			embedded_segments = _split_email_chain(record.body_text)
			for segment in embedded_segments:
				seg_body = segment.body.strip()
				if len(seg_body) < 10:
					continue
				seg_fp = _content_fingerprint(seg_body)
				if seg_fp in seen_fingerprints:
					continue
				seen_fingerprints.add(seg_fp)
				file_counter += 1
				seg_sender = (
					_extract_sender_display_name(segment.sender) if segment.sender else "unknown"
				)
				seg_subject = segment.subject or record.normalized_subject
				filepath = _write_single_email_file(
					output_dir=output_dir,
					file_counter=file_counter,
					timestamp=None,
					date_str_fallback=segment.date_str,
					sender_display=seg_sender,
					subject=seg_subject,
					sender_full=segment.sender,
					to_line="",
					cc_line="",
					message_id="",
					source_path=f"embedded in {record.source_path.name}",
					body=seg_body,
					attachments=None,
				)
				results.append(
					{
						"markdown_path": str(filepath),
						"subject": seg_subject,
						"source_info": f"embedded in {record.source_path.name}",
					}
				)

	return results


def _build_argument_parser():
	parser = argparse.ArgumentParser(
		description="Convert all EML files in a folder into individual markdown files, one per email."
	)
	parser.add_argument(
		"--input-folder",
		default=input_folder,
		help="Folder containing .eml files. Defaults to the configured input_folder.",
	)
	parser.add_argument(
		"--output-folder",
		default=output_folder,
		help="Folder where markdown files and attachments will be written. Defaults to the configured output_folder.",
	)
	return parser


def _interactive_main() -> list[dict[str, str | int]]:
	print("EML to Markdown")
	print("Input folder path. Submit an empty line to use the configured default.")
	input_values = get_input_with_while_cycle(input_prompt="  Input folder> ")
	print("Output folder path. Submit an empty line to use the configured default.")
	output_values = get_input_with_while_cycle(input_prompt="  Output folder> ")

	selected_input = input_values[0] if input_values else input_folder
	selected_output = output_values[0] if output_values else output_folder
	return convert_eml_folder_to_markdown(selected_input, selected_output)


def main() -> None:
	parser = _build_argument_parser()
	args = parser.parse_args()

	try:
		if len(sys.argv) > 1:
			results = convert_eml_folder_to_markdown(args.input_folder, args.output_folder)
		else:
			results = _interactive_main()
	except Exception as exc:
		print(f"Error during conversion: {exc}")
		raise SystemExit(1) from exc

	if not results:
		print("No EML files found.")
		return

	for result in results:
		print(
			f"Markdown output: {result['markdown_path']} | "
			f"Subject: {result['subject']} | "
			f"Source: {result['source_info']}"
		)


if __name__ == "__main__":
	main()