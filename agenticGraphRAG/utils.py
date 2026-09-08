"""Normalization helpers shared by ingestion and identity resolution."""

import re
import unicodedata

import pandas as pd

def clean_date(raw_date):
	"""Return the date portion of a value as ``YYYY-MM-DD`` when available."""
	if pd.isna(raw_date):
		return None
	return str(raw_date)[:10]


def _ascii_lower(value):
	"""Normalize a value to lowercase ASCII without failing on unusual text."""
	text = str(value).lower()
	try:
		return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
	except (TypeError, UnicodeError):
		return text

def clean_text(text):
	"""Return a compact alphanumeric key, or ``unknown`` when text is missing."""
	if pd.isna(text) or str(text).lower() in {"none", "nan"}:
		return "unknown"
	return re.sub(r"[^a-z0-9]", "", _ascii_lower(text)) or "unknown"

def generate_hub_key(text):
	"""Create the token-sorted key used to group orthographic name variants."""
	if pd.isna(text) or str(text).lower() in {"none", "nan", ""}:
		return "unknown"

	words = _ascii_lower(text).split()
	clean_words = [re.sub(r"[^a-z0-9]", "", word) for word in words]
	clean_words = sorted([w for w in clean_words if w])
	return "".join(clean_words) if clean_words else "unknown"
