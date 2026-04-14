import pandas as pd
import unicodedata
import re

def clean_date(raw_date):
	"""
	Cleans a date string to return YYYY-MM-DD.
	Returns None if the date is NA.
	"""
	if pd.isna(raw_date): 
		return None
	return str(raw_date)[:10]

def clean_text(text):
	"""
	Normalizes text: lowercase, remove accents, keep only alphanumeric.
	Returns 'unknown' if text is NA/None/NaN.
	"""
	if pd.isna(text) or str(text).lower() == "none" or str(text).lower() == "nan": 
		return "unknown"
	
	text = str(text).lower()
	# Normalize special chars (e.g. 'ü' -> 'u')
	try:
		text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
	except:
		pass # Fallback if encoding fails
	
	# Remove non-alphanumeric chars
	text = re.sub(r'[^a-z0-9]', '', text)
	return text

def generate_hub_key(text):
	"""
	Creates a token-sorted, normalized key for Name Hubs.
	Alphabetizes words so 'Martin Kauter' and 'Kauter Martin' produce the exact same key.
	"""
	if pd.isna(text) or str(text).lower() in ["none", "nan", ""]: 
		return "unknown"
	
	text = str(text).lower()
	try:
		text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
	except:
		pass
	
	# Split into words, remove non-alphanumeric, sort alphabetically, and join
	words = text.split()
	clean_words = [re.sub(r'[^a-z0-9]', '', w) for w in words]
	clean_words = sorted([w for w in clean_words if w])
	
	return "".join(clean_words) if clean_words else "unknown"