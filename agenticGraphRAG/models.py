import pandas as pd
import hashlib
from .utils import clean_text, clean_date, generate_hub_key

class SHABCompany:
	def __init__(self, uid, name, street=None, house_number=None, seat=None, zip_code=None, 
				 legal_form=None, deletion_date=None, purpose=None, 
				 capital_nominal=None, capital_paid=None, 
				 is_weak=False, source=None):
		
		# --- 1. Identity ---
		self.uid = uid
		self.name = name
		self.is_weak = is_weak
		self.source = source

		# --- 2. Full Address ---
		self.street = street
		self.house_number = house_number
		self.seat = seat
		self.zip_code = zip_code

		# --- 3. Legal & Status ---
		self.legal_form = legal_form
		self.deletion_date = deletion_date
		self.purpose = purpose

		# --- 4. Financials ---
		self.capital_nominal = capital_nominal
		self.capital_paid = capital_paid

	# ==========================================
	#   FACTORY 1: THE STRONG NODE (From CSV)
	# ==========================================
	@classmethod
	def from_row(cls, row):
		"""Creates a Strong Company from a CSV row (Official Register Data)."""
		
		# 1. Identity Extraction
		uid = cls._get_first(row, [
			'content_commonsActual_company_uid', 
			'content_debtor_companies_company_uid',
			'content_commonsNew_company_uid'
		])
		
		if pd.isna(uid):
			return None 

		name = cls._get_first(row, [
			'content_commonsActual_company_name',
			'content_debtor_companies_company_name',
			'content_commonsNew_company_name'
		])

		# 2. Address Extraction
		street = cls._get_first(row, [
			'content_commonsNew_company_address_street',
			'content_commonsActual_company_address_street',
			'content_debtor_companies_company_address_street'
		])
		
		house_number = cls._get_first(row, [
			'content_commonsNew_company_address_houseNumber',
			'content_commonsActual_company_address_houseNumber',
			'content_debtor_companies_company_address_houseNumber'
		])
		
		seat = cls._get_first(row, [
			'content_commonsNew_company_seat',
			'content_commonsActual_company_seat',
			'content_debtor_companies_company_address_town'
		])
		
		zip_code = cls._get_first(row, [
			'content_commonsNew_company_address_swissZipCode',
			'content_commonsActual_company_address_swissZipCode',
			'content_debtor_companies_company_address_swissZipCode'
		])

		# 3. Legal & Status Extraction
		legal_form = cls._get_first(row, [
			'content_commonsActual_company_legalForm',
			'content_debtor_companies_company_legalForm'
		])
		
		deletion_date = cls._get_first(row, [
			'content_transaction_delete_deletionDate',
			'content_proceedingRevocationDate'
		])

		purpose = cls._get_first(row, [
			'content_commonsNew_purpose',
			'content_commonsActual_purpose'
		])

		# 4. Financials Extraction
		capital_nominal = cls._get_first(row, [
			'content_commonsNew_capital_nominal',
			'content_commonsActual_capital_nominal'
		])
		
		capital_paid = cls._get_first(row, [
			'content_commonsNew_capital_paid',
			'content_commonsActual_capital_paid'
		])

		# 5. Return Instance (Strong)
		return cls(uid, name, street, house_number, seat, zip_code, 
				   legal_form, deletion_date, purpose, capital_nominal, capital_paid, 
				   is_weak=False, source='CSV')

	# ==========================================
	#   FACTORY 2: THE WEAK NODE (From Text)
	# ==========================================
	@classmethod
	def from_text(cls, name, event_id):
		"""Creates a Weak Company from an LLM extraction."""
		name = str(name).strip()
		
		# Generate Weak ID (Hash of name)
		unique_str = f"comp_{name.lower().strip()}"
		weak_id = "weak_comp_" + hashlib.md5(unique_str.encode()).hexdigest()[:12]
		
		# Return Instance (Weak - most fields are None)
		return cls(uid=weak_id, name=name, is_weak=True, source=f"Event_{event_id}")

	# ==========================================
	#           HELPERS
	# ==========================================
	@staticmethod
	def _get_first(row, cols):
		for c in cols:
			if c in row and pd.notna(row[c]):
				return row[c]
		return None

	def to_dict(self):
		full_addr = f"{self.street or ''} {self.house_number or ''}, {self.zip_code or ''} {self.seat or ''}".strip()
		
		return {
			'id': self.uid,
			'label': 'Company',
			'properties': {
				'name': self.name,
				'name_normalized': clean_text(self.name),
				'address': full_addr,
				'street': self.street,
				'city': self.seat,
				'zip': self.zip_code,
				'legal_form': self.legal_form,
				'deletion_date': self.deletion_date,
				'purpose': self.purpose,
				'capital_nominal': self.capital_nominal,
				'capital_paid': self.capital_paid,
				'is_weak': self.is_weak,
				'source': self.source
			}
		}

class SHABPerson:
	def __init__(self, pid, firstname, lastname, dob=None, origin=None, town=None, is_weak=False, source=None):
		self.id = pid
		self.firstname = firstname
		self.lastname = lastname
		self.dob = dob
		self.origin = origin
		self.town = town
		self.is_weak = is_weak
		self.source = source
		
		# Define self.name so it can be accessed in loops/exports
		self.name = f"{self.firstname or ''} {self.lastname or ''}".strip()
		
		# Calculate the Name Key for the Hub (Lastname + Firstname for sorting)
		# Define self.name so it can be accessed in loops/exports
		self.name = f"{self.firstname or ''} {self.lastname or ''}".strip()

		# Calculate the Name Key for the Hub (Alphabetized to handle token order issues)
		self.name_key = generate_hub_key(self.name)
	# ==========================================
	#   FACTORY 1: STRONG NODE (From CSV)
	# ==========================================
	@classmethod
	def from_row(cls, row):
		# [CRITICAL FIX] Universal Case-Insensitive Check
		# 1. Convert to string (handles NaNs safely)
		# 2. Lowercase it
		# 3. Check if 'person' is INSIDE the string (matches "Person", "NaturalPerson", "person")
		raw_type = str(row.get('content_debtor_selectType', '')).lower().strip()
		
		if 'person' not in raw_type:
			return None
			
		firstname = row.get('content_debtor_person_prename')
		lastname = row.get('content_debtor_person_name')
		origin = row.get('content_debtor_person_placeOfOrigin')
		town = row.get('content_debtor_person_addressSwitzerland_town')
		
		# 2. Date Cleaning
		raw_dob = row.get('content_debtor_person_dateOfBirth')
		dob = clean_date(raw_dob)

		# 3. VALIDATION (RELAXED)
		# We only require a Lastname. Firstname and DOB are optional.
		if pd.isna(lastname) or str(lastname).strip() == "":
			return None

		# 4. ID Generation (Deterministic)
		clean_fn = clean_text(firstname)
		clean_ln = clean_text(lastname)
		clean_orig = clean_text(origin)
		
		# Use "unknown" string for missing DOB to keep ID format consistent
		dob_str = dob if dob else "unknown"
		
		# ID Format: person_lastname_firstname_dob_origin
		pid = f"person_{clean_ln}_{clean_fn}_{dob_str}_{clean_orig}"

		return cls(pid, firstname, lastname, dob, origin, town, is_weak=False, source='CSV')

	# ==========================================
	#   FACTORY 2: WEAK NODE (From Text)
	# ==========================================
	@classmethod
	def from_text(cls, full_name, event_id):
		full_name = str(full_name).strip()
		parts = full_name.split(' ')
		
		if len(parts) >= 2:
			firstname = parts[0]
			lastname = " ".join(parts[1:]) 
		else:
			firstname = full_name
			lastname = ""
			
		# Weak ID Generation
		clean_name = clean_text(full_name)
		unique_str = f"{clean_name}_{event_id}"
		pid = "weak_" + hashlib.md5(unique_str.encode()).hexdigest()[:12]
		
		return cls(pid, firstname, lastname, dob=None, origin=None, town=None, is_weak=True, source=f"Event_{event_id}")

	def to_dict(self):
		return {
			'id': self.id,
			'label': 'Person',
			'properties': {
				'name': self.name,
				'name_normalized': clean_text(self.name),
				'dob': self.dob,
				'origin': self.origin,
				'town': self.town,
				'is_weak': self.is_weak,
				'source': self.source
			}
		}

class SHABEvent:
	def __init__(self, row):
		# --- 1. Identity & Metadata ---
		self.pub_id = row.get('meta_publicationNumber')
		self.date = row.get('meta_publicationDate')
		self.rubric = row.get('meta_rubric')
		self.sub_rubric = row.get('meta_subRubric')
		self.canton = row.get('meta_cantons')
		
		# --- 2. Details ---
		self.deadline = row.get('content_claimOfCreditors_entryDeadline')
		self.contact_point = row.get('content_contactPointForClaimAndAppeal')
		self.proceeding_type = row.get('content_proceeding_selectType')
		self.circulation_status = row.get('content_typeOfCirculation_selectType')
		
		# --- 3. Assets ---
		self.affected_land = row.get('content_affectedLand')
		self.auction_objects = row.get('content_auctionObjects')

		# --- 4. Changes ---
		self.changes = []
		if row.get('content_transaction_update_changements_capitalChanged_nominal') == True: self.changes.append('CAPITAL_CHANGE')
		if row.get('content_transaction_update_changements_nameChanged') == True: self.changes.append('NAME_CHANGE')
		if row.get('content_transaction_update_changements_seatChanged') == True: self.changes.append('RELOCATION')
		if row.get('content_transaction_update_changements_addressChanged') == True: self.changes.append('ADDRESS_CHANGE')
		if row.get('content_transaction_update_changements_purposeChanged') == True: self.changes.append('PURPOSE_CHANGE')

		# --- 5. Text Construction ---
		self.full_text = self._construct_text(row)

		# --- 6. CONNECTIONS ---
		self.connections = {}
		
		# A. Companies
		self.connections['SUBJECT'] = self._get_first(row, [
			'content_commonsActual_company_uid', 
			'content_debtor_companies_company_uid', 
			'content_commonsNew_company_uid'
		])
		self.connections['PARENT'] = self._get_first(row, ['content_commonsActual_headOffice_uid', 'content_commonsNew_headOffice_uid'])
		self.connections['SELLER'] = row.get('content_transmitting_companies_company_uid')
		self.connections['BUYER'] = row.get('content_acquiring_companies_company_uid')
		self.connections['DISSOLVED'] = row.get('content_dissolved_companies_company_uid')
		self.connections['ASSURANCE'] = row.get('content_assurance_companies_company_uid')

		# B. Person (USING THE NEW LOGIC)
		self.connections['DEBTOR_PERSON'] = self._generate_person_id(row)

	# --- HELPERS ---
	def _generate_person_id(self, row):
		# Replicates SHABPerson logic exactly
		if row.get('content_debtor_selectType') == 'person':
			dob = clean_date(row.get('content_debtor_person_dateOfBirth'))
			firstname = row.get('content_debtor_person_prename')
			lastname = row.get('content_debtor_person_name')
			origin = row.get('content_debtor_person_placeOfOrigin')
			
			if pd.notna(dob) and pd.notna(lastname):
				fn = clean_text(firstname)
				ln = clean_text(lastname)
				orig = clean_text(origin)
				return f"person_{ln}_{fn}_{dob}_{orig}"
		return None
	# -------------------------------------------------------------

	def _construct_text(self, row):
		# (Paste your previous "Universal" _construct_text method here)
		# For brevity, I am using the robust one we just finalized:
		parts = []
		title = self._get_first(row, ['content_title', 'meta_title_de', 'content_publicationTitle'])
		if title: parts.append(f"TITLE: {title}")
		if self.changes: parts.append(f"CHANGES: {', '.join(self.changes)}")
		body = self._get_first(row, ['content_publicationText', 'content_publication'])
		if body: parts.append(f"DETAILS: {body}")
		
		# Juicy Columns
		contact = row.get('content_contactPointForClaimAndAppeal')
		if pd.notna(contact): parts.append(f"CONTACT_POINT: {contact}")
		reg_office = row.get('content_registrationOffice')
		if pd.notna(reg_office): parts.append(f"AUTHORITY_OFFICE: {reg_office}")
		remarks = row.get('content_remarks')
		if pd.notna(remarks): parts.append(f"REMARKS: {remarks}")
		deadline_comment = row.get('content_claimOfCreditors_commentEntryDeadline')
		if pd.notna(deadline_comment): parts.append(f"DEADLINE_NOTE: {deadline_comment}")
		land = row.get('content_affectedLand')
		if pd.notna(land): parts.append(f"AFFECTED_LAND: {land}")
		auction_obj = row.get('content_auctionObjects')
		if pd.notna(auction_obj): parts.append(f"AUCTION_OBJECTS: {auction_obj}")
		liq_type = row.get('content_artOfLiquidation_selectType')
		if pd.notna(liq_type): parts.append(f"LIQUIDATION_TYPE: {liq_type}")
		resolution_date = row.get('content_resolutionDate')
		if pd.notna(resolution_date): parts.append(f"RESOLUTION_DATE: {resolution_date}")
		auth_name = row.get('content_authority_name')
		if pd.notna(auth_name): parts.append(f"AUTHORITY_DECISION_BY: {auth_name}")

		if not body and len(parts) <= 2: 
			 stage = self.circulation_status if pd.notna(self.circulation_status) else "Unknown"
			 proc_type = self.proceeding_type if pd.notna(self.proceeding_type) else "Unknown"
			 parts.append(f"STATUS_SUMMARY: Stage[{stage}] / Type[{proc_type}]")
		return " \n ".join(parts)

	def _get_first(self, row, cols):
		for c in cols:
			if c in row and pd.notna(row[c]): return row[c]
		return None

	def to_dict(self):
		return {
			'id': self.pub_id,
			'label': 'Event',
			'properties': {
				'date': self.date,
				'rubric': self.rubric,
				'sub_rubric': self.sub_rubric,
				'canton': self.canton,
				'full_text': self.full_text,
				'deadline': self.deadline,
				'changes': self.changes
			},
			'connections': self.connections
		}


class SHABEdge:
	def __init__(self, source_id, target_id, relation_type, properties=None):
		self.source = source_id
		self.target = target_id
		self.type = relation_type
		self.properties = properties if properties else {}

	def to_dict(self):
		base = {'source': self.source, 'target': self.target, 'type': self.type}
		if self.properties: base['properties'] = self.properties
		return base
