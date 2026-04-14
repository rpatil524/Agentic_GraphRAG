# Constants used across the SHAB library

KEEP_RUBRICS = ['HR', 'KK', 'ES', 'EK']

# --- NEW CONFIGURATION CONSTANTS ---

# Scope A: The Skeleton (Structure)
# We keep these so the graph has structure (e.g. parents, deletions)
SKELETON_RUBRICS = ['HR', 'KK', 'LS']

# Scope B: The LLM Targets (Enrichment)
# We only pay to enrich these high-value events.
LLM_TARGET_SUBRUBRICS = [
    'HR01',          # New Companies (Founders)
    'KK02', 'KK03', 'KK06', # Bankruptcy (Liquidators)
    'LS01', 'LS02'   # Liquidation (Liquidators)
]

PUBLISHER_STOPLIST = {
    'SHAB', 'SOGC', 'FOSC', 'KAB', 'KANT', 'AMTSBLATT'
}