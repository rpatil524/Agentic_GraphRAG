"""Shared configuration for SHAB graph construction."""

# Rubric families retained for the structured graph skeleton.
SKELETON_RUBRICS = ("HR", "KK", "LS")

# High-value event types enriched with LLM-extracted weak nodes.
LLM_TARGET_SUBRUBRICS = (
    "HR01",  # New company registrations
    "KK02",  # Bankruptcy proceedings
    "KK03",  # Bankruptcy proceedings
    "KK06",  # Bankruptcy proceedings
    "LS01",  # Liquidations
    "LS02",  # Liquidations
)

# Legacy export retained for callers that imported KEEP_RUBRICS.
KEEP_RUBRICS = SKELETON_RUBRICS

PUBLISHER_STOPLIST = {
    "SHAB", "SOGC", "FOSC", "KAB", "KANT", "AMTSBLATT"
}
