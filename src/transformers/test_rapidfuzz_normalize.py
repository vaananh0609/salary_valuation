#!/usr/bin/env python3
"""
Test RapidFuzz-based company name normalization.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.transformers.deduplicator import normalize_company, canonical_companies, company_dict

# Test data from user's specification
test_companies = [
    "Công ty TNHH Samsung Electronics Việt Nam",
    "Samsung Electronics Vietnam",
    "Samsung Electronics VN",
    "Công ty Cổ phần FPT",
    "FPT Software",
    "FPT Telecom",
    "Công ty Cổ phần CMC",
    "CMC Corporation",
    "Công ty Viettel",
    "Viettel",
]

print("=" * 80)
print("RAPIDFUZZ COMPANY NORMALIZATION TEST")
print("=" * 80)
print()

results = []
for company in test_companies:
    normalized = normalize_company(company)
    results.append((company, normalized))
    arrow = " => "
    print(f"{company:45} {arrow} {normalized}")

print()
print("=" * 80)
print(f"SUMMARY: Built {len(canonical_companies)} canonical forms")
print("=" * 80)
print()

# Show company dictionary
print("Company Dictionary (canonical mappings):")
for raw_form, canonical_form in sorted(company_dict.items()):
    if raw_form != canonical_form:
        print(f"  '{raw_form}' → '{canonical_form}'")
    else:
        print(f"  '{raw_form}' (canonical)")

print()
print("Expected output (from user spec):")
print("  Công ty TNHH Samsung Electronics Việt Nam → samsung electronics vietnam")
print("  Samsung Electronics Vietnam → samsung electronics vietnam")
print("  Công ty Cổ phần FPT → fpt")
print("  FPT Software → fpt")
print("  Công ty Cổ phần CMC → cmc")
print("  CMC Corporation → cmc")
print("  Công ty Viettel → viettel")
print("  Viettel → viettel")
