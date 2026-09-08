# Data Source and Use Notice

## SHAB source

The case-study data originate from the Swiss Official Gazette of Commerce
(SHAB/SOGC), made available through the Official Gazettes Portal by the Swiss
State Secretariat for Economic Affairs (SECO). Any use or redistribution of
source-derived data must include appropriate attribution to the Official
Gazettes Portal and/or the responsible publishing authority.

This is not an official publication. The authoritative data are those which
appear on the Official Gazettes Portal and bear the SECO electronic signature
or stamp.

## Terms of use

The legally valid terms are the version published on the
[Official Gazettes Portal](https://www.shab.ch/) at the time of use. The terms
reviewed for this project were dated 7 June 2024, but that archived version does
not replace the current portal version. Users must consult the current terms
regularly and renew their consent every 90 days.

For provenance, the English PDF reviewed for this release had SHA-256 checksum
`cde037343e89e20545dcadaa840c18aa55e8866426b499ac32901735bd115dae`.

The downloader therefore requires two local environment variables before it
will contact the API:

```bash
export SHAB_TERMS_ACCEPTED=true
export SHAB_TERMS_ACCEPTED_DATE="YYYY-MM-DD"
```

These variables are only a local confirmation. They do not accept the terms on
the user's behalf and do not replace any acceptance required by the portal.
For research records, users should retain a dated copy or checksum of the terms
they reviewed and, where appropriate, archive the public terms page. An archive
copy is evidence of the reviewed version, not a substitute for checking the
current terms.

## Data quality and responsibility

SECO does not guarantee uninterrupted portal or application availability, or
the complete and correct transfer of data through the API. Users should account
for missing, delayed, cancelled, duplicated, or malformed records and should
verify important findings against the authoritative publication. Any data-
quality particulars supplied by SECO with the data must also be preserved and
disclosed when the data are used. Data are used and processed at the user's own
risk.

Processing or publishing personal data remains subject to Swiss data-protection
law and any other law applicable where the data are processed. Users are
responsible for evaluating access-period, retention, disclosure, and deletion
requirements before publishing datasets or deploying this software.

## Repository data

Raw SHAB downloads and the populated Neo4j and Chroma databases are not part of
this repository. The included benchmark inputs are research artifacts and may
contain source-derived entity names, facts, and notice text needed to define or
audit questions. Source-derived fields must remain distinguishable from added
questions, reference answers, annotations, and model outputs. Their inclusion
does not make the repository an official SHAB publication.
