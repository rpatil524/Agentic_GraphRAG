"""
download_data.py
================
Download SHAB publication data from the public SHAB API and persist it in two forms:

1. Raw page-level JSON responses, grouped by month.
2. One aggregated monthly CSV file built from the flattened JSON payloads.

The script downloads data week by week and paginates each weekly query. It is resumable:
if a page-level JSON file already exists on disk, that file is reused instead of being
downloaded again.

Default output structure:

    shab_data/
        09_2018/
            data_2018-09-02_to_2018-09-08_page_0.json
            ...
            shab_data_09_2018.csv

Environment variables:
    SHAB_TERMS_ACCEPTED       Set to true after reviewing and accepting the
                              current Official Gazettes Portal terms.
    SHAB_TERMS_ACCEPTED_DATE  Acceptance date in YYYY-MM-DD format. The terms
                              require consent to be renewed every 90 days.
    SHAB_START_DATE           Start date in YYYY-MM-DD format. Default: 2018-09-02
    SHAB_END_DATE             End date in YYYY-MM-DD format. Default: today
    SHAB_OUTPUT_DIR           Output directory. Default: shab_data
    SHAB_DELAY_SEC            Delay between API calls in seconds. Default: 1
    SHAB_MAX_RETRIES          Retries after a failed API request. Default: 3
"""

import datetime
import hashlib
import json
import os
import time
from typing import Optional

import pandas as pd
import requests
from pandas import json_normalize
from tqdm import tqdm


SHAB_API_URL = "https://shab.ch/api/v1/publications"
SHAB_PORTAL_URL = "https://www.shab.ch/"
TERMS_RENEWAL_DAYS = 90

SHAB_TERMS_ACCEPTED = os.getenv("SHAB_TERMS_ACCEPTED", "false")
SHAB_TERMS_ACCEPTED_DATE = os.getenv("SHAB_TERMS_ACCEPTED_DATE", "")
OVERALL_START_DATE = os.getenv("SHAB_START_DATE", "2018-09-02")
OVERALL_END_DATE = os.getenv(
    "SHAB_END_DATE",
    datetime.date.today().strftime("%Y-%m-%d"),
)
BASE_DOWNLOAD_DIRECTORY = os.getenv("SHAB_OUTPUT_DIR", "shab_data")
DELAY_SECONDS = float(os.getenv("SHAB_DELAY_SEC", "1"))
MAX_RETRIES = max(0, int(os.getenv("SHAB_MAX_RETRIES", "3")))


def validate_terms_confirmation(today: Optional[datetime.date] = None) -> None:
    """Require a recent user confirmation before accessing the SHAB API.

    This local check does not accept the terms on the user's behalf. Users must
    review and accept the current terms through the Official Gazettes Portal,
    then record the date through the environment variables documented above.
    """
    accepted_values = {"1", "true", "yes"}
    if SHAB_TERMS_ACCEPTED.strip().lower() not in accepted_values:
        raise RuntimeError(
            "SHAB terms have not been confirmed. Review and accept the current "
            f"Official Gazettes Portal terms at {SHAB_PORTAL_URL}, then set "
            "SHAB_TERMS_ACCEPTED=true and SHAB_TERMS_ACCEPTED_DATE=YYYY-MM-DD."
        )

    try:
        accepted_date = datetime.date.fromisoformat(SHAB_TERMS_ACCEPTED_DATE.strip())
    except ValueError as exc:
        raise RuntimeError(
            "SHAB_TERMS_ACCEPTED_DATE must be a valid date in YYYY-MM-DD format."
        ) from exc

    current_date = today or datetime.date.today()
    if accepted_date > current_date:
        raise RuntimeError("SHAB_TERMS_ACCEPTED_DATE cannot be in the future.")

    age_days = (current_date - accepted_date).days
    if age_days > TERMS_RENEWAL_DAYS:
        raise RuntimeError(
            f"The recorded SHAB terms confirmation is {age_days} days old. "
            f"Review the current terms at {SHAB_PORTAL_URL}, renew consent, and "
            "update SHAB_TERMS_ACCEPTED_DATE before downloading data."
        )


def flatten_response_text_to_df(response_text: str) -> pd.DataFrame:
    """
    Flatten the SHAB API JSON response into a tabular DataFrame.

    The SHAB payload contains nested dictionaries and lists. This helper expands
    dictionaries iteratively, keeps list columns in a compact string form, and
    adds simple null flags for downstream analysis.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        print("      - Warning: could not decode JSON response. Skipping this page.")
        return pd.DataFrame()

    content = data.get("content", [])
    if not content or not isinstance(content, list):
        return pd.DataFrame()

    df = json_normalize(content, sep="_")
    if df.empty:
        return df

    # Iteratively expand nested dictionaries.
    loop_guard = 0
    while loop_guard < 10:
        dict_cols = [col for col in df.columns if df[col].apply(isinstance, args=(dict,)).any()]
        if not dict_cols:
            break

        expanded_dfs = []
        for col in dict_cols:
            dict_rows_mask = df[col].apply(isinstance, args=(dict,))
            if not dict_rows_mask.any():
                continue

            normalized_part = json_normalize(df.loc[dict_rows_mask, col].tolist(), sep="_")
            normalized_part.columns = [f"{col}_{c}" for c in normalized_part.columns]
            normalized_part.set_index(df.index[dict_rows_mask], inplace=True)
            expanded_dfs.append(normalized_part)

        if expanded_dfs:
            df = df.join(pd.concat(expanded_dfs, axis=1))
        df = df.drop(columns=dict_cols)
        loop_guard += 1

    # Convert list fields into compact pipe-separated strings and keep counts.
    list_cols = [col for col in df.columns if df[col].apply(isinstance, args=(list,)).any()]
    if list_cols:
        new_cols_data = {}
        for col in list_cols:
            new_cols_data[f"{col}_count"] = df[col].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            new_cols_data[col] = df[col].apply(
                lambda x: "|".join(map(str, x)) if isinstance(x, list) else x
            )
        df = df.assign(**new_cols_data)

    # Add null-indicator flags for columns that contain missing values.
    null_flag_cols = {
        f"{col}_is_null": df[col].isnull()
        for col in df.columns
        if df[col].isnull().any()
    }
    if null_flag_cols:
        df = pd.concat([df, pd.DataFrame(null_flag_cols)], axis=1)

    # Parse date-like columns when possible.
    for col in df.columns:
        if "date" in col.lower() and "is_null" not in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Move key identifiers to the front for readability.
    front_cols = []
    for candidate in ["meta_id", "id", "meta_publicationNumber"]:
        if candidate in df.columns:
            front_cols.append(candidate)

    if front_cols:
        front_cols = sorted(set(front_cols))
        rest_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + rest_cols]

    return df.reset_index(drop=True).copy()


def fetch_week_page(week_start_str: str, week_end_str: str, page_num: int) -> str:
    """
    Fetch one SHAB API page for the given week window.
    """
    params = {
        "allowRubricSelection": "false",
        "includeContent": "true",
        "pageRequest.page": page_num,
        "pageRequest.size": "100",
        "publicationDate.end": week_end_str,
        "publicationDate.start": week_start_str,
        "publicationStates": "PUBLISHED,CANCELLED",
        "rubrics": "AB,AW,AZ,BB,BH,EK,ES,FM,HR,KK,LS,NA,SB,SR,UP,UV",
        "searchPeriod": "CUSTOM",
    }
    headers = {"User-Agent": "AgenticGraphRAG-research-downloader/1.0"}
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                SHAB_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise last_error


def response_has_publications(response_text: str) -> bool:
    """Return whether an API page contains at least one publication."""
    try:
        content = json.loads(response_text).get("content", [])
    except (json.JSONDecodeError, AttributeError):
        return False
    return isinstance(content, list) and bool(content)


def save_monthly_csv(month_year: str, monthly_dataframes: list[pd.DataFrame], base_dir: str) -> None:
    """
    Persist the aggregated monthly CSV assembled from all downloaded weekly pages.
    """
    if not monthly_dataframes:
        return

    full_month_df = pd.concat(monthly_dataframes, ignore_index=True)
    csv_folder_path = os.path.join(base_dir, month_year)
    os.makedirs(csv_folder_path, exist_ok=True)

    csv_file_name = f"shab_data_{month_year}.csv"
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    full_month_df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"Successfully saved {len(full_month_df)} records to {csv_file_path}")


def download_shab_data() -> None:
    """
    Download SHAB publication data week by week for the configured date range.

    For each week:
    - iterate over paginated API results
    - save the raw JSON response of each page
    - flatten and collect the page content
    - aggregate the collected rows into one monthly CSV
    """
    validate_terms_confirmation()

    print("--- Starting SHAB Data Download ---")
    print(f"Date Range: {OVERALL_START_DATE} to {OVERALL_END_DATE}")
    print(f"Saving data in: '{os.path.abspath(BASE_DOWNLOAD_DIRECTORY)}'")
    print("-" * 35)

    try:
        start_date_obj = datetime.datetime.strptime(OVERALL_START_DATE, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(OVERALL_END_DATE, "%Y-%m-%d").date()
    except ValueError as exc:
        print(f"Error: invalid date format, expected YYYY-MM-DD. Details: {exc}")
        return

    if start_date_obj > end_date:
        raise ValueError("SHAB_START_DATE must be on or before SHAB_END_DATE.")

    os.makedirs(BASE_DOWNLOAD_DIRECTORY, exist_ok=True)

    current_date = start_date_obj
    monthly_dataframes = []
    last_processed_month_year = current_date.strftime("%m_%Y") if current_date <= end_date else None
    total_weeks = ((end_date - start_date_obj).days // 7) + 1

    with tqdm(total=total_weeks, desc="Overall Progress", unit="week", position=0, leave=True) as pbar:
        while current_date <= end_date:
            week_start = current_date
            week_end = min(current_date + datetime.timedelta(days=6), end_date)

            week_start_str = week_start.strftime("%Y-%m-%d")
            week_end_str = week_end.strftime("%Y-%m-%d")
            current_month_year = week_start.strftime("%m_%Y")

            # If the month changes, flush the previous month's collected rows.
            if current_month_year != last_processed_month_year and monthly_dataframes:
                tqdm.write("")
                tqdm.write(f"--- Saving aggregated data for month: {last_processed_month_year} ---")
                save_monthly_csv(last_processed_month_year, monthly_dataframes, BASE_DOWNLOAD_DIRECTORY)
                monthly_dataframes = []

            last_processed_month_year = current_month_year
            pbar.set_description(f"Processing week of {week_start_str}")

            download_path = os.path.join(BASE_DOWNLOAD_DIRECTORY, current_month_year)
            os.makedirs(download_path, exist_ok=True)

            page_num = 0
            seen_page_hashes = set()
            page_bar = tqdm(
                desc=f"Pages for {week_start_str}",
                position=1,
                leave=False,
                unit="page",
            )
            try:
                while True:
                    file_name = f"data_{week_start_str}_to_{week_end_str}_page_{page_num}.json"
                    file_path = os.path.join(download_path, file_name)
                    response_text = None

                    # Reuse valid downloaded pages; retry corrupt cache files.
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "r", encoding="utf-8") as handle:
                                response_text = handle.read()
                            json.loads(response_text)
                        except (OSError, json.JSONDecodeError) as exc:
                            tqdm.write(f"  - Page {page_num}: cached file is invalid; downloading it again ({exc}).")
                            response_text = None

                    if response_text is None:
                        try:
                            response_text = fetch_week_page(week_start_str, week_end_str, page_num)
                            # Validate before replacing a corrupt or absent cache file.
                            json.loads(response_text)
                            with open(file_path, "w", encoding="utf-8") as handle:
                                handle.write(response_text)
                            time.sleep(DELAY_SECONDS)
                        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
                            raise RuntimeError(
                                f"Could not complete page {page_num} for "
                                f"{week_start_str} to {week_end_str}. Rerun the "
                                "downloader to resume from cached pages."
                            ) from exc

                    if not response_has_publications(response_text):
                        break

                    page_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                    if page_hash in seen_page_hashes:
                        raise RuntimeError(
                            f"SHAB API repeated page content for {week_start_str}; "
                            "stopping to avoid an infinite pagination loop."
                        )
                    seen_page_hashes.add(page_hash)

                    try:
                        df = flatten_response_text_to_df(response_text)
                        if not df.empty:
                            monthly_dataframes.append(df)
                    except Exception as exc:
                        tqdm.write(f"  - Page {page_num}: could not read '{file_path}'. Error: {exc}")
                    page_num += 1
                    page_bar.update(1)
            finally:
                page_bar.close()

            current_date += datetime.timedelta(weeks=1)
            pbar.update(1)

    if monthly_dataframes and last_processed_month_year:
        print(f"\n--- Saving final aggregated data for month: {last_processed_month_year} ---")
        save_monthly_csv(last_processed_month_year, monthly_dataframes, BASE_DOWNLOAD_DIRECTORY)

    print("\n--- Download complete. ---")


if __name__ == "__main__":
    download_shab_data()
