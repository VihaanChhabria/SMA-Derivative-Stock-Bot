"""Shared CSV export and console reporting for scanner results."""

from datetime import date, datetime

import pandas as pd

from src.paths import output_path


def export_and_print_scan_results(
    results: pd.DataFrame,
    *,
    latest_date: date | datetime,
    filename: str,
    heading: str,
    no_results_message: str,
    sort_by: str | None = None,
    ascending: bool = True,
    hidden_columns: tuple[str, ...] = (),
) -> None:
    """Export all scan results and print ranked candidates that passed."""
    export_filename = output_path("scans", filename)
    results.to_csv(export_filename, index=False)
    print(f"Full dataset ({len(results)} stocks) exported to '{export_filename}'.\n")

    passed_results = results[results["Passed_Scan"] == True].copy()
    if sort_by is not None:
        passed_results = passed_results.sort_values(
            by=sort_by, ascending=ascending
        )
    passed_results = passed_results.reset_index(drop=True)

    if passed_results.empty:
        print(no_results_message)
        return

    passed_results.index += 1
    passed_results.index.name = "Rank"

    print("=" * 90)
    print(
        heading.format(
            date=latest_date.strftime("%Y-%m-%d"),
            count=len(passed_results),
        )
    )
    print("=" * 90)
    print(passed_results.drop(columns=list(hidden_columns)).to_string())
