# Intent spotcheck and trends notebook

## Data and dependencies

- **Source**: Classified data produced by intent_classification.ipynb: `intent_output/intent_classified.parquet` (or LOAD_FROM_SAVED path). Columns include `conversation_id`, `timestamp`, `conversation` (list of `{role, content, ...}`), `intent_major`, `intent_sub`, `text` (first user message).
- **Timestamp**: Dataset uses **UTC** only. The WildChat README and schema state the timestamp is in UTC; there is no user timezone or locale in the data, so true local time cannot be inferred. The plan is to analyze by **hour (UTC)** and optionally allow a **configurable timezone** (e.g. `US/Pacific`) for "assumed local" plots, with a clear note that this is an assumption.
- **Existing code**: Reuse intent_taxonomy.py `load_classified()` and eda_utils.py `ensure_conversation_parsed()`; insights_utils.py `theme_counts_over_time()` already does period + label counts (can be reused for weekly intent counts and then convert to percentages).

## 1. Spot check: load and view full conversations by category/sub-category

- Load the full classified table from `intent_output`, then run `ensure_conversation_parsed()` so `conversation` is list-of-dicts.
- For each **major** and optionally each **sub** category, take a **random sample** (e.g. 3–5 per category, seed fixed) and display the **full conversation** (all turns) with:
  - One block per message: role (user/assistant) + content.
  - Content wrapped with `textwrap.fill()` (e.g. width 80) for readability.
- Implementation: intent_analysis_utils.py with `load_classified_for_analysis`, `sample_by_category`, `format_conversation`.
- Notebook: control variables at top (`INTENT_OUTPUT_DIR`, `SAMPLE_PER_CATEGORY`, `RUN_SPOTCHECK`, `BY_MAJOR` / `BY_SUB`); one cell that calls these and prints formatted conversations per category.

## 2. Week-by-week intent trends (percentage per category/sub-category)

- Compute **weekly** counts per intent (major and/or sub), then convert to **percentage** per week.
- **Plots**: Line plot and optional stacked area for major intents.
- Implementation: `weekly_intent_counts`, `weekly_intent_percentages`, `plot_weekly_intent_percentages` (with optional `events`).

## 3. Factoring in holidays and major events

- **Constraint**: No event metadata in the dataset. Recommended approach:
  - **Option A (simple)**: A small in-notebook or in-module list of (date or week-end date, label) for known events.
  - **Option B**: Optional **CSV** (e.g. `event_dates.csv` with columns `date`, `event_name`) that the plotting function reads and uses to annotate the same plot.
- Implementation: `plot_weekly_intent_percentages(..., events=None)` where `events` is a list of `(date, label)` or a path to CSV; plot vertical lines or shaded regions at those dates.

## 4. Within-week percentage changes (day-of-week within each week)

- For each **week**, break down by **day** (or weekday 0–6). Compute intent percentage per (week, day).
- **Visualization**: Heatmap (week x day-of-week) for one intent at a time; or a "typical week" profile (average percentage by weekday across all weeks).
- Implementation: `within_week_intent_percentages`, `plot_within_week_heatmap`, `plot_weekday_profile`.

## 5. Hour-of-day trends

- **Primary**: Distribution of conversations by **hour (UTC)** and by intent (count or percentage per hour).
- **Optional "assumed local"**: Configurable timezone (e.g. `TIMEZONE = "US/Pacific"`). If set, convert `timestamp` to that zone and plot by local hour; add a short note that user location is unknown and this is for illustration only.
- Implementation: `hourly_intent_counts`, `hourly_intent_percentages`, `plot_hourly_intent_distribution` (UTC + optional tz).

## 6. Notebook and module layout

- **intent_analysis_utils.py**: Load + parse, spot check, weekly, within-week, hour functions.
- **intent_spotcheck_and_trends.ipynb**: Control variables at top; sections Load data, Spot check, Weekly trends, Within-week, Hour-of-day. Each section guarded by the corresponding `RUN_*` flag so "Run All" respects toggles.
- Follow code-and-notebook-conventions: logic and plotting in .py, notebook imports and calls only; control variables at top of notebook.

## Summary

| Goal                     | Approach                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Spot check conversations | Load classified data, ensure conversation parsed, sample per category, print full convo with `textwrap.fill` |
| Weekly intent % trends   | Period "W", groupby period + label, compute pct per week, line/stacked plot                                  |
| Holidays/events          | Manual list or CSV of (date, name); vertical lines or shaded regions on weekly plot                          |
| Within-week % change     | Group by week + day (or weekday), pct per (week, day); heatmap or weekday profile                            |
| Hour-of-day trends       | Hour from timestamp (UTC); optional tz conversion for "assumed local" with disclaimer                        |
| Local time?              | Not available in data; report UTC and optional single timezone for illustration                              |
