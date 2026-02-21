# Rule-based intent classification (4-category taxonomy)

**Overview:** Add a 4-major-category intent taxonomy (Informational, Navigational, Commercial investigation, Transactional) with your keywords and map existing categories as sub-categories; implement in a new .py module; add a new notebook that uses query-only text, supports sample/full and save/load by category.

---

## Goal

- Keep the existing 6 categories as **sub-categories** where they fit under the 4 major intents.
- Use **query-only** text (first user message) for classification — the current pipeline already does this via `extract_text_column(mode="first_user")`; we will use that explicitly and document it.
- New notebook with **control variables** at top (sample vs full, save/load by category), reading from the **English dataset** (`data/english_chunks/`), and ability to **save/load** classified data by category.

---

## 1. Taxonomy and keyword set

**Major intents (4)** — use the expanded keyword sets below so the rule-based search is comprehensive.

### 1a. Expanded keywords (major intents)

**informational** (learn something):  
how, what, who, where, why, guide, tutorial, resource, help, ideas, tips, learn, examples, explain, meaning of, definition of, understand, describe, tell me about, overview, introduction, intro, basics, step by step, walkthrough, outline, summarize, summary, clarify, concept, theory, process, procedure, difference between, when to use, how it works, can you explain, what does, why do, how do i, where do i, what are, how are, why are, teach me, show me how, get started, beginner, advanced, reference, documentation, faq, frequently asked.

**navigational** (find a site/entity):  
name of, what is the name, what's the name, find me, find a, website for, url for, link to, app for, where can i find, where to find, locate, look up, search for, official site, official website, homepage, login page, sign in page, which app, which website, which site, which service, which company, brand name, product name, service name, store name, restaurant name, address of, contact for, customer service number, phone number for, email for.

**commercial_investigation** (compare before buying):  
best, top, top 10, top 5, pricing, review, reviews, comparison, compare, versus, vs, size, color, alternative, which one, should i get, worth it, recommend, recommendation, cost, price, prices, specs, specification, specifications, rating, ratings, pros and cons, quality, reliable, durable, compare to, compared to, better than, difference between, good for, suitable for, fit for, options for, choices for, alternatives to, instead of, or should i, cheapest, affordable, value for money, user review, customer review, expert review, buying guide, product comparison, side by side.

**transactional** (buy or act):  
buy, coupon, order, purchase, cheap, price, pricing, local, store, shop, discount, cart, checkout, pay, payment, shipping, deliver, delivery, subscribe, sign up, get it, add to cart, buy now, place order, order now, purchase now, in stock, availability, where to buy, buy from, sell, selling, refund, return, promo code, voucher, deal, sale, on sale, free shipping, buy local, near me, in my area, book now, reserve, appointment, sign up for, register for.

### 1b. Expanded keywords (sub-categories)

**education** (within informational):  
explain, how does, what is, why does, learn, teach, lesson, homework, assignment, study, definition of, meaning of, course, textbook, concept, theory, exam, test, quiz, practice, understand, describe, summarize, outline, basics, introduction, step by step, clarify, difference between, when to use, how it works, reference, documentation, faq.

**coding** (within informational):  
code, function, script, python, javascript, programming, debug, api, sql, regex, algorithm, implement, bug, error in my code, variable, loop, class, import, syntax, compile, runtime, function call, return value, array, list, dict, string, integer, boolean, framework, library, package, npm, pip, git, branch, merge, refactor, test case, unit test, exception, stack trace, console.log, print statement, endpoint, request, response, database, query.

**support** (within informational):  
help me, fix my, not working, broken, issue, problem with, error, support, troubleshoot, how do i fix, why is my, fix this, resolve, solution, failed, failure, crash, crashed, doesn't work, won't work, not responding, stuck, hang, freeze, slow, timeout, connection refused, permission denied, invalid, corrupted, restore, recover, reset, reinstall, update failed, install error.

**creative_writing** (within informational):  
write a, write me, story, poem, essay, dialogue, character, plot, fiction, creative, song lyrics, script for, short story, novel, scene, chapter, narrative, protagonist, antagonist, setting, theme, metaphor, rhyme, verse, stanza, draft, rewrite, edit, proofread, prompt for, idea for a story, opening line, ending, twist.

**commercial_product** (within commercial_investigation):  
Same as commercial_investigation major; can add product-specific phrases if desired: which one should i get, best [product type], review of, compare [X] and [Y].

**navigational** (sub): same as navigational major.

**transactional** (sub): same as transactional major.

**casual_other**: no keywords (fallback for informational when no sub matches).


**Sub-categories** (explicitly counted and summarized):

- **informational** → `education`, `coding`, `support` (troubleshooting), `creative_writing`, `casual_other` (fallback when no other sub-match).
- **commercial_investigation** → `commercial_product` (product research / "which one").
- **transactional** → can keep a single sub `transactional` or add later (e.g. `coupon`, `local_store`).
- **navigational** → single sub `navigational` for now.

**Rule order** (first match wins): Transactional and commercial_investigation before informational/navigational so "buy the best X" goes transactional or commercial_investigation; then sub-category rules run within the chosen major intent (e.g. within informational: coding, creative_writing, education, support, then casual_other).

**Output columns**: e.g. `intent_major` (4 values) and `intent_sub` (coding, education, support, creative_writing, commercial_product, navigational, transactional, casual_other). All cells that show counts/summaries will report both and optionally a pivot (major × sub).

---

## 2. New Python module: intent taxonomy and assignment

**File**: `intent_taxonomy.py` (new).

- **Constants**:
  - `MAJOR_INTENT_RULES`: list of `(major_label, list of keywords)` in desired priority order (transactional, commercial_investigation, navigational, informational, then fallback).
  - `SUB_INTENT_RULES`: dict mapping each major intent to a list of `(sub_label, list of keywords)`; include `casual_other` as fallback for informational.
  - Reuse/adapt keywords from `intent_clustering.py` (e.g. coding, creative_writing, education, support, commercial_product) for sub-categories.
- **Functions**:
  - `assign_major_intent(text_series: pd.Series) -> pd.Series`
  - `assign_sub_intent(text_series: pd.Series, major_series: pd.Series) -> pd.Series` (uses major so sub-rules are scoped per major).
  - Optionally: `assign_intent_with_sub(df, text_col="text")` that returns the dataframe with `intent_major` and `intent_sub` added.
- **Design**: Classification uses only the provided text (first user message); the notebook will pass the query column so "use words in query not the assistant answer" is guaranteed.

No change to existing `intent_clustering.py` INTENT_RULES so current `intent_clustering.ipynb` keeps working; the new notebook will use the new module.

---

## 3. New notebook: intent classification (4-category + sub)

**File**: e.g. `intent_classification.ipynb` (new).

- **Control variables (top cell)**
  - `ENGLISH_CHUNKS_DIR = "data/english_chunks"`
  - `USE_SAMPLE = True` (True = sample N for quick distribution look; False = full English set).
  - `SAMPLE_N = 5000` (used only when `USE_SAMPLE` is True).
  - `RANDOM_SEED = 42`
  - `SAVE_BY_CATEGORY = True` — save classified result and optionally per-category subsets.
  - `LOAD_FROM_SAVED = False` — if True, load previously saved classified dataset instead of re-running classification.
  - `INTENT_OUTPUT_DIR = "intent_output"` (or similar) for saved artifacts.
- **Flow**
  1. **Load data**: From `eda_utils.load_english_chunked_parquet` with `ENGLISH_CHUNKS_DIR`. If `USE_SAMPLE` then sample N conversations (e.g. via `prepare_sample` or equivalent); else use full dataframe. Drop rows with empty first user message.
  2. **Text column**: Set `df["text"] = extract_text_column(df, mode="first_user", conversation_col="conversation")` so classification uses **query only** (first user message), not assistant content. Document this in a short markdown.
  3. **Classify**: Call `assign_intent_with_sub(df, text_col="text")` from `intent_taxonomy` to add `intent_major` and `intent_sub`.
  4. **Summarize**:
     - Counts by `intent_major`.
     - Counts by `intent_sub`.
     - Cross-tab `intent_major` × `intent_sub` (all explicitly counted and summarized).
  5. **Save (when `SAVE_BY_CATEGORY`)**:
     - Save full classified dataframe to e.g. `intent_output/intent_classified.parquet`.
     - Optionally save per major (or per sub) subsets as `intent_output/by_major/{intent_major}.parquet` and/or `intent_output/by_sub/{intent_sub}.parquet` so user can load specific categories later.
  6. **Load (when `LOAD_FROM_SAVED`)**: If `LOAD_FROM_SAVED` is True, load from `intent_output/intent_classified.parquet` (and optionally document loading a single category from `by_major/` or `by_sub/` in a later cell).

Notebook stays thin: data load, config, and one-line calls to `eda_utils`, `intent_taxonomy`, and small helpers for save/load; heavier logic in `.py`.

---

## 4. Save/load by category (concrete)

- **Full classified table**: `intent_output/intent_classified.parquet` — all rows with `intent_major`, `intent_sub`, and original columns (e.g. `conversation_id`, `text`, …). Load this when `LOAD_FROM_SAVED` is True to skip re-classification.
- **Per-category subsets**: Write one parquet per major intent, e.g. `intent_output/by_major/informational.parquet`, and optionally per sub, e.g. `intent_output/by_sub/coding.parquet`. This allows "load only commercial_investigation later" without reprocessing. Implement in `intent_taxonomy.py` or a small helper used by the notebook (e.g. `save_classified_by_category(df, output_dir)` / `load_category(category_name, which='major'|'sub')`).

---

## 5. File and output layout

- **New**: `intent_taxonomy.py` (taxonomy constants + assignment functions).
- **New**: `intent_classification.ipynb` (control vars, load English → query column → classify → summarize → save/load by category).
- **New dir**: `intent_output/` for `intent_classified.parquet`, `by_major/*.parquet`, and optionally `by_sub/*.parquet`.
- **Unchanged**: `intent_clustering.py` and `intent_clustering.ipynb` (existing rule set and flow remain as-is).

---

## 6. Sub-category mapping (explicit)

| Major                    | Sub-categories                                             |
| ------------------------ | ---------------------------------------------------------- |
| informational            | education, coding, support, creative_writing, casual_other |
| commercial_investigation | commercial_product                                         |
| transactional            | transactional (single sub for now)                         |
| navigational             | navigational                                               |

All summary tables and any exports will include both `intent_major` and `intent_sub` so you get explicit counts and summaries for each.
