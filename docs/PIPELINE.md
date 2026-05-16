# How the ProphetHacks 2026 Forecasting Agent Works

## What this system does

When a judge's platform asks our agent "Will Bitcoin close above $100,000 on December 31, 2026?", the agent doesn't just ask a single AI model and return whatever it says. Instead it runs a full pipeline: three different AI models weigh in simultaneously, Snowflake looks up the most similar questions ever resolved in the past and tells us what actually happened, a trained calibrator adjusts the probabilities based on what history says, and finally a learned policy decides how much to trust our own prediction versus just echoing the market price. The whole thing runs in under a minute and returns a standard JSON response the judge's scoring system can read directly.

The reason we built all of this instead of just calling one LLM is a specific insight from Prof. Haifeng Xu's scoring research: **Brier score** (how accurate your probabilities are in absolute terms) and **AVER** (how much you beat the market's price) measure completely different things. If you just copy the Kalshi market price, you get decent Brier but zero AVER — you can't beat the market by being the market. If you blindly trust a single LLM, you might deviate from the market in the wrong direction and do badly on both metrics. Our system is designed to find the cases where we should deviate from the market price, and by how much.

---

## What goes in and what comes out

**The input** is a standard OpenAI-format chat request. The judge's platform sends a POST request to `/v1/chat/completions` with a message that contains the forecasting question in plain text. Our parser reads that message and extracts: the question itself (looking for lines that start with "Question:" or "Will..."), the list of possible outcomes (numbered or bulleted lines), the current Kalshi market price if provided, a resolution date, and a market ID. If none of those markers are present, the whole message becomes the question and outcomes default to Yes/No.

**The output** is also standard OpenAI format. The response contains a short explanation followed by a line of JSON with the probabilities: `{"probabilities": [0.48, 0.52]}`. The probabilities are in the same order as the outcomes that were passed in. There's also a `prophet_debug` field attached to the response with every intermediate value — what each individual model said, how many historical neighbors we found, what the base rate was, what the calibrator output, and how long the whole thing took. This debug field is what powers the Streamlit demo.

---

## The six stages of inference

### Stage 1 — Reading the question

The parser (`src/prophet_agent/parser.py`) takes the raw text and turns it into structured data. It uses regular expressions to find question headers, outcome lists, price lines, and date markers. If it can't find a structured question, it uses the whole message. If it can't find outcomes, it assumes binary Yes/No. This is intentionally forgiving — the judge's format might vary, and we'd rather make a reasonable guess than crash.

### Stage 2 — Three AI models vote in parallel

`src/prophet_agent/llm/wafer.py` fires off calls to three models at the same time via the Wafer.ai API: GLM-5.1, Qwen3.5-397B, and Qwen3.6-35B. Each one gets the question, the outcomes, and the current market price, and is instructed to return a JSON probability vector. The three responses come back in parallel — the total time is roughly the slowest of the three, not the sum.

Each model is a "reasoning model" meaning it thinks before answering, which requires a generous token budget (3,500 tokens by default, increasing on retry). If a model fails or returns garbled output, it falls back to a uniform distribution so the ensemble always has something to work with. The final ensemble output is the mean probability across all three models, plus a **disagreement variance** — how much the three models differed from each other. High variance means uncertainty; low variance means they all agreed.

### Stage 3 — Looking up similar past questions in Snowflake

`src/prophet_agent/retrieval/base_rate.py` takes the question text, embeds it into a 768-dimensional vector using Snowflake's built-in `EMBED_TEXT_768` function (so the embedding computation happens entirely inside the warehouse — no data leaving Snowflake), and then searches `HISTORICAL_MARKETS` for the 15 most similar resolved questions using cosine similarity.

From those 15 neighbors it computes two things: the **base rate** (what fraction of similar past questions resolved Yes vs No) and the **Kalshi residual** (how much the market price was off on average for questions like this one). Both of these become features for the calibrator. If Snowflake is unreachable or the table is empty, this stage returns a uniform prior and the pipeline keeps going.

### Stage 4 — The AutoML calibrator adjusts the probabilities

`src/prophet_agent/calibrator.py` takes everything computed so far — the ensemble mean, the Kalshi market price, the historical base rate, the disagreement variance, and the question category — and feeds it into a trained model that outputs calibrated probabilities.

For each possible outcome, the calibrator sees five numbers: the Kalshi price for that outcome, what the ensemble thinks, how much the models disagreed, what history says the base rate is, and what category this question falls into. A GradientBoosting classifier was trained on hundreds of resolved historical markets to learn the mapping from these features to actual outcomes. The raw per-outcome scores are then softmax-normalized so they sum to 1.

At startup the calibrator tries to load a Snowflake AutoML model first, then falls back to a local sklearn `.joblib` file, and if neither exists it just passes the ensemble probabilities through unchanged. So the system degrades gracefully — it always returns something, it's just better when the model is trained.

### Stage 5 — The α-policy decides how much to trust our prediction

`src/prophet_agent/shading.py` computes a single number, **α**, that controls how much weight to give to our calibrated prediction versus the Kalshi market price. The final output is `p_final = α × p_meta + (1-α) × q_kalshi`. At α=1 we fully trust the calibrator. At α=0 we exactly match the market. Most of the time it lands somewhere in between.

α is not a fixed constant — it's predicted per market by a second trained model. The features it uses are: how much the three models disagreed (higher disagreement → probably trust the market more), how many similar historical neighbors we found (more neighbors → more confident in base rate → willing to deviate more), how similar those neighbors were, how wide the market's spread is, and the category. The α-policy was trained using a line-search over historical markets to find what α would have maximized return on each one.

Like the calibrator, the α-policy loads a Snowflake AutoML regression model first, then a local sklearn file, then a single global α number saved from training, then hardcoded 0.5 as a last resort.

### Stage 6 — Format the response and log to Snowflake

`src/prophet_agent/server.py` takes the pipeline result and wraps it in a standard OpenAI JSON response. It also attempts a best-effort INSERT into `AGENT_PREDICTIONS` in Snowflake to log the request, all the intermediate values, and the final output. If that INSERT fails for any reason — bad connection, expired credentials, anything — the failure is logged to stderr and the HTTP response goes out normally. Logging never blocks or crashes a request.

---

## The Snowflake database

There are five tables and two training views.

**`HISTORICAL_MARKETS`** is the institutional memory. It holds every resolved Kalshi market we ingested: the question text, outcomes, category, open and resolution timestamps, the market price at open and 24 hours before resolution, which outcome actually happened, and a 768-dimensional vector embedding of the question text. This table is what Stage 3 searches.

**`ENSEMBLE_BACKFILL`** and **`BASE_RATE_BACKFILL`** are offline feature caches. For each historical market, `ENSEMBLE_BACKFILL` stores what the Wafer ensemble would have said if the question had been asked live (we ran it retroactively). `BASE_RATE_BACKFILL` stores what the Snowflake retrieval would have returned. Together these are the features we train the calibrator on.

**`META_PREDICTIONS_BACKFILL`** stores the calibrator's output on every historical market, plus the `alpha_star` value (what α would have been optimal on that market). This is what the α-policy trains on.

**`AGENT_PREDICTIONS`** is the live audit trail — every request the running agent handles gets logged here with all intermediate values. Reserved columns for `realized_outcome`, `brier`, and `payoff` can be filled in later once markets resolve.

The **`META_TRAIN` view** (created by the calibrator training script) flattens the historical markets into one row per outcome, joining all the backfill tables. This is the direct training input for the calibrator. The **`ALPHA_TRAIN` view** does the same for the α-policy.

---

## How to build the database from scratch

Everything runs in order. Each step is idempotent — safe to re-run.

```
bootstrap_snowflake.py   →  creates all tables (empty)
import_kalshi_history.py →  fills HISTORICAL_MARKETS with resolved markets
backfill_candlesticks.py →  updates real Kalshi prices (open + 24h-pre)
embed_questions.py       →  computes question_embedding for each row
backfill_features.py     →  fills ENSEMBLE_BACKFILL + BASE_RATE_BACKFILL
04_train_automl_calibrator.py  →  trains calibrator, fills META_PREDICTIONS_BACKFILL
06_train_alpha_policy.py       →  trains α-policy, writes Pareto plot
```

The first four steps are pure data collection and can run once. Steps 5 and 6 call the Wafer API (costs money per call), so you control the limit. Steps 7 and 8 can be re-run as more backfill data becomes available to improve the models.

---

## What the calibrator is actually learning

The calibrator sees five numbers per outcome and learns to predict whether that outcome actually happened:

**`q` — the Kalshi market price** for this outcome at 24 hours before resolution. This is the market's collective best guess. It's a strong baseline on its own.

**`p_llm` — what the ensemble predicted** for this outcome. Sometimes this agrees with the market, sometimes it disagrees.

**`disagreement`** — how much the three models varied from each other on this question. When models disagree a lot, it's a signal to be more conservative and stay closer to the market price.

**`base_rate`** — what fraction of historically similar questions had this outcome. This is the long-run empirical frequency, independent of both the current market and the current LLM opinion.

**`category`** — Elections, Politics, Economics, etc. Different types of questions have different calibration needs: political prediction markets might be systematically miscalibrated in different ways than crypto price markets.

---

## Why the Brier–AVER tradeoff is the core design problem

**Brier score** measures squared error between your probabilities and the realized outcome. Lower is better. If you predict 0.55 and the event happens, your Brier for that outcome is (0.55-1)² = 0.2025. The market price (Kalshi's `q_kalshi`) is already aggregating a lot of information, so just copying it gives a reasonable Brier.

**AVER** measures your log-payoff relative to the market price. If the event happens and you predicted 0.55 while the market said 0.50, your AVER is log(0.55/0.50) = +0.095. If you just copy the market (`p=q`), your AVER is exactly zero — you have no edge because you are the market.

The tension is: to get positive AVER you have to deviate from the market price in the right direction. But every deviation you make that turns out to be wrong hurts your Brier. The α parameter is the knob: high α means you trust your calibrator and deviate aggressively; low α means you stay close to the market and stay safe on Brier. Our α-policy tries to predict, for each individual market, where on that frontier to stand — based on how confident our signals are.

---

## Current status

| Component | Status |
|---|---|
| Parser, Wafer ensemble, Snowflake retrieval | Working in production |
| Calibrator (sklearn fallback) | Active after running `04_train_automl_calibrator.py` |
| α-policy (sklearn fallback) | Active after running `06_train_alpha_policy.py` |
| Snowflake AutoML (`ML.CLASSIFICATION` / `ML.REGRESSION`) | Attempted at training time; falls back to sklearn on trial tier |
| Agent endpoint | Serving real requests; neighbor_count=15, mean_similarity≈0.73 confirmed live |
| `AGENT_PREDICTIONS` logging | Best-effort; failures are silent (printed to stderr, not returned as errors) |
