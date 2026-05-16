# ProphetHacks 2026 — pitch scripts

Structured for the eight story beats in `whimsical-kindling-hamster.md`. Read out loud; trim only if the moderator cuts you off.

---

### 30-second elevator (5 lines)

We compete in the **Forecasting Track** at ProphetHacks.
Naïve LLMs **optimize Brier accidentally** and basically **never optimize AVER**.
Prof. Xu’s PM-RANK scoring explainer makes that split official: *“the Brier score is an **absolute metric**, while the averaged return score is a **relative metric**.”*
We built a **market-aware meta-calibrator** on Snowflake: ensemble → institutional memory → planned dual AutoML → OpenAI-compatible agent.
**Let me show you the Streamlit dashboard and a live `/v1/chat/completions` call.**

---

### 2-minute pitch (~300 words)

**Beat 1 — opener.**  
We started by reading your scoring doc, not by reading Twitter. PM-RANK 0.3.1 says Brier and averaged return are different *species* of score. Quote the line we cared about: **“the Brier score is an absolute metric, while the averaged return score is a relative metric.”** That is the design brief. If the metric is wrong for the story you want to tell, **the whole model is wrong**.

**Beat 7 — Snowflake as memory *and* model host.**  
Snowflake is not our filing cabinet. It is where we **embed** every resolved market, **retrieve** neighbors with vector similarity, and train **two** AutoML models: a **classifier** for calibration on real binary labels, and a **regressor** for a per-market **α-policy** that decides how hard to shade toward or away from the posted Kalshi price. Same warehouse. Same governance story the side prize asks for. One team will paste “we used Snowflake” in a slide footnote—we built the **path**: tables, vectors, model slots, UI.

**Beat 8 — demo handoff.**  
The agent is **FastAPI**, OpenAI-compatible, exactly what the harness wants. `./run.sh`, `POST /v1/chat/completions`, probabilities in the assistant message. **Streamlit-in-Snowflake** is the four-tab demo over the same tables—ensemble, neighbors, calibrator hook, Pareto story. You can be skeptical of every sentence I just said. **You cannot be skeptical of a curl that returns JSON and a dashboard that queries your own tables.**

We will not waste your time with vibes. **Let me show you the Streamlit dashboard.**

---

### 5-minute pitch (~700 words)

**Open with Prof. Xu, verbatim.**  
*"A has a higher Brier Score, but lower returns."*  
That is not a footnote in PM-RANK 0.3.1. It is the whole reason “just ask GPT for a number” is a trap under your utility.

**Beat 1 — we read the scoring doc.**  
The write-up makes the yardsticks explicit. **Brier** is an **absolute** calibration metric against truth. **AVER** is **relative**—it is about beating the baseline price students see in the arena. A system can look elegant on one and lose on the other. We treated that as the **first** requirement, not the last slide.

**Beat 2 — naïve LLMs lose both ways.**  
If the model hugs the market, Brier gets easy points but **AVER goes flat**. If it drifts without grounding, Brier blows up—and **AVER still does not rescue you**. Neither default is a strategy.

**Beat 3 — institutional memory.**  
We ingest **resolved** Kalshi history into Snowflake. We **embed** questions with Cortex. When a new market arrives, we pull **similar past markets** and aggregate what actually happened—base rates and neighbor structure—not vibes from a single prompt.

**Beat 4 — ensemble, not oracle.**  
Three diverse frontier models through Wafer—**GLM-5.1**, **Qwen 397B**, **Qwen 35B**—in parallel. Their **spread** is a uncertainty signal we feed forward. One model’s typo should not become everyone’s position.

**Beat 5 — labels that exist.**  
The calibrator trains on **resolved** markets only. Binary outcomes. Held-out validation by time. The point is simple: **no synthetic labels, no pretend supervision**—only prices and resolutions we can audit in Snowflake. When eval lands, the slide line is explicit: *Brier: AGENT=`{CALIB_BRIER}` vs KALSHI_BASELINE=`{KALSHI_BRIER}` vs RAW_ENSEMBLE=`{ENSEMBLE_BRIER}`.*

**Beat 6 — not a hand-tuned α.**  
We are not stopping at “λ times Brier plus AVER.” We want a **policy** for shading: where the ensemble disagrees, where neighbors are thin, where the spread is wide—that is where blending with **q** matters. The **α-model** is the second AutoML piece; it is how we avoid shipping a single brittle dial. The **Pareto** chart is not decoration—it is **`{PARETO_POLICY_POINT}`** against **`{PARETO_SWEEP_LABEL}`** once numbers exist.

**Beat 7 — dual AutoML on first-party Snowflake.**  
**Classification** for calibration. **Regression** for α. **AI_EMBED** for retrieval. **Streamlit-in-Snowflake** for the judge-facing surface. Two AutoML calls is deliberate: **memory** plus **two model hosts** is the cleanest reading of the Snowflake prize rubric without stretching definitions.

**Beat 8 — live demo.**  
Watch a **`/v1/chat/completions`** call with the sample fixture: parsed question, outcomes, posted price, ensemble, neighbors, `prophet_debug`, JSON probabilities in the assistant body. Then **Streamlit**: same pipeline as a dashboard you can deploy in **Snowsight**.

Close where we started: your own scoring explainer told us **Brier and AVER diverge**. **Our system is built to respect both—because you wrote that they would.**

---

### 10-minute pitch (~1200 words)

**Slide 0 — the quote.**  
Open with Prof. Xu: *"A has a higher Brier Score, but lower returns."*  
Hold the room for one breath. Then name the two metrics **by category**: absolute vs relative. This is beat **1** and **2** welded together: we did not improvise the objective. We **read the doc**.

**Phase / beat 1–2 — problem framing.**  
PM-RANK also says: *“the Brier score is an **absolute metric**, while the averaged return score is a **relative metric**.”*  
Translate for non-theorists. Students see **prices**. Graders see **outcomes**. A naïve LLM that echoes prices can look calibrated-ish on Brier and still **earn nothing** on AVER. A brave LLM without memory burns **both**. **Our wedge is structure**, not charisma.

**Phase / beat 3 — data ingest and memory (Snowflake).**  
We pull public Kalshi history, normalize into **`HISTORICAL_MARKETS`**, and run **candlestick backfill** so `q` at open and near resolution is **real**, not a constant placeholder—critical for AutoML to mean anything. Judges can replay `bootstrap_snowflake.py` → import → candlesticks → embed. That is the **substrate** every later slide stands on. *After your run, replace talking-point counts with* **`{N_HISTORICAL_MARKETS}`**, **`{N_WITH_EMBEDDINGS}`**, **`{N_BACKFILLED_ENSEMBLE}`**—grep-friendly tokens the same as **`{CALIB_BRIER}`**.

**Phase / beat 3b — why idempotent scripts matter for judges.**  
Hackathon demos love **snowflake** demos that only work on one laptop. We pushed **repeatable** commands in the README because Prof. Xu’s team should be able to **re-run** the boring parts without calling us. If your unpacked zip lacks a populated warehouse, you still get a green **`./run.sh`** and a honest **`prophet_debug`** trace—just expect weaker retrieval until **`embed_questions.py`** finishes.

**Phase / beat 4 — ensemble.**  
Inference fans out **three** models in parallel through Wafer. We collect per-model probability vectors, mean, and **variance**. That variance is not trivia—it is the **confidence feature** feeding calibration and the **α** head. This is also our latency story: we budget to wall-clock **max**, not **sum**.

**Phase / beat 4b — model identity as a feature, not swagger.**  
We name **GLM-5.1**, **Qwen 397B**, **Qwen 35B** because diversity is the **engineering** claim—if three clones agreed always, **disagreement** would not exist and Beat 6 would be fiction. If one model goes dark under rate limit, our integration tests still let the server boot—production should retry, not crash.

**Phase / beat 5 — retrieval.**  
We embed the live question, search neighbors with **cosine similarity**, filter by category hygiene, exclude the current market id when known, aggregate neighbor outcomes into **base rates**, and return **neighbor_count** and **mean_similarity** for debugging. Memory is not “RAG paragraphs.” It is **tabular forensics** on **resolved** markets similar to this one.

**Phase / beat 5 continued — AutoML calibrator (numbers placeholders).**  
Train **`SNOWFLAKE.ML.CLASSIFICATION`** on `META_TRAIN`—per-outcome rows with **realized labels**. Report holdout Brier against baselines. **Planned eval line for the slide:**  
*Brier: AGENT=`{CALIB_BRIER}` vs KALSHI_BASELINE=`{KALSHI_BRIER}` vs RAW_ENSEMBLE=`{ENSEMBLE_BRIER}`.*  
If the zip you are judging is **pre-Phase-4**, say plainly: **stubs today; numbers after training**. Honesty beats vapor.

**Phase / beat 6–7 — α-policy and second AutoML.**  
Offline we search **α*** per historical market using realized outcomes and posted prices—utility is the teacher. Features like **disagreement**, **neighbor_count**, **similarity**, **spread**, **category** feed **`SNOWFLAKE.ML.REGRESSION`**. At inference: **p_final = α · p_meta + (1 − α) · q**. **Pareto slide placeholders:**  
*Global α sweep vs learned α — Brier axis `{PARETO_BRIER_AXIS}`, AVER axis `{PARETO_AVER_AXIS}`; our point `{PARETO_POLICY_POINT}`.*  
Mark every numeric token **`{LIKE_THIS}`** so the team can grep-replace after notebooks **04** and **06** land.

**Phase / beat 8 — demo.**  
**Agent:** FastAPI, **`POST /v1/chat/completions`**, OpenAI-shaped JSON; probabilities as **`{"probabilities": [...]}`** inside assistant `content`; **`prophet_debug`** for judges who want the full trace.  
**Streamlit-in-Snowflake:** four tabs—**live forecast**, **Pareto**, **by category**, **memory search**—see `streamlit_app/README.md` for Snowsight deploy. Narrate one **curl** and one **UI** path so both command-line and visual judges are satisfied.

**Extra slide — data ingest as credibility.**  
Walk **`bootstrap_snowflake.py`** once. It is boring on purpose. Boring DDL means the demo is reproducible. Then **import**, **candlesticks**, **embed**, **backfill**—each script is idempotent enough for a judge laptop. If you skip this slide, someone will assume we hand-loaded five rows. We did not build for five rows.

**Extra slide — ensemble is latency-aware.**  
Three calls, one winner’s clock. That matters inside a **10-minute** Arena window—we are not stacking serial 40-second monologues per model. Disagreement variance becomes a scalar feature cheap enough to log every request.

**Extra slide — retrieval as “institutional memory” without poetry.**  
Neighbors are not “vibes from similar questions.” They are **rows** with **realized_outcome**, **q** at two timestamps, and cosine **similarity** scores. Streamlit’s **Memory** tab is literally that query with a text box.

**Extra slide — calibration honesty.**  
`META_TRAIN` is the promise: labels come from **what resolved**, not from self-evaluation. If notebook **04** is missing from a zip, say it. Judges respect the DDL stub more than fake graphs.

**Extra slide — α and the second AutoML story.**  
Regression on **α*** is how Beat 6 stays technical. Global sweeps are fine baselines—they belong on the **Pareto** slide as **`{PARETO_SWEEP_LABEL}`**—but the product claim is **per-market** policy when the regressor ships.

**Close — side prize and integrity.**  
Five Snowflake-native surfaces—**embeddings**, **vector retrieval**, **AutoML classification**, **AutoML regression**, **SiS**—in one forecasting story. We are not claiming magic. We are claiming **alignment** between the platform, the labels, and the **Brier↔AVER** tension **PM-RANK** names explicitly.

When the training cells finish, paste the real scalars into **`{CALIB_BRIER}`**, **`{KALSHI_BRIER}`**, **`{ENSEMBLE_BRIER}`**, and the **`{PARETO_*}`** tuple. Until then, the story still holds: **architecture first, numbers second**.

---

### If they cut your mic early (priority order)

1. **Beat 1 quote** — absolute vs relative metrics from PM-RANK 0.3.1.  
2. **Beat 7** — two AutoML models on Snowflake, not one.  
3. **Beat 5** — labels == resolutions, **`{CALIB_BRIER}`** placeholders ok.  
4. **Beat 8** — `curl` + Streamlit path.  
5. Everything else is supporting evidence.

### If someone asks “is this just RAG?”

No. **RAG** usually retrieves *documents*. We retrieve **labeled market rows** with **prices** and **outcomes** and aggregate **empirical frequencies** plus **Kalshi residuals**—that is a **base rate**, not a quote machine. The **AutoML** head is what turns those features into **calibrated** probabilities under **Brier**, then **α** revisits **payoff** shape.

### If someone asks “prove Snowflake matters”

Show **`prophet_debug.neighbor_count`** on a live call with a populated warehouse. Show **`AGENT_PREDICTIONS`** DDL. Show **SiS** reading the same table names. **Five surfaces** is a rhetoric hinge, but the **shared catalog** is the substantive claim—no second shadow database.

### If someone asks about trial-tier AutoML limits

Say the truth: **`calibrator.py`** already sketches a **sklearn** shadow path so inference survives if **`SNOWFLAKE.ML.CLASSIFICATION`** is unavailable. The **pitch** favors first-party AutoML; the **engineers** refuse a demo that dies on account SKU.

