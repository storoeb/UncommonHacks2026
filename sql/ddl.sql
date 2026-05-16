-- ProphetHacks 2026 — Snowflake schema
-- Run via: python scripts/bootstrap_snowflake.py
--
-- All tables live in the schema specified by SNOWFLAKE_DATABASE.SNOWFLAKE_SCHEMA
-- (defaults to TRAINING_DB.TRAININGLAB per the user's .env). Names are unprefixed
-- so the same DDL works regardless of the configured schema.

-- ============================================================================
-- 1. Resolved historical markets — the institutional memory.
--    Source: Kalshi public API (and any other resolved-market source we add).
-- ============================================================================
CREATE TABLE IF NOT EXISTS HISTORICAL_MARKETS (
    market_id            STRING        NOT NULL,           -- Kalshi ticker or similar
    source               STRING        NOT NULL,           -- 'kalshi' | 'polymarket' | ...
    category             STRING,                            -- 'sports'|'politics'|'crypto'|'science'|'tech'|'econ'|'other'
    series_ticker        STRING,                            -- Kalshi series ticker
    question_text        STRING        NOT NULL,
    outcomes             ARRAY,                             -- [string, ...]
    open_ts              TIMESTAMP_NTZ,
    resolve_ts           TIMESTAMP_NTZ,
    q_kalshi_at_open     ARRAY,                             -- [float, ...] per outcome
    q_kalshi_at_24h_pre  ARRAY,                             -- [float, ...] per outcome 24h before resolution
    realized_outcome     INT,                               -- index into outcomes
    question_embedding   VECTOR(FLOAT, 768),                -- AI_EMBED('snowflake-arctic-embed-m-v1.5', ...)
    raw_payload          VARIANT,                           -- original Kalshi JSON for debugging
    ingested_at          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_historical_markets PRIMARY KEY (market_id, source)
);


-- ============================================================================
-- 2. Backfill tables — features computed retrospectively over historical markets
--    so the calibrator has training data.
-- ============================================================================

-- One row per historical market: the ensemble's prediction had it been live.
CREATE TABLE IF NOT EXISTS ENSEMBLE_BACKFILL (
    market_id         STRING NOT NULL,
    source            STRING NOT NULL,
    model_outputs     VARIANT,                              -- [{model, probabilities, latency_s}, ...]
    p_ensemble        ARRAY,                                -- mean probabilities
    p_ensemble_var    FLOAT,                                -- avg per-outcome variance across models
    backfilled_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_ensemble_backfill PRIMARY KEY (market_id, source)
);

-- One row per historical market: the retrieval features had it been live.
-- (When backfilling a historical market, exclude self from neighbor search.)
CREATE TABLE IF NOT EXISTS BASE_RATE_BACKFILL (
    market_id         STRING NOT NULL,
    source            STRING NOT NULL,
    neighbor_count    INT,
    mean_similarity   FLOAT,
    base_rate         ARRAY,                                -- aggregated outcome rate from neighbors
    kalshi_residual   ARRAY,                                -- mean(o_ik - q_kalshi_at_open[k]) across neighbors
    backfilled_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_base_rate_backfill PRIMARY KEY (market_id, source)
);

-- One row per historical market: the calibrator's output (populated in Phase 4).
-- Used as input to the α-policy trainer in Phase 6.
CREATE TABLE IF NOT EXISTS META_PREDICTIONS_BACKFILL (
    market_id      STRING NOT NULL,
    source         STRING NOT NULL,
    p_meta         ARRAY,                                   -- calibrated probabilities
    alpha_star     FLOAT,                                   -- α* from line-search (target for α-policy)
    backfilled_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT pk_meta_predictions_backfill PRIMARY KEY (market_id, source)
);


-- ============================================================================
-- 3. Live predictions log — every request the agent has served.
--    Resolved fields populated later by a sweep against actual outcomes.
-- ============================================================================
CREATE TABLE IF NOT EXISTS AGENT_PREDICTIONS (
    request_id       STRING NOT NULL,
    request_ts       TIMESTAMP_NTZ NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    market_id        STRING,
    category         STRING,
    question_text    STRING,
    outcomes         ARRAY,
    q_kalshi         ARRAY,
    p_ensemble       ARRAY,
    p_ensemble_var   FLOAT,
    base_rate        ARRAY,
    p_meta           ARRAY,
    p_final          ARRAY,
    alpha            FLOAT,
    realized_outcome INT,                                   -- populated later when resolved
    brier            FLOAT,
    payoff           FLOAT,
    CONSTRAINT pk_agent_predictions PRIMARY KEY (request_id)
);


-- ============================================================================
-- 4. Training view for the AutoML calibrator (one row per (market, outcome)).
--    Created in Phase 4 once backfill tables are populated.
--    Left as a comment for now — depends on rows existing.
-- ============================================================================
-- CREATE OR REPLACE VIEW META_TRAIN AS
-- SELECT
--     m.category,
--     m.q_kalshi_at_24h_pre[f.index]::FLOAT  AS q,
--     e.p_ensemble[f.index]::FLOAT           AS p_llm,
--     e.p_ensemble_var                       AS disagreement,
--     b.base_rate[f.index]::FLOAT            AS base_rate,
--     CASE WHEN m.realized_outcome = f.index THEN 1 ELSE 0 END AS y
-- FROM HISTORICAL_MARKETS m
-- JOIN ENSEMBLE_BACKFILL e USING (market_id, source)
-- JOIN BASE_RATE_BACKFILL b USING (market_id, source),
-- LATERAL FLATTEN(input => m.outcomes) f
-- WHERE m.realized_outcome IS NOT NULL;


-- ============================================================================
-- 5. Training view for the AutoML α-policy (one row per market).
--    Created in Phase 6 once META_PREDICTIONS_BACKFILL is populated.
-- ============================================================================
-- CREATE OR REPLACE VIEW ALPHA_TRAIN AS
-- SELECT
--     m.market_id,
--     m.category,
--     e.p_ensemble_var                                       AS disagreement,
--     b.neighbor_count                                       AS neighbor_count,
--     b.mean_similarity                                      AS neighbor_sim,
--     ARRAY_MAX(m.q_kalshi_at_24h_pre)::FLOAT
--       - ARRAY_MIN(m.q_kalshi_at_24h_pre)::FLOAT            AS q_spread,
--     mp.alpha_star                                          AS alpha_star
-- FROM HISTORICAL_MARKETS m
-- JOIN ENSEMBLE_BACKFILL e USING (market_id, source)
-- JOIN BASE_RATE_BACKFILL b USING (market_id, source)
-- JOIN META_PREDICTIONS_BACKFILL mp USING (market_id, source)
-- WHERE mp.alpha_star IS NOT NULL;
