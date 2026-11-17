# All_ArchetypePlan: Universal Autonomous Data Science Pipeline

**Document Version:** 1.0.0  
**Target Milestone:** Post-Sprint 3 (Scheduled immediately following Security Hardening, 5-Stage Pruning, and Swarm Vega-Lite Integration)  
**Author:** Antigravity (Autonomous Systems Architect & Tech Lead)  
**Reference Codebase:** `Sorrel / DSA Agent` (`d:\(Dev2)DSA_AGENT`)  

---

## 1. Executive Vision & Strategic Purpose

The ultimate objective of Sorrel is to operate as a **universal, autonomous senior data scientist** capable of delivering sector-grade, mathematically rigorous, and prescriptive analysis on **any nature of structured data** without requiring manual feature engineering, dataset reshaping, or domain prompting from the user.

Whether a user drops an e-commerce transaction log, a high-frequency financial tick series, an industrial sensor stream, a hospital clinical trial, or an HR employee census, the agent must:
1. **Deterministically classify the dataset's structural archetype** in Stage 1 without LLM hallucinations.
2. **Automatically execute canonical pre-modeling transformations** (e.g., non-stationary prices to log returns, raw checkout lines to customer RFM tables).
3. **Seed sector-grade hypotheses** in Stage 2 that reflect industry-standard questions rather than trivial bivariate correlations.
4. **Route dynamically to specialized quantitative tools** already present in `src/tools/` (Apriori, Cox survival, mixed-effects models, FFT multi-seasonality, Bradley-Terry rankings).
5. **Produce prescriptive simulations** in Stage 5 via the 2x2 Action Prioritization Matrix ("What lever should the business pull, and what will the outcome be?").

---

## 2. Chronological Placement in the Engineering Roadmap

This plan builds directly on top of the active sprint work:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CURRENT ACTIVE SPRINTS (P0/P1 Launch Readiness)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Sprint 1 (Committed: 8f99a1f): Security & Host Containment (H-1..H-8)      │
│ • Sprint 2 (Underway): Core Engine Pruning & 5-Stage Workflow Formalization │
│ • Sprint 3 (Next): Swarm Vega-Lite Live Binding & Answers Tab UX Polish     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ALL_ARCHETYPEPLAN (Universal Domain Expansion — Next Milestone)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Phase 4.1: Structural Archetype Signature Classifier                      │
│ • Phase 4.2: Automated Pre-Modeling Reshaping Pipeline                     │
│ • Phase 4.3: Universal Domain Pack Expansion & KPI Standards                │
│ • Phase 4.4: Archetype-Native Question Agenda Seeding                       │
│ • Phase 4.5: Prescriptive What-If Simulation Expansion                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The 10 Universal Data Archetypes (MECE Classification)

Every structured or semi-structured dataset in business, science, and engineering maps to one of ten canonical archetypes:

```
                                  DATASET INGESTED (STAGE 1)
                                              │
          ┌───────────────────────────────────┼───────────────────────────────────┐
          │                                   │                                   │
   [EVENT & TEMPORAL]                  [ENTITY & DURATION]                 [RELATIONAL & SPATIAL]
   1. Transactional Event Stream       4. Cross-Sectional Entity           7. Market Basket Sets
   2. Continuous Time-Series           5. Survival / Censored Duration     8. Network Graph
   3. Panel / Longitudinal Repeated    6. Survey / Psychometric Ordinal    9. Geospatial Field
                                                                          10. Hybrid Text-Tabular
```

### Detailed Archetype Specification

| # | Archetype | Real-World Domains | Structural Fingerprint (Profiler) | Mandatory Transformations | Canonical Tool Pipeline |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Transactional Event Stream** | E-commerce checkouts, POS retail, bank card transactions, web clickstreams. | `order_id` + `customer_id` + `timestamp` + `amount`/`quantity`. Granularity: row-per-event. | Aggregate to Customer RFM (Recency, Frequency, Monetary) & Daily Sales matrices. | `cohort_analysis`, `concentration_analysis` (Gini/Pareto 80/20), `elasticity`. |
| **2** | **Continuous Time-Series & Signals** | Stock prices, crypto, forex, environmental air quality, power grid load, server CPU/RAM. | Monotonic timestamp index + continuous numeric values. Granularity: uniform time intervals. | Differentiate non-stationary prices to log returns: $r_t = \ln(P_t/P_{t-1})$; rolling volatility. | `time_series`, `forecast` (ARIMA/Prophet), `multi_seasonal_analysis` (FFT), `financial_analysis`. |
| **3** | **Panel / Longitudinal Repeated** | Multi-store sales over 24 months, clinical patient visits, SaaS accounts across quarters. | `entity_id` + `time_period` + metric measures. Multiple observations per entity over time. | Check entity balance, compute within-entity deltas and lag features. | `mixed_model` (hierarchical fixed/random effects), `causal_analysis` (Difference-in-Differences). |
| **4** | **Cross-Sectional Entity Snapshot** | HR employee census, loan applications, house prices, B2B deal CRM. | One row per unique entity. Rich mix of numerical features and categorical factors. | One-hot / target encoding, standardization of numerical features. | `regression`, `ml_pipeline` (CV tournaments), `segment_comparison`, `equity` (fairness audits). |
| **5** | **Survival / Censored Duration** | Customer churn, industrial equipment wear (MTBF), clinical disease recurrence, employee tenure. | `duration`/`tenure` column + binary status event flag (`churned`, `failed`, `dead`). | Verify non-negative duration; mark right-censored observations. | `survival` (Kaplan-Meier curves, Cox Proportional Hazards, log-rank tests). |
| **6** | **Survey / Psychometric / Ordinal** | Customer satisfaction (CSAT), NPS, employee feedback (1–5 Likert), clinical assessments. | Bounded integer/ordinal scales (1–5, 1–7, 1–10), respondent ID, demographic splits. | Align polarity (reverse-code negative questions), calculate Cronbach's alpha reliability. | `statistical_analysis` (Mann-Whitney, Kruskal-Wallis non-parametric), ordinal regressions. |
| **7** | **Market Basket / Combinatorial Sets** | Supermarket basket receipts, clinical co-prescriptions, content playlists, job skills. | `basket_id` / `transaction_id` + `item_id`. One-to-many relationship per basket. | Pivot long transactional list to sparse boolean item-incidence matrix. | `basket` (Apriori association rules, Support, Confidence, Lift calculations). |
| **8** | **Network / Graph / Relational** | Supply chain routes, inter-bank money transfers, management org trees, telecommunications. | `source_id` + `target_id` + optional `weight`/`latency`/`volume`. | Build adjacency list, compute in/out degree distributions. | `graph_analysis` (degree centrality, PageRank, community detection, bottleneck cuts). |
| **9** | **Geospatial / Spatial Field** | Delivery logistics, ride-share pickups, retail store catchment, disease outbreaks. | `latitude` + `longitude` coordinate pairs or standard Zip/Postal codes. | Project coordinates, calculate spatial distance matrices. | `geospatial` (spatial DBSCAN clustering, Moran's I spatial autocorrelation). |
| **10**| **Hybrid Text-Tabular** | Customer service Zendesk tickets, Amazon product reviews, medical clinical notes. | High-cardinality unstructured free text alongside structured ratings and dates. | TF-IDF vectorization, VADER/sentiment polarity extraction, keyword n-grams. | `text_analysis` (sentiment vs. rating correlation, keyword impact regression). |

---

## 4. Technical Architecture: Wiring into the 5-Stage Engine

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION & ARCHETYPE RECOGNITION (src/core/profiler.py & reshaping.py)  │
│ 1. Structural Fingerprint Classifier: Detect DatasetArchetype enum               │
│ 2. Automated Reshaping: Apply canonical transforms before model fitting          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ STAGE 2: ARCHETYPE-NATIVE HYPOTHESES (src/core/agenda.py & domain_packs.py)      │
│ Dynamic injection of sector KPIs, regulatory bounds, and domain hypothesis seeds  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ STAGES 3 & 4: ARCHETYPE TOOL ROUTING (src/core/controller_planning.py)           │
│ Confine tool selection to mathematically sound methods (ban OLS on non-stationary)│
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ STAGE 5: PRESCRIPTIVE 2x2 ACTION MATRIX (src/core/controller_report.py)          │
│ Run what-if simulations (ScenarioSimulationTool, ThresholdAnalysisTool)          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1. Stage 1: Structural Signature Classifier (`src/core/profiler.py`)
Add an explicit `DatasetArchetype` enum and deterministic detector to `profiler.py`:

```python
class DatasetArchetype(str, Enum):
    TRANSACTIONAL = "transactional"
    CONTINUOUS_SERIES = "continuous_series"
    PANEL = "panel"
    CROSS_SECTIONAL = "cross_sectional"
    SURVIVAL = "survival"
    SURVEY = "survey"
    BASKET = "basket"
    NETWORK = "network"
    GEOSPATIAL = "geospatial"
    HYBRID_TEXT = "hybrid_text"
```

**Heuristic Decision Ladder (Deterministic & Fast):**
1. **Geospatial Check:** Are there latitude/longitude float pairs? $\rightarrow$ `GEOSPATIAL`.
2. **Network Check:** Are there explicit `source`/`target` or `from_node`/`to_node` columns? $\rightarrow$ `NETWORK`.
3. **Survival Check:** Is there a positive numeric duration/tenure column AND a binary event column (`churn`, `died`, `failed`, `cancelled`)? $\rightarrow$ `SURVIVAL`.
4. **Basket Check:** Is there a transaction ID with high cardinality repetition (>3 items per ID) and categorical item names? $\rightarrow$ `BASKET`.
5. **Time-Series / Continuous Check:** Is there a monotonic datetime index/column with numeric metrics and no repeated entity IDs? $\rightarrow$ `CONTINUOUS_SERIES`.
6. **Panel Check:** Does an entity identifier (`store_id`, `patient_id`, `account_id`) repeat regularly across consecutive time periods? $\rightarrow$ `PANEL`.
7. **Survey Check:** Are >60% of numeric columns bounded small integers (e.g., min $\ge 1$, max $\le 10$)? $\rightarrow$ `SURVEY`.
8. **Hybrid Text Check:** Does the dataset contain free-text columns with average character length $> 50$? $\rightarrow$ `HYBRID_TEXT`.
9. **Default:** If none of the above $\rightarrow$ `CROSS_SECTIONAL`.

### 4.2. Stage 1: Automated Reshaping Pipeline (`src/core/reshaping.py`)
Raw data is converted into its mathematically required format before modeling:
- **Financial / Continuous Data:** Raw prices $\rightarrow$ computed log returns ($r_t$) and rolling 30-period volatilities.
- **Transactional Data:** Raw order items $\rightarrow$ aggregated into Customer RFM (Recency, Frequency, Monetary) profiles.
- **Basket Data:** Long transactions $\rightarrow$ pivoted into a binary presence-absence matrix for association rules.
- **Hybrid Text Data:** Unstructured text $\rightarrow$ tokenized sentiment scores and top TF-IDF features appended as numerical columns.

### 4.3. Stage 2: Archetype-Specific Question Agenda (`src/core/agenda.py`)
Replace generic questions with sector-standard inquiries:
- **E-Commerce / Shopping:**
  - *"What is the 90-day repurchase rate, and which customer cohorts exhibit the highest LTV?"*
  - *"Which product bundles generate cross-sell lift $> 2.0$ without cannibalizing flagship sales?"*
- **Financial Markets:**
  - *"Are asset returns stationary (ADF test)? Is volatility clustered around specific macro regimes?"*
  - *"What is the maximum drawdown duration, and what is the annualized Sharpe/Sortino ratio?"*
- **Customer Churn & Retention:**
  - *"What is the median customer half-life, and which contractual levers significantly reduce hazard rates?"*
- **Operations & Manufacturing:**
  - *"Are sensor signals exhibiting non-linear threshold tipping points before component failure?"*

### 4.4. Stages 3 & 4: Safe Quantitative Routing (`src/core/controller_planning.py`)
Prevent mathematically invalid modeling:
- **Anti-Pattern Guard:** Banning ordinary least squares (OLS) regression on raw non-stationary time series (enforcing difference stationarity first).
- **Survival Guard:** Banning standard logistic regression on right-censored customer churn (routing to Cox Proportional Hazards).
- **Panel Guard:** Requiring cluster-robust standard errors or random effects when analyzing multi-store repeated measures.

### 4.5. Stage 5: Prescriptive What-If Simulation (`src/tools/scenario_simulation.py`)
The final dossier must produce a **2x2 Action Prioritization Matrix** tailored to the domain:
- **E-Commerce:** Simulates revenue impact under a 5% discount using estimated price elasticity.
- **Finance:** Simulates portfolio drawdown under volatility shocks.
- **Churn:** Simulates retention gain from onboarding intervention on high-hazard cohorts.

---

## 5. Implementation Roadmap (Post-Sprint 3)

| Phase | Milestone | Primary Deliverable | Target Files |
| :---: | :--- | :--- | :--- |
| **Phase 4.1** | **Archetype Classifier** | Deterministic detection of all 10 archetypes in Stage 1 profiling. | `src/core/profiler.py`, `tests/test_archetype_detector.py` |
| **Phase 4.2** | **Reshaping Engines** | Automatic pre-modeling transformations (log returns, RFM, basket incidence). | `src/core/reshaping.py`, `src/core/data_processing.py` |
| **Phase 4.3** | **Domain Pack Expansion** | Sector-standard KPIs for Retail, SaaS, Finance, and Healthcare. | `src/core/domain_packs.py`, `src/core/domains.py` |
| **Phase 4.4** | **Agenda & Hypotheses** | Domain-tailored question generation in Stage 2 planning. | `src/core/agenda.py`, `src/core/prompt_manager.py` |
| **Phase 4.5** | **Prescriptive Simulation** | Domain-native driver interventions in the 2x2 Action Matrix. | `src/tools/scenario_simulation.py`, `src/core/controller_report.py` |
| **Phase 4.6** | **Universal Test Matrix** | End-to-end integration test with 10 benchmark datasets representing each archetype. | `tests/test_universal_archetypes.py` |

---

## 6. Verification & Quality Gates

Each archetype must be verified by an automated integration test against a canonical reference dataset:
1. `AirQualityUCI.csv` $\rightarrow$ Classified as `CONTINUOUS_SERIES` / Sensor Telemetry.
2. `Retail_Orders.csv` $\rightarrow$ Classified as `TRANSACTIONAL` / Basket.
3. `Telco_Churn.csv` $\rightarrow$ Classified as `SURVIVAL`.
4. `Store_Weekly_Sales.csv` $\rightarrow$ Classified as `PANEL`.
5. `Employee_Attrition.csv` $\rightarrow$ Classified as `CROSS_SECTIONAL`.

Every phase must pass all existing repository gates:
- `ruff check .` with **0 violations**.
- `mypy src/` with **0 errors across all source files**.
- 100% of existing unit and regression tests passing.
