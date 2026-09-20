# FutureScope

Systems that could raise the **quality of the analysis** without slowing the pipeline down, for **any kind of data**. Revised 2026-09-21 (second revision, after Round 9; see `IMPROVEMENTS.md`).

Nothing here is started.

**What this revision changes.** The first revision generalised the wording but was checked against only six kinds of data. This one starts from a systematic taxonomy of all kinds of data (six independent axes, section 3), cross-checks it against 16 domain groups (section 3.7), derives every property the pipeline would need to recognise (section 4), and then revises the systems accordingly (section 5). It adds roughly 30 properties and about 40 systems that the first revision did not have.

## 1. Scope: what "all kinds of data" means here

The pipeline analyses **tables** (CSV, Excel, Parquet, JSON, Stata/SAS/SPSS, Feather, HDF5, NetCDF, up to a row cap). "All kinds of data" therefore means every structure that can be expressed as, or derived from, a table:

| Tier | Meaning | Examples |
|---|---|---|
| **1. Native tables** | One row per unit, columns of variables. | cross-sections, surveys, lab batches, rosters |
| **2. Structures carried in tables** | The table is a container for a richer structure that analysis must recognise. | time series, panels, event logs, transactions, multi-table relational data, edge lists (graphs), coordinates and regions (spatial), trajectories, categorical sequences, curves and spectra (one curve per row or per group), text columns, nested JSON, wide "time in the header" tables, pivot-style report tables, aggregated rows (mean / SD / n per group), embedding-vector columns |
| **3. Raw non-tabular modalities** | Not analysed directly. | images, audio, video, point clouds, raw documents, binaries |

**Tier 3 is an explicit non-goal for analysis.** The system should still *detect* it (path or URL columns, blob columns), say clearly that it is not analysed, and analyse whatever tabular metadata or extracted features accompany it.

## 2. Design principles

1. **Key every system on a data property, never on a domain.** Each row below reads "when the data has property P, do X". If a system cannot be phrased that way, it is domain-specific and belongs in the domain-packs extension point (principle 2).
2. **Domain knowledge enters through exactly one door: domain packs.** Data files (reference tables, limits, KPI definitions, standard methods) with one schema. With no matching pack the core still works.
3. **Gate by property, cost nothing otherwise.** Every system decides applicability from the profile (the existing `applies_to` mechanism). Absent property: no run, no prompt space, no time.
4. **Statistical honesty is universal.** Dependence, seasonality, regime changes, multiplicity, confounding, heavy tails, selection and design corrupt conclusions on every kind of data. These checks are core.
5. **Deterministic first, LLM optional.** Every system works with the LLM off; the LLM chooses among, explains and requests systems.
6. **Disclose, don't silently repair.** Any repair (nulling a placeholder, reshaping, sampling, demoting a finding) is stated with numbers.
7. **A general system replaces a special case.** Generalised periodicity detection replaces the hour-of-day and weekday profiles added in Round 9.
8. **Never assume what a generic pipeline gets wrong.** Section 8 lists the hidden assumptions (header on row 1, one table per sheet, independent rows, meaningful means, a time axis, ...). Each is a test case.
9. **Prove generality on a corpus, not on one dataset** (section 7).
10. **Say when the method does not apply.** A census, a deterministic simulation and an aggregate table each make some standard tests meaningless. The right output is a plain-language "this test does not apply because ...", not a p-value.

## 3. A taxonomy of all kinds of data

Six independent axes. Any real dataset is one point on each axis, so together they cover more than any list of domains can.

### 3.1 Axis A: structure (what the rows and columns mean)

| Kind | Description | Examples across domains |
|---|---|---|
| Cross-section | one row per unit, one moment | HR snapshot, survey wave, lab batch, customer table |
| Time series (regular) | one series, fixed spacing | hourly air quality, 15-min load, daily prices, monthly indices |
| Time series (irregular) | timestamps at uneven spacing | tick data, asynchronous sensors, clinical visits |
| Many series | thousands of parallel series | SKUs, meters, tickers, stations, sensors on a plant |
| Panel / longitudinal | many units observed repeatedly | employees over years, patients, stores, firms |
| Event log / point process | timestamped events, no fixed spacing | clicks, incidents, logins, crimes, transactions, alarms |
| Transactions / baskets | events grouped into orders | retail, banking, telecom usage |
| Relational (multi-table) | several tables linked by keys | orders to customers to products, plant to unit to sensor |
| Graph / network | edges between entities | payments, social, citations, call detail, supply chain |
| Spatial (points, regions) | coordinates or area codes | air-quality stations, real estate, crime, census areas |
| Spatio-temporal / trajectories | position over time | GPS, vessels, aircraft, animal tracking, delivery routes |
| Gridded / raster | values on a lattice | climate, remote sensing, medical imaging as arrays |
| Categorical sequences | ordered symbols | clickstreams, genomes, treatment pathways, log templates |
| Functional / curve | each unit is a curve | spectra, growth curves, ECG, load profiles, dose-response |
| Compositional | parts of a whole | budget shares, chemical composition, vote shares, time use |
| Contingency / matrix | the table is a cross-tab or a flow matrix | origin-destination, confusion matrices, mobility tables |
| Aggregated rows | rows are group summaries (mean, SD, n) | published statistics, meta-analysis, cohort tables |
| Report-style | human-readable layout, not a clean table | multi-row headers, totals, notes, several tables per sheet |
| Wide "time in the header" | one column per year, month, week or visit | World Bank style indicators, repeated measures `t1..tn` |
| Free text | long text columns | reviews, tickets, clinical notes, open answers |
| Nested / semi-structured | JSON or list-valued cells | event payloads, tags, multi-valued diagnoses |
| Vector-valued | embeddings or fixed arrays in a cell | model embeddings, feature vectors |
| Simulation / model output | inputs and outputs of a computed model | parameter sweeps, ensembles, design of experiments |

### 3.2 Axis B: variable type (measurement scale)

Nominal · ordinal (Likert, grades, ranks) · interval · ratio · count · rate or ratio (with an exposure denominator) · proportion or bounded [0,1] · percentage · **compositional** · semi-continuous (many zeros plus a positive continuous part) · **circular or angular** (wind direction, hour of day, month, phase) · duration · time-to-event · monetary (with currency, real vs nominal) · date, time, time zone, calendar (fiscal, trading, business days) · identifier and quasi-identifier · boolean flag · multi-valued set · hierarchical code (ICD, NAICS, SKU tree, geography levels) · geometry · free text · value with attached uncertainty (± error, standard error, interval, range like "10-20") · censored value (`<LOD`, top-coded, `>100`).

### 3.3 Axis C: how the data was generated (design)

Census (whole population) · probability sample with weights and strata · cluster or multistage sample · convenience or found or web-scraped data (selection bias) · observational cohort · case-control · randomised experiment (A/B, factorial, crossover, blocked, dose-response, sequential) · quasi-experiment or natural experiment · continuous sensors · administrative or transactional records · simulation or computed output (no sampling noise) · synthetic data · multi-source harmonised data (different definitions per source).

### 3.4 Axis D: quality regime (what is wrong with the data)

Placeholder codes · detection limits and censoring · truncation · measurement error and instrument drift · **batch effects** (plates, lots, instruments) · **heaping and digit preference** (ages ending in 0 or 5) · duplicates and near-duplicates · **reporting delay and right-truncation** (recent periods incomplete) · **revisions and vintages** (same period reported several times) · informative missingness · outage blocks · unit or currency mixing inside a column · schema drift between files · **subtotal and total rows mixed with data** · implicit row order (no timestamp but order matters) · structural zeros · sampling frame gaps · fabricated or manipulated data (Benford, terminal-digit tests) · PII.

### 3.5 Axis E: what the user is asking

Describe · compare groups · associate · predict · explain or estimate a cause · forecast · detect anomalies or changes · segment · rank or score · monitor or control a process · audit or find fraud · assess fairness · benchmark against a standard · optimise or decide · track drift between versions.

### 3.6 Axis F: scale and size

Tiny (n < 30) · small · medium · large (10^5 to 10^7 rows) · wide (thousands of columns, p >> n) · many tables · streaming or appended in batches.

### 3.7 Cross-check: 16 domain groups and the properties they bring

If a domain group brings a property that is not in sections 3.1 to 3.6, the taxonomy has a hole. None did after the additions above.

| Domain group | Typical properties (axes A to F) |
|---|---|
| Environment, climate, hydrology, agriculture, ecology | regular and irregular time series, shared cycles, gridded and spatial, extremes, sensor drift, placeholders, seasonal |
| Physical sciences, astronomy, chemistry, engineering experiments | measurement error and replicates, curves and spectra, detection limits, batch and instrument effects, simulation output, design of experiments |
| Life sciences, genomics, proteomics, lab assays | wide p >> n, batch effects, compositional, counts, replicates, dose-response, sequences |
| Clinical, epidemiology, public health, pharma | time-to-event, competing risks, longitudinal panels, case-control, counts with exposure, reporting delay, PII, protected attributes |
| Finance, markets, crypto, insurance, actuarial | irregular and regular series, heavy tails and extremes, regimes, volatility clustering, trading calendars, corporate-action adjustments, claim counts and semi-continuous amounts, reporting triangles |
| Economics, official statistics, census, labour | weights and design, seasonal adjustment, vintages and revisions, mixed frequency, aggregated rows, report-style tables, wide time-in-header, compositional, real vs nominal |
| Retail, e-commerce, marketing, CRM, SaaS | transactions and baskets, many series with intermittent demand, funnels and sessions, experiments, counts, hierarchies (SKU trees), multi-currency |
| Operations, supply chain, logistics, transport, aviation, maritime | relational multi-table, graphs and flows (origin-destination), trajectories, event logs, calendars and time zones, capacity and constraints |
| Manufacturing, industrial IoT, energy, utilities, buildings | many sensor series, stable processes with targets, load curves, regimes, spatial layout, constraints (mass and energy balance), predictive maintenance events |
| IT operations, cybersecurity, software engineering | event logs, sequences, graphs, counts, bursty arrivals, nested JSON, schema drift, class imbalance |
| People analytics, education, psychology, social science | panels, ordinal scales and Likert, surveys with weights, clustered and nested data (students in schools), sensitive attributes, small groups, PII |
| Sports, games, esports, ratings | pairwise comparisons and rankings, sequences, small n per unit, home effects, tracking trajectories |
| Media, web, social, search, advertising | events and sessions, text, graphs, counts, heavy tails, experiments, selection bias |
| Geospatial, urban planning, traffic, real estate, emergency services | spatial dependence, trajectories, regions and hierarchies, rasters, aggregation-level effects |
| Government, legal, elections, policy, public administration | aggregated rows and report-style tables, ordinal and categorical, compositional (vote shares), revisions, small areas, fairness |
| Humanities, culture, archives, genealogy, linguistics | text, hierarchical codes, irregular dates and uncertain dates, small n, entity resolution across records, network of relations |

## 4. The property vocabulary, with honest status

What the profile would need to recognise. **Status as of Round 9**, from the tool inventory and profile fields; not re-audited line by line. `Have` = detected and used; `Partial` = detected or used in a limited way; `Missing` = not present; `Unaudited` = I have not checked.

### 4.1 Structure

| Property | How it can be detected | Status |
|---|---|---|
| Time axis and grain | datetime column, median gap | Have |
| Regular vs irregular spacing, gaps | gap distribution | Partial |
| Many parallel series | entity column x time, many groups | Partial (panel detected) |
| Panel / entity | repeated id column | Have |
| Event log | timestamp plus entity, uneven gaps | Have (archetype), limited analysis |
| Transactions / baskets | repeated order id with item rows | Have (basket tool) |
| Multi-table relational | several files, key overlap | Have (auto-join, review) |
| Graph edge list | two id-like columns drawn from the same value set | Missing |
| Spatial coordinates or area codes | lat/lon names and ranges, region codes | Have (coordinates), Missing (area codes, spatial dependence) |
| Trajectories | id x time x coordinates | Missing |
| Gridded / raster | array-shaped file, lat/lon/time dimensions | Partial (NetCDF read, capped) |
| Categorical sequences | ordered symbols per id | Missing |
| Functional / curves | one row per curve, or repeated x per group | Partial (single curve fit) |
| Compositional | share-like columns whose row sums are about constant | Missing |
| Contingency / flow matrix | row and column headers with counts, square symmetric | Missing |
| Aggregated rows (mean, SD, n) | `n` or count column beside mean, SD, SE columns | Missing |
| Report-style layout | multi-row headers, blank spacer rows, notes below, several tables per sheet | Unaudited (likely first table only) |
| Wide "time in the header" | column names that are years, months, weeks or `t1..tn` | Missing |
| Free text columns | long strings | Have (light text analysis) |
| Nested / multi-valued cells | JSON or delimited lists inside cells | Partial (JSON flattened at read) |
| Vector-valued cells | fixed-length numeric arrays | Missing |
| Simulation / model output | parameter columns with a deterministic output, no noise | Missing |
| Implicit row order | monotone index, sequentially ordered measurements without timestamps | Missing |

### 4.2 Variable type

| Property | How it can be detected | Status |
|---|---|---|
| Ordinal (Likert, grades) | few integer levels with ordered meaning | Have (role), Partial (methods) |
| Counts and overdispersion | non-negative integers, variance above the mean | Missing |
| Rates with exposure | events plus a population, hours or area column | Missing |
| Proportions / bounded | values in [0,1] or 0 to 100 | Partial (unit hint) |
| Semi-continuous (zero-inflated) | mass at zero plus continuous positives | Missing |
| Circular / angular | direction, angle, hour, month, phase names and ranges | Missing |
| Heavy tails and extremes | tail index, kurtosis | Partial (skew only) |
| Heaping / digit preference | last-digit distribution far from uniform | Missing |
| Digit-law anomalies (Benford) | monetary or naturally scaled amounts | Missing |
| Value with attached uncertainty | `±` in values, adjacent SE, CI or range columns | Partial (`±` parsed) |
| Censored values | `<LOD`, `>max`, top-coding | Have (placeholders and detection limits) |
| Monetary and currency | currency symbols or codes, real vs nominal | Partial (unit hint) |
| Time zones, fiscal / trading / business calendars, holidays | offsets, weekday gaps, fixed non-trading days | Missing |
| Hierarchical codes | code prefixes that nest (ICD, NAICS, geography) | Missing |
| Unit heterogeneity inside a column | mixed suffixes or magnitudes | Missing |
| Identifier vs quasi-identifier | uniqueness, key names | Have (identifier), Partial (quasi) |

### 4.3 Design and dependence

| Property | How it can be detected | Status |
|---|---|---|
| Serial dependence | autocorrelation | Partial |
| Shared cycle or common driver across columns | correlated cycles or residuals | Missing |
| Cluster / nested groups | grouping columns with repeated members | Partial (mixed model) |
| Spatial dependence | Moran's I on coordinates or areas | Missing |
| Network dependence | shared endpoints between rows | Missing |
| Batch / plate / lot / instrument effects | categorical batch variable with mean shifts | Missing |
| Survey weights and strata | weight column, stratum and cluster ids | Missing |
| Experimental design (blocks, factors, crossover, multiple arms) | arm, block and period columns | Partial (two-arm test) |
| Census or whole population | rows equal the stated population | Missing |
| Selection bias / found data | unrepresentative sampling frame clues | Missing |
| Regime changes / non-stationarity | level and variance shifts | Partial (level shift detector, unvalidated) |
| Reporting delay / right-truncation | recent periods systematically lower | Partial (final period excluded) |
| Revisions and vintages | repeated periods with differing values | Missing |
| Drift between snapshots | same schema across uploads | Missing |

### 4.4 Quality and scale

| Property | How it can be detected | Status |
|---|---|---|
| Placeholder codes | repeated extreme values | Have |
| Missingness structure (blocks, time-related, informative) | run lengths, association with other columns | Partial (share only) |
| Duplicates and near-duplicates | exact and fuzzy match | Partial (exact) |
| Subtotal / total rows | rows equal to the sum of others, labels like Total, All | Missing |
| Constraints between columns | inequality and sum relations | Partial (formula relations) |
| Sensitive grouping attributes, PII | names and value sets | Have |
| Tiny n, wide p >> n, large n | shape | Partial (sufficiency flag, sampling above 200k) |
| Many related tables | key graph | Have |

## 5. Systems, keyed on the properties above

Each system names the property that triggers it and gives examples from several domains. **Speed** and **effort** tags: `faster` removes work or hides waiting, `~0` negligible, `small` under about 0.5 s on typical data, `+1 call` one extra LLM call. Effort: Low is a few hours of agent work, Medium a day, High multiple days.

### 5.1 Understand the layout before analysing (the widest-reaching group)

Real uploads are messy spreadsheets long before they are statistical problems. These systems run once, at read time, and only fire when the property is present.

| System | Triggered by | What it does | Speed | Effort |
|---|---|---|---|---|
| **Table and layout detection** | report-style layout | Find the actual table in a sheet: header rows (including multi-row and merged), spacer rows, footnotes, several tables per sheet; offer each as a dataset. | ~0 | Medium |
| **Wide-to-long reshape** | time or category in the header | Recognise year, month, week or `t1..tn` columns and reshape to a proper time series or panel so time-series and panel tools apply. | small | Medium |
| **Subtotal and total-row detection** | rows that equal sums of others | Exclude them from aggregation and totals; report what was excluded (double counting is the most common silent error in statistical tables). | small | Low |
| **Aggregated-row detection** | n, SD, SE beside means | Analyse with weights and warn about the ecological fallacy; support meta-analysis style pooling. | small | Medium |
| **Implicit-order detection** | monotone index without timestamps | Treat row order as a sequence, with a warning. | ~0 | Low |
| **Multi-valued and nested cell expansion** | delimited lists, JSON cells | Explode to a long form for co-occurrence and counts. | small | Low-Medium |
| **Hierarchical-code roll-ups** | prefix-nested codes | Analyse at several levels; pool rare categories by parent. | small | Medium |
| **Unit and currency harmonisation** | mixed suffixes, currencies | Convert within a column; real vs nominal on request; use packs for rates. | small | Medium |
| **Calendar and time-zone normalisation** | offsets, weekday gaps, fiscal or trading calendars | Normalise to one zone; treat non-trading days as absent, not zero; feed holiday and calendar effects to periodicity and change systems. | small | Medium |
| **Vintage and revision handling** | repeated periods with differing values | Keep the latest vintage, measure revisions, flag revised periods. | small | Medium |
| **Snapshot drift check** | same schema in several uploads | Distribution drift per column (population stability, KS), schema change, new categories. | small | Low-Medium |
| **Data dictionary ingest** | user-supplied notes | See section 5.9. | ~0 | Medium |

### 5.2 Variable-aware methods (use the right method for the measurement scale)

Applying a mean and a Pearson correlation to everything is the commonest wrong-method error.

| System | Triggered by | What it does | Examples | Speed | Effort |
|---|---|---|---|---|---|
| **Count models** | counts, overdispersion, exposure | Poisson or negative-binomial with an exposure offset; rates per unit exposure, not raw counts. | crime, claims, defects, page views, infections | small | Medium |
| **Zero-inflated and two-part handling** | semi-continuous or zero-inflated | Model the zero mass and the positive part separately; report both. | insurance claims, spend, rainfall | small | Medium |
| **Compositional handling** | share-like columns with a constant sum | Log-ratio transforms; do not correlate raw shares (spurious negative bias). | budgets, chemistry, vote shares, time use | small | Medium |
| **Circular statistics** | angular or periodic variables | Circular mean and dispersion; hour 23 is next to hour 0. | wind direction, time of day, phase | ~0 | Low |
| **Ordinal-aware methods** | ordinal scales | Rank-based tests, ordinal effect sizes; no means on Likert items; polychoric correlation. | surveys, grades, ratings | small | Low-Medium |
| **Pairwise ranking models** | pairwise outcomes | Bradley-Terry or Elo-style ratings with uncertainty. | sports, preferences, tournaments | small | Medium |
| **Proportion / bounded handling** | bounded values | Logit or beta models; intervals that respect the bounds. | rates, occupancy, humidity | ~0 | Low |
| **Robust statistics and extremes** | heavy tails | Medians and trimmed means, log or return transforms, tail estimates (peaks over threshold, block maxima) reported separately. | returns, floods, claims, latency | small | Medium |
| **Heaping and digit-preference test** | last-digit imbalance | Detect rounding to 0 or 5; warn that fine-grained differences are unreliable; suggest binning. | ages, incomes, lab readings | ~0 | Low |
| **Benford and integrity forensics** | amounts spanning several magnitudes | First-digit and last-digit tests, duplicate sequences; flag for review, not accusation. | accounting, expenses, scientific data integrity, elections | small | Low-Medium |
| **Uncertainty-aware values** | attached SE, CI or range | Inverse-variance weights; propagate interval values instead of treating midpoints as exact. | published estimates, instrument readings | small | Medium |

### 5.3 Dependence- and design-aware inference

Rows are usually not independent and the data-generating design changes the valid method.

| System | Triggered by | What it does | Examples | Speed | Effort |
|---|---|---|---|---|---|
| **Deseasonalised and partial correlation** | shared cycle or common driver | Correlation after removing the shared cycle and after conditioning on a stated common driver. | pollutants, load and temperature, intraday volumes | small | Low |
| **Autocorrelation-aware inference** | serial dependence | Effective sample size or block bootstrap for correlations and trend tests. | any regular series | small | Medium |
| **Cluster- and hierarchy-aware inference** | repeated entities or nested groups | Cluster-robust intervals, design effects. | patients, students in schools, stores | small | Medium |
| **Spatial dependence** | coordinates or areas | Moran's I, spatially aware cross-validation, hot spots; note aggregation-level effects. | air quality, crime, real estate, epidemiology | small | Medium |
| **Network dependence** | graph structure | Do not treat connected rows as independent; permutation on the graph. | fraud rings, social, contacts | small | Medium |
| **Batch-effect detection and correction** | batch, plate, lot or instrument variable | Detect mean shifts by batch; normalise or include as a factor; use QC samples where flagged. | omics, assays, multi-instrument sensors | small | Medium |
| **Design-aware analysis** | experimental design columns | Blocking, factorial interactions, crossover carry-over, multi-arm multiplicity, sequential-peeking warning; existing sample-ratio check kept. | A/B tests, clinical trials, lab factorial designs | small | Medium |
| **Survey-weight and design-effect handling** | weights, strata, clusters | Weighted estimates with design-based intervals. | official statistics, opinion surveys | small | Medium |
| **"No inference" mode** | census, whole population, deterministic simulation | Suppress significance testing; use descriptive comparison, effect sizes and sensitivity analysis instead; say why. | census tables, model ensembles | ~0 | Low |
| **Selection-bias and found-data caveats** | convenience or scraped data | Add a standing caveat where the sampling frame is unknown; avoid prevalence claims. | web data, reviews, admin extracts | ~0 | Low |
| **Regime and level-shift handling** | non-stationarity | Detect regimes with false-positive control, analyse within a regime, report a shift with date and size instead of an anomaly. Applies to the unchecked NOx finding, a tariff change or a market break. | prices, tariffs, policy, recalibration | small | Medium |
| **Reporting-delay (nowcasting) adjustment** | recent periods systematically incomplete | Mark or adjust the most recent periods; do not report their drop as real. | claims, infections, orders | small | Medium |
| **Split-half replication label** | any sample large enough to halve | Mark each finding "holds in both halves" or not. | all | small | Medium |
| **Confounder / Simpson's-paradox check** | segment comparisons | Re-test stratified by the strongest confounder. | pay gaps, churn, outcomes by site | small | Medium |
| **Causal-claim guard** | "explain / cause" questions | Identify the design (randomised, observational, quasi-experiment), list uncontrolled confounders, downgrade language from cause to association when needed. | all | ~0 | Low |
| **Global false-discovery control and empirical calibration** | many tests in one run | One correction across all findings; shuffle once or twice to calibrate thresholds to the dataset. | wide tables, sweeps | small | Low-Medium |
| **Small-n and high-dimensional safeguards** | tiny n or p >> n | Exact and permutation tests, shrinkage, regularised selection, stronger multiplicity control; refuse flexible models on tiny n. | pilot studies, omics, spectra | small | Medium |
| Power and minimum detectable effect | null results | "No effect" versus "not enough data to tell". | all | ~0 | Low |
| Time- and group-aware cross-validation | temporal or repeated-entity targets | Prevent leakage that inflates scores. | forecasting, churn, attrition | ~0 | Low-Med |

### 5.4 Analyses that the structure itself calls for

| System | Triggered by | What it does | Examples | Speed | Effort |
|---|---|---|---|---|---|
| **Graph analysis** | edge list | Degree distribution, components, centrality, communities, unusual hubs. | payments, social, supply chain, call records | small | Medium |
| **Spatial analysis** | coordinates or area codes | Hot spots, spatial autocorrelation, regional rollups, choropleth. | crime, real estate, air quality | small | Medium |
| **Trajectory analysis** | id x time x position | Segment into trips and stops; speed, dwell, distance; origin-destination flows. | mobility, vessels, deliveries | small | Medium |
| **Event-log analysis** | event log | Sessionisation, funnels, inter-arrival and burstiness, rates over time, next-event patterns. | web, IT operations, security | small | Medium |
| **Sequence analysis** | categorical sequences | Transition matrices, frequent patterns, entropy, run-length. | clickstreams, care pathways, genomes | small | Medium |
| **Functional / curve analysis** | curves per unit | Smoothing, peak detection, curve PCA and clustering, curve alignment. | spectra, growth curves, ECG, load profiles | small | Medium-High |
| **Many-series analysis** | thousands of parallel series | Cluster series by shape, hierarchical reconciliation, intermittent-demand handling; batch processing so it stays fast. | SKUs, meters, sensors | faster | Medium-High |
| **Mixed-frequency alignment** | series at different frequencies | Aggregate or disaggregate to a common grain with stated assumptions. | economic indicators, sensors plus daily logs | small | Medium |
| **Text analysis extensions** | text columns | Language, topics, near-duplicates, PII in text, length outliers. | reviews, tickets, clinical notes | small | Medium |
| **Vector-column handling** | embedding cells | Treat as high-dimensional; clustering, nearest neighbours, drift. | embeddings from any model | small | Medium |
| **Contingency and flow analysis** | cross-tab or flow matrix | Correspondence analysis, standardised residuals, flow diagrams. | origin-destination, confusion matrices | small | Medium |
| **Sensitivity analysis** | simulation or model output | Rank inputs by influence (variance-based), emulator fit; no p-values. | ensembles, engineering models | small | Medium |
| **Generalised periodicity detection** | regular series | FFT or autocorrelation over every measure and grain; replaces the hard-coded hour and weekday profiles. | pollution, load, volumes, hiring | faster | Medium |
| **Series decomposition** | regular series with trend and cycle | Trend / seasonal / residual (STL-style, official seasonal-adjustment style). | indices, load, prices | small | Medium |
| **Time-varying variance** | volatility changes | Rolling or clustered volatility; scale anomaly thresholds. | returns, prices, sensor noise | small | Medium |
| **Stable-process control charts** | stable process with a target | EWMA, CUSUM, Shewhart. | manufacturing, KPIs, latency | small | Medium |
| **Quick-insights sweep** | any wide or mixed table | One vectorised pass scores many candidate insights on a common scale. | all | small | Medium-High |
| **Anomaly root-cause drill-down** | anomalies plus other columns | Which measures or slices spike together. | all | small | Low |

### 5.5 Data integrity

| System | Triggered by | What it does | Speed | Effort |
|---|---|---|---|---|
| **Value-plausibility and continuity pass** | numeric or timestamped data | Bounds from unit parsing, flatlines, stuck values, gaps, duplicated timestamps, irregular sampling. | small | Low |
| **Logical-constraint discovery** | ordered or additive column relations | Learn `end >= start`, `high >= low`, `part <= total`, sum identities; list violating rows. Extends the existing relation scan. | small | Low |
| **Missingness mechanism check** | missing values | Blocks, time-related, related to other columns; says whether missingness biases results. | small | Low |
| **Entity resolution and near-duplicates** | repeated records | Fuzzy duplicates within and across tables; better join keys. | small-moderate | Medium |
| **Hierarchy and dependency detection** | grouping columns | Which column determines which; safe rollups. | ~0 | Low |
| **Grain awareness across tables** | joined data | Track the grain of each column; pick the correct aggregation level and small-group rules. | small | Medium |

### 5.6 Never miss what the user asked for

| System | What it does | Speed | Effort |
|---|---|---|---|
| **Deliverable contract** | Parse the objective into required deliverables (chart type, driver analysis, comparison, forecast). After the run, check each was delivered; run a deterministic recipe for any that was not. | ~0 | Medium |
| **Objective-to-plan compiler** | Map common intents to tool calls so a small model does not have to remember the goal. | ~0 | Medium |
| **Question-type router** | Choose the analysis family from axis E (describe, compare, associate, predict, explain, forecast, detect, segment, rank, monitor, audit, fairness, benchmark, decide). | ~0 | Medium |

### 5.7 The one domain-specific door: domain packs

Principle 2 in practice. A pack is a data file, not code.

- **Schema:** name, activation rule (inferred domain or user choice), reference tables, thresholds and limits (with units and source), KPI definitions (name, formula over columns, direction of good), standard methods to prefer.
- **Categories of content:**
  - *Reference tables:* calendars and holidays per market or country, currencies and inflation indices, country and region codes and hierarchies, unit registry, code hierarchies (ICD, NAICS).
  - *Limits and reference ranges:* air-quality standards and averaging windows, clinical reference ranges, regulatory limits, detection limits.
  - *KPI libraries:* finance (returns, drawdown, volatility, Sharpe), energy (peak demand, load factor, baseload), retail and SaaS (conversion, retention, churn, LTV), HR (attrition, headcount, pay-gap thresholds), manufacturing (OEE, yield), IT (availability, latency percentiles), sports (ratings).
  - *Standard methods:* seasonal adjustment conventions, index-number weighting, trading-day rules, survival conventions.
- **Effect:** turns "CO reached 3.7" into "exceeded the limit on X% of days", "peak demand rose 12%", "drawdown reached 18%", "attrition is above the sector reference". The same core systems run either way.
- **Cost:** ~0 (a lookup); Medium for the loader and schema, Low per additional pack.

### 5.8 LLM path quality and latency

Properties here are properties of the model and the workload, not of the data.

| System | When | What it does | Speed | Effort |
|---|---|---|---|---|
| **Speculative baseline execution** | LLM planning is slower than the baseline tools | Start the deterministic tools immediately while the LLM plans; the LLM then adds or removes steps. Live calls take about a minute; the tools take seconds. | **faster** | Medium |
| **Chatty-model handling** | the model answers in prose or exhausts its token budget before JSON | One tiny JSON probe at startup; if the model answers in prose, mark it chatty and switch to a compact plan schema, structured-output or tool-calling mode where available, and a lower reasoning setting; fall back to the deterministic plan sooner. Observed in the live run. | faster | Medium |
| Optional critic pass | one extra call is affordable | Reads findings and narrative; flags contradictions and overclaims. | +1 call | Low |
| Citations to finding ids in the narrative | any narrative | Verified deterministically. | ~0 | Low |
| Progressive report | slow model | Deterministic findings first, narrative when it arrives. | faster (perceived) | Low-Medium |
| Grounded follow-up Q&A | a finished run | See section 5.9. | +1 call on demand | Medium |

### 5.9 Retrieval (RAG): recommendation

Not a general-purpose RAG. The gaps found so far were analysis logic, now in tested code, not missing knowledge. Two narrow retrieval features are worth it, and both are domain-neutral:

1. **User-supplied data dictionary or notes (highest value).** Upload or paste text; deterministic parsing extracts units, declared missing codes, valid ranges, code meanings and column descriptions, feeding placeholder detection, role overrides, unit parsing and chart formatting, with short column notes in the prompt. No vector database: keyword ranking against column names and the objective covers long documents. Extracted claims (for example "-200 means missing") are validated against the data before use, the same way LLM-proposed roles are.
2. **Grounded follow-up questions.** Retrieval over the run's own findings and report, with an LLM answer that cites finding ids. The corpus is small and trusted.

Not recommended: a vector store over literature or the web (embedding dependency, latency, and retrieved noise hurts small-window models); retrieval over the tool catalog (`applies_to` gating and compact prompts already do this); statistical-guidance retrieval (that belongs in tested code). Uploaded documents are untrusted text and get the same injection guarding as column data.

### 5.10 Performance foundations

Properties: wide or long tables, repeated work across tools, many series, LLM latency.

| System | What it does | Speed | Effort |
|---|---|---|---|
| **Shared analysis context** | Compute the correlation matrix, aggregates, resamples, ranks and group indexes once and reuse them across tools. Coerced-frame caching already helped. | **faster** | Medium |
| **Cost-aware scheduler** | Rank steps by expected value per second; skip low-value expensive ones. Question priorities exist; cost is not used. | faster | Medium |
| Polars or Arrow for heavy group-bys | `polars` is already in `requirements.txt`. | faster | Medium |
| Cross-run cache | Keyed on file hash and parameters. | faster | Low |
| Adaptive sampling with error bounds | Fixed sampling above 200,000 rows already exists (`DSA_ANALYSIS_SAMPLE_ROWS`). | faster | Medium |
| Batch processing for many series and many tables | Vectorised across series; parallel where safe. | faster | Medium |

## 6. Suggested order (by breadth of data affected, then cost)

1. **Layout, reshape and total-row detection** (5.1). Real uploads are messy spreadsheets; wide time-in-header tables and totals mixed with data silently corrupt everything downstream, across every domain.
2. **Deliverable contract and question-type router** (5.6). Fixes the request that went undelivered; helps all data.
3. **Shared analysis context and speculative baseline execution** (5.10, 5.8). Make it faster.
4. **Variable-aware defaults** (5.2): counts, ordinal, proportions, compositional, circular, robust and heavy-tailed handling. Using the wrong method for the scale is the broadest correctness risk.
5. **Dependence- and design-aware core** (5.3): deseasonalised correlation, autocorrelation-aware, cluster-aware, split-half replication, regime handling, "no inference" mode, causal-claim guard.
6. **Integrity pass and constraint discovery** (5.5).
7. **Domain-pack schema with two initial packs**, and the **cross-shape corpus** (section 7) so each later system is proven general.
8. **Structure-specific analyses** (5.4): graph, spatial, event-log, many-series and functional analysis, each only when a real user need shows up.

## 7. Proving generality: the cross-shape corpus

A system counts as general only when it passes three kinds of test on structurally different data: a **planted case** that must be found, a **null case** that must not, and **two structurally different datasets** where it must not silently misfire (the `applies_to` negative tests already do this for the specialist tools).

- **One realistic dataset per property combination**, chosen to span the taxonomy, for example: sub-daily environmental sensors (have: UCI Air Quality); regular electricity load; stock prices (irregular, heavy tails, regimes); employee roster and history; laboratory assays with replicates and batches; official statistics (weights, seasonal adjustment, revisions, report-style layout, wide time-in-header); retail transactions (multi-table, hierarchy, intermittent demand); clinical time-to-event; count data with exposure; compositional data; a graph edge list; a spatial point set; an event log; a curve or spectrum set; a survey with Likert items; a factorial experiment; a census table (no-inference mode); a simulation sweep.
- **Track** precision and recall per commit, with the benchmark harness for speed.
- **Provenance:** every number traceable to the tool call and parameters (evidence fields cover part of this).

### Applicability by domain group

`●` typically applies by property, `·` usually does not. Section 3.7 lists the properties behind each row. The systems are grouped as in section 5.

| Domain group | 5.1 Layout | 5.2 Variable-aware | 5.3 Dependence / design | 5.4 Structure-specific | 5.5 Integrity | 5.6 Ask | 5.7 Packs |
|---|---|---|---|---|---|---|---|
| Environment, climate, ecology | ● | ● | ● | ● | ● | ● | ● |
| Physical sciences, engineering | ● | ● | ● | ● | ● | ● | ● |
| Life sciences, omics, lab assays | ● | ● | ● | ● | ● | ● | ● |
| Clinical, epidemiology, health | ● | ● | ● | ● | ● | ● | ● |
| Finance, markets, insurance | ● | ● | ● | ● | ● | ● | ● |
| Economics, official statistics | ● | ● | ● | ● | ● | ● | ● |
| Retail, e-commerce, marketing | ● | ● | ● | ● | ● | ● | ● |
| Operations, supply chain, transport | ● | ● | ● | ● | ● | ● | ● |
| Manufacturing, IoT, energy | ● | ● | ● | ● | ● | ● | ● |
| IT operations, security, software | ● | ● | ● | ● | ● | ● | ● |
| People, education, social science | ● | ● | ● | ● | ● | ● | ● |
| Sports, games, ratings | ● | ● | ● | ● | ● | ● | ● |
| Media, web, social, advertising | ● | ● | ● | ● | ● | ● | ● |
| Geospatial, urban, real estate | ● | ● | ● | ● | ● | ● | ● |
| Government, legal, elections | ● | ● | ● | ● | ● | ● | ● |
| Humanities, culture, archives | ● | ● | · | ● | ● | ● | ● |

The point of the matrix is what it shows: at the level of system *groups*, every domain group needs almost every group, because the groups are keyed on properties that every domain has some of. The differences are inside the groups, and the property gates (section 5) decide which individual system runs.

## 8. Hidden assumptions a generic pipeline must not make

Each line is a test case: build a dataset that breaks the assumption and confirm the system either handles it or says it cannot.

- The header is row 1, there is one table per sheet, and there are no merged cells or footnotes.
- Every row is a unit; none is a total, subtotal, unit label or note.
- Columns are variables; time may actually be in the column headers.
- Rows are independent (they may share a person, a place, a network edge, a batch or a time neighbourhood).
- There is a time axis (and, if so, regular, in one time zone, with every day a business day).
- Numeric columns are on an interval scale and a mean means something (ordinal, circular, compositional, count and bounded variables all break this).
- Zero means "none" (it may mean missing, not applicable, or below detection).
- Missing values are random.
- A large sample means precision (it may be a census, a duplicated frame or serially dependent data).
- A significant result means an effect (multiplicity, dependence, selection and design all break this).
- Values are exact (they may be rounded, censored, heaped, revised or carry an error).
- One currency and one unit per column; one vintage per period; one meaning per code across sources.
- Text is English and free of personal data.
- The data fits in memory and in one file.
- The user's question matches the data's structure (a heatmap by month needs a time axis).

## 9. Non-goals and honest limits

- Raw images, audio, video and point clouds are not analysed (tier 3); accompanying tables and feature columns are.
- Causal claims from observational data are never made, only guarded (5.3).
- Streaming ingestion is not planned; appended data is handled as snapshots with a drift check.
- Domain packs contain data, not analysis code; a domain that needs a genuinely new method needs a new property-keyed system.
- The status column in section 4 is from Round 9 inventory and has not been audited line by line; an audit is the first step of any item that depends on it.

## 10. Implementation runbook (how to run this plan with Claude Code)

Written from Round 9, which used about 60 agents. Numbers are from that session, not from any plan limit: check `/usage` before choosing a phase size.

### Models
- **Orchestrator: the most capable model available (Opus 5 or Fable 5.1).** Its job is judgment: verifying agent claims, resolving conflicts, deciding what is real. Round 9's mistakes were orchestration mistakes (accepting an unverified "pre-existing failure" claim, reading logs only when prompted). It uses far fewer tokens than the workers, so the premium is small.
- **Workers: Sonnet 5**, with a tight spec and disjoint file ownership. Haiku for mechanical jobs (docs, lint fixes). The orchestrator's model for subtle statistics (dependence-aware inference, regime detection).
- **`/advisor`** at the start of each phase and before declaring it done.

### Concurrency and budget
- 3 to 4 agents at a time; four or more large agents together hit the session limit twice in Round 9.
- One or two systems per agent; agents used roughly 50k to 200k tokens each.
- About 8 to 12 agents per phase. Stop after each phase, verify, commit.
- Use `isolation: "worktree"` for risky refactors (for example the shared analysis context).

### Phases (from section 6)
| Phase | Content | Agents |
|---|---|---|
| 0 | Audit the status column in section 4; build the hidden-assumption test scaffolding (section 8) | 2 |
| 1 | Layout detection, wide-to-long reshape, total-row detection (share `io.py`: run in sequence) | 3 |
| 2 | Deliverable contract, question router, shared analysis context, speculative baseline execution | 4 |
| 3 | Variable-aware methods (counts, compositional, ordinal, heavy tails, heaping, uncertainty) | 4 |
| 4 | Dependence- and design-aware core | 4-5 |
| 5-7 | Integrity pass; domain packs plus the test corpus; structure-specific analyses on demand | 2-3 each |

### Prompt to paste at the start of a session
```
Implement FutureScope.md phase <N>. Read FutureScope.md and IMPROVEMENTS.md (Round 9) first.
Rules: every system is keyed on a data property, gated by applies_to, and costs nothing when absent;
no domain-specific code outside domain packs. Each system ships with a planted test, a null test and
two structurally different datasets. Testing IS approved: run pytest, ruff, mypy, scripts/validate.py,
scripts/dry_run.py and scripts/bench.py --check after every wave; the deterministic run must not
get slower. Enter plan mode first and ask me about ambiguities. Use /advisor before the plan and
before finishing. Workers: Sonnet, 3-4 at a time, disjoint file ownership, reports under 150 words.
Verify agent claims yourself. Don't commit; write a "Round 10" status block in IMPROVEMENTS.md
after every wave so a fresh session can resume. Stop after the phase.
```

### Guardrails learned in Round 9
- **Testing permission is per task.** The saved memory rule is "no testing until asked", so state it in the first message.
- **Resume, don't restart.** After a limit, resume the cut-off agent with `SendMessage`; it keeps its context.
- **State lives in the repo**, not in the conversation: a status block in `IMPROVEMENTS.md` after every wave.
- **Live LLM runs go last** in a phase; they are slow and use quota.
- **Agents must not edit a test to make it pass** without stating why; check such changes yourself.
- **Verify claims** that decide trust (a "pre-existing failure", a "fixed" item) with a direct check.
- **Run the gates yourself** between waves; agents report on their own files only.

### Ending a long session
Start a **new session** for a new phase rather than compacting: the design conclusions are in this file, the state is in `IMPROVEMENTS.md`, and a fresh context is cheaper per turn and avoids compaction losing detail. Before ending the old session: let any live run finish and record its results in `IMPROVEMENTS.md`, commit the work, and update the memory notes.

## 11. Known open items from Round 9

- Live LLM run on `AirQualityUCI.csv` not yet reviewed end to end (chart-design pass, driver analysis for CO, month-by-pollutant heatmap, coverage report). Its first reply was prose again and hit the token cap; the retry path fired.
- Controller-side role wiring (`_apply_column_roles`) has no test of its own; it needs an LLM run.
- Join review cards, group-size input and run-comparison expander in the Streamlit UI were only syntax-checked, not looked at.
- Local-model path (8k context, compact prompts, reasoning-tag stripping, concurrency limit) tested with fakes only; needs a real Ollama run with `LOCAL_LLM_CONTEXT=8192`.
- `cohort_analysis` RFM finding names a segment without a group-size check.
- `.github/workflows/ci.yml` was overwritten (now Windows and Ubuntu, Python 3.13); review whether to keep it.
- README, `AGENTS.md` and `HANDOVER.md` do not yet describe the Round 9 work.
