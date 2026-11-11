# GPU Acceleration and Streamlit Run Resilience Plan

**Date:** 2026-10-17  
**Incident:** `lte_train.csv` UI run created at 16:40:39  
**Scope:** selective GPU acceleration, bounded statistical workloads, worker isolation,
durable run recovery, and operational telemetry.

## 1. Confirmed incident facts

- The affected dataset is about 65 MB, 47,970 rows, and roughly 100 columns.
- Stage 2 initial reasoning completed and `clean_data` wrote its output.
- No report was finalized for the affected run.
- The Streamlit Python process remained alive and consumed about one CPU core continuously,
  with approximately 3.8 GB private memory and 2.6 GB resident memory.
- Streamlit and analysis currently share one Python process. A CPU-bound tool can therefore
  delay WebSocket heartbeats and UI polling even though its worker thread is still alive.
- A disconnected Streamlit session is eligible for cleanup after 120 seconds. The active run
  lives only in that session's `st.session_state`, so a later browser session cannot reattach.
- The machine has an NVIDIA GeForce RTX 2050 with 4 GB VRAM. XGBoost already selects CUDA;
  pandas/SciPy correlation and most sklearn estimators remain CPU-only.

## 2. Target behavior

1. A heavy analysis must never make the Streamlit server unresponsive.
2. Every run receives a durable run ID and can be reattached after refresh or disconnect.
3. Statistical work is bounded by explicit row, column, memory, and wall-clock budgets.
4. `compute_backend=auto` uses a GPU only when the operation supports it, the workload is
   large enough to repay transfer overhead, and estimated peak VRAM remains below a safe cap.
5. GPU absence, incompatibility, or out-of-memory errors cause a recorded deterministic CPU
   fallback; they never fail the whole analysis.
6. Reports disclose sampling, approximation, compute backend, fallbacks, and timeouts.
7. CPU and GPU implementations produce equivalent results within documented numeric tolerance.

## 3. Compute backend contract

Add a typed, immutable compute policy in `src/core/compute_policy.py`:

```text
mode: auto | cpu | cuda
gpu_min_cells: workload threshold below which CPU is preferred
gpu_memory_fraction: default 0.60 of currently free VRAM
max_stat_rows: default 50,000 for ordinary exploratory statistics
max_corr_rows: default 20,000 for wide correlation
max_corr_columns: default 64 before objective-led column selection
tool_timeout_s: per-tool deadline, distinct from sandbox timeout
```

Each eligible tool returns these fields:

```json
{
  "compute_backend": "cpu|cuda",
  "compute_device": "device name or cpu",
  "backend_reason": "selection or fallback reason",
  "sampled_from": 47970,
  "sampled_to": 20000,
  "approximation": "none|seeded_sample|rank_then_pearson"
}
```

Backend selection must be based on public capability probes and estimated memory, not merely
the presence of `nvidia-smi`. The probe must verify that the actual library can allocate and
execute a small operation on the device.

### GPU eligibility matrix

| Workload | Initial backend | GPU path | Fallback |
| --- | --- | --- | --- |
| XGBoost classification/regression | Existing CUDA path | XGBoost `device=cuda` | Retry once on CPU |
| Dense Pearson correlation | CPU BLAS for small work | Optional CuPy matrix path | Vectorized NumPy |
| Spearman without missing values | CPU for small work | Rank once, then GPU Pearson | Rank once, then CPU Pearson |
| Pairwise-missing Spearman | Bounded CPU sample | No unsafe semantic substitution | Seeded sample plus caveat |
| PCA/SVD on dense numeric matrices | CPU for small work | Optional GPU array backend | sklearn/NumPy |
| Random forest, classical regression, stats tests | CPU | None in phase 1 | Bounded CPU execution |
| Report rendering, LLM calls, file I/O | CPU | Never | CPU |

CuPy must be an optional `gpu` dependency, not a mandatory install. RAPIDS/cuDF is not the
first implementation because native Windows support and 4 GB VRAM make it a poor default for
this deployment. The system must remain fully functional without the optional extra.

## 4. Work packages

### WP1 — Stop the current statistical runaway

Files: `src/tools/data_processing.py`, `src/core/agenda.py`, corresponding tests.

- Replace the global 200,000-row correlation cap with a width-aware budget.
- Estimate work as rows multiplied by selected-column pairs.
- When the budget is exceeded, choose columns deterministically from the objective, target,
  semantic roles, completeness, and variance; record excluded columns.
- For auto-Spearman, never calculate a full pairwise-missing Spearman matrix on a wide frame.
  Use a seeded bounded sample and clearly label it exploratory.
- Continue calculating exact full-row p-values only for the small number of reported top pairs,
  with their own maximum-row guard where required.
- Add cooperative cancellation checkpoints between profiling, Pearson, Spearman, pair testing,
  and finding construction.

**Exit:** the incident-shaped fixture finishes correlation within 30 seconds on CPU and keeps
peak incremental memory below 750 MB.

### WP2 — Central automatic compute policy

Files: new `src/core/compute_policy.py`, `src/core/run_config.py`, `app.py`, `.env.example`.

- Implement typed capability and workload decisions with no tool imports into the controller.
- Add `COMPUTE_BACKEND=auto|cpu|cuda`, default `auto`.
- Detect physical VRAM, free VRAM, supported library, and allocation success.
- Cache stable hardware facts but refresh free-memory facts for each decision.
- Add UI choices `Auto`, `CPU only`, and `CUDA when beneficial`; never expose API keys or
  process-global configuration through these settings.

**Exit:** decisions are deterministic for a supplied capability snapshot and every fallback has
a machine-readable reason.

### WP3 — GPU-enable suitable numeric kernels

Files: `src/tools/data_processing.py`, `src/tools/dimensionality.py`,
`src/tools/ml_pipeline.py`, optional dependency metadata.

- Preserve the existing XGBoost CUDA path, but route it through the shared policy and report the
  actual backend used.
- Implement an optional CuPy Pearson matrix kernel for complete dense numeric matrices.
- Implement tie-correct rank-once Spearman before the Pearson kernel; verify parity with pandas.
- Estimate matrix, workspace, and output sizes before allocation. Keep at least 40% of free VRAM
  unused on the 4 GB device.
- Catch CUDA initialization, unsupported dtype, and OOM failures; free GPU pools and retry once
  using the bounded CPU implementation.
- Do not silently move unsupported statistics to GPU.

**Exit:** GPU tests pass when CUDA is available; the same suite skips cleanly without CUDA; CPU
and GPU matrices agree within `1e-6`, including ties and constant columns.

### WP4 — Isolate analysis from Streamlit

Files: `ui/run.py`, new worker/registry modules under `src/core/`, `app.py`.

- Replace the in-process daemon analysis thread with a spawned worker process.
- Keep Streamlit responsible only for submission, polling, and rendering.
- Limit worker CPU threads explicitly so BLAS/sklearn cannot consume the server's event-loop
  capacity through oversubscription.
- Enforce per-run RSS, wall-clock, and child-process limits. A timed-out tool becomes a structured
  execution issue and the controller continues or finalizes a partial report according to
  objective criticality.
- Preserve the existing `BaseTool.run()` boundary and seven-stage controller workflow.
- Cancellation first requests cooperative stop, then terminates only the exact validated worker
  process after a grace period.

**Exit:** deliberately CPU-saturating a worker leaves `/_stcore/health` responsive and UI polls
under two seconds.

### WP5 — Durable run registry and reconnection

Files: new `src/core/run_registry.py`, `ui/run.py`, `app.py`.

- Create `<run>/run_status.json` atomically before ingestion, with run ID, stage, heartbeat,
  process ID, timestamps, artifact paths, and terminal status.
- Never store credentials, prompt text, or raw rows in the registry.
- Update heartbeat and current stage from the worker process.
- Store the current run ID in both session state and a URL query parameter.
- On refresh/reconnect, validate the run ID against the registry and reattach to `running`,
  `succeeded`, `partial`, `cancelled`, or `failed` state.
- Detect abandoned workers by heartbeat plus process identity, avoiding PID-reuse mistakes.
- Retain completed public artifacts using the existing allowlist.

**Exit:** closing the browser for more than 120 seconds and reopening the run URL restores live
progress or the finished report without launching a duplicate run.

### WP6 — Resource cleanup and observability

Files: `ui/run.py`, controller finalization, logging configuration, public diagnostics UI.

- Write rotating Streamlit/server logs with WebSocket and session lifecycle events.
- Record per-tool wall time, CPU time, peak RSS, backend, GPU peak allocation, sampling, and
  fallback reason.
- Release controller, DataFrame, model, and GPU-pool references after artifacts are finalized.
- Bound dataframe/profile caches by byte size, not only entry count.
- Show a plain-language message when the browser is disconnected while the worker continues.

**Exit:** two sequential incident-sized runs do not produce monotonic multi-gigabyte process
growth, and a future disconnect has enough evidence to identify its exact close path.

## 5. Implementation order

1. WP1 CPU bounds and regression fixture — removes the immediate runaway independently of GPU.
2. WP2 compute policy and telemetry contract.
3. WP4 worker-process isolation — protects the UI before introducing another compute backend.
4. WP5 durable registry and reconnect flow.
5. WP3 optional GPU kernels and shared XGBoost routing.
6. WP6 cleanup, logs, and operational UI.

GPU work must not precede statistical bounds or process isolation. A faster unbounded operation
can still exhaust VRAM or make the application unavailable.

## 6. Test plan

### Unit and parity tests

- Backend decision table: absent GPU, unsupported library, insufficient VRAM, small workload,
  eligible workload, forced CPU, and forced CUDA failure.
- Pearson and Spearman CPU/GPU parity, including NaNs, ties, constants, infinities, and mixed
  numeric dtypes.
- Deterministic column selection and seeded sampling.
- GPU OOM fallback returns the same contract and a recorded reason.
- Registry atomicity, path validation, heartbeat expiry, PID reuse, and credential exclusion.

### Integration tests

- Reproduce the 47,970-row wide-data case with planted correlations and missingness.
- Disconnect a browser session for more than the configured TTL, then reattach.
- Kill a worker and verify `failed` or `partial` rather than an endless spinner.
- Force a tool timeout and confirm report finalization plus structured issue reporting.
- Run two concurrent analyses and verify CPU, RAM, and VRAM admission controls.

### Mandatory repository gates

- `ruff check .`
- `mypy src/`
- `pytest tests/ -v`
- Seven-stage sample workflow and populated Markdown report
- CPU-only run on a machine with no CUDA packages
- CUDA smoke and parity run when the optional GPU environment is installed
- No credentials in registry, logs, artifacts, or tracked files

## 7. Rollout and rollback

- Ship WP1 behind safe defaults with no feature flag: unsafe wide pairwise work remains bounded.
- Ship process isolation and registry behind `RUNNER_BACKEND=process|thread`, defaulting to
  `process` after integration tests pass. Keep `thread` for one release as rollback only.
- Ship GPU support with `COMPUTE_BACKEND=auto`; changing it to `cpu` is the immediate rollback.
- Record backend selection rates, timeouts, fallbacks, peak memory, and reconnect success before
  raising workload thresholds.

## 8. Definition of done

- The incident-shaped dataset completes or produces a truthful bounded partial result; it never
  spins indefinitely.
- Streamlit remains responsive throughout CPU- or GPU-heavy work.
- A disconnected browser can reattach to the same durable run.
- Supported large numeric kernels use CUDA automatically when beneficial and safe.
- Unsupported or small work stays on CPU without pretending GPU acceleration occurred.
- All fallbacks, sampling, timeouts, and backend choices are visible in audit and report output.
