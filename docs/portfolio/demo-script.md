# Five-minute local demonstration script

**Status:** executable Week 11 walkthrough prepared for review. The final 4-6
minute MP4, fresh-checkout rehearsal and owner viewing decision are Week 12 work.
Show synthetic data only. No real customer outreach, model fitting or historical
cohort scoring occurs.

## Prepare before the timed walkthrough

Use the recorded implementation commit, Python 3.12 and the locked environment.
Follow the [setup guide](../../README.md#local-development), restore and verify
the model, and authenticate Release B in the separate historical checkout at
`7e571fc4fbb4e6cb99b0d66f8e7d72b24feccff4` using the fixed
[release anchors](../../README.md#release-b-sign-off). Return to the current
implementation checkout. Do not run reference building, benchmarking, model
selection or final-test commands. Historical verification is separate from
current implementation CI and live synthetic checks.

Require a clean committed working tree (`git status --porcelain` prints nothing).
The helper intentionally refuses dirty implementation. Its reference digest is
`5f2a43675cbd9f6ed44bf4a421df6647d3e780cf5b2e9eae6ba782e94849c4ac`;
it authenticates the existing reference rather than regenerating it.

Run these native Windows commands from the repository root in separate terminals:

```powershell
# Terminal A: API, bound to this machine only.
.venv/Scripts/python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080
```

```powershell
# Terminal B: UI, using that API.
$env:CREDIT_RISK_API_URL = 'http://127.0.0.1:8080'
.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open `http://127.0.0.1:8501`. Use the Predictor page with the exact
[synthetic API fixture](../../tests/fixtures/prediction_request.json): credit
limit **1,000,000**, repayment codes **1, 0, -1, -1, -1, -1** in lag order, all
billing amounts **4,000**, all payment amounts **1,500**. Other UI defaults do
not represent this fixture. Keep terminals showing only synthetic outputs;
hide credentials, unrelated tabs and customer material before recording.

## Timed walkthrough

| Time | Show and say |
| --- | --- |
| 0:00-0:35 | README opening: the human-owned monthly review decision, individual ownership and local-use boundary. Identify historical ranking metrics as evidence, not prevented defaults. |
| 0:35-1:35 | Live API and Streamlit Predictor using the fixture. Click Predict; show API `0.190382` (the unchanged UI rounds this to `0.1904`), the standard band and both model-attribution categories. These are synthetic integration checks, not causal explanations. |
| 1:35-2:45 | Execute the helper below with a fresh run ID. Show 400 valid synthetic rows, 40 selected rows, verified output hashes and two different invocation traces for the same batch ID. Reuse preserves file bytes and modification times. |
| 2:45-3:35 | Open the new monitoring summary. Show `investigate` and `automatic_model_change: false`. Explain that repeated synthetic fixture patterns differ from the historical reference. Retain completed scores; a human owns assessment and any restoration of real use. |
| 3:35-4:30 | Rendered architecture and linked Linux CI evidence. Explicitly label the platform persistence/recovery evidence as recorded Linux CI, separate from the API/UI running live on Windows. SQLite demonstrates promotion/rollback; PostgreSQL demonstrates fixed bootstrap/recovery. |
| 4:30-5:00 | Show hypothetical outreach design and calculator. The next-cycle missed-payment endpoint differs from the dataset label. Close with G3 conditions, open G5 and the remaining Release C owner acceptance. |

In Terminal C, select a new run ID for each complete helper run:

```powershell
.venv/Scripts/python.exe -m credit_risk.portfolio.demo --run-id week11-demo-001
```

The helper uses the real batch CLI twice, the existing API client and the existing
monitoring publisher/verifier. It checks readiness, prediction/model identity,
explanation structure, input/output lineage, counts, no-rewrite reuse and the
expected investigation state. Success exits 0; controlled failures exit 1;
argument-parser failures exit 2. A failed run is retained for inspection: correct
the cause and choose a fresh ID. Never delete historical evidence to make it pass.

The ignored receipt is
`experiment/portfolio/week11/week11-demo-001/receipt.json`. It records the exact
implementation commit, fixture and model hashes, batch identity, invocation
traces and monitoring manifest digest. It is synthetic demonstration evidence,
not a new release approval. The CSV repeats 20 reviewed fixture patterns under
400 unique synthetic IDs; it does not contain independent customer observations.

Show a few queue entries using the same run ID:

```powershell
Import-Csv experiment/portfolio/week11/week11-demo-001/batches/2026-09-30/week11-demo-001/scores.csv |
  Where-Object selected_for_review -EQ 'true' |
  Select-Object -First 5 account_id,probability_of_default,risk_band,primary_reason_category
Get-Content reports/monitoring/release_c_demo/week11-demo-001/summary.json
```

Batch dates are frozen synthetic identifiers, not the date of an actual customer
snapshot. Row-level CSVs and event receipts remain under ignored `experiment/`;
new monitoring packages remain in the narrowly ignored
`reports/monitoring/release_c_demo/` namespace. Existing destinations are refused.
No authenticated Release B output is overwritten or committed anew.

## Evidence shown on screen and Week 12 handoff

Use the [claims inventory](claims-inventory.md) to connect every spoken metric to
its source. The [verified baseline Linux run](https://github.com/shaloy-lewis/credit-card-default-prediction/actions/runs/36594629955)
binds commit `9b0635559ec80d9f0d1198be5f978fd0999bff24`; the Week 11 PR must
add its own final-revision CI links before the package is handed off. Do not
present baseline CI as a measurement of a different revision.

A rehearsal must record checkout/environment, commands, run ID, receipt/digest,
claims review and failures. Stop the native processes you started after the
walkthrough. Week 12 repeats this from a fresh checkout, records the full MP4,
checks duration/privacy/claims and records its SHA-256, recording commit and
out-of-Git location. The owner watches and accepts the completed package
separately. The [Release C checklist](release-c-acceptance-plan.md#completion-checklist-and-owner-decision)
remains unchecked until that evidence and decision exist.
