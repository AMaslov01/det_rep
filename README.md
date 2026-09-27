# det_rep

This repository prepares a paired answer-correction experiment. It joins fixed RAGTruth sources with annotated Llama answers, generates entity and claim feedback through the shared Gemini gateway, and sends matched correction prompts to vLLM. Inputs, model weights, credentials, scientific caches, and run results live outside the repository. The original `hallu_smiles` project remains separate.

**Current status:** the E/C experiment is being prepared; no new model or server run has been started. Before any server work, data transfer, Docker/GPU use, or storage cleanup, read [the server rules](docs/server-resource-rules.md). Execution on caniculus needs a separate command from the owner. The staged procedure is in [the current runbook](docs/caniculus-runbook.md); the completed 12-arm QA100 is [historical](docs/history/qa100-20260919.md).

## Correction protocol

Use exactly the same 100 available train `source_id` values as the previous QA100. The deterministic selection is balanced across the input CSV labels (50 each); those labels were supplied by GPT-4o and are used to select examples, not to judge corrected answers. All four conditions begin from the same original answer and receive the same question and source evidence. Each makes one stateless correction request with the same Llama checkpoint and generation settings:

| Arm | Additional feedback visible to the corrector |
| --- | --- |
| `B` | None |
| `E` | Entity diagnostics from the source and answer KG/matching pipeline |
| `C` | VeriScore-extracted claims checked by the existing four-way Gemini verifier |
| `EC` | Both entity and claim diagnostics |

Every VeriScore claim enters C regardless of whether it mentions an entity. C uses `veriscore.claim_verifier.labels: critical` and the four-way verifier configured under `critical.claim_verifier`; its scientific cache protocol is `det-rep-four-way-verdict-v1`. E and C are prepared independently so a failure is attributed to its component; B does not require Gemini. Relation (`R`) and claim-link (`X`) treatments, the former atomic-claim extractor, detector metrics, and evaluator implementation are outside this run. The independent evaluator and human audit are tasks for another team.

A complete run contains **400 trajectories** (100 sources × four arms), zero final failures, and `replay.missing=0`. The blind export contains every unique correction; identical answers from different arms share one request. Its row count is determined after the run and is not padded to 400.

## Inputs and local checks

Use Python 3.12. `prepare` joins RAGTruth `source_info.jsonl` to the annotated Llama 3.1 8B CSV by `llama31_8b_<source_id>` and writes a content-hashed `det-rep-ec-v1` input manifest and fixed split to an external work directory. The fixed split orders all 989 source IDs by SHA-256 of `42\0<source_id>`: 791 train and 198 test. `prepare`, `smoke`, and `replay` enforce SHA-256 checks for the exact QA100 source and answer snapshots; `smoke` also checks the frozen hash of the selected 100 IDs before model preflight. The older manifest has a different schema and must not be reused. Neither the test split nor the historical QA100 results are inputs to this correction series.

```bash
python3.12 -m pip install -e '.[test]'
python3.12 -m pytest -q

python3.12 -m det_rep prepare \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --work-dir /absolute/external/ec100-work
```

The S-BERT snapshot needs a local `config.json`. The gateway key is read from `HALLU_GATEWAY_API_KEY` or, in the experiment container, the read-only `HALLU_GATEWAY_API_KEY_FILE`. The exact gateway URL, vLLM endpoint, served checkpoint, config, image digest, and model revisions are runtime inputs. Do not store keys or input data here. Capture and validate the authenticated gateway manifest before the first correction request. `smoke --gateway-manifest /path/to/frozen-gateway-manifest.json` can pin that snapshot without a live manifest fetch; an existing run automatically reuses its `gateway_manifest.json` and rejects a conflicting supplied manifest. If Gemini is unavailable during resume, B can proceed while E/C record typed component failures.

## Future authorized run and handoff

The following CLI stages describe a future authorized run, not a command to execute during repository preparation. Run scientific Python in the user's unprivileged Docker container on caniculus under [the resource rules](docs/server-resource-rules.md). Use a new external run directory; the science cache is isolated at `work-dir/cache/ec-veriscore-v1`. Do not mix the new VeriScore C results with QA100 artifacts. `smoke` fixes 100 train IDs and one iteration; `--max-sources 1` permits a gated first-ID pass in the same run directory when separately authorized.

```bash
python3.12 -m det_rep smoke \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec100-work/input_manifest-HASH.json \
  --work-dir /absolute/external/ec100-work \
  --run-dir /absolute/external/ec100-work/runs/ec100 \
  --config /absolute/path/to/frozen-config.yaml \
  --embedding-path /absolute/path/to/sbert-snapshot \
  --gateway-url https://your-gateway.example \
  --vllm-url http://127.0.0.1:18000/v1 \
  --checkpoint EXACT_LLAMA_COMMIT

python3.12 -m det_rep replay \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec100-work/input_manifest-HASH.json \
  --work-dir /absolute/external/ec100-work \
  --run-dir /absolute/external/ec100-work/runs/ec100

python3.12 -m det_rep package-export \
  --run-dir /absolute/external/ec100-work/runs/ec100 \
  --work-dir /absolute/external/ec100-work \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec100-work/input_manifest-HASH.json \
  --repo-dir /absolute/path/to/frozen-repo \
  --config /absolute/path/to/frozen-config.yaml \
  --environment-dir /absolute/external/ec100-environment \
  --out-dir /absolute/external/ec100-package
```

The run identity pins source IDs, four arms, one iteration, inputs, code, model/feedback identity, and generation settings. Resume only when all pinned values remain identical. `replay` verifies the recorded artifacts without inference; `package-export` requires a complete replayed run and produces a full private owner archive plus a separate blind package for the evaluation team.

The private archive includes exact inputs and selected IDs, code and environment snapshot, configuration and model provenance, evidence, feedback and scientific caches, every trajectory and prompt hash, failures, replay, treatment assignment/provenance, and checksums. The blind package contains only an opaque request ID, context, question, original answer, revised answer, and a format description for every unique request. It excludes source IDs, arms, feedback, and the private map. This repository does not score the corrections or perform a human audit.

## Shared Gemini gateway

`gemini_gateway` is a small Cloud Run service that exposes one OpenAI-compatible
model, `openai/gemini-3.5-flash`. It authenticates a caller with an individually
revocable bearer key and uses its Cloud Run service identity to call Vertex AI in
the EU; neither a Vertex API key nor a service-account key is given to recipients.
It supports text chat, streaming, and `response_format` JSON schemas. Images,
files, function tools, and other unsupported OpenAI fields fail explicitly.

Deploy it only after authenticating `gcloud` with an account that can administer
`project-fe2f39ea-f456-4e8f-8e8`:

```bash
gcloud auth login
./scripts/deploy_gateway.sh
python3.12 -m gemini_gateway.manage_keys \
  --project project-fe2f39ea-f456-4e8f-8e8 create alice
```

The last command prints Alice's raw `gk_…` key exactly once. Send it via an
appropriate private channel. To revoke it, run:

```bash
python3.12 -m gemini_gateway.manage_keys \
  --project project-fe2f39ea-f456-4e8f-8e8 revoke alice
```

The service refreshes Secret Manager key records at most once a minute, so a
new key can take up to a minute to become usable and a revocation takes effect
within the same window, without a redeploy. The deploy script creates a €50
monthly project budget alert at 50%, 90%, and 100%; it is an alert, not a hard
spending limit.

### Recipient quickstart: one file, one command

The canonical recipient interface is
[`gemini_recipient.py`](gemini_recipient.py). Send that one file and the
recipient's individual `gk_…` key; they do not clone this repository, install a
package, configure Google Cloud, or need the endpoint URL. The file uses only
the Python standard library and contains the gateway's stable public URL.

```bash
GEMINI_GATEWAY_API_KEY='gk_alice_…' python3 gemini_recipient.py 'Explain Bayes theorem in one sentence.'
```

It is equally usable from their own code as one function:

```python
from gemini_recipient import ask_gemini

answer = ask_gemini("Write a haiku about a clean API.", api_key="gk_alice_…")
print(answer)
```

The key stays outside the file so it can be revoked independently. For users
who prefer an HTTP client or the OpenAI Python SDK, the exact endpoint, key,
and model contract remains available below.

### Direct HTTP or OpenAI SDK

```bash
export GEMINI_GATEWAY_URL='https://your-gateway.europe-west4.run.app'
export GEMINI_GATEWAY_API_KEY='gk_alice_...'
curl "$GEMINI_GATEWAY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $GEMINI_GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gemini-3.5-flash","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ["GEMINI_GATEWAY_URL"].rstrip("/") + "/v1",
    api_key=os.environ["GEMINI_GATEWAY_API_KEY"],
)
reply = client.chat.completions.create(
    model="openai/gemini-3.5-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```
