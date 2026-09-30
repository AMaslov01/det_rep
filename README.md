# det_rep

This repository prepares a paired answer-correction experiment. It joins fixed RAGTruth sources with annotated Llama answers, generates entity and claim feedback through the shared Gemini gateway, and sends matched correction prompts to vLLM. Inputs, model weights, credentials, scientific caches, and run results live outside the repository. The original `hallu_smiles` project remains separate.

**Current status (2026-10-01):** the owner authorized the 750-answer run on caniculus. The frozen scientific snapshot is commit `b95181b`; its one-container R pilot completed 1/1 without a final error, and the full R sweep is running in `/mnt/ssd/a.maslov/det_rep/ec750-b95181b`. Correction and archive stages are gated on complete R verification. Before further server work, data transfer, Docker/GPU use, or storage cleanup, read [the server rules](docs/server-resource-rules.md). The staged procedure and live status paths are in [the current runbook](docs/caniculus-runbook.md); the completed 12-arm QA100 is [historical](docs/history/qa100-20260919.md).

## Correction protocol

Use all **750** annotated Llama answers available in the pinned CSV (599 in the fixed train split and 151 in test). The original CSV labels are provenance only; they do not select answers or judge corrections. Before correction, run one KGGen extraction over the context, question, and original answer for each of those 750 IDs. Save only the resulting R relation triples in a private artifact. E later reads the same KG cache and cannot make another KG extraction call. R is not a correction arm. All four correction conditions begin from the same original answer and receive the same question and source evidence. Each makes one stateless request with the same Llama checkpoint and generation settings:

| Arm | Additional feedback visible to the corrector |
| --- | --- |
| `B` | None |
| `E` | Entity diagnostics from the source and answer KG/matching pipeline |
| `C` | VeriScore-extracted claims checked by the existing four-way Gemini verifier |
| `EC` | Both entity and claim diagnostics |

Every VeriScore claim enters C regardless of whether it mentions an entity. C uses `veriscore.claim_verifier.labels: critical` and the four-way verifier configured under `critical.claim_verifier`; its scientific cache protocol is `det-rep-four-way-verdict-v2`. E and C are prepared independently so a failure is attributed to its component; B does not require Gemini. R is extracted once for another experiment and is never shown to the corrector. The independent evaluator and human audit are tasks for another team.

A complete run contains **750 R source artifacts** and **3000 trajectories** (750 answers × four arms), zero final failures, and `replay.missing=0`. The private condition map has 3000 rows. The blind export contains every unique correction; identical public answer pairs share one request. Its row count is measured after the run and is never padded. Because all available answers are included, the fixed train/test split is recorded for provenance and is not used as a held-out quality estimate.

## Inputs and local checks

Use Python 3.12. `prepare` joins the 989 RAGTruth QA sources to the 750 annotated Llama 3.1 8B answers by `llama31_8b_<source_id>` and writes a content-hashed `det-rep-ec-r-v1` manifest and fixed split to an external work directory. The split orders all 989 source IDs by SHA-256 of `42\0<source_id>`: 791 train and 198 test. The CLI pins the exact source and answer file hashes and the hash of all 750 selected IDs. Old QA100 manifests and results cannot be reused.

```bash
python3.12 -m pip install -e '.[test]'
python3.12 -m pytest -q

python3.12 -m det_rep prepare \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --work-dir /absolute/external/ec750-work
```

The S-BERT snapshot needs a local `config.json`. The gateway key is read from `HALLU_GATEWAY_API_KEY` or the experiment container's read-only `HALLU_GATEWAY_API_KEY_FILE`. The exact gateway URL, vLLM endpoint, served checkpoint, config, image digest, and model revisions are runtime inputs. Capture the authenticated gateway manifest before R extraction. `extract-r --gateway-manifest /path/to/frozen.json` can pin that snapshot without a live manifest fetch; `run` reuses the R stage's frozen manifest and rejects a conflicting revision. If Gemini is unavailable during correction, B can continue while E/C record typed component failures.

## Future authorized run and handoff

The following CLI stages run in one unprivileged Docker container on caniculus under [the resource rules](docs/server-resource-rules.md). Its image includes separate Python environments for the experiment and vLLM. The GPU is mounted by its checked UUID, but the vLLM process starts only after R extraction is verified. Inputs and models are read-only mounts; work, scientific caches, and results are on the user's SSD outside the container. Use separate new R and correction directories; the science cache is isolated at `work-dir/cache/ec-veriscore-r750-v1`. `extract-r` must finish and pass integrity checks before `run`; E then uses its KG cache in cache-only mode. Both stages support resumable `--max-sources 1` when separately authorized.

```bash
python3.12 -m det_rep extract-r \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec750-work/input_manifest-HASH.json \
  --work-dir /absolute/external/ec750-work \
  --relation-dir /absolute/external/ec750-work/runs/r750 \
  --config /absolute/path/to/frozen-config.yaml \
  --embedding-path /absolute/path/to/sbert-snapshot \
  --gateway-url https://your-gateway.example

python3.12 -m det_rep verify-r \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec750-work/input_manifest-HASH.json \
  --work-dir /absolute/external/ec750-work \
  --relation-dir /absolute/external/ec750-work/runs/r750

python3.12 -m det_rep run \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec750-work/input_manifest-HASH.json \
  --work-dir /absolute/external/ec750-work \
  --relation-dir /absolute/external/ec750-work/runs/r750 \
  --run-dir /absolute/external/ec750-work/runs/ec750 \
  --config /absolute/path/to/frozen-config.yaml \
  --embedding-path /absolute/path/to/sbert-snapshot \
  --gateway-url https://your-gateway.example \
  --vllm-url http://127.0.0.1:18000/v1 \
  --checkpoint EXACT_LLAMA_COMMIT

python3.12 -m det_rep replay \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec750-work/input_manifest-HASH.json \
  --work-dir /absolute/external/ec750-work \
  --run-dir /absolute/external/ec750-work/runs/ec750

python3.12 -m det_rep package-export \
  --run-dir /absolute/external/ec750-work/runs/ec750 \
  --relation-dir /absolute/external/ec750-work/runs/r750 \
  --work-dir /absolute/external/ec750-work \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/ec750-work/input_manifest-HASH.json \
  --repo-dir /absolute/path/to/frozen-repo \
  --config /absolute/path/to/frozen-config.yaml \
  --environment-dir /absolute/external/ec750-environment \
  --out-dir /absolute/external/ec750-package
```

The R and correction identities pin the same 750 IDs, inputs, code, Gemini runtime, and cache namespace; the correction identity also pins the four arms, one iteration, R fingerprint, corrector model, and generation settings. Resume only when all pinned values remain identical. `replay` verifies recorded corrections without inference; `package-export` requires a complete R sweep and correction replay and produces a private owner archive plus a separate blind package.

The private archive includes the 750 R artifacts, exact inputs and IDs, code and environment snapshot, configuration and model provenance, evidence, feedback and scientific caches, every trajectory and prompt hash, failures, replay, assignment/provenance, and checksums. The blind package contains only an opaque request ID, context, question, original answer, revised answer, and a format description for every unique request. It excludes source IDs, arms, R triples, feedback, and the private map. This repository does not score corrections or perform a human audit.

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
