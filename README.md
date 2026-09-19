# det_rep

Minimal shared code for a matched hallucination detection and correction experiment. The original `hallu_smiles` repository and its historical archives are independent. This folder is not a Git repository until the team initializes it.

## Data and split

Inputs stay outside this folder. `prepare` joins RAGTruth `source_info.jsonl` to the annotated Llama 3.1 8B CSV by `llama31_8b_<source_id>`. It validates the CSV columns, prompt/context/query membership, unique sources, binary label, and annotation provenance. The current CSV has 750 answers; 239 of the 989 QA sources are pending. Source `12448` is included.

The split is fixed over all 989 source IDs before more answers arrive: SHA-256 order of `42\0<source_id>`, first 791 train and last 198 test. `source_split.json` is immutable within the external work directory. Each CSV revision receives its own hashed input manifest. The existing 750 rows occupy 599 train and 151 test slots under this new split. Because historical results informed the protocol, this split does not create an independent confirmation.

## Setup and entry points

Use Python 3.12. The local S-BERT snapshot must have `config.json`; the Gemini gateway key is read only from `HALLU_GATEWAY_API_KEY`. The vLLM endpoint and exact served checkpoint are runtime settings. No key, source data, cache, or output belongs in this repository.

```bash
python -m pip install -e '.[test]'
python -m pytest -q

python -m det_rep prepare \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --work-dir /absolute/external/work

export HALLU_GATEWAY_API_KEY='set outside this repository'
export DET_REP_VLLM_BASE_URL='http://your-server:8000/v1'
export DET_REP_VLLM_CHECKPOINT='exact-served-checkpoint-or-weight-hash'
python -m det_rep smoke \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/work/input_manifest-HASH.json \
  --work-dir /absolute/external/work \
  --run-dir /absolute/external/work/runs/smoke-01 \
  --embedding-path /absolute/path/to/sbert-snapshot

python -m det_rep replay \
  --sources /absolute/path/to/source_info.jsonl \
  --answers /absolute/path/to/ragtruth_llama31_annotated.csv \
  --manifest /absolute/external/work/input_manifest-HASH.json \
  --work-dir /absolute/external/work \
  --run-dir /absolute/external/work/runs/smoke-01
```

`smoke` selects ten available train examples, balanced by the current CSV label, and runs all twelve arms with one revision by default. `--iterations` accepts 1, 2, or 3; choose and freeze the main value using train data. The CLI does not run held-out test cases in this first release. `replay` only reads local artifacts and verifies the cache inventory; it makes no inference calls. A live smoke requires working gateway and vLLM access and incurs model usage.

## Shared contracts

The versioned Python records live in `det_rep/contracts.py`. `Example` is the source/answer unit. `EvidencePack` contains all context and query sentences with stable `C<n>` and `Q<n>` IDs and is built once per source. `FeedbackRecord` contains four separate blocks: entity diagnostics E, directed relation diagnostics R, atomic claim verdicts C, and claim-to-sentence links X. Feedback and prompt hashes, model identity, use, status, and every iteration are stored in `Trajectory` files. `EvaluationRequest` contains an opaque request ID, source text, query, original answer, and final revision; source IDs and treatment labels live in a separate assignment file. The request IDs use a run-specific random salt. All records use `det-rep-v1`.

The arms are `B E R ER C EC RC ERC CX ECX RCX ERCX`. B receives only the shared evidence, query, and current answer. X is available only with C. Every arm starts from the same original answer in a fresh stateless vLLM request. Gemini recomputes feedback on the current answer at each revision; content-addressed caches reuse identical work. Failure files contain source, arm, iteration, stage, and exception type, without raw prompts or credentials. A failed arm is not silently scored as successful.

Team ownership follows the contracts: role 1 owns orchestration and Gemini/vLLM integration; role 2 implements `ClaimAligner`; role 3 implements `FrozenEvaluator`, metrics, and human audit; role 4 implements `SpanAssessor` using separately supplied original span annotations; role 5 implements `AnswerModelKG` in a distinct model/cache namespace; role 6 supplies the remaining answers through `AnswerSource` or the same CSV schema. Scientific implementations for roles 2–6 are intentionally absent. Tests use fakes only to verify integration. The copied `det_rep/core` modules preserve the existing Gemini extraction, S-BERT matching, claim verification, structured output, retries, and scientific cache behavior.

Before a held-out scientific run, the team must freeze the evaluator, prompt/schema versions, Llama checkpoint, iteration count, failure policy, and full 989-answer manifest using training data. The old `strict`, `support`, and `support-critical` results remain historical baselines, not new results from this repository.
