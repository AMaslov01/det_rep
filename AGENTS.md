# det_rep

This repository coordinates a research comparison of hallucination detection feedback and answer correction. It joins fixed RAGTruth sources with annotated Llama answers, builds Gemini feedback, runs matched correction conditions through vLLM, and prepares blinded requests for independent evaluation. The Gemini gateway service lives in the same repository; the original `hallu_smiles` project remains separate.

Inputs, model weights, caches, credentials, and run results live outside this repository. The README describes the current interfaces and commands.

**Shared server rule:** Before planning or executing work on an Applied AI server, transferring data or models, using Docker or GPU, or cleaning storage, read [docs/server-resource-rules.md](docs/server-resource-rules.md). It includes the centre's rules and the operator's additional constraints for caniculus. Never act on another user's files, images, containers, or processes.

The staged procedure and current compatibility checks for caniculus are in [docs/caniculus-runbook.md](docs/caniculus-runbook.md). Server execution needs the owner's separate command.
