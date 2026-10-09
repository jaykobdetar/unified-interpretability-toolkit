# Documentation

## Inspect local weights

Start with the [static-analysis walkthrough](STATIC-ANALYSIS.md), then use these references as needed:

| Task | Guide |
| --- | --- |
| Build, launch, configure a cache, or diagnose startup | [Launch and troubleshooting](LAUNCH.md) |
| Understand supported storage, color scales, and pooled pixels | [Formats and color semantics](FORMATS.md) |
| Compare matching checkpoints and inspect differences | [Checkpoint comparison](COMPARISON.md) |
| Export exact original values with source metadata | [Region exports](REGION-EXPORT.md) |
| Follow a bounded real-weight observation | [Qwen weight windows](WORKED-EXAMPLE.md) |
| Understand screenshot sources and reproduce the views | [Screenshot notes](screenshots/README.md) |
| Choose explicit standalone resource budgets | [Resource configuration](RESOURCES.md) |

## Optional modes and qualification limits

[Inference](INFERENCE.md) covers the separate pinned CPU workflow. [Regional analytics](analytics/CONTRACT.md) and [whole-slice profiles](profile-worker/PROFILE-WORKER-INTERFACE.md) have their own source, dependency, and resource restrictions. Static file support does not qualify a model for those modes. [Feature status](FEATURE-STATUS.md) and [workflow qualification](DATA-WORKFLOW-QUALIFICATION.md) retain implemented-versus-qualified distinctions.

Registered static inspection uses the separate owner/registry layer: [static host integration](STATIC-HOST-INTEGRATION.md), [dense admission](DENSE-STATIC-ADMISSION.md), and [static evidence and limits](STATIC-QUALIFICATION.md). It is not required by the standalone viewer quickstart. Raw inspection readiness does not establish color or inference readiness.

## Develop and integrate

Read [architecture](ARCHITECTURE.md), [development and testing](DEVELOPMENT.md), [extension recipes](EXTENDING.md), and the [progressive API](API-PROGRESSIVE.md). The [base API](API-v1.md) retains the earlier interface contract; use the progressive reference for current readiness behavior.

The `*-CONTRACT.md` files retain detailed source-binding, edits, observations, logging, host, and profile semantics. Contracts describe required behavior; qualification documents state what was exercised and what remains unverified. Plans such as [the model-inference plan](MODEL-INFERENCE-PLAN.md) are not support promises.

`models/` contains operational source pins/configuration and retained model notices required by verification and tests. `screenshots/` contains genuine application captures with their source descriptions. Historical machine receipts and private review archives remain outside the release tree.
