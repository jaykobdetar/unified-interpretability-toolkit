# Registered dense static-view policy

`StaticPolicy` validates installed dense model data for static inspection. The owner registry supplies the name, repository, pinned revision, license and saved file hashes. Catalog polling does not download, register, enable, calibrate or execute a model.

This policy depends on the bounded built-in configuration descriptor and owner-acquisition contracts. Its host integration is described in [STATIC-HOST-INTEGRATION.md](STATIC-HOST-INTEGRATION.md). The narrower real-model admission is described in [DENSE-STATIC-ADMISSION.md](DENSE-STATIC-ADMISSION.md). Evidence and qualification limits are described in [STATIC-QUALIFICATION.md](STATIC-QUALIFICATION.md).

## Public state and identity

`catalog()` uses a bounded registry snapshot, exact directory inventory and current regular-file fingerprints. It does not open configuration, tensor headers or payload content. `static_view_candidate` means an enabled owner-expected receipt contains configuration data and can be considered for explicit activation. Architecture and storage validation still occur during activation.

Readiness requires the same registry's private `BoundStatic`, a current receipt, the selected owned context and an unexpired lease. The host supplies its last coordinator observation of reader health. Catalog polling does not sample or reap processes. A current fingerprint mismatch or expired lease can make the catalog not ready without starting lifecycle work.

Public identity is the opaque `m_<content_digest>`. Catalog entries expose the owner name, repository, full revision, license ID, receipt provenance and installed bytes. Private source paths and capabilities are excluded. Inference, editing, downloads and measured-fit permissions remain false.

`weight-atlas-static-binding-v1` binds the owner content digest, descriptor digest, native source identity, native model identity and opaque model ID. Its evidence consists of saved owner verification, current fingerprints, bounded metadata validation and complete native catalog correspondence. `fresh_payload_hashes_recomputed=false` distinguishes the saved receipt from a new payload hash.

The configuration descriptor remains `weight-atlas-model-descriptor-v1`. Static projection uses `static_model_descriptor` and `weight-atlas-static-descriptor-v1`; it does not invent legacy inference head layouts. Projection removes private paths, legacy inference annotations, unknown fields and raw calibration errors. Experiment-v1 logs retain their existing closed source tuple and FP32/greedy/seed-zero semantics.

## Explicit preparation and binding

`prepare(model_id)` runs under the existing serial owner/resource admission. It requires an enabled, current owner-expected receipt, the same canonical directory and an exact listed inventory. Extra shards, unlisted model API metadata and canonical-root replacement are refused.

Preparation reads at most 64 KiB of configuration, 2 MiB for each tensor header or shard index, and 8 MiB total activation metadata. Each file open uses no-follow/nonblocking flags and regular-file/fingerprint checks before and after reading. Unbuffered weight-header reads consume the length prefix and declared header only. Configuration and index bytes must match saved hashes. Weight payloads are not read or rehashed by this policy.

The built-in descriptor covers a narrow dense full-attention subset of Llama, Qwen2 and Qwen3 configurations reviewed against Transformers 4.56.2. Tensor names, shapes, vectors and tied aliases must match exactly. BF16, F16 and F32 tensors require explicit dense widths, contiguous extents and complete coverage. Mixed dense dtypes are admissible for static viewing; they establish no inference precision policy. The existing 200000 display-axis limit remains in force.

Packed or quantized encodings, custom model code, unsupported configuration, missing/extra/repeated tensors, duplicate JSON keys, nonfinite metadata, gaps, overlaps and index/header disagreements are refused. Index metadata is bounded flat JSON scalar data without floating values; integer values must fit native JSON's signed or unsigned 64-bit range. No pickle, model-shipped executable code or `trust_remote_code` path is introduced.

Preparation derives native identities from the canonical root, sorted shards, exact headers, file-stat identities, data starts and optional parsed index. `bind()` then requires complete correspondence with the private native model response: API version, owner revision, source/model identities, byte and parameter totals, and every tensor's ID, name, shape, axes, dtype, width, count, shard, offset, availability and display/slice semantics. A mismatch refuses readiness.

`PreparedStatic` and `BoundStatic` remain private Python objects. They cannot be supplied through HTTP, browser selections or persisted leases. A new reader context requires fresh preparation and complete binding even when reopening the same model ID.

## Ownership and publication

The host keeps the existing opaque acquire request, context capabilities, single-reader slot, lease limits, serial coordinator and uncertain-cleanup behavior. A candidate does not itself authorize a renderer factory. The returned reader handle is installed before fallible initialization.

Before every static read, the host requires the current lease/context, receipt and full binding. It repeats the source/context checks after channel I/O. Publication also retains the selected reader and original resource grant through serialization, bounded writing, watch joining and settlement. Close, expiry, disable, source change, observed reader loss or read failure immediately revoke readiness and proofs. Unconfirmed cleanup retains ownership and blocks replacement.

Static profiles and inference remain disabled. Browser picker permission is an admission hint; numeric requests wait for a verified static model response bound to the selected context. Catalog polling never performs a payload scan or calibration. Qualification of one pinned source does not enable arbitrary registered models.
