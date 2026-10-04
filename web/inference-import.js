(() => {
    "use strict";
    // Closed, bounded archive import. No DOM, transport, storage, or worker ownership.
    const AtlasExperimentImport = (() => {
        const utf8Size = text => (new TextEncoder).encode(text).length;
        const MAX_BYTES = 1048576, MAX_RUNS = 8;
        const check = (ok, msg) => {
            if (!ok) throw new Error(msg);
        };
        const keys = (v, allowed) => check(v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).every((k => (typeof allowed === "string" ? allowed.split(" ") : allowed).includes(k))), "Unknown or invalid experiment fields");
        const integer = (v, lo, hi) => Number.isSafeInteger(v) && v >= lo && v <= hi;
        const timestamp = v => typeof v === "string" && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(v) && Number.isFinite(Date.parse(v)) && new Date(v).toISOString() === v;
        const canonical = v => JSON.stringify(v, ((_, x) => x && typeof x === "object" && !Array.isArray(x) ? Object.fromEntries(Object.keys(x).sort().map((k => [ k, x[k] ]))) : x));
        function parse(text) {
            check(typeof text === "string" && (new TextEncoder).encode(text).length <= MAX_BYTES, "Import exceeds 1 MiB");
            // Detect duplicate names before JSON.parse can overwrite their evidence.
            const tokens = text.match(/"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"|[{}\[\]:,]|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null/g) || [], stack = [];
            for (let i = 0; i < tokens.length; i++) {
                const t = tokens[i];
                if (t === "{" || t === "[") {
                    stack.push(t === "{" ? new Set : null);
                    check(stack.length <= 24, "Import nesting exceeds limit");
                } else if (t === "}" || t === "]") stack.pop(); else if (t[0] === '"' && tokens[i + 1] === ":") {
                    const k = JSON.parse(t), set = stack.at(-1);
                    check(set && !set.has(k), "Duplicate JSON field");
                    set.add(k);
                    check(![ "__proto__", "prototype", "constructor" ].includes(k), "Unsafe JSON field");
                }
            }
            const value = JSON.parse(text);
            let count = 0;
            const bounded = (x, depth) => {
                check(depth <= 24 && ++count <= 1e5, "Import structure exceeds limit");
                if (typeof x === "number") check(Number.isFinite(x), "Nonfinite experiment data");
                if (typeof x === "string") check(x.length <= 16384, "Experiment text exceeds limit");
                if (Array.isArray(x)) check(x.length <= 576, "Experiment array exceeds limit");
                if (x && typeof x === "object") for (const [k, v] of Object.entries(x)) {
                    check(![ "session", "session_id", "capability", "path", "source_directory", "process_id", "pid", "cleanup_pending" ].includes(k), "Worker ownership or filesystem fields cannot be imported");
                    bounded(v, depth + 1);
                }
            };
            bounded(value, 0);
            return value;
        }
        function validateRecord(record, {experimentRecord: experimentRecord}) {
            keys(record, "schema created_at status complete worker_cleanup_confirmed termination privacy request settings runtime limits summary baseline edited steps sweep_plan");
            check(record.runtime && !Array.isArray(record.runtime) && typeof record.runtime === "object" && record.summary && !Array.isArray(record.summary) && typeof record.summary === "object", "Invalid runtime or summary");
            check(record.schema === "weight-atlas-experiment-v1" && timestamp(record.created_at), "Invalid experiment schema or timestamp");
            check([ "complete", "cancelled", "error", "resource_limit", "time_limit", "client_timeout", "connection_lost" ].includes(record.status) && record.complete === (record.status === "complete") && typeof record.worker_cleanup_confirmed === "boolean", "Only terminal archived runs can be imported");
            const request = record.request;
            keys(request, "mode max_new_tokens layer activation_site source_model edits observation prompt prompt_ids prompts preview_digest plan_digest positions targets operation scale seed capture_layer");
            check([ undefined, "generate", "comparison", "prompt_pair", "sweep" ].includes(request.mode), "Unsupported archived mode");
            keys(request.source_model, "repo revision weights_sha256");
            check(Object.keys(request.source_model).length === 0 || [ "repo", "revision", "weights_sha256" ].every((k => typeof request.source_model[k] === "string" && request.source_model[k].length > 0 && request.source_model[k].length <= 512)), "Invalid archived model identity");
            check(Array.isArray(record.steps) && record.steps.length <= 32, "Invalid archived trace cap");
            let equivalents = 0, width = null;
            for (const [index, step] of record.steps.entries()) {
                check(step.index === index && Array.isArray(step.activation) && step.activation.length > 0 && step.activation.length <= 576 && step.activation.every(Number.isFinite), "Invalid archived activation");
                width ??= step.activation.length;
                check(step.activation.length === width, "Activation widths differ");
                equivalents += step.activation.length;
                for (const table of [ step.top_logits, step.candidates, step.logit_lens?.candidates, step.sweep?.candidates ]) if (table !== undefined) check(Array.isArray(table) && table.length <= 10, "Candidate cap exceeded");
                if (step.prompt_pair) for (const side of [ "a", "b" ]) {
                    const vector = step.prompt_pair[side]?.activation;
                    check(Array.isArray(vector) && vector.length === width && vector.every(Number.isFinite), "Invalid paired activation");
                    equivalents += vector.length;
                }
                if (step.attention) {
                    const a = step.attention;
                    check(Array.isArray(a.probabilities) && Array.isArray(a.key_positions) && a.probabilities.length <= 160 && a.probabilities.length === a.key_positions.length && a.probabilities.every((x => Number.isFinite(x) && x >= 0 && x <= 1)), "Invalid attention archive");
                }
            }
            check(equivalents <= 18432, "Archived vector budget exceeded");
            check(Array.isArray(request.edits || []) && (request.edits || []).length <= 8, "Archived edit cap exceeded");
            if (request.mode === "sweep") check(record.sweep_plan && Array.isArray(record.sweep_plan.cases) && record.sweep_plan.cases.length <= 19 && Array.isArray(request.targets) && request.targets.length <= 2 && record.limits?.prompts <= 2, "Archived sweep cap exceeded");
            const edit = e => {
                const element = e.kind === "element", fields = [ "tensor", "shape", "kind", "operation", ...element ? [ "row", "col" ] : [ "start", "end" ], ...e.operation === "scale" ? [ "scale" ] : [] ];
                keys(e, fields);
                check(Object.keys(e).length === fields.length && typeof e.tensor === "string" && e.tensor.length > 0 && Array.isArray(e.shape) && e.shape.length === 2 && e.shape.every((n => integer(n, 1, Number.MAX_SAFE_INTEGER))) && Number.isSafeInteger(e.shape[0] * e.shape[1]) && [ "element", "rows", "columns" ].includes(e.kind) && [ "zero", "scale" ].includes(e.operation), "Invalid archived edit");
                check(element ? integer(e.row, 0, e.shape[0] - 1) && integer(e.col, 0, e.shape[1] - 1) : integer(e.start, 0, e.shape[e.kind === "rows" ? 0 : 1] - 1) && integer(e.end, e.start + 1, e.shape[e.kind === "rows" ? 0 : 1]), "Edit coordinates exceed native shape");
                if (e.operation === "scale") check(Number.isFinite(e.scale) && Math.abs(e.scale) <= 100, "Invalid edit scale");
            };
            (request.edits || []).forEach(edit);
            const target = t => {
                const fields = [ "kind", "layer", ...t.kind === "offset" ? [ "heads", "offset" ] : t.kind === "layer_heads" ? [] : [ "head" ] ];
                keys(t, fields);
                check(Object.keys(t).length === fields.length && [ "head", "query_head", "offset", "layer_heads" ].includes(t.kind) && integer(t.layer, 0, 29) && (t.kind === "layer_heads" || (t.kind === "offset" ? Array.isArray(t.heads) && t.heads.length > 0 && t.heads.length <= 8 && new Set(t.heads).size === t.heads.length && t.heads.every((h => integer(h, 0, 8))) && integer(t.offset, 0, 63) : integer(t.head, 0, 8))), "Invalid archived target");
            };
            const candidates = a => a?.forEach(c => {
                keys(c, "id piece baseline_logit edited_logit delta");
                check(Object.keys(c).length === 5 && integer(c.id, 0, 49151) && typeof c.piece === "string" && utf8Size(c.piece) <= 1024, "Invalid candidate fields");
                check([c.baseline_logit, c.edited_logit, c.delta].every(x => x === null || Number.isFinite(x)) && c.delta === (c.baseline_logit === null || c.edited_logit === null ? null : c.edited_logit - c.baseline_logit), "Invalid candidate scores");
            });
            const coverage = v => {
                if (v && typeof v === "object") {
                    for (const k of [ "planned_ids", "completed_ids", "unrun_ids" ]) check(Array.isArray(v[k]) && v[k].length <= 32 && v[k].every((x => typeof x === "string")) && new Set(v[k]).size === v[k].length, "Invalid coverage IDs");
                    check(typeof v.complete === "boolean" && (v.interrupted_id === null || typeof v.interrupted_id === "string") && Array.isArray(v.unfinished_heads) && v.unfinished_heads.every((h => integer(h, 0, 8))), "Invalid coverage state");
                    keys(v, "planned_ids completed_ids unrun_ids interrupted_id complete finished_targets unfinished_targets unfinished_heads");
                    check(Object.keys(v).length === 8 && [v.finished_targets, v.unfinished_targets].every(a => Array.isArray(a) && a.length <= 9), "Invalid target coverage");
                    for (const a of [ v.finished_targets, v.unfinished_targets ]) a?.forEach(target);
                }
            };
            coverage(record.summary?.coverage);
            coverage(record.summary?.sweep_coverage);
            request.targets?.forEach(target);
            for (const step of record.steps) {
                candidates(step.candidates);
                if (step.sweep) {
                    keys(step.sweep.metrics, "logit_delta_rms logit_delta_max_abs softmax_total_variation baseline_argmax_id baseline_argmax_logit_delta edited_argmax_id context semantics");
                    candidates(step.sweep.candidates);
                }
            }
            if (record.sweep_plan) {
                const plan = record.sweep_plan;
                keys(plan, "version scope architecture intervention_semantics control_semantics seed control_version source_model targets cases prompt_count records prefills capture_layer activation_site coverage limits");
                keys(plan.source_model, "repo revision weights_sha256");
                plan.targets.forEach(target);
                for (const c of plan.cases) {
                    keys(c, "id role target edits selected_cells control_geometry");
                    if (c.target) target(c.target);
                    check(Array.isArray(c.edits) && c.edits.length <= 8, "Invalid case edits");
                    c.edits.forEach(edit);
                }
                const architecture = a => {
                    const dimensions = {width:576, layers:30, query_heads:9, kv_heads:3, head_dim:64, queries_per_kv:3, vocab_size:49152, intermediate_size:1536};
                    check(Object.entries(dimensions).every(([k,v]) => a[k] === v) && a.model_type === 'llama' && a.layout === 'llama-eager-head-major-v1', 'Invalid archived architecture');
                    keys(a, "model_type width layers query_heads kv_heads head_dim queries_per_kv vocab_size intermediate_size capture_sites layout head_ablation query_intervention");
                    keys(a.capture_sites, "block attention mlp");
                    check(Object.keys(a).length === 13 && Object.keys(a.capture_sites).length === 3 && Object.values(a.capture_sites).every(x => typeof x === 'string') && a.head_ablation === 'o_proj columns' && a.query_intervention === 'q_proj rows', 'Invalid capture architecture');
                };
                architecture(plan.architecture);
                keys(plan.limits, "targets interventions_including_controls prompts subset_limits probes_per_prompt records prefills edits selected_cells wall_seconds worker_cpu_seconds control_version full_layer_records trace_cap architecture scope");
                keys(plan.limits.subset_limits, "targets records prefills");
                architecture(plan.limits.architecture);
                const caps = {targets:9, interventions_including_controls:19, prompts:2, probes_per_prompt:1, records:19, prefills:38, edits:8, selected_cells:65536, wall_seconds:120, worker_cpu_seconds:90, full_layer_records:19, trace_cap:32};
                check(Object.entries(caps).every(([k,v]) => plan.limits[k] === v) && canonical(plan.limits.subset_limits) === canonical({targets:2, records:10, prefills:20}) && plan.control_version === 'weight-atlas-sweep-control-v2' && plan.limits.control_version === plan.control_version && canonical(plan.architecture) === canonical(plan.limits.architecture), 'Invalid sweep limits');
                check(['intervention_semantics','control_semantics','coverage'].every(k => typeof plan[k] === 'string') && typeof plan.limits.scope === 'string', 'Invalid sweep metadata');
            }
            // Semantic constraints are independent of the exporter field projection.
            const pair = request.mode === "prompt_pair", sweep = request.mode === "sweep", budget = pair || sweep ? 0 : request.max_new_tokens;
            check([ "block", "attention", "mlp" ].includes(request.activation_site) && integer(sweep ? request.capture_layer : request.layer, 0, 29), "Invalid capture request");
            const modeFields = [ "mode", "source_model", "activation_site", ...pair ? [ "layer", "positions", "prompts", "preview_digest" ] : sweep ? [ "targets", "operation", "scale", "seed", "capture_layer", "prompts", "plan_digest" ] : [ "layer", "max_new_tokens", "edits", "observation", "prompt", "prompt_ids" ] ];
            keys(request, modeFields);
            if (!pair && !sweep) check(integer(budget, 1, 32) && record.steps.length <= budget, "Invalid generation budget");
            if (pair) {
                check(Array.isArray(request.positions) && request.positions.length > 0 && request.positions.length <= 8 && record.steps.length <= request.positions.length && new Set(request.positions.map((p => p.a + "," + p.b))).size === request.positions.length, "Invalid paired positions");
                for (const p of request.positions) {
                    keys(p, "a b");
                    check(integer(p.a, 0, 127) && integer(p.b, 0, 127), "Invalid paired position");
                }
            }
            if (sweep) {
                check(request.targets.length > 0 && integer(request.seed, 0, 4294967295) && [ "zero", "scale" ].includes(request.operation) && (request.operation === "scale" ? Number.isFinite(request.scale) && Math.abs(request.scale) <= 100 : request.scale === undefined), "Invalid sweep request");
                check(!request.targets.some((t => t.kind === "layer_heads")) || request.targets.length === 1 && request.operation === "zero", "Invalid all-head request");
                const p = record.sweep_plan;
                check(p.version === "weight-atlas-sweep-plan-v2" && [ "subset", "layer_heads" ].includes(p.scope) && integer(p.prompt_count, 1, 2) && p.prompt_count === record.limits.prompts && p.records === p.cases.length * p.prompt_count && p.prefills === 2 * p.records && integer(p.records, 1, 32) && record.steps.length <= p.records && p.seed === request.seed && p.capture_layer === request.capture_layer && p.activation_site === request.activation_site && canonical(p.source_model) === canonical(request.source_model), "Invalid sweep plan counts or capture");
            }
            if (request.prompts !== undefined) check(Array.isArray(request.prompts) && request.prompts.length === (pair ? 2 : record.limits.prompts) && request.prompts.every((p => typeof p === "string" && p.trim() && utf8Size(p) <= 2048)), "Invalid private prompts");
            if (request.prompt !== undefined) check(typeof request.prompt === "string" && request.prompt.trim() && utf8Size(request.prompt) <= 2048, "Invalid private prompt");
            for (const k of [ "preview_digest", "plan_digest" ]) if (request[k] !== undefined) check(typeof request[k] === "string" && /^[a-f0-9]{64}$/.test(request[k]), "Invalid prompt/plan digest");
            if (request.observation) {
                keys(request.observation, "kind head");
                check(request.observation.kind === "attention" ? integer(request.observation.head, 0, 8) && request.activation_site === "attention" : request.observation.kind === "logit_lens" && request.observation.head === undefined && request.activation_site === "block", "Invalid observation request");
            }
            const ids = (v, cap) => check(Array.isArray(v) && v.length <= cap && v.every((id => integer(id, 0, 49151))), "Invalid archived token IDs or budget");
            if (request.prompt_ids !== undefined) ids(request.prompt_ids, 128);
            const rules = {}, rule = (names, test) => names.split(" ").forEach((k => rules[k] = test));
            rule("token_id input_token_id id lens_argmax_id final_argmax_id baseline_argmax_id edited_argmax_id", (x => integer(x, 0, 49151)));
            rule("eos token_equal prefix_equal restoration_verified deterministic_algorithms", (x => typeof x === "boolean"));
            rule("compute_ms compute_total_ms load_ms a_l2 b_l2 delta_l2 parameter_delta_l2 logit_delta_rms logit_delta_max_abs", (x => Number.isFinite(x) && x >= 0));
            rule("value lens_logit final_logit delta_lens_minus_final baseline_argmax_logit_delta", Number.isFinite);
            rule("baseline_logit edited_logit delta", (x => x === null || Number.isFinite(x)));
            rule("generated_text token_piece piece reason error context semantics python torch transformers tokenizers safetensors platform machine device dtype sampling attention_backend termination", (x => typeof x === "string"));
            rule("generated_tokens", (x => integer(x, 0, budget)));
            rule("record_count", (x => integer(x, 0, 32)));
            rule("selected_cells changed_cells", (x => integer(x, 0, 65536)));
            rule("layer", x => integer(x, 0, 29));
            rule("activation_site", x => ["block", "attention", "mlp"].includes(x));
            rule("phase", x => ["prefill", "decode"].includes(x));
            rule("activation_kind activation_branch alignment score_kind comparison_phase sweep_current", x => typeof x === "string");
            rule("position query_position", (x => integer(x, 0, 159)));
            rule("seed", (x => integer(x, 0, 4294967295)));
            rule("numerical_threads", (x => integer(x, 1, 64)));
            rule("cosine", (x => x === null || Number.isFinite(x) && Math.abs(x) <= 1));
            rule("softmax_total_variation", (x => Number.isFinite(x) && x >= 0 && x <= 1 + 1e-6));
            const leaves = v => {
                if (v && typeof v === "object") for (const [k, x] of Object.entries(v)) {
                    if (k === "generated_ids" || k === "key_token_ids") ids(x, k === "generated_ids" ? budget : 160); else if (rules[k]) check(rules[k](x), "Invalid archived " + k);
                    leaves(x);
                }
            };
            for (const v of [ record.runtime, record.summary, record.baseline, record.edited, ...record.steps ]) leaves(v);
            for (const step of record.steps) {
                check(pair ? !!step.prompt_pair && !step.sweep : sweep ? !!step.sweep && !step.prompt_pair : !step.prompt_pair && !step.sweep, "Trace mode differs from request");
                if (step.layer !== undefined) check(step.layer === (sweep ? request.capture_layer : request.layer), "Trace layer differs");
                if (step.activation_site !== undefined) check(step.activation_site === request.activation_site, "Trace site differs");
                if (step.prompt_pair) for (const k of [ "a", "b" ]) check(step.prompt_pair[k].position === request.positions[step.index][k], "Pair trace position differs");
                for (const side of [ step.baseline, step.edited ]) if (side) {
                    ids(side.generated_ids, Math.min(budget, step.index + 1));
                    check(side.generated_ids.length === step.index + 1 && side.token_id === side.generated_ids.at(-1), "Invalid branch sequence");
                }
                if (step.attention) check(step.attention.key_positions.every(((p, i) => p === i)), "Invalid attention positions");
            }
            if (sweep) {
                const p = record.sweep_plan, planned = p.cases.flatMap((c => Array.from({
                    length: p.prompt_count
                }, ((_, i) => c.id + "/prompt-" + (i + 1)))));
                check(p.targets.length === (p.scope === "layer_heads" ? 9 : request.targets.length) && p.cases.length === 1 + 2 * p.targets.length && new Set(planned).size === planned.length, "Invalid sweep expansion");
                for (const [i, c] of p.cases.entries()) check(c.id === (i === 0 ? "empty" : (i % 2 ? "target" : "matched_control") + "-" + Math.ceil(i / 2)) && c.role === (i === 0 ? "empty_control" : i % 2 ? "target" : "matched_control") && integer(c.selected_cells, 0, 65536) && (i === 0 ? c.edits.length === 0 && c.target === null && c.selected_cells === 0 : !!c.target), "Invalid sweep case");
                for (const v of [ record.summary.coverage, record.summary.sweep_coverage ]) if (v && typeof v === "object") check(canonical(v.planned_ids) === canonical(planned) && canonical(v.completed_ids) === canonical(planned.slice(0, record.steps.length)) && canonical(v.unrun_ids) === canonical(planned.slice(record.steps.length)) && v.complete === (record.steps.length === p.records) && (v.interrupted_id === null || v.interrupted_id === planned[record.steps.length]), "Coverage differs from trace");
                for (const step of record.steps) {
                    const v = step.sweep, c = p.cases[Math.floor(step.index / p.prompt_count)];
                    check(v.record_id === planned[step.index] && v.case_id === c.id && v.role === c.role && v.prompt_index === step.index % p.prompt_count && v.selected_cells === c.selected_cells && integer(v.changed_cells, 0, c.selected_cells) && v.restoration_verified === true && Object.keys(v.metrics).length === 8, "Sweep trace differs from plan");
                }
            }
            for (const step of record.steps) {
                for (const table of [ step.top_logits, step.candidates, step.logit_lens?.candidates, step.sweep?.candidates ]) if (table) check(table.length > 0 && new Set(table.map((c => c.id))).size === table.length, "Invalid candidate IDs");
                if (step.prompt_pair) check((!step.prompt_pair.prefix_equal || step.prompt_pair.token_equal && step.prompt_pair.a.position === step.prompt_pair.b.position) && (!record.privacy.prompt_included || step.prompt_pair.token_equal === (step.prompt_pair.a.token_id === step.prompt_pair.b.token_id)) && Object.keys(step.prompt_pair.metrics).length === 4 && typeof step.prompt_pair.token_equal === "boolean" && typeof step.prompt_pair.prefix_equal === "boolean", "Invalid pair metrics");
                for (const c of step.sweep?.candidates || []) check(Number.isFinite(c.baseline_logit) && Number.isFinite(c.edited_logit) && c.delta === c.edited_logit - c.baseline_logit, "Invalid sweep score delta");
            }
            check(typeof record.privacy?.prompt_included === "boolean" && (record.termination === null || typeof record.termination === "string"), "Invalid consent or termination");
            if (record.privacy.prompt_included) {
                check(pair || sweep ? Array.isArray(request.prompts) : typeof request.prompt === "string", "Missing consented prompt");
                if (pair || sweep) check(typeof request[pair ? 'preview_digest' : 'plan_digest'] === 'string', 'Missing consented plan digest');
            }
            for (const step of record.steps) {
                for (const c of step.top_logits || []) {
                    keys(c, 'id value');
                    check(Object.keys(c).length === 2 && integer(c.id, 0, 49151) && Number.isFinite(c.value), 'Invalid top logit');
                }
                for (const c of step.logit_lens?.candidates || []) {
                    keys(c, 'id piece lens_logit final_logit delta_lens_minus_final');
                    check(Object.keys(c).length === 5 && integer(c.id, 0, 49151) && typeof c.piece === 'string' && utf8Size(c.piece) <= 1024 && Number.isFinite(c.lens_logit) && Number.isFinite(c.final_logit) && c.delta_lens_minus_final === c.lens_logit - c.final_logit, 'Invalid lens candidate');
                }
            }
            // Rebuild through the exporter: removes unknown fields and recomputes derived
            // status/privacy/settings/limits. Equality refuses anything it would discard.
            const req = {
                ...request,
                ...request.mode === "sweep" && !request.prompts ? {
                    prompts: Array(record.limits.prompts).fill("")
                } : {}
            };
            const details = {
                ...record.summary,
                runtime: record.runtime,
                baseline: record.baseline,
                edited: record.edited,
                sweep_plan: record.sweep_plan,
                prompt_ids: request.prompt_ids
            };
            const rebuilt = experimentRecord(req, {
                status: record.status,
                worker_alive: record.worker_cleanup_confirmed ? false : null,
                steps: record.steps,
                details: details,
                termination: record.termination
            }, {
                includePrompt: record.privacy.prompt_included,
                createdAt: record.created_at
            });
            check(canonical(rebuilt) === canonical(record), "Archive differs from the supported export contract");
            return rebuilt;
        }
        function read(text, {experimentRecord: experimentRecord, redactExperiment: redactExperiment, includePrompt: includePrompt = false} = {}) {
            check(typeof experimentRecord === "function" && typeof redactExperiment === "function", "Experiment codec unavailable");
            const value = parse(text);
            let records;
            if (value.schema === "weight-atlas-session-log-v1") {
                keys(value, "schema created_at persistence limits records");
                keys(value.limits, "bytes runs");
                check(value.limits.bytes <= MAX_BYTES && value.limits.runs <= MAX_RUNS && Number.isSafeInteger(value.limits.bytes) && Number.isSafeInteger(value.limits.runs) && value.limits.bytes > 0 && value.limits.runs > 0, "Invalid archive limits");
                check(timestamp(value.created_at) && value.persistence === "browser session memory; durable only after explicit file download" && utf8Size(text) <= value.limits.bytes && Array.isArray(value.records) && value.records.length <= value.limits.runs, "Invalid archive envelope or declared cap");
                records = value.records;
            } else records = [ value ];
            check(Array.isArray(records) && records.length > 0 && records.length <= MAX_RUNS, "Import requires 1–8 archived runs");
            return records.map((r => {
                const valid = validateRecord(r, {
                    experimentRecord: experimentRecord
                });
                return includePrompt ? valid : redactExperiment(valid);
            }));
        }
        function append(log, records, {beforeCommit: beforeCommit = (() => {})} = {}) {
            check(Array.isArray(records) && records.length > 0, "Missing imported records");
            const next = [ ...log.records, ...records ], text = JSON.stringify(log.envelope(next), null, 2);
            check(next.length <= Math.min(log.maxRuns, MAX_RUNS) && (new TextEncoder).encode(text).length <= Math.min(log.maxBytes, MAX_BYTES), "Import would exceed the log cap; existing records were unchanged");
            const detached = JSON.parse(JSON.stringify(next));
            beforeCommit();
            log.records = detached;
            return records.length;
        }
        return {
            MAX_BYTES: MAX_BYTES,
            MAX_RUNS: MAX_RUNS,
            read: read,
            append: append
        };
    })();
    if (typeof module !== "undefined") module.exports = AtlasExperimentImport; else globalThis.AtlasExperimentImport = AtlasExperimentImport;
})();