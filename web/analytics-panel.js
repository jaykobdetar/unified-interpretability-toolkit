// Standalone component; no global app state, dependencies, or geometry mutation.
export function mountAnalytics(root, {load, jump}) {
  let epoch = 0, report = null, disposed = false;
  const doc = root.ownerDocument;
  const el = (tag, text, cls) => {
    const node = doc.createElement(tag);
    if (text != null) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const panel = el('section', null, 'atlas-analytics');
  panel.append(el('h2', 'Native-region analytics'));
  const controls = el('div', null, 'analytics-controls');
  const seedLabel = el('label', 'Shuffle seed ');
  const seed = el('input'); seed.type = 'number'; seed.min = '0'; seed.max = '4294967295'; seed.value = '1';
  seedLabel.append(seed); controls.append(seedLabel);
  const viewLabel = el('label', 'View '), view = el('select');
  view.setAttribute('aria-label', 'View');
  for (const [key, label] of [['strength', 'Mean absolute strength'], ['outliers', 'Largest absolute values'],
    ['heads', 'Head boundaries and folded offsets'], ['vector', 'Vector bars'], ['svd', 'Singular energy and residual (64 detail)'], ['svd_summary', 'SVD spectrum and residual preview (128 summary)']]) {
    const option = el('option', label); option.value = key; view.append(option);
  }
  viewLabel.append(view); controls.append(viewLabel);
  const sortLabel = el('label', 'Sort strips '), sort = el('input'); sort.type = 'checkbox'; sortLabel.prepend(sort); controls.append(sortLabel);
  const refresh = el('button', 'Compare original / shuffle'); refresh.type = 'button'; controls.append(refresh);
  const svdButton = el('button', 'Compute bounded SVD comparison'); svdButton.type = 'button'; controls.append(svdButton);
  const summaryButton = el('button', 'Compute 128-window SVD summary'); summaryButton.type = 'button'; controls.append(summaryButton);
  panel.append(controls);
  const status = el('p'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite'); panel.append(status);
  panel.append(el('p', 'Same-region exact-multiset shuffle; statistics fit separately. A shuffle alone does not prove pattern meaning. Sorting can manufacture gradients. Matrix geometry remains in native order.', 'analytics-caution'));
  const body = el('div'); panel.append(body); root.append(panel);
  const fmt = x => x == null ? 'unavailable' : Number(x).toPrecision(4);
  const button = (text, action) => { const b = el('button', text); b.type = 'button'; b.onclick = action; return b; };

  function pairBars(title, original, shuffled, axis, signed = false, sortable = false) {
    body.append(el('h3', title));
    // Sort each population independently: the UI explicitly exposes the artificial
    // gradient in both. Each bar retains its own native/control-position index.
    let a = original, b = shuffled;
    const value = x => x.mean_abs ?? x.value ?? 0;
    if (sort.checked && sortable) {
      const ordering = items => [...items].sort((x, y) => Math.abs(value(y))-Math.abs(value(x)) || x.index-y.index);
      a = ordering(a); b = ordering(b);
    }
    const scale = Math.max(0, ...a.map(x => Math.abs(value(x))), ...b.map(x => Math.abs(value(x))));
    const columns = el('div', null, 'analytics-pair');
    for (const [label, items] of [['Original', a], ['Shuffled control positions', b]]) {
      const box = el('div'); box.append(el('h4', label));
      const canvas = el('canvas'); canvas.width = 800; canvas.height = 160;
      canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${title}: ${label}; ${items.length} bars, shared scale ${fmt(scale)}`);
      const ctx = canvas.getContext('2d');
      const baseline = signed ? 80 : 150;
      ctx.strokeStyle = '#8293a6'; ctx.beginPath(); ctx.moveTo(0, baseline); ctx.lineTo(800, baseline); ctx.stroke();
      items.forEach((item, index) => {
        const v = value(item), height = scale ? Math.abs(v)/scale*(signed ? 70 : 140) : 0;
        ctx.fillStyle = v < 0 ? '#d88b41' : '#4da6c9';
        ctx.fillRect(index*800/items.length, v < 0 ? baseline : baseline-height, Math.max(0.5, 800/items.length), height);
      });
      if (axis && label === 'Original') {
        canvas.onclick = event => {
          const rect = canvas.getBoundingClientRect();
          const item = items[Math.min(items.length-1, Math.max(0, Math.floor((event.clientX-rect.left)/rect.width*items.length)))];
          if (item) jump({axis, index: item.index});
        };
      }
      box.append(canvas);
      const details = el('details'); details.append(el('summary', 'Exact values and index map (first 64 displayed)'));
      const list = el('ol');
      for (const item of items.slice(0, 64)) {
        const li = el('li'); const text = `${item.index}: ${fmt(item.mean_abs ?? item.value)}${item.count != null ? ` (${item.count} values)` : ''}`;
        li.append(axis && label === 'Original' ? button(text, () => jump({axis, index: item.index})) : el('span', text)); list.append(li);
      }
      details.append(list); box.append(details); columns.append(box);
    }
    body.append(columns);
  }

  function foldedMatrices(original, shuffled) {
    body.append(el('h3', 'Signed mean across head groups (other axis preserved)'));
    if (!original.available || !shuffled.available) { body.append(el('p', original.reason ?? shuffled.reason)); return; }
    const scale = Math.max(0, ...original.mean.map(x => Math.abs(x ?? 0)), ...shuffled.mean.map(x => Math.abs(x ?? 0)));
    const pair = el('div', null, 'analytics-pair');
    for (const [label, matrix] of [['Original', original], ['Shuffled control', shuffled]]) {
      const box = el('div'); box.append(el('h4', label));
      box.append(el('p', `${matrix.rows} × ${matrix.cols}; rows: ${matrix.row_axis}; columns: ${matrix.column_axis}. Gray cells have no observations.`));
      const canvas = el('canvas'); canvas.width = 800; canvas.height = 320;
      canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${label} signed mean folded matrix; shared absolute scale ${fmt(scale)}`);
      const ctx = canvas.getContext('2d');
      matrix.mean.forEach((value, index) => {
        const intensity = value == null || !scale ? 0 : Math.min(1, Math.abs(value)/scale);
        ctx.fillStyle = value == null ? '#555d66' : value < 0 ? `rgb(${30+200*intensity}, ${30+80*intensity}, 30)` : `rgb(30, ${30+130*intensity}, ${30+200*intensity})`;
        ctx.fillRect(index%matrix.cols*800/matrix.cols, Math.floor(index/matrix.cols)*320/matrix.rows, Math.ceil(800/matrix.cols), Math.ceil(320/matrix.rows));
      });
      box.append(canvas);
      const details = el('details'); details.append(el('summary', 'Folded means / contributor counts (first 64 cells)'));
      const list = el('ol'); matrix.mean.slice(0,64).forEach((value, index) => list.append(el('li', `offset-grid [${Math.floor(index/matrix.cols)}, ${index%matrix.cols}]: ${fmt(value)} / ${matrix.counts[index]} contributors`)));
      details.append(list); box.append(details); pair.append(box);
    }
    body.append(pair);
  }

  function render() {
    body.replaceChildren();
    svdButton.hidden = view.value !== 'svd';
    summaryButton.hidden = view.value !== 'svd_summary';
    sortLabel.hidden = view.value !== 'strength';
    if (!report) return;
    if (report.schema === 'weight-atlas.svd-window-summary.v2') { renderSummary(); return; }
    if (view.value === 'svd_summary') { body.append(el('p', 'Compute the explicit summary for this selected native window. Full spectrum, 16 × 16 residual preview; 64-detail analysis stays separate.')); return; }
    const {original: a, shuffled: b, coverage, heads} = report;
    status.textContent = `${report.tensor} • ${coverage.visited_values.toLocaleString()} / ${coverage.total_tensor_values.toLocaleString()} tensor values • window rows [${report.region.row}, ${report.region.row+report.region.rows}), columns [${report.region.col}, ${report.region.col+report.region.cols}) • ${coverage.full_tensor ? 'full tensor' : 'partial native window'} • seed ${report.control.seed}`;
    if (view.value === 'strength') {
      pairBars('Row mean |weight|', a.rows, b.rows, 'row', false, true);
      pairBars('Column mean |weight|', a.columns, b.columns, 'column', false, true);
      const label = el('label', 'Jump to native row/column '), axis = el('select'), index = el('input');
      for (const key of ['row', 'column']) { const op = el('option', key); op.value = key; axis.append(op); }
      index.type = 'number'; index.min = '0'; index.value = '0';
      label.append(axis, index, button('Jump', () => { const n = Number(index.value); const max = axis.value === 'row' ? (report.shape.length === 1 ? 1 : report.shape[0]) : report.shape.at(-1); if (Number.isInteger(n) && n >= 0 && n < max) jump({axis: axis.value, index: n}); })); body.append(label);
    } else if (view.value === 'outliers') {
      body.append(el('p', 'Ranked by raw absolute magnitude in this window. Row and column rankings are available through sorted strength strips. Control locations describe shuffled positions, not new source outlier addresses.'));
      for (const [label, stats] of [['Original source coordinates', a], ['Shuffled positions → source coordinates', b]]) {
        body.append(el('h3', label)); const list = el('ol');
        for (const item of stats.top_values) list.append(el('li', `${JSON.stringify(item.native_indices ?? item.control_position)}${item.source_native_indices ? ` → ${JSON.stringify(item.source_native_indices)}` : ''}: ${fmt(item.value)}`));
        body.append(list);
      }
    } else if (view.value === 'vector') {
      if (!a.vector) body.append(el('p', 'Vector bars require a native one-dimensional tensor.'));
      else pairBars('Signed vector values', a.vector, b.vector, 'column', true, false);
    } else if (view.value === 'heads') {
      if (!heads.available) body.append(el('p', heads.reason));
      else {
        body.append(el('h3', heads.label));
        const label = el('label', 'Jump to head group '), selected = el('select');
        selected.setAttribute('aria-label', 'Jump to head group');
        heads.boundaries.slice(0, -1).forEach((start, i) => { const option = el('option', `${i}: ${heads.axis} ${start}–${heads.boundaries[i+1]-1}`); option.value = start; selected.append(option); });
        label.append(selected, button('Jump', () => jump({axis: heads.axis, index: Number(selected.value)}))); body.append(label);
        body.append(el('p', `Fold coverage: ${a.folded.full_head_axis ? 'full head axis' : 'partial head axis'}; head groups ${a.folded.covered_heads.join(', ')}. Other-axis coverage remains the declared native window.`));
        foldedMatrices(a.folded.matrix, b.folded.matrix);
        pairBars('Mean |weight| by within-head offset', a.folded.offsets, b.folded.offsets, null);
      }
    } else if (view.value === 'svd') {
      const svd = report.svd;
      if (!svd.available) body.append(el('p', svd.reason));
      else {
        body.append(el('p', `${svd.scope}; uncentered SVD, separately fitted shuffled control. Original rank-one residual energy: ${fmt(svd.results.original.rank_one_residual_energy_fraction)}; control: ${fmt(svd.results.shuffled.rank_one_residual_energy_fraction)}.`));
        const series = result => result.energy_fractions.map((value, index) => ({index, value}));
        pairBars('Singular energy fraction', series(svd.results.original), series(svd.results.shuffled), null);
        const residual = result => result.rank_one_residual.map((value, index) => ({index, value}));
        pairBars('Leading rank-one residual (row-major window)', residual(svd.results.original), residual(svd.results.shuffled), null, true);
      }
    }
  }

  function renderSummary() {
    const c = report.coverage, p = report.preview, r = report.region;
    status.textContent = `${report.tensor} • ${c.visited_values.toLocaleString()} / ${c.total_tensor_values.toLocaleString()} tensor values (${fmt(100*c.tensor_fraction)}%) • rows [${r.row}, ${r.row+r.rows}), columns [${r.col}, ${r.col+r.cols}) • ${c.full_tensor ? 'full tensor' : 'partial native window'}; no full-model claim • seed ${report.control.seed}`;
    if (view.value !== 'svd_summary') { body.append(el('p', 'This report contains a spectrum and residual preview. Choose the summary view, or recompute regional analysis for this view.')); return; }
    const a = report.results.original, b = report.results.shuffled;
    body.append(el('p', `Uncentered SVD fitted independently on both complete selected windows. Full-window rank-one residual energy fraction: original ${fmt(a.rank_one_residual_energy_fraction)}, shuffled ${fmt(b.rank_one_residual_energy_fraction)}. Zero energy gives unavailable fractions.`));
    const series = (result, key) => result[key].map((value, index) => ({index, value}));
    pairBars('Complete selected-window singular spectrum', series(a,'singular_values'), series(b,'singular_values'), null);
    pairBars('Complete selected-window singular energy fractions', series(a,'energy_fractions'), series(b,'energy_fractions'), null);
    body.append(el('p', `Residual preview only: top-left ${p.shape[0]} × ${p.shape[1]} positions (${p.displayed_values} / ${c.visited_values}; ${fmt(100*p.displayed_values/c.visited_values)}%). ${p.omitted_values} residual positions omitted per side; preview does not describe the omitted residuals.`));
    const residual = result => result.rank_one_residual_preview.map((value, i) => ({index:p.positions[i],value}));
    pairBars('Rank-one residual preview (native window positions)', residual(a), residual(b), null, true);
    const details = el('details'); details.append(el('summary', 'All displayed residual positions and shuffled source coordinates'));
    const list = el('ol');
    p.positions.forEach((position, i) => {
      const source = report.control.preview_position_to_source[i];
      const native = index => JSON.stringify(report.shape?.length === 1 ? [r.col+index%r.cols] : [r.row+Math.floor(index/r.cols),r.col+index%r.cols]);
      list.append(el('li', `window [${Math.floor(position/r.cols)}, ${position%r.cols}], native ${native(position)}: original ${fmt(a.rank_one_residual_preview[i])}; shuffled ${fmt(b.rank_one_residual_preview[i])} → source ${native(source)}`));
    });
    details.append(list); body.append(details);
    body.append(el('p', `Complete permutation SHA-256: ${report.control.permutation_sha256}; ${report.control.digest_encoding}. ${report.control.omitted_mapping_entries} mapping entries omitted. Schema ${report.schema}; algorithm ${report.algorithm}; source ${report.source_identity}; revision binding ${report.model_identity}; cache key ${report.cache_key}.`));
  }

  async function reload(withSvd = false) {
    const value = Number(seed.value);
    if (!Number.isInteger(value) || value < 0 || value > 4294967295) { status.textContent = 'Seed must be an integer from 0 to 4294967295.'; return; }
    const requestEpoch = ++epoch;
    report = null; body.replaceChildren(); status.textContent = 'Reading bounded native region…';
    try {
      const next = await load(withSvd === 'svd_summary' ? {seed:value, scope:'svd_summary'} : {seed: value, svd: withSvd});
      if (disposed || requestEpoch !== epoch) return;
      if (next.schema !== (withSvd === 'svd_summary' ? 'weight-atlas.svd-window-summary.v2' : 'weight-atlas.analytics.v1')) throw new Error('Unsupported analytics schema');
      report = next; if (withSvd === 'svd_summary') view.value = 'svd_summary'; render();
    } catch (error) { if (!disposed && requestEpoch === epoch) status.textContent = `Analytics unavailable: ${error.message}`; }
  }
  view.onchange = render; sort.onchange = render; refresh.onclick = () => reload(false); svdButton.onclick = () => reload(true); summaryButton.onclick = () => reload('svd_summary');
  render();
  return {reload, setReport(next) { ++epoch; report = next; if (!next) status.textContent = ''; render(); }, destroy() { disposed = true; ++epoch; panel.remove(); }};
}

// Separate renderer accepts exact model-wide coverage; never fills absent coordinates.
export function renderModelOutliers(root, report) {
  root.replaceChildren();
  const doc = root.ownerDocument, c = report.coverage;
  const add = (tag, text) => { const node = doc.createElement(tag); node.textContent = text; root.append(node); return node; };
  add('h3', `${c.full_model ? 'Full-model' : 'Partial-model'} outlier ranking`);
  add('p', `${c.visited_values} / ${c.total_values} values; ${c.visited_tensors} / ${c.total_tensors} tensors. ${c.selection}. ${report.warning}`);
  add('p', `Paired control: ${report.control.kind}; seed ${report.control.seed}; ${report.control.statistics}. A shuffled ranking alone does not establish pattern meaning.`);
  for (const [side, rankings] of Object.entries(report.rankings)) {
    add('h3', side === 'original' ? 'Original source' : 'Shuffled positions');
    for (const [kind, items] of Object.entries(rankings)) {
      add('h4', kind); const list = add('ol', '');
      for (const item of items) {
        const li = doc.createElement('li');
        const coordinate = item.native_indices ?? item.control_position ?? item.index;
        const source = item.source_native_indices ? ` → source ${JSON.stringify(item.source_native_indices)}` : '';
        li.textContent = `${item.tensor}: ${JSON.stringify(coordinate)}${source} — ${item.abs ?? item.mean_abs} (${item.score_scope}${item.full_axis == null ? '' : `; ${item.full_axis ? 'full axis' : 'partial axis'}`})`;
        list.append(li);
      }
    }
  }
  const details = add('details', ''), summary = doc.createElement('summary'); summary.textContent = 'Exact tensor coverage and exclusions'; details.append(summary);
  for (const item of c.tensors) { const p = doc.createElement('p'); p.textContent = `${item.tensor}: ${item.visited_values}/${item.total_values}; ${item.excluded_reason ?? 'full tensor'}; region ${JSON.stringify(item.region)}`; details.append(p); }
}
