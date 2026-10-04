# Original region exports

Select an explicit tensor slice and inclusive native bounds under **Save, annotate & export a region**. Choose Raw CSV or Numeric NumPy + metadata, then click **Export original values**. Each export reads at most 256 native cells, has a 30-second deadline, and checks exact source/revision/tensor/slice identity before, during and after reading. Cancellation or navigation produces no file. Nonfinite values, mismatched decimals/bytes, unsupported storage and changed sources are refused.

CSV supports original BF16, F16 and F32 values. The first row contains JSON metadata, including original tensor shape, selected leading indices, region dimensions and source identity. Each remaining data row retains native indices, exact decimal and original little-endian bytes. Numeric values keep signed zero. No normalization or color transform is applied.

NumPy export downloads two files from the same checked snapshot: `weight-atlas-region.npy` and `weight-atlas-region.metadata.json`. Keep both. Browsers may require permission for the second download. The metadata includes the original source bytes, native region, leading slice indices and source/revision binding. The array alone cannot identify its model or source region.

The array is NPY v1.0, C order, two-dimensional with shape `(selected_rows, selected_columns)`. Vector selections therefore remain `(1, columns)`; higher-rank tensors require explicit leading indices. F16 uses `<f2` and F32 uses `<f4`, retaining the original stored bits. BF16 uses `<f4` with an exact widening, including signed zero. BF16's original two-byte encoding remains in the metadata. No object array or pickle is produced.

```python
import json
import numpy as np

values = np.load('weight-atlas-region.npy', allow_pickle=False)
with open('weight-atlas-region.metadata.json', encoding='utf-8') as file:
    metadata = json.load(file)
assert list(values.shape) == metadata['numpy']['shape']
assert metadata['values'] == 'original, untransformed'
```

Each downloaded file is capped at 256 KiB. Existing NumPy is needed only to consume the file, not to export it. The committed independent numerical fixture checks CSV and NumPy bit roundtrips for all three dtypes; browser workflow qualification is separate.
