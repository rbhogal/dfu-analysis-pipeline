## How each teammate uses it:

Teammate A imports like this at the top of their module during development:

```python
from mock_data import get_mock_pred_mask, get_mock_area, get_mock_iou, get_mock_mean_brightness

# test WoundAreaAnalyzer
mask = get_mock_pred_mask()
contour = get_mock_contour()
analyzer = WoundAreaAnalyzer()
print(analyzer.calc_area(mask))           # should print ~7285
print(analyzer.classify_iou_performance(0.54))  # should print "partial"

```

Teammate B imports like this:

```python
from mock_data import get_mock_pred_mask, get_mock_contour, get_mock_dataframe

# test MorphologyAnalyzer
contour = get_mock_contour()
mask = get_mock_pred_mask()
morphology = MorphologyAnalyzer()
print(morphology.calc_metrics(contour, mask))

# test visualization
df = get_mock_dataframe()
plot_iou_histogram(df)
```

Three things to note:

`get_mock_gt_mask()` uses pixel value `1` not `255` — this matches the real FUSeg format and is important for testing compute_iou correctly.

`get_mock_pred_mask()` uses `255` — this matches what your OpenCV segmentation actually produces after thresholding.

`get_mock_dataframe()` includes one failed row (`mock_005`) on purpose — your teammates need to test that their functions handle `None` values correctly without crashing, which is exactly what will happen with real failures in the pipeline.
