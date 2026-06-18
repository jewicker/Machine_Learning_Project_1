import numpy as np
# Monkeypatch NumPy 2.0 compatibility for older libraries (like evidently)
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

from us_visa.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()