def test_import_lightweight():
    import sys
    import cebmf  # noqa: F401
    assert 'rpy2' not in sys.modules, "rpy2 imported at package import time"
    assert 'torch' not in sys.modules, "torch imported at package import time"
    assert 'fancyimpute' not in sys.modules, "fancyimpute imported at package import time"
