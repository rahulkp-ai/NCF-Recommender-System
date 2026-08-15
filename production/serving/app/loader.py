"""
production/serving/app/loader.py
Loads NCF weights and engines once at startup.
NOTE: sys.path and pickle shim are set up in main.py before this is called.
"""


def load_hybrid_engine():
    from production.recommenders.hybrid.engine import HybridEngine
    return HybridEngine.load()