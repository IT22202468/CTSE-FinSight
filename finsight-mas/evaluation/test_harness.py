# evaluation/test_harness.py
"""
Unified test harness — runs all 4 member test modules.
Run: pytest evaluation/ -v --cov=tools --cov-report=term-missing
"""
from evaluation.test_fetcher   import *  # noqa
from evaluation.test_sentiment import *  # noqa
from evaluation.test_correlator import * # noqa
from evaluation.test_briefing  import *  # noqa
