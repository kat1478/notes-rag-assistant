"""Judge cache keys: a corrected dataset must not reuse grades given against other key points."""
from app import run_judge


def test_cache_key_depends_on_expected_key_points():
    args = ("openrouter:x/y", "val016", "Adam Mickiewicz wrote it [Source 2].")
    v2 = run_judge.cache_key(*args, ["Adam Mickiewicz wrote Pan Tadeusz", "lies outside the scope"])
    v21 = run_judge.cache_key(*args, ["Adam Mickiewicz wrote Pan Tadeusz"])
    assert v2 != v21
    assert v21 == run_judge.cache_key(*args, ["Adam Mickiewicz wrote Pan Tadeusz"])  # stable across runs
