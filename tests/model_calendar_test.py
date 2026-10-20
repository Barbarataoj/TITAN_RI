"""analysis/model_calendar.py must agree with titan.calibration.lethality's convention (one owner, rule 6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import model_calendar as mc                      # noqa: E402
from titan.calibration import lethality          # noqa: E402


def test_agrees_with_lethality():
    assert mc.FIRST_YEAR == lethality.FIRST_YEAR and mc.STEPS_PER_YEAR == lethality.STEPS_PER_YEAR
    for y in (2012, 2022, 2023, 2027, 2036):
        assert mc.start_step(y) == lethality.start_step(y)


def test_windows():
    assert mc.year_of_step(1) == 2012 and mc.year_of_step(12) == 2012 and mc.year_of_step(13) == 2013
    assert mc.steps_of_year(2022) == (121, 132)
    assert mc.steps_of_year(2027) == (181, 192) and mc.steps_of_year(2036) == (289, 300)
    assert mc.last_full_year(300) == 2036 and mc.last_full_year(240) == 2031 and mc.last_full_year(241) == 2031
