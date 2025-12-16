from typing import Dict, ClassVar

from . import base_feature
from .. import agent
from .. import population
from .. import model
from ..parse_params import ObjMap


class HIVTesting(base_feature.BaseFeature):
    """
    HIV testing feature.

    - Applies HIV testing to agents at each time step.
    - If an infected, undiagnosed agent is tested, they are diagnosed via
      HIV.diagnose(), which sets hiv.dx, hiv.dx_time, and updates dx_counts.
    - HAART then uses hiv.dx and hiv.dx_time to enroll agents on treatment.
    """

    name = "hiv_testing"
    # We only add one new stat: total number of HIV tests performed.
    # New HIV diagnoses are already tracked by HIV.set_stats via "hiv_dx_new".
    stats = ["hiv_tests"]

    # Optional class-level counts (by race/sex_type) for total tests
    test_counts: ClassVar[Dict] = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        # Per-step counter (reset in set_stats)
        self.tests_this_step = 0

    # ================= CLASS-LEVEL METHODS =================

    @classmethod
    def init_class(cls, params: "ObjMap"):
        """
        Initialize the counts dictionaries for the races and sex_types in the model.

        args:
            params: the population params
        """
        cls.test_counts = {
            race: {sex_type: 0 for sex_type in params.classes.sex_types}
            for race in params.classes.races
        }

    # ================= AGENT INITIALIZATION =================

    def init_agent(self, pop: "population.Population", time: int):
        """
        Initialize the agent for this feature.

        HIV exposure already initializes:
          - hiv.active
          - hiv.dx
          - hiv.dx_time

        We do not change those here; we only drive *new* diagnoses over time.
        """
        self.tests_this_step = 0

    # ================= PER-TIME-STEP UPDATE =================

    def update_agent(self, model: "model.TITAN"):
        """
        Apply HIV testing to this agent at the current time step.

        args:
            model: the instance of TITAN currently being run
        """
        # Reset per-step counter for this agent
        self.tests_this_step = 0

        # Agent must have the HIV exposure feature
        if not hasattr(self.agent, "hiv"):
            return

        hiv_feat = self.agent.hiv  # type: ignore[attr-defined]

        # Global testing params (top-level block: params.hiv_testing)
        testing_params = getattr(model.params, "hiv_testing", None)
        if testing_params is None:
            return

        # Respect HIV start_time for testing as well (optional but consistent)
        if model.time < model.params.hiv.start_time:
            return

        base_prob = getattr(testing_params, "base_prob", 0.0)
        high_risk_mult = getattr(testing_params, "high_risk_mult", 1.0)
        oud_mult = getattr(testing_params, "oud_mult", 1.0)
        interval = getattr(testing_params, "interval", 1)

        # Only test on specific time steps if interval > 1
        if interval > 1 and (model.time % interval != 0):
            return

        # ---- Compute testing probability for this agent ----
        prob = base_prob

        # Increase for high_risk agents (assuming high_risk is a feature with .active)
        high_risk_feat = getattr(self.agent, "high_risk", None)
        if high_risk_feat is not None and getattr(high_risk_feat, "active", False):
            prob *= high_risk_mult

        # Increase for OUD agents (Inj / NonInj)
        if getattr(self.agent, "drug_type", None) in ("Inj", "NonInj"):
            prob *= oud_mult

        # Clamp to [0, 1]
        prob = max(0.0, min(1.0, prob))

        # Decide if a test occurs
        if model.run_random.random() >= prob:
            return

        # A test happened
        self.tests_this_step += 1
        self.test_counts[self.agent.race][self.agent.sex_type] += 1

        # ---- Determine infection + diagnosis status ----

        # Infection flag in HIV exposure is `active`
        is_infected = bool(hiv_feat.active)

        # If not infected, nothing to diagnose (but test still counted)
        if not is_infected:
            return

        # Already diagnosed? Nothing more to do here; HIV.set_stats will handle hiv_dx_new
        if hiv_feat.dx:
            return

        # NEW diagnosis: use HIV.diagnose() so dx, dx_time, and dx_counts are consistent
        hiv_feat.diagnose(model)

        # We do NOT manually modify dx_counts or hiv_dx_new here; HIV.exposure
        # will update hiv_dx_new in HIV.set_stats() when dx_time == current time.

    # ================= STATS =================

    def set_stats(self, stats: Dict[str, int], time: int):
        """
        Contribute HIV testing stats for this agent to the global stats dict.

        stats keys:
          * hiv_tests
        """
        if self.tests_this_step > 0:
            stats.setdefault("hiv_tests", 0)
            stats["hiv_tests"] += self.tests_this_step

        # Reset per-step counter
        self.tests_this_step = 0
