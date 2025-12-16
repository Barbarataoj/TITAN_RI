from typing import Dict, ClassVar

from . import base_feature
from .. import agent
from .. import population
from .. import model
from ..parse_params import ObjMap


class MOUD(base_feature.BaseFeature):
    """
    Medication for Opioid Use Disorder (MOUD) feature.

    Represents pharmacologic treatment for OUD (e.g., buprenorphine, methadone).
    Agents can initiate, discontinue, and re-initiate MOUD over time.
    """

    name = "moud"
    stats = ["moud"]
    """
        MOUD collects the following stats:

        * moud - number of agents currently on MOUD
    """

    counts: ClassVar[Dict] = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)

        self.active = False  # currently on MOUD
        self.ever = False    # ever been on MOUD

    # ================= CLASS-LEVEL METHODS =================

    @classmethod
    def init_class(cls, params: "ObjMap"):
        """
        Initialize the counts dictionary for the races and sex_types in the model.

        args:
            params: the population params
        """
        cls.counts = {
            race: {sex_type: 0 for sex_type in params.classes.sex_types}
            for race in params.classes.races
        }

    @classmethod
    def add_agent(cls, agent: "agent.Agent"):
        """
        Increment MOUD counts for the given agent's race and sex_type.
        """
        cls.counts[agent.race][agent.sex_type] += 1

    @classmethod
    def remove_agent(cls, agent: "agent.Agent"):
        """
        Decrement MOUD counts for the given agent's race and sex_type.
        """
        cls.counts[agent.race][agent.sex_type] -= 1

    # ================= AGENT INITIALIZATION =================

    def init_agent(self, pop: "population.Population", time: int):
        """
        Initialize MOUD status during population creation.

        Only OUD agents (drug_type in {Inj, NonInj}) are eligible. They are
        placed on MOUD at baseline with probability `moud.init` from the
        demographic params.

        args:
            pop: the population this agent is a part of
            time: the current time step
        """
        if not self._is_oud():
            return

        moud_params = (
            self.agent.location.params.demographics[self.agent.race]
            .sex_type[self.agent.sex_type]
            .drug_type[self.agent.drug_type]
            .moud
        )

        if pop.pop_random.random() < moud_params.init:
            self.initiate(pop.pop_random, moud_params, "init")

    # ================= PER-TIME-STEP UPDATE =================

    def update_agent(self, model: "model.TITAN"):
        """
        Update MOUD status for this agent at each time step.

        MOUD is available only after `model.params.moud.start_time`.
        Agents must have OUD (drug_type in {Inj, NonInj}).

        args:
            model: the TITAN model instance
        """
        if not self._is_oud():
            # If agent is no longer OUD, turn off MOUD and adjust counts
            if self.active:
                self.active = False
                self.remove_agent(self.agent)
            return

        moud_global = getattr(model.params, "moud", None)
        if moud_global is not None:
            start_time = getattr(moud_global, "start_time", 0)
            use_reinit = getattr(moud_global, "use_reinit", True)
        else:
            # Sensible defaults if not specified
            start_time = 0
            use_reinit = True

        # Before MOUD becomes available
        if model.time < start_time:
            return

        moud_params = (
            self.agent.location.params.demographics[self.agent.race]
            .sex_type[self.agent.sex_type]
            .drug_type[self.agent.drug_type]
            .moud
        )

        # Go on (or back on) MOUD
        if not self.active:
            # Re-init behavior if configured globally
            if self.ever and use_reinit:
                if model.run_random.random() < moud_params.reinit.prob:
                    self.initiate(model.run_random, moud_params, "prob")
            else:
                # New initiation based on time-varying enrollment prob
                enroll_prob = self._get_enroll_prob(model.time, moud_params)
                if model.run_random.random() < enroll_prob:
                    self.initiate(model.run_random, moud_params, "prob")

        # Already on MOUD: possible discontinuation
        else:
            if model.run_random.random() < moud_params.discontinue:
                self.active = False
                self.remove_agent(self.agent)

    # ================= STATS & SCALING =================

    def set_stats(self, stats: Dict[str, int], time: int):
        """
        Add MOUD counts to the global stats accumulator.
        """
        if self.active:
            stats["moud"] += 1

    def get_overdose_risk_multiplier(self) -> float:
        """
        Multiplier for overdose risk when on MOUD.

        This allows the Overdose feature to query the MOUD effect.
        By default, it delegates to params.moud.overdose_scale.active/inactive.

        If you haven't set those yet, you can just return a fixed value
        (e.g., 0.4 when active, 1.0 when not active).
        """
        params = self.agent.location.params
        # If you add a calibration block for MOUD, you can use that here.
        moud_params = getattr(params, "moud", None)
        if moud_params is not None:
            if self.active:
                return moud_params.overdose_scale.active
            else:
                return moud_params.overdose_scale.inactive
        # Default: 60% reduction in overdose risk when on MOUD
        return 0.4 if self.active else 1.0

    def get_injection_risk_multiplier(self) -> float:
        """
        Return a multiplier for injection-related HIV/HCV risk.

        Example (from YAML):
        moud.injection_scale.active   ~ 0.7 (30% reduction)
        moud.injection_scale.inactive ~ 1.0
        """
        params = self.agent.location.params
        moud_global = getattr(params, "moud", None)

        if moud_global is not None:
            if self.active:
                return moud_global.injection_scale.active
            else:
                return moud_global.injection_scale.inactive

        # Fallback: 30% reduction if active, no change if not
        return 0.7 if self.active else 1.0

    

    # ================= HELPER METHODS =================

    def _is_oud(self) -> bool:
        """
        Define who is considered to have OUD: Inj or NonInj drug_type.
        """
        return getattr(self.agent, "drug_type", None) in ("Inj", "NonInj")

    def _get_enroll_prob(self, time: int, moud_params: ObjMap) -> float:
        """
        Time-varying enrollment probability based on `moud.enroll` windows.

        moud:
          enroll:
            1:
              prob: ...
              start: ...
              stop: ...
            2:
              prob: ...
              start: ...
              stop: ...
        """
        enroll_prob = 0.0
        for window in moud_params.enroll.values():
            if window.start <= time < window.stop:
                enroll_prob = window.prob
                break

        return max(0.0, min(1.0, enroll_prob))

    def initiate(self, rand_gen, moud_params: ObjMap, init_or_prob: str):
        """
        Initiate an agent on MOUD and update class-level counts.

        args:
            rand_gen: random number generator (pop_random or run_random)
            moud_params: demographic-specific MOUD params
            init_or_prob: "init" vs "prob" to index adherence, etc. if needed
        """
        # For now we don't distinguish 'adherent' vs 'non-adherent' MOUD;
        # you can extend this later if you want subtypes.
        self.active = True
        self.ever = True
        self.add_agent(self.agent)
