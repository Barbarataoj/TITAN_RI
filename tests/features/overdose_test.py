from typing import Dict, ClassVar

from . import base_feature
from .. import agent
from .. import population
from .. import model
from ..parse_params import ObjMap


class Overdose(base_feature.BaseFeature):
    """
    Overdose feature.

    - Applies only to OUD agents (drug_type in {Inj, NonInj}).
    - Simulates non-fatal and fatal overdoses.
    - Fatal overdoses trigger death via exit_enter.
    """

    name = "overdose"
    # Stats we will contribute to:
    #   * overdose_events: number of overdoses (fatal + non-fatal) this step
    #   * overdose_deaths: number of fatal overdoses this step
    stats = ["overdose_events", "overdose_deaths"]

    # Class-level cumulative counts by race/sex_type (optional)
    event_counts: ClassVar[Dict] = {}
    death_counts: ClassVar[Dict] = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)

        # Agent-level counters
        self.overdose_count = 0  # cumulative overdoses over the agent's life
        self.events_this_step = 0
        self.deaths_this_step = 0

    # ================= CLASS-LEVEL METHODS =================

    @classmethod
    def init_class(cls, params: "ObjMap"):
        """
        Initialize the counts dictionaries for the races and sex_types in the model.

        args:
            params: the population params
        """
        cls.event_counts = {
            race: {sex_type: 0 for sex_type in params.classes.sex_types}
            for race in params.classes.races
        }
        cls.death_counts = {
            race: {sex_type: 0 for sex_type in params.classes.sex_types}
            for race in params.classes.races
        }

    # ================= AGENT INITIALIZATION =================

    def init_agent(self, pop: "population.Population", time: int):
        """
        Initialize agent-level overdose tracking variables.
        """
        self.overdose_count = 0
        self.events_this_step = 0
        self.deaths_this_step = 0

    # ================= PER-TIME-STEP UPDATE =================

    def update_agent(self, model: "model.TITAN"):
        """
        Simulate overdose events for this agent at the current time step.

        args:
            model: the instance of TITAN currently being run
        """
        # Reset per-step counters for this agent
        self.events_this_step = 0
        self.deaths_this_step = 0

        # Only agents with OUD (Inj / NonInj) are eligible
        if getattr(self.agent, "drug_type", None) not in ("Inj", "NonInj"):
            return

        overdose_params = getattr(model.params, "overdose", None)
        if overdose_params is None:
            return

        # Base risk by drug_type
        if self.agent.drug_type == "Inj":
            base_prob = getattr(overdose_params, "base_prob_inj", 0.0)
        else:
            base_prob = getattr(overdose_params, "base_prob_noninj", 0.0)

        fatal_frac = getattr(overdose_params, "fatal_frac", 0.0)
        high_risk_mult = getattr(overdose_params, "high_risk_mult", 1.0)
        ssp_rr = getattr(overdose_params, "ssp_rr", 1.0)

        # Start from base probability
        prob = base_prob

        # MOUD effect: ask the MOUD feature for an overdose risk multiplier
        moud_feat = getattr(self.agent, "moud", None)
        if moud_feat is not None and hasattr(moud_feat, "get_overdose_risk_multiplier"):
            prob *= moud_feat.get_overdose_risk_multiplier()

        # High-risk multiplier
        if hasattr(self.agent, "high_risk") and self.agent.high_risk:  # type: ignore[attr-defined]
            prob *= high_risk_mult

        # Syringe services (if you believe SSP reduces overdose risk, e.g. via naloxone)
        if hasattr(self.agent, "syringe_services") and self.agent.syringe_services.active:  # type: ignore[attr-defined]
            prob *= ssp_rr

        # Clamp probability into [0, 1]
        prob = max(0.0, min(1.0, prob))

        # Decide whether an overdose occurs this step
        if model.run_random.random() >= prob:
            return

        # Overdose occurred
        self.events_this_step += 1
        self.overdose_count += 1
        self.event_counts[self.agent.race][self.agent.sex_type] += 1

        # Decide if fatal
        if model.run_random.random() < fatal_frac:
            self.deaths_this_step += 1
            self.death_counts[self.agent.race][self.agent.sex_type] += 1

            # Mark this agent for death via exit_enter.
            # NOTE: adjust field names here to match your actual exit_enter implementation.
            self.agent.exit_class = "death"
            # Optional: label reason for later analysis
            self.agent.death_reason = "overdose"

    # ================= STATS =================

    def set_stats(self, stats: Dict[str, int], time: int):
        """
        Contribute overdose stats for this agent to the global stats dict.

        stats keys:
          * overdose_events
          * overdose_deaths
        """
        if self.events_this_step > 0:
            stats.setdefault("overdose_events", 0)
            stats["overdose_events"] += self.events_this_step

        if self.deaths_this_step > 0:
            stats.setdefault("overdose_deaths", 0)
            stats["overdose_deaths"] += self.deaths_this_step

        # Reset per-step counters after stats collection
        self.events_this_step = 0
        self.deaths_this_step = 0
