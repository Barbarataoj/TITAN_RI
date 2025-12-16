# features/oud_state.py

from . import base_feature
import math

class OUDState(base_feature.BaseFeature):
    """
    Dynamic OUD status and injection status transitions.

    - Uses agent.has_oud and agent.drug_type
    - Keeps Population.pwid_agents in sync with agent.drug_type == "Inj"
    - Exposes OUD status stats for reporting (oud_active, oud_remission, oud_none)
    """
    name = "oud_state"

    # These become columns in the basic report
    stats = ["oud_active", "oud_remission", "oud_none"]

    @classmethod
    def init_class(cls, params):
        # No class-level bookkeeping yet
        pass

    def __init__(self, agent):
        super().__init__(agent)
        # Always "active" so update_agent runs each timestep
        self.active = True

    def init_agent(self, pop, time):
        """
        Called once at agent creation.

        Baseline OUD status is set in Population.create_agent
        via agent.has_oud and agent.drug_type, so nothing else is needed here
        for now.
        """
        # If you ever want to track baseline OUD duration, you could initialize it here.
        pass

    def update_agent(self, model):
        """
        Per-timestep OUD dynamics:
        - Remission from active OUD
        - Injection <-> non-injection transitions among people with OUD

        Transition rates are taken from location.params.oud.transitions:
            remission_rate_per_year
            inject_start_rate_per_year
            inject_stop_rate_per_year
        """
        # If OUD transitions aren't configured for this location, do nothing.
        oud_params = getattr(self.agent.location.params, "oud", None)
        if oud_params is None or not hasattr(oud_params, "transitions"):
            return

        transitions = oud_params.transitions

        # Convert annual hazards to per-step probabilities
        steps_per_year = model.params.model.time.steps_per_year
        dt_years = 1.0 / steps_per_year

        def hazard_to_prob(hazard_per_year: float) -> float:
            if hazard_per_year <= 0.0:
                return 0.0
            return 1.0 - math.exp(-hazard_per_year * dt_years)

        p_remission = hazard_to_prob(transitions.remission_rate_per_year)
        p_inject_start = hazard_to_prob(transitions.inject_start_rate_per_year)
        p_inject_stop = hazard_to_prob(transitions.inject_stop_rate_per_year)

        # If agent does not currently have OUD, you may (optionally) model incident OUD here later.
        if not getattr(self.agent, "has_oud", False):
            return

        # 1. Remission from active OUD
        if model.run_random.random() < p_remission:
            self.agent.has_oud = False
            self.agent.is_in_remission = True
            # In remission, treat as non-injecting
            self.agent.drug_type = "NonInj"

            # If they were in PWID set, remove them
            if self.agent in model.pop.pwid_agents.members:
                model.pop.pwid_agents.remove_agent(self.agent)

            # Once remitted, skip injection transitions this step
            return

        # 2. Injecting vs non-injecting transitions among people with OUD
        if self.agent.drug_type != "Inj":
            # Non-injecting OUD → start injecting
            if model.run_random.random() < p_inject_start:
                self.agent.drug_type = "Inj"
                # Ensure they are in the PWID set used for partnering
                model.pop.pwid_agents.add_agent(self.agent)
        else:
            # Injecting OUD → stop injecting (still has OUD)
            if model.run_random.random() < p_inject_stop:
                self.agent.drug_type = "NonInj"
                # Remove from PWID set
                if self.agent in model.pop.pwid_agents.members:
                    model.pop.pwid_agents.remove_agent(self.agent)

    def set_stats(self, stats_dict: dict, time: int):
        """
        Add this agent to one of the OUD status counters for the basic report.
        Called once per agent per timestep.
        """
        has_oud = getattr(self.agent, "has_oud", False)
        in_remission = getattr(self.agent, "is_in_remission", False)

        if has_oud and not in_remission:
            stats_dict["oud_active"] += 1
        elif has_oud and in_remission:
            stats_dict["oud_remission"] += 1
        else:
            stats_dict["oud_none"] += 1
