from typing import List

from . import base_feature
from .. import agent
from .. import population
from .. import model


class Insurance(base_feature.BaseFeature):
    """
    Insurance status as agent state (2026-10-24; work record §19.34).

    The claims database the care layer is fitted to sees only insured people. With this
    feature on, a share of agents is uninsured: they visit care less (the care_seeking
    setting's `visit_mult_uninsured`), accept MOUD and PrEP offers less often
    (`insurance.moud_acceptance_mult`, `insurance.prep_acceptance_mult`), and their
    activity is not billed to a payer in the cash flow. Their OUD, injecting, overdose and
    HIV dynamics are unchanged.

    The feature publishes two attributes on every agent, kept in step by `set_status`:
    `agent.insured` (bool; True by default, also when the feature is off, so the hooks in the
    other features are no-ops unless an agent is uninsured) and `agent.insured_state`
    ("insured" / "uninsured"), the value the reports stratify on when `insured_states` is in
    `outputs.classes`. That stratification is how the scorer and the cash flow read the insured
    agents' counts: there are no insured-only copies of other features' stats to keep in step.

    Churn runs in `update_pop`, so a change of status in a month precedes that month's visits
    (CareSeeking rolls them in its own `update_pop`, which runs after this one because this
    feature is registered first in `titan/features/__init__.py`).
    """

    name = "insurance"
    stats: List[str] = []      # reported through the insured_states output class instead

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        self.set_status(True)   # everyone insured unless the feature decides otherwise

    def set_status(self, insured: bool):
        """Set both published attributes together."""
        self.agent.insured = bool(insured)
        self.agent.insured_state = "insured" if insured else "uninsured"

    def init_agent(self, pop: "population.Population", time: int):
        """Draw insurance status at initialisation (and at entry): uninsured with the drug-type share."""
        u = self.uninsured_share(self.agent, self.agent.location.params.insurance)
        self.set_status(not (u > 0.0 and pop.pop_random.random() < u))
        self.active = True

    @staticmethod
    def uninsured_share(agent: "agent.Agent", params) -> float:
        """The stationary uninsured share for this agent's drug type (the entry probability)."""
        if getattr(agent, "drug_type", "NonInj") == "Inj":
            return float(getattr(params, "init_uninsured_prob_inj", 0.0))
        return float(getattr(params, "init_uninsured_prob_noninj", 0.0))

    @staticmethod
    def lose_prob(gain: float, u: float) -> float:
        """Monthly loss probability that keeps the uninsured share `u` stationary under gain `gain`.

        Derived, not stored (one owner): with g the gain probability, the share is stationary when
        (1 - u) lose = u g, i.e. lose = g u / (1 - u).
        """
        if u <= 0.0 or gain <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return min(1.0, gain * u / (1.0 - u))

    @classmethod
    def update_pop(cls, model: "model.TITAN"):
        """Monthly churn for every agent, before the care layer rolls the month's visits."""
        params = model.params.insurance
        gain = float(getattr(params, "gain_prob_per_month", 0.0))
        for a in model.pop.all_agents:
            a.insurance.churn(model, params, gain)

    def churn(self, model: "model.TITAN", params, gain: float):
        """One agent's monthly draw: the insured may lose coverage, the uninsured may regain it."""
        if self.agent.insured:
            lose = self.lose_prob(gain, self.uninsured_share(self.agent, params))
            if lose > 0.0 and model.run_random.random() < lose:
                self.set_status(False)
        elif gain > 0.0 and model.run_random.random() < gain:
            self.set_status(True)

    @staticmethod
    def acceptance_mult(agent: "agent.Agent", params, service: str) -> float:
        """Multiplier the other features apply to an offer's acceptance: 1.0 for insured agents or when the feature is off."""
        if getattr(agent, "insured", True) or not getattr(params.features, "insurance", False):
            return 1.0
        return float(getattr(params.insurance, f"{service}_acceptance_mult", 1.0))
