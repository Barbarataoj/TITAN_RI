import math
from typing import Dict, List

from . import base_feature
from .. import agent
from .. import population
from .. import model

COMPONENTS = ("housing", "food", "employment")      # unstable housing, food insecurity, unemployment
RISK_HOOKS = ("sharing", "condomless_sex", "inject_start", "overdose")


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class SDH(base_feature.BaseFeature):
    """
    Social determinants of health as agent state (2026-10-24; work record §19.39; proposal Table 2).

    Hardship index: `agent.hardship` in 0..3 counts unstable housing, food insecurity and unemployment.
    The three are drawn from one latent propensity per agent (Gaussian copula: z_k = sqrt(rho) z0 +
    sqrt(1 - rho) eps_k, component present when Phi(z_k) < its marginal), so they correlate as the
    surveys show while each marginal is exact. An episode ends with probability
    `episode_end_prob_per_month`, when its eps_k is re-drawn: states persist, marginals stay stationary,
    the correlation persists through z0. Sex work (`agent.sex_work`) is a fourth component on the same
    latent with its own marginal by sex type and injecting status.

    What depends on it -- and nothing else does: the care setting's visit rate
    (`visit_mult_hardship_per_point` ^ k, care_seeking), MOUD and PrEP discontinuation and the ART
    adherence transitions (`mult_per_point`, read through `SDH.mult`), the PrEP indication (`sex_work`
    in prep.target_model) and the sexual partner number of agents in sex work (once, at
    initialisation). Transmission, overdose and OUD dynamics are untouched: they are calibrated, and the
    proposal asks for the determinants' effect on implementation.

    With every marginal at 0 (the defaults) no random number is drawn, so a run with the feature on and
    nobody in hardship is identical to one with it off.
    """

    name = "sdh"
    stats: List[str] = ["hardship_0", "hardship_1", "hardship_2", "hardship_3", "sex_work"]
    # population mean of m^k per risk hook, recomputed each month in update_pop (mean-preserving risk multipliers)
    norm: Dict[str, float] = {h: 1.0 for h in RISK_HOOKS}

    @classmethod
    def init_class(cls, params):
        cls.norm = {h: 1.0 for h in RISK_HOOKS}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        self.z0 = 0.0
        self.eps: Dict[str, float] = {}
        self.components: Dict[str, bool] = {c: False for c in COMPONENTS}
        self.agent.hardship = 0
        self.agent.sex_work = False

    @staticmethod
    def marginals(agent: "agent.Agent", params) -> Dict[str, float]:
        """The four marginals for this agent's drug type and sex type."""
        h = params.sdh.hardship
        block = h.prob_inj if getattr(agent, "drug_type", "NonInj") == "Inj" else h.prob_noninj
        out = {c: float(getattr(block, c, 0.0) or 0.0) for c in COMPONENTS}
        sw = params.sdh.sex_work.prob[agent.sex_type]
        out["sex_work"] = float(getattr(sw, "inj" if getattr(agent, "drug_type", "NonInj") == "Inj" else "noninj", 0.0) or 0.0)
        return out

    def init_agent(self, pop: "population.Population", time: int):
        params = self.agent.location.params
        p = self.marginals(self.agent, params)
        self.active = True
        if max(p.values()) <= 0.0:
            return                              # nothing to draw: identical to the feature being off
        rho = float(params.sdh.hardship.rho)
        self.z0 = pop.pop_random.gauss(0.0, 1.0)
        for k in (*COMPONENTS, "sex_work"):
            self.eps[k] = pop.pop_random.gauss(0.0, 1.0)
        self._apply(p, rho)
        self._scale_partners(params)

    def _latent(self, k: str, rho: float) -> float:
        return math.sqrt(rho) * self.z0 + math.sqrt(1.0 - rho) * self.eps[k]

    def _apply(self, p: Dict[str, float], rho: float):
        for c in COMPONENTS:
            self.components[c] = _phi(self._latent(c, rho)) < p[c]
        self.agent.hardship = sum(1 for v in self.components.values() if v)
        self.agent.sex_work = _phi(self._latent("sex_work", rho)) < p["sex_work"]

    def _scale_partners(self, params):
        """Agents in sex work carry more sexual partners (once, at initialisation; the rest of the run keeps it)."""
        if not self.agent.sex_work:
            return
        m = float(params.sdh.sex_work.partner_mult)
        if m == 1.0:
            return
        for bond, bond_def in params.classes.bond_types.items():
            if "sex" in bond_def.acts_allowed:
                n = self.agent.mean_num_partners.get(bond, 0)
                if n > 0:
                    self.agent.mean_num_partners[bond] = int(math.ceil(n * m))
                    self.agent.target_partners[bond] = self.agent.mean_num_partners[bond]

    @classmethod
    def update_pop(cls, model: "model.TITAN"):
        """Monthly: episodes end and are re-drawn, for every agent, before the care layer rolls the month's visits;
        then the mean-preserving normalisers of the risk multipliers are refreshed for the month."""
        params = model.params
        q = float(params.sdh.hardship.episode_end_prob_per_month)
        rho = float(params.sdh.hardship.rho)
        if q > 0.0:
            for a in model.pop.all_agents:
                a.sdh.churn(model, params, q, rho)
        cls.refresh_norms(model.pop.all_agents, params)

    @classmethod
    def refresh_norms(cls, agents, params):
        """Population mean of m^hardship per risk hook (1.0 when a hook's multiplier is 1 or nobody has hardship)."""
        mults = getattr(params.sdh.hardship, "risk_mult_per_point", None)
        n = 0
        sums = {h: 0.0 for h in RISK_HOOKS}
        ms = {h: float(getattr(mults, h, 1.0)) for h in RISK_HOOKS}
        for a in agents:
            k = int(getattr(a, "hardship", 0) or 0)
            n += 1
            for h in RISK_HOOKS:
                sums[h] += ms[h] ** k
        cls.norm = {h: (sums[h] / n if n else 1.0) for h in RISK_HOOKS}

    @classmethod
    def risk_mult(cls, agent: "agent.Agent", params, hook: str) -> float:
        """Relative multiplier on a risk behaviour for `hook` in RISK_HOOKS: m^hardship over the population
        mean of m^hardship (mean-preserving, `sdh.hardship.mean_preserving`), 1.0 when the feature is off."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        mults = getattr(getattr(getattr(params, "sdh", None), "hardship", None), "risk_mult_per_point", None)
        m = float(getattr(mults, hook, 1.0) or 1.0) if mults is not None else 1.0
        if m == 1.0:
            return 1.0
        k = int(getattr(agent, "hardship", 0) or 0)
        raw = m ** k
        if getattr(params.sdh.hardship, "mean_preserving", True):
            return raw / cls.norm.get(hook, 1.0)
        return raw

    def churn(self, model: "model.TITAN", params, q: float, rho: float):
        if not self.eps:
            return                              # never drawn (every marginal 0 at entry)
        p = self.marginals(self.agent, params)
        changed = False
        for k in self.eps:
            if model.run_random.random() < q:
                self.eps[k] = model.run_random.gauss(0.0, 1.0)
                changed = True
        if changed:
            self._apply(p, rho)

    @staticmethod
    def mult(agent: "agent.Agent", params, hook: str) -> float:
        """Multiplier the other features apply for `hook` (moud_discontinue, prep_discontinue, art_nonadherence,
        art_readherence): m ^ hardship, 1.0 when the feature is off or the agent has no hardship."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        k = int(getattr(agent, "hardship", 0) or 0)
        if k <= 0:
            return 1.0
        return float(getattr(params.sdh.hardship.mult_per_point, hook, 1.0)) ** k

    def set_stats(self, stats: Dict[str, int], time: int):
        stats[f"hardship_{min(int(self.agent.hardship), 3)}"] += 1
        if self.agent.sex_work:
            stats["sex_work"] += 1
