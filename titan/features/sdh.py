import math
from typing import Dict, List, Optional, Tuple

from . import base_feature
from .. import agent
from .. import population
from .. import model

COMPONENTS = ("housing", "food", "employment")      # unstable housing, food insecurity, unemployment
RISK_HOOKS = ("sharing", "condomless_sex", "inject_start", "overdose")
CARE_HOOKS = ("moud_discontinue", "prep_discontinue", "art_nonadherence", "art_readherence")


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

    Every effect is a RELATIVE multiplier per index point, m^k, divided each month by the mean of m^k
    over the agents the hook acts on (`mean_preserving`): risk hooks by drug type (sharing, condomless
    sex, starting to inject, overdose); the visit multiplier over the INSURED agents of the drug type,
    per setting (the claims-solved rates describe insured members); MOUD and PrEP discontinuation over
    the agents on them; the ART transitions over the adherent / non-adherent agents on ART. The
    population average of each calibrated or anchored behaviour is therefore unchanged and the
    determinants decide who carries it. Sex work multiplies the sexual partner number while the state
    holds (stochastic rounding) and is a PrEP indication (`sex_work` in prep.target_model).

    With every marginal at 0 (the defaults) no random number is drawn and every multiplier is exactly
    1, so a run with the feature on and nobody affected is identical to one with it off.
    """

    name = "sdh"
    stats: List[str] = ["hardship_0", "hardship_1", "hardship_2", "hardship_3", "sex_work"]
    norm: Dict[Tuple, float] = {}       # (hook, group) -> mean of m^k over that group, refreshed monthly

    @classmethod
    def init_class(cls, params):
        cls.norm = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        self.z0 = 0.0
        self.eps: Dict[str, float] = {}
        self.components: Dict[str, bool] = {c: False for c in COMPONENTS}
        self.partner_base: Dict[str, int] = {}
        self.agent.hardship = 0
        self.agent.sex_work = False

    # ------------------------------------------------------------------ marginals and the latent
    @staticmethod
    def marginals(agent: "agent.Agent", params) -> Dict[str, float]:
        """The four marginals for this agent's current drug type and sex type."""
        h = params.sdh.hardship
        inj = getattr(agent, "drug_type", "NonInj") == "Inj"
        block = h.prob_inj if inj else h.prob_noninj
        out = {c: float(getattr(block, c, 0.0) or 0.0) for c in COMPONENTS}
        sw = params.sdh.sex_work.prob[agent.sex_type]
        out["sex_work"] = float(getattr(sw, "inj" if inj else "noninj", 0.0) or 0.0)
        return out

    @staticmethod
    def any_marginal(params) -> bool:
        """Whether any group has a non-zero marginal: the latent is drawn for everyone when so, because an
        agent's drug type can change and its marginals with it."""
        h = params.sdh.hardship
        vals = [float(getattr(h.prob_inj, c, 0.0) or 0.0) for c in COMPONENTS]
        vals += [float(getattr(h.prob_noninj, c, 0.0) or 0.0) for c in COMPONENTS]
        for st in params.sdh.sex_work.prob.values():
            vals += [float(getattr(st, "inj", 0.0) or 0.0), float(getattr(st, "noninj", 0.0) or 0.0)]
        return max(vals) > 0.0

    def init_agent(self, pop: "population.Population", time: int):
        params = self.agent.location.params
        self.active = True
        if not self.any_marginal(params):
            return                              # nothing to draw: identical to the feature being off
        rho = float(params.sdh.hardship.rho)
        self.z0 = pop.pop_random.gauss(0.0, 1.0)
        for k in (*COMPONENTS, "sex_work"):
            self.eps[k] = pop.pop_random.gauss(0.0, 1.0)
        self._apply(params, rho, pop.pop_random)

    def _latent(self, k: str, rho: float) -> float:
        return math.sqrt(rho) * self.z0 + math.sqrt(1.0 - rho) * self.eps[k]

    def _apply(self, params, rho: float, rng):
        p = self.marginals(self.agent, params)
        for c in COMPONENTS:
            self.components[c] = _phi(self._latent(c, rho)) < p[c]
        self.agent.hardship = sum(1 for v in self.components.values() if v)
        sex_work = _phi(self._latent("sex_work", rho)) < p["sex_work"]
        if sex_work != self.agent.sex_work:
            self.agent.sex_work = sex_work
            self._scale_partners(params, rng)

    def _scale_partners(self, params, rng):
        """Entering sex work multiplies the sexual partner number (stochastic rounding, so the realised
        multiplier equals the parameter on average); leaving it restores the base."""
        m = float(params.sdh.sex_work.partner_mult)
        if m == 1.0:
            return
        for bond, bond_def in params.classes.bond_types.items():
            if "sex" not in bond_def.acts_allowed:
                continue
            if self.agent.sex_work:
                base = int(self.agent.mean_num_partners.get(bond, 0))
                if base <= 0:
                    continue
                self.partner_base[bond] = base
                x = base * m
                n = int(math.floor(x)) + (1 if rng.random() < x - math.floor(x) else 0)
                self.agent.mean_num_partners[bond] = n
                self.agent.target_partners[bond] = n
            elif bond in self.partner_base:
                self.agent.mean_num_partners[bond] = self.partner_base.pop(bond)
                self.agent.target_partners[bond] = self.agent.mean_num_partners[bond]

    # ------------------------------------------------------------------ monthly step
    @classmethod
    def update_pop(cls, model: "model.TITAN"):
        """Monthly: episodes end and are re-drawn, for every agent, before the care layer rolls the month's
        visits; then the mean-preserving normalisers are refreshed for the month."""
        params = model.params
        q = float(params.sdh.hardship.episode_end_prob_per_month)
        rho = float(params.sdh.hardship.rho)
        if q > 0.0:
            for a in model.pop.all_agents:
                a.sdh.churn(model, params, q, rho)
        cls.refresh_norms(model.pop.all_agents, params)

    def churn(self, model: "model.TITAN", params, q: float, rho: float):
        if not self.eps:
            return                              # never drawn: every marginal was 0
        changed = False
        for k in self.eps:
            if model.run_random.random() < q:
                self.eps[k] = model.run_random.gauss(0.0, 1.0)
                changed = True
        if changed:
            self._apply(params, rho, model.run_random)

    @classmethod
    def refresh_norms(cls, agents, params):
        """Mean of m^hardship per hook over the agents the hook acts on (see the class docstring)."""
        h = params.sdh.hardship
        risk = {k: float(getattr(getattr(h, "risk_mult_per_point", None), k, 1.0) or 1.0) for k in RISK_HOOKS}
        care = {k: float(getattr(getattr(h, "mult_per_point", None), k, 1.0) or 1.0) for k in CARE_HOOKS}
        settings = getattr(getattr(params, "care_seeking", None), "setting", None) or {}
        visit = {s: float(getattr(st, "visit_mult_hardship_per_point", 1.0) or 1.0) for s, st in settings.items()}
        sums: Dict[Tuple, float] = {}
        cnt: Dict[Tuple, int] = {}

        def add(key, value):
            sums[key] = sums.get(key, 0.0) + value
            cnt[key] = cnt.get(key, 0) + 1

        for a in agents:
            k = int(getattr(a, "hardship", 0) or 0)
            dt = getattr(a, "drug_type", "NonInj")
            for hook, m in risk.items():
                if m != 1.0:
                    add((hook, dt), m ** k)
            if getattr(a, "insured", True):
                for s, m in visit.items():
                    if m != 1.0:
                        add(("visit", s, dt), m ** k)
            moud = getattr(a, "moud", None)
            if care["moud_discontinue"] != 1.0 and moud is not None and getattr(moud, "active", False):
                add(("moud_discontinue", None), care["moud_discontinue"] ** k)
            prep = getattr(a, "prep", None)
            if care["prep_discontinue"] != 1.0 and prep is not None and getattr(prep, "active", False):
                add(("prep_discontinue", None), care["prep_discontinue"] ** k)
            haart = getattr(a, "haart", None)
            if haart is not None and getattr(haart, "active", False):
                if getattr(haart, "adherent", False):
                    if care["art_nonadherence"] != 1.0:
                        add(("art_nonadherence", None), care["art_nonadherence"] ** k)
                elif care["art_readherence"] != 1.0:
                    add(("art_readherence", None), care["art_readherence"] ** k)
        cls.norm = {key: sums[key] / cnt[key] for key in sums if cnt[key]}

    # ------------------------------------------------------------------ hooks read by the other features
    @classmethod
    def _relative(cls, params, m: float, k: int, key: Tuple) -> float:
        if m == 1.0 or k <= 0 and not getattr(params.sdh.hardship, "mean_preserving", True):
            return 1.0
        raw = m ** k
        if getattr(params.sdh.hardship, "mean_preserving", True):
            return raw / cls.norm.get(key, 1.0)
        return raw

    @classmethod
    def risk_mult(cls, agent: "agent.Agent", params, hook: str) -> float:
        """Relative multiplier on a risk behaviour (`hook` in RISK_HOOKS), normalised within the agent's drug type."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        mults = getattr(getattr(getattr(params, "sdh", None), "hardship", None), "risk_mult_per_point", None)
        m = float(getattr(mults, hook, 1.0) or 1.0) if mults is not None else 1.0
        k = int(getattr(agent, "hardship", 0) or 0)
        return cls._relative(params, m, k, (hook, getattr(agent, "drug_type", "NonInj")))

    @classmethod
    def mult(cls, agent: "agent.Agent", params, hook: str) -> float:
        """Relative multiplier on a care transition (`hook` in CARE_HOOKS), normalised over the agents on
        that treatment (adherent / non-adherent for the ART transitions)."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        mults = getattr(getattr(getattr(params, "sdh", None), "hardship", None), "mult_per_point", None)
        m = float(getattr(mults, hook, 1.0) or 1.0) if mults is not None else 1.0
        k = int(getattr(agent, "hardship", 0) or 0)
        return cls._relative(params, m, k, (hook, None))

    @classmethod
    def visit_mult(cls, agent: "agent.Agent", params, setting_name: str, setting) -> float:
        """Relative multiplier on the setting's visit rate, normalised over the insured agents of the drug type."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        m = float(getattr(setting, "visit_mult_hardship_per_point", 1.0) or 1.0)
        k = int(getattr(agent, "hardship", 0) or 0)
        return cls._relative(params, m, k, ("visit", setting_name, getattr(agent, "drug_type", "NonInj")))

    def set_stats(self, stats: Dict[str, int], time: int):
        stats[f"hardship_{min(int(self.agent.hardship), 3)}"] += 1
        if self.agent.sex_work:
            stats["sex_work"] += 1
