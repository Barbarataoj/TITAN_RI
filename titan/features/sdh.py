import math
from typing import Dict, List, Optional, Tuple

from . import base_feature
from .. import agent
from .. import population
from .. import model

COMPONENTS = ("housing", "food", "employment")      # unstable housing, food insecurity, unemployment
RISK_HOOKS = ("sharing", "condomless_sex", "inject_start", "overdose")
CARE_HOOKS = ("moud_discontinue", "prep_discontinue", "art_nonadherence", "art_readherence")


class _Table:
    """A small attribute bag (component -> value) for the visit multipliers read off a setting."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


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

    Every effect is a RELATIVE multiplier: the product, over the components the agent has, of that
    component's multiplier for the behaviour (`risk_mult`, `care_mult`, the settings' `visit_mult_hardship_<component>`;
    component-specific since 2026-10-25, work record §19.46 -- with equal values it is the former m^k per
    index point), divided each month by the mean of the product over the agents the hook acts on
    (`mean_preserving`): risk hooks by drug type (sharing, condomless
    sex, starting to inject, overdose); the visit multiplier over the INSURED agents of the drug type,
    per setting (the claims-solved rates describe insured members); MOUD and PrEP discontinuation over
    the agents on them; the ART transitions over the adherent / non-adherent agents on ART. The
    population average of each calibrated or anchored behaviour is therefore unchanged and the
    determinants decide who carries it. Sex work redistributes sexual partners the same way (2026-10-25,
    work record §19.42 finding 2): the survey-anchored partner distributions already contain sex workers,
    so an agent in sex work has `partner_mult` times the partners of one who is not, within the group's
    unchanged mean -- factor m / (1 + p (m - 1)) in sex work and 1 / (1 + p (m - 1)) otherwise, with p
    the group's sex-work marginal (sex type x injecting status), stochastic rounding on the agent's drawn
    base -- and sex work in the past `sex_work.prep_indication_months` months is a PrEP indication
(`sex_work` in prep.target_model; `agent.sex_work_recent`, refreshed every month by this feature).

    With every marginal at 0 (the defaults) no random number is drawn and every multiplier is exactly
    1, so a run with the feature on and nobody affected is identical to one with it off.
    """

    name = "sdh"
    stats: List[str] = ["hardship_0", "hardship_1", "hardship_2", "hardship_3", "sex_work",
                        "hardship_housing", "hardship_food", "hardship_employment"]   # agents with the component (2026-10-25)
    norm: Dict[Tuple, float] = {}       # (hook, group) -> mean of m^k over that group, refreshed monthly

    @classmethod
    def init_class(cls, params):
        cls.norm = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        self.z0 = 0.0
        self.eps: Dict[str, float] = {}
        self.components: Dict[str, bool] = {c: False for c in COMPONENTS}
        self.partner_base: Dict[str, int] = {}      # the agent's drawn partner numbers, the base the factor acts on
        self.last_drug: Optional[str] = None        # drug type the states were last drawn against
        self.sex_work_last: Optional[int] = None    # last step the agent was in sex work (the PrEP indication window)
        self.agent.hardship = 0
        self.agent.hardship_on: Tuple[str, ...] = ()   # the components present, read by the hooks
        self.agent.sex_work = False
        self.agent.sex_work_recent = False          # in sex work within the past prep_indication_months months

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
        self._stamp(params, time)

    def _latent(self, k: str, rho: float) -> float:
        return math.sqrt(rho) * self.z0 + math.sqrt(1.0 - rho) * self.eps[k]

    def _apply(self, params, rho: float, rng):
        p = self.marginals(self.agent, params)
        for c in COMPONENTS:
            self.components[c] = _phi(self._latent(c, rho)) < p[c]
        self.agent.hardship = sum(1 for v in self.components.values() if v)
        self.agent.hardship_on = tuple(c for c in COMPONENTS if self.components[c])
        sex_work = _phi(self._latent("sex_work", rho)) < p["sex_work"]
        drug = getattr(self.agent, "drug_type", "NonInj")
        first = self.last_drug is None
        group_changed = self.last_drug != drug
        self.last_drug = drug
        if sex_work != self.agent.sex_work or group_changed or first:
            self.agent.sex_work = sex_work
            self._set_partners(params, rng, p["sex_work"])

    def _stamp(self, params, time: int):
        """Record this month if the agent is in sex work and refresh the PrEP-indication flag: in sex work in any of
        the past `prep_indication_months` months, this one included (window 1 = the current state only)."""
        if self.agent.sex_work:
            self.sex_work_last = time
        w = int(params.sdh.sex_work.prep_indication_months)
        self.agent.sex_work_recent = self.sex_work_last is not None and 0 <= time - self.sex_work_last < w

    @staticmethod
    def partner_factor(params, sex_work: bool, p_sw: float) -> float:
        """Relative partner-number factor: m / (1 + p (m - 1)) in sex work, 1 / (1 + p (m - 1)) otherwise, so the
        group's mean partner number is unchanged for the parameter marginal p; 1.0 when m is 1 or nobody in the
        group does sex work."""
        m = float(params.sdh.sex_work.partner_mult)
        if m == 1.0 or p_sw <= 0.0:
            return 1.0
        mean = 1.0 + p_sw * (m - 1.0)
        return (m if sex_work else 1.0) / mean

    def _set_partners(self, params, rng, p_sw: float):
        """Set the sexual partner numbers to the agent's drawn base times the factor for its current state and group
        (stochastic rounding, so the realised factor equals the parameter on average). The base is captured the
        first time, before any factor is applied."""
        f = self.partner_factor(params, self.agent.sex_work, p_sw)
        for bond, bond_def in params.classes.bond_types.items():
            if "sex" not in bond_def.acts_allowed:
                continue
            if bond not in self.partner_base:
                self.partner_base[bond] = int(self.agent.mean_num_partners.get(bond, 0))
            base = self.partner_base[bond]
            if base <= 0:
                continue
            if f == 1.0:
                n = base
            else:
                x = base * f
                n = int(math.floor(x)) + (1 if rng.random() < x - math.floor(x) else 0)
            self.agent.mean_num_partners[bond] = n
            self.agent.target_partners[bond] = n

    # ------------------------------------------------------------------ monthly step
    @classmethod
    def update_pop(cls, model: "model.TITAN"):
        """Monthly: episodes end and are re-drawn, for every agent, before the care layer rolls the month's
        visits; then the mean-preserving normalisers are refreshed for the month."""
        params = model.params
        q = float(params.sdh.hardship.episode_end_prob_per_month)
        rho = float(params.sdh.hardship.rho)
        for a in model.pop.all_agents:                  # every month: episodes may end, drug-type changes re-read
            a.sdh.churn(model, params, q, rho)
        cls.refresh_norms(model.pop.all_agents, params)

    def churn(self, model: "model.TITAN", params, q: float, rho: float):
        if not self.eps:
            return                              # never drawn: every marginal was 0
        changed = False
        if q > 0.0:                                     # no random numbers consumed when episodes never end
            for k in self.eps:
                if model.run_random.random() < q:
                    self.eps[k] = model.run_random.gauss(0.0, 1.0)
                    changed = True
        if changed or self.last_drug != getattr(self.agent, "drug_type", "NonInj"):
            self._apply(params, rho, model.run_random)      # a drug-type change re-reads the states against its marginals
        self._stamp(params, model.time)

    # ------------------------------------------------------------------ the product form
    @staticmethod
    def product(table, hook: Optional[str], on) -> float:
        """Product over the agent's components present (`on`) of the component's multiplier: table.<component>.<hook>,
        or table.<component> itself when `hook` is None (the settings' visit table). 1.0 when the table is missing."""
        if table is None:
            return 1.0
        w = 1.0
        for c in on:
            block = getattr(table, c, None)
            if block is None:
                continue
            v = block if hook is None else getattr(block, hook, 1.0)
            w *= float(v if v is not None else 1.0)
        return w

    @staticmethod
    def active(table, hook: Optional[str]) -> bool:
        """Whether any component's multiplier for `hook` differs from 1 (an inert hook has no normaliser and no effect)."""
        if table is None:
            return False
        for c in COMPONENTS:
            block = getattr(table, c, None)
            if block is None:
                continue
            v = block if hook is None else getattr(block, hook, 1.0)
            if float(v if v is not None else 1.0) != 1.0:
                return True
        return False

    @staticmethod
    def visit_table(setting):
        """The setting's hardship multipliers as a component table (fields visit_mult_hardship_<component>)."""
        return _Table(**{c: float(getattr(setting, f"visit_mult_hardship_{c}", 1.0) or 1.0) for c in COMPONENTS})

    @classmethod
    def refresh_norms(cls, agents, params):
        """Mean of the product multiplier per hook over the agents the hook acts on (see the class docstring)."""
        h = params.sdh.hardship
        risk_t = getattr(h, "risk_mult", None)
        care_t = getattr(h, "care_mult", None)
        risk_on = {k: cls.active(risk_t, k) for k in RISK_HOOKS}
        care_on = {k: cls.active(care_t, k) for k in CARE_HOOKS}
        settings = getattr(getattr(params, "care_seeking", None), "setting", None) or {}
        visit_t = {s: cls.visit_table(st) for s, st in settings.items()}
        visit_on = {s: cls.active(t, None) for s, t in visit_t.items()}
        sums: Dict[Tuple, float] = {}
        cnt: Dict[Tuple, int] = {}

        def add(key, value):
            sums[key] = sums.get(key, 0.0) + value
            cnt[key] = cnt.get(key, 0) + 1

        for a in agents:
            on = getattr(a, "hardship_on", ())
            dt = getattr(a, "drug_type", "NonInj")
            for hook in RISK_HOOKS:
                if risk_on[hook]:
                    add((hook, dt), cls.product(risk_t, hook, on))
            if getattr(a, "insured", True):
                for s, t in visit_t.items():
                    if visit_on[s]:
                        add(("visit", s, dt), cls.product(t, None, on))
            moud = getattr(a, "moud", None)
            if care_on["moud_discontinue"] and moud is not None and getattr(moud, "active", False):
                add(("moud_discontinue", None), cls.product(care_t, "moud_discontinue", on))
            prep = getattr(a, "prep", None)
            if care_on["prep_discontinue"] and prep is not None and getattr(prep, "active", False):
                add(("prep_discontinue", None), cls.product(care_t, "prep_discontinue", on))
            haart = getattr(a, "haart", None)
            if haart is not None and getattr(haart, "active", False):
                if getattr(haart, "adherent", False):
                    if care_on["art_nonadherence"]:
                        add(("art_nonadherence", None), cls.product(care_t, "art_nonadherence", on))
                elif care_on["art_readherence"]:
                    add(("art_readherence", None), cls.product(care_t, "art_readherence", on))
        cls.norm = {key: sums[key] / cnt[key] for key in sums if cnt[key]}

    # ------------------------------------------------------------------ hooks read by the other features
    @classmethod
    def _relative(cls, params, table, hook: Optional[str], agent, key: Tuple) -> float:
        if not cls.active(table, hook):
            return 1.0
        raw = cls.product(table, hook, getattr(agent, "hardship_on", ()))
        if getattr(params.sdh.hardship, "mean_preserving", True):
            return raw / cls.norm.get(key, 1.0)
        return raw

    @classmethod
    def risk_mult(cls, agent: "agent.Agent", params, hook: str) -> float:
        """Relative multiplier on a risk behaviour (`hook` in RISK_HOOKS), normalised within the agent's drug type."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        table = getattr(getattr(getattr(params, "sdh", None), "hardship", None), "risk_mult", None)
        return cls._relative(params, table, hook, agent, (hook, getattr(agent, "drug_type", "NonInj")))

    @classmethod
    def mult(cls, agent: "agent.Agent", params, hook: str) -> float:
        """Relative multiplier on a care transition (`hook` in CARE_HOOKS), normalised over the agents on
        that treatment (adherent / non-adherent for the ART transitions)."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        table = getattr(getattr(getattr(params, "sdh", None), "hardship", None), "care_mult", None)
        return cls._relative(params, table, hook, agent, (hook, None))

    @classmethod
    def visit_mult(cls, agent: "agent.Agent", params, setting_name: str, setting) -> float:
        """Relative multiplier on the setting's visit rate, normalised over the insured agents of the drug type."""
        if not getattr(getattr(params, "features", None), "sdh", False):
            return 1.0
        return cls._relative(params, cls.visit_table(setting), None, agent, ("visit", setting_name, getattr(agent, "drug_type", "NonInj")))

    def set_stats(self, stats: Dict[str, int], time: int):
        stats[f"hardship_{min(int(self.agent.hardship), 3)}"] += 1
        if self.agent.sex_work:
            stats["sex_work"] += 1
        for c in getattr(self.agent, "hardship_on", ()):
            stats[f"hardship_{c}"] += 1
