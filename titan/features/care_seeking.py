import logging
import math
from typing import Dict, List, ClassVar, Optional

from . import base_feature
from .. import agent
from .. import population
from .. import model
from ..parse_params import ObjMap


SERVICES = ("hiv_test", "prep", "moud", "counseling")


class CareSeeking(base_feature.BaseFeature):
    """
    Care-setting and provider feature.

    Agents visit clinical settings (primary care, emergency department, substance
    use treatment, mental health) at setting-specific rates. Each agent is attached
    to a provider in each setting they use, and that attachment persists, so an
    agent whose provider does not deliver a service stays unserved across repeated
    visits.

    Providers are not agents. They are records in a class-level pool, carrying a
    willingness flag and an offer intensity per service. This is enough to express
    provider-directed interventions (train some fraction of the pool) and to
    produce the clustering that a flat per-visit probability cannot, without adding
    a second agent type to the model.

    Every visit-rolling and offer-resolution happens in `update_pop`, not
    `update_agent`. `TITAN.update_all_agents` runs every enabled feature's
    `update_pop` (population level) before any agent's `update_agent` (agent
    level) - see `base_feature.BaseFeature.update_pop`. Doing the work here
    means `care_offer` is guaranteed to reflect *this* time step by the time
    `prep`, `moud`, and `hiv_testing` read it in their own `update_agent`,
    regardless of where `care_seeking` sits in `titan/features/__init__.py`'s
    import order (which is what `BaseFeature.__subclasses__()` - and so
    `TITAN.features` - is ordered by).

    This feature publishes what happened during a time step onto the agent:

    * `agent.care_encounter` - dict of setting name to number of visits this step
    * `agent.care_offer` - dict of service name to whether it was offered this
      step, at any setting (this is what `prep`/`moud`/`hiv_testing` read)
    * `agent.care_offer_by_setting` - dict of setting name to dict of service
      name to whether it was offered in that setting this step, preserving the
      provenance `agent.care_offer` collapses away, for reporting by setting

    `counseling` is one of the four offered services, but nothing consumes it
    as a multiplier or prerequisite for another service - in particular,
    `hiv_testing` does not condition on it. CDC opt-out testing does not
    require prevention counseling, so keep it that way; a counseling/testing
    cross-tab belongs in descriptive validation, not in the transition logic.
    """

    name = "care_seeking"
    stats: ClassVar[List[str]] = []

    providers: ClassVar[Dict[str, List[Dict]]] = {}
    """Provider pool, keyed by setting. Built once per population."""

    provider_weights: ClassVar[Dict[str, List[float]]] = {}
    """Cumulative attachment weights per setting, for patient assignment."""

    provider_shares: ClassVar[Dict[str, List[float]]] = {}
    """Non-cumulative attachment shares per setting, kept for `panel_concentration`."""
    pool_draw_p: ClassVar[Dict[str, Dict[str, float]]] = {}
    """The p_willing each setting's pool was last drawn or refreshed against, per service."""

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)

        self.active = False
        self.provider_id: Dict[str, int] = {}
        self.visit_frailty: Dict[str, float] = {}
        # The overdose step for which the post-overdose visit has already been
        # rolled, so one event yields at most one extra visit per setting even
        # though the window below spans two steps.
        self.od_visit_rolled: Dict[str, Optional[int]] = {}
        # patient-facing programme (reach_* fields): reached per setting, and
        # whether the reach draw has been made
        self.reached: Dict[str, bool] = {}
        self.reach_rolled: Dict[str, bool] = {}

    # ================= CLASS-LEVEL METHODS =================

    @classmethod
    def init_class(cls, params: "ObjMap"):
        """
        Reset the provider pool and declare this run's stats keys. The pool is
        built lazily on the first agent initialization, because building it
        needs a random number generator and this method is not passed one.

        Stats are declared per setting and service (rather than as a fixed
        list) because settings are a params-defined, arbitrarily-named
        collection, not a fixed set of classes.

        args:
            params: the population params
        """
        cls.providers = {}
        cls.provider_weights = {}
        cls.provider_shares = {}

        cls.stats = ["care_visits", "care_reached", "care_reach_visits", "prep_monitoring_visits"]
        for setting_name in params.care_seeking.setting:
            cls.stats.append(f"care_encounters_{setting_name}")
            for service in SERVICES:
                cls.stats.append(f"care_offer_{setting_name}_{service}")

    @classmethod
    def build_pool(cls, params: "ObjMap", rand_gen, num_pop: Optional[int] = None):
        """
        Create the provider pool, one list of provider records per setting.

        Pool size is scaled from the real Rhode Island provider count down to
        the simulated population, so that the number of patients per provider
        is preserved as the population size changes. The scaled count is
        rounded stochastically (probability of rounding up equal to the
        fractional part) rather than with `round()`, so that a setting whose
        scaled count is below 1 gets a proportional chance of a single
        provider instead of deterministically collapsing to an empty pool for
        every small simulated population.

        Willingness is drawn per service. When `shared_trait_prob` is above
        zero, some providers use a single shared draw across all services, so
        that a provider willing to deliver one service tends to deliver the
        others. That parameter is what the APCD service-pattern pull informs.

        args:
            params: the population params
            rand_gen: random number generator to draw provider attributes with
        """
        care_params = params.care_seeking
        # num_pop is passed explicitly rather than read off params.model.num_pop:
        # population_io sets num_pop to 0 before constructing a loaded
        # Population, and that 0 is deep-copied into every Location's params and
        # never restored -- which silently scaled every provider pool to zero.
        if num_pop is None:
            num_pop = params.model.num_pop
        scale = num_pop / care_params.n_ri_oud
        rho = care_params.shared_trait_prob

        for setting_name, setting in care_params.setting.items():
            n_providers = cls.scaled_provider_count(setting.n_providers_ri, scale, rand_gen)

            pool: List[Dict] = []
            for _ in range(n_providers):
                pool.append(cls.draw_provider(setting, rho, rand_gen))

            cls.providers[setting_name] = pool
            cls.pool_draw_p[setting_name] = {
                service: getattr(setting, f"p_willing_{service}") for service in SERVICES
            }
            cumulative, shares = cls.draw_weights(
                len(pool), setting.panel_skew, rand_gen
            )
            cls.provider_weights[setting_name] = cumulative
            cls.provider_shares[setting_name] = shares

    @staticmethod
    def scaled_provider_count(n_providers_ri: int, scale: float, rand_gen) -> int:
        """
        Scale a real-world provider count down to the simulated population,
        rounding stochastically so that small simulated populations get a
        proportional chance at a provider instead of always rounding to zero.

        args:
            n_providers_ri: real-world provider count for this setting
            scale: num_pop / n_ri_oud
            rand_gen: random number generator

        returns:
            number of providers to draw for this setting's pool
        """
        expected = n_providers_ri * scale
        n_providers = int(math.floor(expected))
        if rand_gen.random() < (expected - n_providers):
            n_providers += 1
        return n_providers

    @classmethod
    def draw_provider(cls, setting: "ObjMap", rho: float, rand_gen) -> Dict:
        """
        Draw one provider record.

        args:
            setting: params for the setting this provider practices in
            rho: probability that this provider's willingness across services
                is drawn from a single shared draw rather than independently
                per service - a mixture weight, not a correlation coefficient
            rand_gen: random number generator

        returns:
            dict of service name to (willing, intensity)
        """
        shared = rand_gen.random() if rand_gen.random() < rho else None

        record: Dict = {}
        for service in SERVICES:
            p_willing = getattr(setting, f"p_willing_{service}")
            draw = shared if shared is not None else rand_gen.random()
            record[service] = {
                "willing": draw < p_willing,
                "intensity": getattr(setting, f"intensity_{service}"),
            }
        return record

    @classmethod
    def draw_weights(
        cls, n_providers: int, skew: float, rand_gen
    ) -> "tuple[List[float], List[float]]":
        """
        Draw attachment weights so that some providers carry more patients than
        others.

        args:
            n_providers: number of providers in this setting's pool
            skew: gamma shape; lower concentrates patients on fewer providers
            rand_gen: random number generator

        returns:
            (cumulative shares, non-cumulative shares), both normalized to
            1.0, or a pair of empty lists if the pool is empty
        """
        if n_providers == 0:
            return [], []

        weights = [rand_gen.gammavariate(skew, 1.0) for _ in range(n_providers)]
        total = sum(weights)
        if total <= 0:
            shares = [1.0 / n_providers for _ in range(n_providers)]
        else:
            shares = [weight / total for weight in weights]

        cumulative = []
        running = 0.0
        for share in shares:
            running += share
            cumulative.append(running)
        return cumulative, shares

    @classmethod
    def sample_provider(cls, setting_name: str, rand_gen) -> Optional[int]:
        """
        Pick a provider index in a setting, weighted so that busier providers are
        more likely to be picked.

        args:
            setting_name: which setting to draw from
            rand_gen: random number generator

        returns:
            index into the setting's provider pool, or None if the pool is empty
        """
        cumulative = cls.provider_weights.get(setting_name, [])
        if not cumulative:
            return None

        draw = rand_gen.random()
        for idx, threshold in enumerate(cumulative):
            if draw <= threshold:
                return idx
        return len(cumulative) - 1

    @classmethod
    def panel_concentration(cls, setting_name: str) -> Optional[float]:
        """
        Herfindahl-Hirschman index of patient-share concentration across
        providers in a setting: the sum of squared shares, ranging from
        1 / n_providers (patients split evenly) to 1 (all patients on one
        provider). For validating against the APCD panel-size concentration
        pull (Pull 2) - a population snapshot, not a per-timestep additive
        count, so it is exposed as a direct classmethod rather than folded
        into `stats`.

        args:
            setting_name: which setting to report concentration for

        returns:
            the HHI, or None if the setting has no providers
        """
        shares = cls.provider_shares.get(setting_name)
        if not shares:
            return None
        return sum(share * share for share in shares)

    @classmethod
    def update_pop(cls, model: "model.TITAN"):
        """
        Population-level step, run before any agent's `update_agent` (see the
        class docstring for why that ordering is load-bearing here):

        1. Retire providers and replace them with fresh draws, reassigning
           attached patients to a new provider if `reassign_on_attrition` is
           set.
        2. Roll every attached agent's visits for this time step and resolve
           what was offered.

        args:
            model: the instance of TITAN currently being run
        """
        care_params = model.params.care_seeking
        cls.ensure_initialized(model)
        cls.refresh_pools(care_params, model)
        cls.process_attrition(care_params, model)
        cls.resolve_all_visits(care_params, model)

    @classmethod
    def refresh_pools(cls, care_params: "ObjMap", model: "model.TITAN"):
        """
        Re-draw a setting's provider records against the current parameters at
        `setting.pool_refresh_time`.

        Provider records freeze `willing` and `intensity` when drawn, so a
        parameter changed mid-run (timeline_scaling at an arm's start date)
        would otherwise reach only the replacements attrition draws. Here every
        provider's intensity is set to the current parameter, and willingness is
        topped up (or trimmed) to the new share without re-rolling providers who
        already deliver: a non-willing provider becomes willing with probability
        (p_new - p_old) / (1 - p_old); when the parameter fell, a willing one
        stops with probability (p_old - p_new) / p_old. Added 2026-10-02 for the
        dated Phase 7 arms.

        args:
            care_params: the care_seeking params
            model: the instance of TITAN currently being run
        """
        for setting_name, setting in care_params.setting.items():
            if getattr(setting, "pool_refresh_time", -9999) != model.time:
                continue
            pool = cls.providers.get(setting_name, [])
            if not pool:
                continue
            drawn = cls.pool_draw_p.setdefault(setting_name, {})
            for service in SERVICES:
                p_new = getattr(setting, f"p_willing_{service}")
                p_old = drawn.get(service, p_new)
                intensity = getattr(setting, f"intensity_{service}")
                if p_new > p_old and p_old < 1.0:
                    p_up = (p_new - p_old) / (1.0 - p_old)
                    p_down = 0.0
                elif p_new < p_old and p_old > 0.0:
                    p_up = 0.0
                    p_down = (p_old - p_new) / p_old
                else:
                    p_up = p_down = 0.0
                for record in pool:
                    rec = record[service]
                    if not rec["willing"] and p_up > 0 and model.run_random.random() < p_up:
                        rec["willing"] = True
                    elif rec["willing"] and p_down > 0 and model.run_random.random() < p_down:
                        rec["willing"] = False
                    rec["intensity"] = intensity
                drawn[service] = p_new
            logging.info(
                "care_seeking: refreshed %s provider pool (%d providers) at t=%d",
                setting_name, len(pool), model.time,
            )

    @classmethod
    def ensure_initialized(cls, model: "model.TITAN"):
        """
        Build the provider pool and attach any agent that has no care_seeking
        state yet.

        All of this feature's state is normally created in `init_agent`, which
        only runs from `Population.create_agent`. A population restored from
        disk is assembled with `Population.add_agent` instead (and with
        `num_pop` temporarily set to 0), so none of it ran: the pool stayed
        empty, every agent stayed inactive, and because all three consumers read
        the offers through `getattr(..., {})`, the whole feature silently did
        nothing -- no PrEP initiation, no MOUD initiation, no HIV testing, with
        a report that still looked plausible.

        args:
            model: the instance of TITAN currently being run
        """
        pop = model.pop
        num_pop = pop.all_agents.num_members()

        if not any(cls.providers.values()):
            if not cls.providers:
                logging.info(
                    "care_seeking: building provider pool for %d agents "
                    "(no pool was initialized, e.g. a population loaded from disk)",
                    num_pop,
                )
            cls.providers = {}
            cls.provider_weights = {}
            cls.provider_shares = {}
            cls.build_pool(model.params, pop.pop_random, num_pop=num_pop)

        for agent in pop.all_agents:
            care = getattr(agent, "care_seeking", None)
            if care is None or care.active:
                continue
            # agent never had init_agent run for this feature
            if not care.provider_id:
                care.init_agent(pop, model.time)

    @classmethod
    def process_attrition(cls, care_params: "ObjMap", model: "model.TITAN"):
        """
        Retire providers and replace them with fresh draws.

        When `reassign_on_attrition` is false, a replaced record keeps its
        index, so agents attached to it stay attached and simply find a
        different provider there: the practice continues, the clinician has
        changed. When true, patients attached to a retired index are instead
        resampled to a (possibly different) provider in the same setting, to
        represent them leaving with their clinician rather than staying with
        the practice.

        args:
            care_params: the care_seeking params
            model: the instance of TITAN currently being run
        """
        steps_per_year = model.params.model.time.steps_per_year
        rho = care_params.shared_trait_prob

        for setting_name, setting in care_params.setting.items():
            if setting.attrition_annual <= 0:
                continue

            pool = cls.providers.get(setting_name, [])
            if not pool:
                continue

            prob = cls.annual_to_step(setting.attrition_annual, steps_per_year)
            retired = [
                idx for idx in range(len(pool)) if model.run_random.random() < prob
            ]
            if not retired:
                continue

            for idx in retired:
                pool[idx] = cls.draw_provider(setting, rho, model.run_random)

            if not care_params.reassign_on_attrition:
                continue

            retired_set = set(retired)
            for a in model.pop.all_agents:
                care = getattr(a, cls.name, None)
                if care is None or not care.active:
                    continue
                if care.provider_id.get(setting_name) in retired_set:
                    new_idx = cls.sample_provider(setting_name, model.run_random)
                    if new_idx is not None:
                        care.provider_id[setting_name] = new_idx

    @classmethod
    def resolve_all_visits(cls, care_params: "ObjMap", model: "model.TITAN"):
        """
        Roll this time step's visits for every attached agent.

        args:
            care_params: the care_seeking params
            model: the instance of TITAN currently being run
        """
        steps_per_year = model.params.model.time.steps_per_year
        for a in model.pop.all_agents:
            care = getattr(a, cls.name, None)
            if care is None or not care.active:
                continue
            care.roll_visits(care_params, steps_per_year, model)

    @staticmethod
    def annual_to_step(annual_prob: float, steps_per_year: int) -> float:
        """
        Convert an annual probability to a per-time-step probability, via the
        hazard rate. Matches the conversion used in `hiv_testing`.

        args:
            annual_prob: probability of the event over one year
            steps_per_year: number of time steps per year

        returns:
            probability of the event over one time step
        """
        if annual_prob <= 0:
            return 0.0
        if annual_prob >= 1:
            return 1.0

        hazard = -math.log(1.0 - annual_prob)
        return 1.0 - math.exp(-hazard / steps_per_year)

    # ================= AGENT-LEVEL METHODS =================

    def init_agent(self, pop: "population.Population", time: int):
        """
        Attach the agent to a provider in each setting, draw its per-setting
        visit-frailty (agent-level utilization heterogeneity), and publish
        empty encounter and offer records so downstream features can read them
        safely before the first time step.

        args:
            pop: the population this agent is a part of
            time: the current time step
        """
        self.agent.care_encounter = {}
        self.agent.care_offer = {service: False for service in SERVICES}
        self.agent.care_offer_by_setting = {}
        self.agent.care_offer_mult = {service: 1.0 for service in SERVICES}

        params = self.agent.location.params
        if not self.providers:
            self.build_pool(params, pop.pop_random)

        for setting_name, setting in params.care_seeking.setting.items():
            provider_idx = self.sample_provider(setting_name, pop.pop_random)
            if provider_idx is not None:
                self.provider_id[setting_name] = provider_idx

            self.visit_frailty[setting_name] = self.draw_frailty(setting, pop.pop_random)

        self.active = bool(self.provider_id)

    @staticmethod
    def draw_frailty(setting: "ObjMap", rand_gen) -> float:
        """
        Draw this agent's persistent visit-rate frailty for a setting.

        Mixing a homogeneous Poisson rate with a Gamma-distributed per-agent
        frailty (mean 1) produces a Negative-Binomial visit-count distribution
        in cross-section, which is what over-dispersed utilization data (some
        patients are consistently heavier or lighter users than the average)
        needs and a homogeneous Poisson cannot produce. `visit_heterogeneity_shape`
        is the Gamma shape parameter: smaller values mean more heterogeneity
        (heavier-tailed utilization), and 0 (the default) disables this,
        reproducing the original homogeneous-Poisson behavior exactly (frailty
        pinned to 1).

        args:
            setting: params for the setting to draw frailty for
            rand_gen: random number generator

        returns:
            a multiplier on this agent's visit rate in this setting, drawn
            once and held for the agent's lifetime
        """
        shape = getattr(setting, "visit_heterogeneity_shape", 0.0)
        if shape <= 0:
            return 1.0
        return rand_gen.gammavariate(shape, 1.0 / shape)

    def roll_visits(
        self, care_params: "ObjMap", steps_per_year: int, model: "model.TITAN"
    ):
        """
        Roll this time step's visits across every setting and resolve what was
        offered at each. Nothing is applied to the agent here: whether an
        offered service is taken up, and what it does, belongs to the feature
        that owns that service.

        args:
            care_params: the care_seeking params
            steps_per_year: number of time steps per year
            model: the instance of TITAN currently being run
        """
        self.agent.care_encounter = {}
        self.agent.care_offer = {service: False for service in SERVICES}
        self.agent.care_offer_by_setting = {}
        # acceptance multiplier carried with the offer: the largest among the
        # settings that offered the service this step (1.0 = the global
        # acceptance scalar unchanged); consumed by hiv_testing, prep and moud
        self.agent.care_offer_mult = {service: 1.0 for service in SERVICES}
        self.agent.care_reach_visits = 0
        self.agent.prep_monitoring_visit = False
        # PrEP-side effects of the programme(s) that reached this agent,
        # consumed by titan/features/prep.py
        self.agent.care_reach = self.reach_effects(care_params)

        for setting_name, setting in care_params.setting.items():
            provider_idx = self.provider_id.get(setting_name)
            if provider_idx is None:
                continue

            # The post-overdose visit does not depend on the routine rate, so a
            # scenario that zeroes a setting's routine visits keeps the link
            # between an overdose and an emergency encounter.
            n_visits = self.visit_after_overdose(setting, setting_name, model)
            monitoring = self.prep_monitoring_visit(setting_name, model)
            n_visits += monitoring

            if setting.visit_rate_annual > 0:
                rate = self.visit_rate(setting, setting_name, steps_per_year)
                # visit_rate_annual is a count of visit-days, not a probability,
                # so the number of visits in a step is drawn as a Poisson count.
                # Rolling a single yes/no instead would cap an agent at one visit
                # per step and undercount the busier settings, where an agent
                # can plausibly attend more than once a month.
                n_visits += int(model.np_random.poisson(rate))
            if n_visits == 0:
                continue

            self.agent.care_encounter[setting_name] = n_visits

            # the setting's patient-facing programme: reach draw at the first
            # visit on/after its start, then an "active" visit roll
            active = self.roll_reach(setting, setting_name, model)
            if active:
                self.agent.care_reach_visits += 1

            # each visit is a separate chance to be offered a service
            for _ in range(n_visits):
                self.resolve_offers(setting_name, provider_idx, model, active=active)

            if monitoring:
                # the monitoring visit carries its own HIV test, accepted
                self.agent.prep_monitoring_visit = True
                self.agent.care_offer["hiv_test"] = True
                self.agent.care_offer_by_setting.setdefault(
                    setting_name, {service: False for service in SERVICES}
                )["hiv_test"] = True
                self.agent.care_offer_mult["hiv_test"] = max(
                    self.agent.care_offer_mult.get("hiv_test", 1.0), 1e6
                )

    def roll_reach(self, setting: "ObjMap", setting_name: str, model: "model.TITAN") -> bool:
        """
        Whether this visit is an active programme visit: draw reach at the first
        visit on or after reach_start_time (retry at later visits with
        reach_retry_prob), then roll reach_visit_prob for reached agents.

        returns:
            True if the programme is active at this visit
        """
        p_reach = getattr(setting, "reach_prob", 0.0)
        if p_reach <= 0 and not self.reached.get(setting_name, False):
            return False
        if model.time < getattr(setting, "reach_start_time", -9999):
            return False
        if not self.reached.get(setting_name, False):
            if not self.reach_rolled.get(setting_name, False):
                self.reach_rolled[setting_name] = True
                p = p_reach
            else:
                p = getattr(setting, "reach_retry_prob", 0.0)
            if p > 0 and model.run_random.random() < p:
                self.reached[setting_name] = True
            else:
                return False
        return model.run_random.random() < getattr(setting, "reach_visit_prob", 1.0)

    def reach_effects(self, care_params: "ObjMap") -> Dict[str, float]:
        """
        The PrEP-side multipliers of the programmes that reached this agent:
        the strongest across settings (lowest discontinuation multiplier,
        highest of the others). Neutral values when the agent is reached nowhere.
        """
        eff = {"prep_discontinue_mult": 1.0, "prep_adherence_mult": 1.0,
               "prep_restart_prob": 0.0, "prep_monitoring_mult": 1.0}
        for setting_name, is_reached in self.reached.items():
            if not is_reached:
                continue
            setting = care_params.setting.get(setting_name)
            if setting is None:
                continue
            eff["prep_discontinue_mult"] = min(
                eff["prep_discontinue_mult"], getattr(setting, "reach_prep_discontinue_mult", 1.0))
            eff["prep_adherence_mult"] = max(
                eff["prep_adherence_mult"], getattr(setting, "reach_prep_adherence_mult", 1.0))
            eff["prep_restart_prob"] = max(
                eff["prep_restart_prob"], getattr(setting, "reach_prep_restart_prob", 0.0))
            eff["prep_monitoring_mult"] = max(
                eff["prep_monitoring_mult"], getattr(setting, "reach_prep_monitoring_mult", 1.0))
        return eff

    def prep_monitoring_visit(self, setting_name: str, model: "model.TITAN") -> int:
        """
        One PrEP monitoring visit in prep.monitoring_setting when the agent is
        on oral PrEP and a monitoring interval has elapsed since initiation,
        attended with probability prep.monitoring_attendance (scaled by the
        agent's reach_prep_monitoring_mult). 0 when monitoring is off.
        """
        prep_params = getattr(model.params, "prep", None)
        interval = getattr(prep_params, "monitoring_interval_steps", 0) if prep_params is not None else 0
        if interval <= 0 or setting_name != getattr(prep_params, "monitoring_setting", "primary_care"):
            return 0
        prep = getattr(self.agent, "prep", None)
        if prep is None or not getattr(prep, "active", False) or getattr(prep, "type", "") != "Oral":
            return 0
        started = getattr(prep, "time", None)
        if started is None:
            return 0
        elapsed = model.time - started
        if elapsed <= 0 or elapsed % interval != 0:
            return 0
        attend = getattr(prep_params, "monitoring_attendance", 1.0)
        attend = min(1.0, attend * getattr(self.agent, "care_reach", {}).get("prep_monitoring_mult", 1.0))
        return 1 if model.run_random.random() < attend else 0

    def visit_after_overdose(
        self, setting: "ObjMap", setting_name: str, model: "model.TITAN"
    ) -> int:
        """
        One extra visit to this setting after a nonfatal overdose, with
        probability `visit_prob_after_overdose`.

        The overdose feature records `agent.last_overdose_t` in its
        `update_agent`, which runs after this feature's `update_pop` within a
        step, so the event is first visible here on the following step; the
        window accepts the event step too in case the ordering ever changes.
        Each overdose is rolled at most once per setting. Added 2026-10-02:
        before this nothing linked an overdose to an emergency encounter, so
        post-overdose MOUD linkage was unmodelled while the care layer bypasses
        `moud.post_overdose_rr`.

        args:
            setting: params for the setting
            setting_name: name of the setting
            model: the instance of TITAN currently being run

        returns:
            1 if the extra visit happens this step, else 0
        """
        prob = getattr(setting, "visit_prob_after_overdose", 0.0)
        if prob <= 0:
            return 0
        last_od = getattr(self.agent, "last_overdose_t", None)
        if last_od is None or not (0 <= model.time - last_od <= 1):
            return 0
        if self.od_visit_rolled.get(setting_name) == last_od:
            return 0
        self.od_visit_rolled[setting_name] = last_od
        return 1 if model.run_random.random() < prob else 0

    def visit_rate(
        self, setting: "ObjMap", setting_name: str, steps_per_year: int
    ) -> float:
        """
        This agent's per-step expected visit count for a setting: the
        setting's base rate, adjusted by this agent's stratified multiplier
        and persistent frailty.

        args:
            setting: params for the setting
            setting_name: name of the setting, to key this agent's frailty
            steps_per_year: number of time steps per year

        returns:
            expected number of visits this agent makes to this setting this
            time step (the Poisson mean `roll_visits` draws from)
        """
        annual_rate = setting.visit_rate_annual * self.stratified_multiplier(setting)
        return (annual_rate / steps_per_year) * self.visit_frailty.get(setting_name, 1.0)

    def stratified_multiplier(self, setting: "ObjMap") -> float:
        """
        Multiply the setting's base visit rate by whichever agent-state
        stratifiers this setting configures.

        The APCD encounter pull (which informs `visit_rate_annual`) asks for
        stratification by MOUD status, injection status, HIV status, age, sex,
        race/ethnicity, and insurance. Only MOUD status, injection status, and
        HIV status are wired here, as `visit_mult_moud`, `visit_mult_inj`, and
        `visit_mult_hiv_positive` - each a single setting-level multiplier,
        since those are binary agent states. Age, sex, race/ethnicity, and
        insurance each have more than two categories and are not yet backed by
        setting-level rates; add them here the same way (one multiplier field
        per category) once those rates exist. Insurance in particular isn't
        modeled as agent state anywhere else in the model yet.

        args:
            setting: params for the setting

        returns:
            a multiplier on `setting.visit_rate_annual`, 1.0 if none of the
            configured stratifiers apply to this agent
        """
        multiplier = 1.0

        if getattr(self.agent, "on_moud", False):
            multiplier *= getattr(setting, "visit_mult_moud", 1.0)

        if self.agent.drug_type == "Inj":
            multiplier *= getattr(setting, "visit_mult_inj", 1.0)

        hiv_feat = getattr(self.agent, "hiv", None)
        if hiv_feat is not None and getattr(hiv_feat, "active", False):
            multiplier *= getattr(setting, "visit_mult_hiv_positive", 1.0)

        return multiplier

    def resolve_offers(self, setting_name: str, provider_idx: int, model: "model.TITAN",
                       active: bool = False):
        """
        Work out which services the agent's provider in this setting offered
        during a visit, recording both the collapsed (`care_offer`) and
        setting-attributed (`care_offer_by_setting`) views.

        An offer requires both that the provider is willing to deliver the
        service at all, and that they do so at this particular visit. The first
        is the term a provider-directed intervention moves.

        At an active programme visit (`active`, see roll_reach) the setting's
        reach_* fields apply: an offer floor reach_offer_prob_<service> that
        applies whatever the provider's willingness (a test order prompted by
        the summary; a prepared patient's request acted on by a non-prescribing
        provider), the provider's intensities scaled by
        reach_intensity_mult_<service>, and the acceptance multiplier scaled by
        reach_acceptance_mult_<service>.

        args:
            setting_name: the setting the visit happened in
            provider_idx: index of the agent's provider in this setting's pool
            model: the instance of TITAN currently being run
            active: whether the setting's patient-facing programme is active at this visit
        """
        pool = self.providers.get(setting_name, [])
        if provider_idx >= len(pool):
            return

        provider = pool[provider_idx]
        setting_offers = self.agent.care_offer_by_setting.setdefault(
            setting_name, {service: False for service in SERVICES}
        )
        # the setting's acceptance multipliers; fall back to the agent's
        # location params so a harness without `model.params` still resolves
        care = getattr(getattr(model, "params", None), "care_seeking", None)
        if care is None:
            care = getattr(getattr(self.agent, "location", None), "params", None)
            care = getattr(care, "care_seeking", None)
        setting = None
        if care is not None:
            try:
                setting = care.setting[setting_name]
            except (KeyError, AttributeError, TypeError):
                setting = None
        for service in SERVICES:
            offered = False
            if provider[service]["willing"]:
                intensity = provider[service]["intensity"]
                if active and setting is not None:
                    intensity = min(1.0, intensity * getattr(setting, f"reach_intensity_mult_{service}", 1.0))
                offered = model.run_random.random() < intensity
            if not offered and active and setting is not None:
                # the programme's floor on the offer whatever the provider's
                # own willingness: the summary prompts a test order; a prepared
                # patient's request gets a non-prescribing provider to act
                floor = getattr(setting, f"reach_offer_prob_{service}", -1.0)
                if floor >= 0:
                    offered = model.run_random.random() < floor
            if not offered:
                continue
            self.agent.care_offer[service] = True
            setting_offers[service] = True
            mult = getattr(setting, f"acceptance_mult_{service}", 1.0) if setting is not None else 1.0
            if active and setting is not None:
                mult *= getattr(setting, f"reach_acceptance_mult_{service}", 1.0)
            offer_mult = getattr(self.agent, "care_offer_mult", None)
            if offer_mult is not None and mult > offer_mult.get(service, 1.0):
                offer_mult[service] = mult

    # ================= STATS =================

    def set_stats(self, stats: Dict[str, int], time: int):
        """
        Report this agent's encounters and offers, by setting and service.

        Per-service *completion* (was the offered PrEP/MOUD/test actually
        taken up) is intentionally not reported here: `prep`, `moud`, and
        `hiv_testing` already report that themselves (`prep`/`prep_new`,
        `moud`/`moud_active`, `hiv_tests`), and reading their per-step counters
        from here would recreate the same feature-ordering hazard this file's
        `update_pop` restructuring exists to remove, since `set_stats` is
        called across features in the same `TITAN.features` order.

        args:
            stats: the dictionary to update with this agent's feature statistics
            time: the time step of the model when the stats are set
        """
        encounters = getattr(self.agent, "care_encounter", None)
        if encounters:
            stats["care_visits"] += sum(encounters.values())
            for setting_name, n_visits in encounters.items():
                stats[f"care_encounters_{setting_name}"] += n_visits
        # written only when non-zero, like the counters above, so a stats dict
        # built without this feature's init_class (tests/output_test.py) is safe
        if any(self.reached.values()):
            stats["care_reached"] += 1
        reach_visits = getattr(self.agent, "care_reach_visits", 0)
        if reach_visits:
            stats["care_reach_visits"] += reach_visits
        if getattr(self.agent, "prep_monitoring_visit", False):
            stats["prep_monitoring_visits"] += 1

        for setting_name, offers in getattr(
            self.agent, "care_offer_by_setting", {}
        ).items():
            for service, offered in offers.items():
                if offered:
                    stats[f"care_offer_{setting_name}_{service}"] += 1
