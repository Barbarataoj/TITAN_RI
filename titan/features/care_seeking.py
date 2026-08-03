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

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)

        self.active = False
        self.provider_id: Dict[str, int] = {}
        self.visit_frailty: Dict[str, float] = {}

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

        cls.stats = ["care_visits"]
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
        cls.process_attrition(care_params, model)
        cls.resolve_all_visits(care_params, model)

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

        for setting_name, setting in care_params.setting.items():
            if setting.visit_rate_annual <= 0:
                continue

            provider_idx = self.provider_id.get(setting_name)
            if provider_idx is None:
                continue

            rate = self.visit_rate(setting, setting_name, steps_per_year)

            # visit_rate_annual is a count of visit-days, not a probability, so
            # the number of visits in a step is drawn as a Poisson count.
            # Rolling a single yes/no instead would cap an agent at one visit
            # per step and undercount the busier settings, where an agent can
            # plausibly attend more than once a month.
            n_visits = int(model.np_random.poisson(rate))
            if n_visits == 0:
                continue

            self.agent.care_encounter[setting_name] = n_visits

            # each visit is a separate chance to be offered a service
            for _ in range(n_visits):
                self.resolve_offers(setting_name, provider_idx, model)

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

    def resolve_offers(self, setting_name: str, provider_idx: int, model: "model.TITAN"):
        """
        Work out which services the agent's provider in this setting offered
        during a visit, recording both the collapsed (`care_offer`) and
        setting-attributed (`care_offer_by_setting`) views.

        An offer requires both that the provider is willing to deliver the
        service at all, and that they do so at this particular visit. The first
        is the term a provider-directed intervention moves.

        args:
            setting_name: the setting the visit happened in
            provider_idx: index of the agent's provider in this setting's pool
            model: the instance of TITAN currently being run
        """
        pool = self.providers.get(setting_name, [])
        if provider_idx >= len(pool):
            return

        provider = pool[provider_idx]
        setting_offers = self.agent.care_offer_by_setting.setdefault(
            setting_name, {service: False for service in SERVICES}
        )
        for service in SERVICES:
            if not provider[service]["willing"]:
                continue
            if model.run_random.random() < provider[service]["intensity"]:
                self.agent.care_offer[service] = True
                setting_offers[service] = True

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

        for setting_name, offers in getattr(
            self.agent, "care_offer_by_setting", {}
        ).items():
            for service, offered in offers.items():
                if offered:
                    stats[f"care_offer_{setting_name}_{service}"] += 1
