from typing import List, Dict, Optional, Set
import math  # NEW: for exp() in background FOI calculations

from . import base_exposure
from .. import agent
from .. import population
from .. import model
from .. import utils


class HIV(base_exposure.BaseExposure):
    name: str = "hiv"
    stats: List[str] = [
        "hiv",
        "hiv_dx",
        "hiv_aids",
        "hiv_new",
        "hiv_dx_new",
        "hiv_new_inj",
        "hiv_dx_new_inj",
        "hiv_new_sex",
        "hiv_dx_new_sex",
        "hiv_new_background",
        "hiv_dx_new_background",
        # Expected infections (2026-10-09): the sum, over every exposure of a
        # susceptible agent this step, of the probability of conversion the
        # engine then rolled -- the hazard, before the coin flip. Its total
        # has the same expectation as hiv_new and none of the Bernoulli noise,
        # which at Rhode Island's incidence is 60% of the variance of a run's
        # infection count. _averted_prep is the part of the hazard that the
        # agent's PrEP removed: the probability the same exposures would have
        # carried without it, minus the probability they carried.
        "hiv_hazard",
        "hiv_hazard_inj",
        "hiv_hazard_sex",
        "hiv_hazard_background",
        "hiv_hazard_averted_prep",
        "hiv_hazard_averted_prep_inj",
        # source-status attribution of the expected infections (2026-10-13):
        # who the hazard came from, and how much of it the source's viral
        # suppression (adherent ART) would remove
        "hiv_hazard_src_undx",
        "hiv_hazard_src_dx_noart",
        "hiv_hazard_src_art_nonadh",
        "hiv_hazard_src_art_adh",
        "hiv_hazard_avert_sup_undx",
        "hiv_hazard_avert_sup_dx_noart",
        "hiv_hazard_avert_sup_art_nonadh",
    ]
    """
        HIV collects the following stats:

        * hiv - number of agents with active hiv
        * hiv_dx - number of agents with diagnosed hiv
        * hiv_aids - number of agents with aids
        * hiv_new - number of agents converted to hiv this timestep
        * hiv_dx_new - number of agents with diagnosed with hiv this timestep
        * hiv_new_inj / hiv_dx_new_inj - of those, acquired by the injection route
        * hiv_new_sex / hiv_dx_new_sex - acquired by sex on the modelled network
        * hiv_new_background / hiv_dx_new_background - acquired from the
          non-network force of infection

        The `_inj` stats exist to compare against RIDOH's IDU transmission
        category, which is assigned from case-investigation risk-factor
        interviews and counts only infections attributed to injection. Since the
        sexual route went live, `hiv_new` among PWID also contains infections
        RIDOH would classify as GBMSM or heterosexual, so it is not the right
        comparator for that target. `hiv_dx_new_inj` is the closer match of the
        two, because RIDOH counts diagnoses rather than true conversions.

        `_sex` and `_background` were split apart on 2026-08-04. They had been
        pooled as "everything that is not injection", which hid the quantity that
        matters most for a scenario analysis: the background route represents
        partners outside the modelled population, so no modelled intervention can
        touch it. If it carries a large share of infections, every scenario
        contrast is biased toward the null, and until these counters existed
        there was no way to see that. Seeded prevalence is not counted under any
        of the three -- it is recoverable as hiv_new minus their sum.
    """

    ACQUISITION_ROUTES: Set[str] = {"injection", "sex", "background", "seed"}
    """
        How an agent's HIV was acquired. "injection" and "sex" come from a
        network transmission event and carry that event's interaction type;
        "background" is the non-network force of infection, which represents
        partners outside the modelled OUD population and is sexual by
        construction; "seed" is initial prevalence or agent zero, i.e. not
        incident within the run at all.

        The `external_exposure` feature also converts agents, through a generic
        `convert(model)` call it shares with the knowledge and monkeypox
        exposures, so it cannot pass a route without changing their signatures
        too. Those infections therefore land as "seed". The feature defaults to
        false and is not enabled in any RI scenario; if it is ever turned on,
        its infections will be counted in `hiv_new` but excluded from
        `hiv_new_inj`, which is the right side to err on -- an out-of-model
        exposure is not injection-attributed evidence.
    """

    dx_counts: Dict[str, Dict[str, int]] = {}
    """Counts of diagnosed agents by race and sex_type"""

    agents: Set["agent.Agent"] = set()
    """Agents with active hiv"""

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)

        self.active = False
        self.time: Optional[int] = None
        self.dx = False
        self.dx_time: Optional[int] = None
        self.aids = False
        self.acquisition_route: Optional[str] = None
        # per-step expected-infection accumulators (see stats); reset whenever a
        # new time is seen, so the order of expose() and update_agent() within
        # a step does not matter
        self._hazard_time: Optional[int] = None
        self._hazard: Dict[str, float] = {}

    def record_hazard(self, time: int, route: str, p: float, averted: float = 0.0,
                      source: str = "", avert_sup: float = 0.0):
        """
        Accumulate this step's expected infections for a susceptible agent:
        `p`, the conversion probability the engine rolls for one exposure, and
        `averted`, the probability PrEP removed from that exposure. `source`
        (undx / dx_noart / art_nonadh / art_adh) attributes the hazard to the
        HIV-positive partner's care status and `avert_sup` is the part of it
        that the partner's viral suppression would have removed (2026-10-13).
        Pure bookkeeping, no random draw.
        """
        if self._hazard_time != time:
            self._hazard_time = time
            self._hazard = {}
        h = self._hazard
        h["hiv_hazard"] = h.get("hiv_hazard", 0.0) + p
        key = {"injection": "hiv_hazard_inj", "sex": "hiv_hazard_sex",
               "background": "hiv_hazard_background"}.get(route)
        if key:
            h[key] = h.get(key, 0.0) + p
        if averted:
            h["hiv_hazard_averted_prep"] = h.get("hiv_hazard_averted_prep", 0.0) + averted
            if route == "injection":
                h["hiv_hazard_averted_prep_inj"] = h.get("hiv_hazard_averted_prep_inj", 0.0) + averted
        if source:
            h[f"hiv_hazard_src_{source}"] = h.get(f"hiv_hazard_src_{source}", 0.0) + p
            if avert_sup and source != "art_adh":
                h[f"hiv_hazard_avert_sup_{source}"] = h.get(f"hiv_hazard_avert_sup_{source}", 0.0) + avert_sup

    def care_status(self) -> str:
        """The HIV-positive agent's care status for the hazard attribution:
        undx, dx_noart, art_nonadh or art_adh."""
        if not self.dx:
            return "undx"
        haart = getattr(self.agent, "haart", None)
        if haart is None or not getattr(haart, "active", False):
            return "dx_noart"
        return "art_adh" if getattr(haart, "adherent", False) else "art_nonadh"

    @classmethod
    def init_class(cls, params):
        """
        Initialize any diagnosis counts and the agents set.

        args:
            params: parameters for this population
        """
        cls.dx_counts = {
            race: {so: 0 for so in params.classes.sex_types}
            for race in params.classes.races
        }
        cls.agents = set()

        cls._check_sex_acts_reachable(params)
        cls._check_deprecated_background_keys(params)

    @staticmethod
    def _check_deprecated_background_keys(params):
        """
        Refuse to run on `hiv.background_sex.per_act_transmission_prob`.

        The background route now reads the same sex-type/role-stratified
        `partnership.sex.acquisition` table the network route uses, so a params
        file that still sets this key would be silently ignored -- and the key
        was, until 2026-07-30, both sex-type-blind and fitted, which is exactly
        the pair of defects F1 and F9 describe. Same treatment as the
        oud_state window keys.
        """
        bg = getattr(getattr(params, "hiv", None), "background_sex", None)
        if bg is None:
            return

        found = getattr(bg, "per_act_transmission_prob", -1.0)
        if found is not None and found >= 0.0:
            raise ValueError(
                "hiv.background_sex.per_act_transmission_prob is deprecated "
                f"(found {found}). The background force-of-infection now uses "
                "partnership.sex.acquisition[sex_type][sex_role], the same "
                "per-act table as the network sexual route, because it is the "
                "same biological quantity. The old key was a single blended "
                "value applied to every sex type AND a free calibration "
                "parameter, so the exogenous route competed with the network "
                "routes for the PWID incidence target -- see "
                "EPIDEMIC_DYNAMICS_AUDIT_2026-07-29b.md F9. Remove the key; "
                "set the magnitude through rate_nonoud_per_timestep and "
                "partner_hiv_prevalence_nonoud, which are observable inputs."
            )

    @staticmethod
    def _check_sex_acts_reachable(params):
        """
        Raise if a sex-carrying bond can never generate a single sex act.

        `partnership.sex.frequency` defaults to a single bin of `min: 0, max: 1`,
        and `utils.safe_random_int` returns a value on the half-open `[min, max)`,
        so that default yields exactly zero acts forever.
        `BaseInteraction.interact` then returns before calling `expose`, and the
        entire sexual transmission pathway is silently absent -- no error, no
        warning, just an injection-only epidemic. Every RI parameter file omitted
        this block and so ran that way (see EPIDEMIC_DYNAMICS_MAP_2026-07-29.md,
        finding P1).

        Sexual transmission being off is a legitimate modelling choice, but it has
        to be an explicit one, so require the zero to be written down as a zero.
        """
        sex_bonds = [
            bond
            for bond, bond_def in params.classes.bond_types.items()
            if "sex" in bond_def.acts_allowed
        ]

        dead = []
        for bond in sex_bonds:
            freq = params.partnership.sex.frequency[bond]
            if freq.type == "bins":
                # reachable iff some bin can return >= 1, i.e. has max > 1
                if not any(b.max > 1 for b in freq.bins.values()):
                    dead.append(
                        f"{bond} (bins, all max <= 1; safe_random_int is "
                        f"half-open so these yield 0 acts)"
                    )
            # A `distribution` form is never dead: writing one out is itself the
            # explicit choice this guard is asking for, including the all-zero
            # case that means "injection-only epidemic, on purpose".

        if dead and len(dead) == len(sex_bonds):
            raise ValueError(
                "partnership.sex.frequency can never produce a sex act for any "
                f"sex-carrying bond ({'; '.join(dead)}), so sexual HIV "
                "transmission would be silently disabled. Set an explicit "
                "frequency, or -- if an injection-only epidemic is intended -- "
                "say so with `type: distribution` and a zero-valued "
                "distribution, which is accepted."
            )

    def init_agent(self, pop: "population.Population", time: int):
        """
        Initialize the agent for this exposure during population initialization (`Population.create_agent`).  Called on only exposures that are enabled per the params.

        Based on demographic params for the agent, stochastically determine if hiv is active, and if active, at what past time point was the agent converted, if the agent is diagnosed, and if the agent has aids.

        args:
            pop: the population this agent is a part of
            time: the current time step
        """
        agent_params = (
            self.agent.location.params.demographics[self.agent.race]
            .sex_type[self.agent.sex_type]
            .drug_type[self.agent.drug_type]
        )

        # HIV
        if (
            pop.pop_random.random() < agent_params.hiv.init
            and time >= pop.params.hiv.start_time
        ):
            self.active = True
            # Prevalent at population creation, not incident within the run, so
            # it must not count toward the injection-attributed incidence stats.
            self.acquisition_route = "seed"

            # if HIV, when did the agent convert? Random sample
            self.time = utils.safe_random_int(
                time - self.agent.location.params.hiv.max_init_time,
                time,
                pop.pop_random,
            )

            if pop.pop_random.random() < agent_params.hiv.aids.init:
                self.aids = True

            if pop.pop_random.random() < agent_params.hiv.dx.init:
                self.dx = True
                # agent was diagnosed at a random time between conversion and now
                self.dx_time = utils.safe_random_int(self.time, time, pop.pop_random)

            # add agent to class
            self.add_agent(self.agent)

    def update_agent(self, model: "model.TITAN"):
        """
        Update the agent for this exposure for a time step.  Called once per time step in `TITAN.update_all_agents`. Agent level updates are done after population level updates.   Called on only exposures that are enabled per the params.

        If the agent is hiv+ and the model time is past the hiv start_time, determine if the agent becomes diagnosed if not yet diagnosed, and if the agent has progressed to aids.

        If the agent is hiv- and the model time is past the hiv start_time, apply background
        risk of HIV acquisition from non-OUD partners (force-of-infection term), in addition
        to any relationship-based exposures handled via `expose`.
        """
        # NEW: Do nothing before HIV modeling starts
        if model.time < model.params.hiv.start_time:
            return

        # Case 1: agent is already HIV+  --> keep existing behavior
        if self.active:
            # This implicit per-timestep diagnosis hazard is the legacy stand-in
            # for a testing process, from before `hiv_testing` existed. Once
            # `hiv_testing` is enabled it is the explicit testing/diagnosis
            # mechanism (see its own `update_agent`), so this pathway steps
            # aside rather than diagnosing the same agent a second, independent
            # way. Kept active for scenarios that still run without
            # `hiv_testing` enabled.
            if not self.dx and not model.params.features.hiv_testing:
                # Top-level hiv.dx.prob (calibration override) takes precedence;
                # else fall back to demographic-level (legacy) value. Mirrors the
                # moud.post_overdose_rr / moud.discontinue_per_month pattern in moud.py.
                hiv_global = getattr(model.params, "hiv", None)
                dx_override = getattr(getattr(hiv_global, "dx", None), "prob", -1.0)
                if dx_override is not None and dx_override >= 0.0:
                    test_prob = float(dx_override)
                else:
                    test_prob = (
                        self.agent.location.params.demographics[self.agent.race]
                        .sex_type[self.agent.sex_type]
                        .drug_type[self.agent.drug_type]
                        .hiv.dx.prob
                    )

                # Rescale based on calibration param
                test_prob *= model.calibration.test_frequency

                if model.run_random.random() < test_prob:
                    self.diagnose(model)

            self.progress_to_aids(model)
            return

        # Case 2: agent is HIV-  --> apply background FOI from non-OUD partners
        p_background = self.get_background_foi_probability(model)  # NEW

        # expected-infection bookkeeping (no random draw)
        averted = 0.0
        if getattr(getattr(self.agent, "prep", None), "active", False):
            averted = max(0.0, self.get_background_foi_probability(model, skip_prep=True) - p_background)
        self.record_hazard(model.time, "background", p_background, averted)

        if p_background > 0.0 and model.run_random.random() < p_background:
            # Background FOI stands in for partners outside the modelled OUD
            # population and is sexual by construction, so it is not attributed
            # to the injection route.
            self.convert(model, "background")

    @classmethod
    def add_agent(cls, agent: "agent.Agent"):
        """
        Add an agent to the class (not instance).  This can be useful if tracking population level statistics or groups, such as counts or newly active agents.

        Add the agent to the `agents` set and if the agent is diagnosed, updated the `dx_counts`

        args:
            agent: the agent to add to the class attributes
        """
        cls.agents.add(agent)

        if agent.hiv.dx:  # type: ignore[attr-defined]
            cls.dx_counts[agent.race][agent.sex_type] += 1

    @classmethod
    def remove_agent(cls, agent: "agent.Agent"):
        """
        Remove an agent from the class (not instance).  This can be useful if tracking population level statistics or groups, such as counts.

        Remove the agent from the `agents` set and decrement the `dx_counts` if the agent was diagnosed.

        args:
            agent: the agent to remove from the class attributes
        """
        cls.agents.remove(agent)

        if agent.hiv.dx:  # type: ignore[attr-defined]
            cls.dx_counts[agent.race][agent.sex_type] -= 1

    def set_stats(self, stats: Dict[str, int], time: int):
        if self._hazard_time == time:
            for key, value in self._hazard.items():
                stats[key] += value
        if self.active:
            stats["hiv"] += 1
            # One counter per incident route. "seed" is deliberately not counted:
            # it is prevalence at initialisation, not an event, and it is
            # recoverable as hiv_new minus the three incident routes.
            route = self.acquisition_route
            suffix = {
                "injection": "_inj",
                "sex": "_sex",
                "background": "_background",
            }.get(route)
            if self.time == time:
                stats["hiv_new"] += 1
                if suffix:
                    stats[f"hiv_new{suffix}"] += 1
            if self.aids:
                stats["hiv_aids"] += 1
            if self.dx:
                stats["hiv_dx"] += 1
                if self.dx_time == time:
                    stats["hiv_dx_new"] += 1
                    if suffix:
                        stats[f"hiv_dx_new{suffix}"] += 1

    @staticmethod
    def expose(
        model: "model.TITAN",
        interaction: str,
        rel: "agent.Relationship",
        num_acts: int,
    ):
        """
        Expose a relationship to the exposure for a number of acts of a specific interaction type.  Typically, this is determining if the exposure can cause conversion/change in one of the agents, then if so determining the probability of that and then converting the succeptible agent.

        For hiv, one agent must be active and the other not for an exposure to cause conversion.

        args:
            model: The running model
            interaction: The type of interaction (e.g. sex, injection)
            rel: The relationship where the interaction is occuring
            num_acts: The number of acts of that interaction
        """
        # Agent 1 is HIV+, Agent 2 is not, Agent 2 is succept
        if rel.agent1.hiv.active and not rel.agent2.hiv.active:  # type: ignore[attr-defined]
            agent_ = rel.agent1
            partner = rel.agent2
        # If Agent 2 is HIV and Agent 1 is not, Agent 1 is succept
        elif not rel.agent1.hiv.active and rel.agent2.hiv.active:  # type: ignore[attr-defined]
            agent_ = rel.agent2
            partner = rel.agent1
        else:  # neither agent is HIV or both are
            return

        p = agent_.hiv.get_transmission_probability(  # type: ignore[attr-defined]
            model, interaction, partner, num_acts
        )

        # expected-infection bookkeeping (no random draw): the hazard this
        # exposure carries, and the share PrEP removed from it
        averted = 0.0
        if getattr(getattr(partner, "prep", None), "active", False):
            p_unprotected = agent_.hiv.get_transmission_probability(  # type: ignore[attr-defined]
                model, interaction, partner, num_acts, skip_prep=True
            )
            averted = max(0.0, p_unprotected - p)
        # who the hazard came from, and what the source's suppression would remove
        source = agent_.hiv.care_status()  # type: ignore[attr-defined]
        avert_sup = 0.0
        if source != "art_adh" and p > 0.0:
            p_suppressed = agent_.hiv.get_transmission_probability(  # type: ignore[attr-defined]
                model, interaction, partner, num_acts, force_suppressed=True
            )
            avert_sup = max(0.0, p - p_suppressed)
        partner.hiv.record_hazard(model.time, interaction, p, averted, source=source, avert_sup=avert_sup)  # type: ignore[attr-defined]

        if model.run_random.random() < p:
            # if agent HIV+ partner becomes HIV+
            partner.hiv.convert(model, interaction)  # type: ignore[attr-defined]

    def get_transmission_probability(
        self,
        model: "model.TITAN",
        interaction: str,
        partner: "agent.Agent",
        num_acts: int,
        skip_prep: bool = False,
        force_suppressed: bool = False,
    ) -> float:
        """
        Determines the probability of an hiv transmission event from agent to partner based on
            interaction type and numer of acts. For sex acts, transmission probability is a
            function of the acquisition probability of the HIV- agent's sex role
            and the HIV+ agent's haart adherence, acute status, and dx risk reduction.
            force_suppressed evaluates the HIV+ agent as if on adherent HAART
            (the counterfactual for the hazard attribution; bookkeeping only).

        args:
            model: The running model
            interaction : "injection" or "sex"
            partner: HIV- Agent
            num_acts: The number of exposure interactions the agents had this time step

        returns:
            probability of transmission from agent to partner
        """
        # if this isn't an interaction where hiv can transmit, return 0% prob
        if interaction not in ("injection", "sex"):
            return 0.0

        # Logic for if needle or sex type interaction
        p: float

        # get baseline probabilities
        if interaction == "injection":
            p = model.params.partnership.injection.transmission.base
        elif interaction == "sex":
            agent_sex_role = self.agent.sex_role
            partner_sex_role = partner.sex_role

            # get partner's sex role during acts
            if partner_sex_role == "versatile":  # versatile partner takes
                # "opposite" position of agent
                if agent_sex_role == "insertive":
                    partner_sex_role = "receptive"
                elif agent_sex_role == "receptive":
                    partner_sex_role = "insertive"
                else:
                    partner_sex_role = "versatile"  # if both versatile, can switch
                    # between receptive and insertive by act

            # get probability of sex acquisition given HIV- partner's position
            p = partner.location.params.partnership.sex.acquisition[partner.sex_type][
                partner_sex_role
            ]

        # feature specific risk adjustment
        for feature in model.features:
            agent_feature = getattr(self.agent, feature.name)
            if force_suppressed and feature.name == "haart":
                p *= agent_feature.get_transmission_risk_multiplier(self.time, interaction, adherent=True)
            else:
                p *= agent_feature.get_transmission_risk_multiplier(self.time, interaction)

            partner_feature = getattr(partner, feature.name)
            # the partner's protection at the CURRENT step (2026-10-02): this
            # passed self.time, the source agent's infection time, which every
            # time-dependent acquisition multiplier (injectable PrEP decay,
            # vaccine waning) would have evaluated at the wrong moment; oral
            # PrEP ignores the argument, so calibrated results are unchanged
            if skip_prep and feature.name == "prep":
                continue  # the counterfactual without PrEP (expected-infection bookkeeping)
            p *= partner_feature.get_acquisition_risk_multiplier(model.time, interaction)

        # Scaling parameter for acute HIV infections
        if self.get_acute_status(model.time):
            p *= self.agent.location.params.hiv.acute.infectivity

        # Scaling parameter for positively identified HIV agents
        if self.dx:
            p *= 1 - self.agent.location.params.hiv.dx.risk_reduction[interaction]

        # Racial calibration parameter to attain proper race incidence disparity
        p *= partner.location.params.demographics[partner.race].hiv.transmission

        # Scaling parameter for per act transmission.
        p *= model.calibration.acquisition

        return utils.total_probability(p, num_acts)

    def convert(self, model: "model.TITAN", acquisition_route: str = "seed"):
        """
        Agent becomes HIV agent. Update all appropriate attributes, sets and dictionaries.

        args:
            model: The model being run
            acquisition_route: how the agent acquired HIV, one of
                ACQUISITION_ROUTES. Recorded so incidence can be attributed to
                the injection route and compared against RIDOH's IDU
                transmission category. Defaults to "seed" so that callers
                representing initial prevalence or agent zero -- which are not
                incident infections -- do not have to pass it explicitly.
        """
        if acquisition_route not in self.ACQUISITION_ROUTES:
            raise ValueError(
                f"unknown HIV acquisition_route {acquisition_route!r}; "
                f"expected one of {sorted(self.ACQUISITION_ROUTES)}"
            )

        if not self.active:
            self.active = True
            self.time = model.time
            self.acquisition_route = acquisition_route
            self.agent.vaccine.active = False  # type: ignore[attr-defined]
            self.add_agent(self.agent)

        if self.agent.prep.active:  # type: ignore[attr-defined]
            self.agent.prep.progress(model, force=True)  # type: ignore[attr-defined]

    def diagnose(self, model: "model.TITAN"):
        """
        Mark the agent as diagnosed.

        args:
             model: the running model
        """
        self.dx = True
        self.dx_time = model.time
        self.add_agent(self.agent)

    # ============================ HELPER METHODS ==============================

    def get_acute_status(self, time: int) -> bool:
        """
        Get acute status of agent at time

        args:
            time: The current time step

        returns:
            whether an agent is acute
        """
        if self.active and self.time is not None:
            hiv_duration = time - self.time

            if self.agent.location.params.hiv.acute.duration >= hiv_duration >= 0:
                return True

        return False

    def progress_to_aids(self, model: "model.TITAN"):
        """
        Model the progression of HIV agents to AIDS agents

        args:
             model: the running model
        """
        aids_prob = self.agent.location.params.hiv.aids.prob
        p = self.agent.haart.aids_scale()  # type: ignore[attr-defined]

        if model.run_random.random() < p * aids_prob:
            self.aids = True

    def _background_override(self, bg):
        """The agent's sex-type entry of hiv.background_sex.by_sex_type, or None.

        Tolerates both the engine's ObjMap (item access) and the plain
        namespaces the unit tests build (attribute access).
        """
        by_sex = getattr(bg, "by_sex_type", None)
        if by_sex is None:
            return None
        sex_type = getattr(self.agent, "sex_type", None)
        if sex_type is None:
            return None
        try:
            return by_sex[sex_type]
        except (KeyError, TypeError, IndexError):
            return getattr(by_sex, sex_type, None)

    # NEW: helper for background HIV risk from non-OUD partners
    def get_background_foi_probability(self, model: "model.TITAN", skip_prep: bool = False) -> float:
        """
        Compute per-timestep HIV acquisition probability from non-OUD partners
        (background force-of-infection) for HIV-susceptible agents.

        Uses parameters in location.params.hiv.background_sex. If that block is not
        defined, returns 0.0 and has no effect.
        """
        # If no background_sex params defined, do nothing
        hiv_params = self.agent.location.params.hiv
        bg = getattr(hiv_params, "background_sex", None)
        if bg is None:
            return 0.0

        # Average number of sex acts with non-OUD partners per timestep
        rate_nonoud = bg.rate_nonoud_per_timestep

        # HIV prevalence among those non-OUD partners
        prev_nonoud = bg.partner_hiv_prevalence_nonoud

        # Per-act transmission probability for these acts.
        #
        # This is the same biological quantity the network route reads out of
        # partnership.sex.acquisition, so it is looked up the same way rather
        # than carried as an independent scalar. Until 2026-07-30 this was
        # bg.per_act_transmission_prob -- one sex-type-blind number, and one that
        # sat in parameters_to_optimize, which meant:
        #   * it reproduced finding F1 (a single blended per-act value applied to
        #     everyone) inside the background route after F1 had been fixed in the
        #     network route -- it was fitted to 0.002805, between the heterosexual
        #     value (0.0004-0.0008) and MSM receptive (0.015), and applied to both;
        #   * the exogenous route had its own free lever competing with
        #     calibration.sex.act and partnership.injection.transmission.base for
        #     the same PWID incidence target. It won that competition: 12 of 21
        #     infections in the post-rebuild run arrived this way. An exogenous
        #     route is the one route no modelled intervention can touch, so a
        #     fitted one biases every scenario contrast toward the null.
        # See EPIDEMIC_DYNAMICS_AUDIT_2026-07-29b.md F9 and the 2026-07-30
        # calibration-design writeup.
        #
        # There is no tracked source agent, so the susceptible's own role is used
        # as-is; `versatile` is already the blended entry in that table.

        # Sex-type overrides (added 2026-09-29). MSM agents' non-OUD partners are
        # drawn from the MSM community, not from all Rhode Island adults, and
        # they have more of them: the all-adult prevalence and rate applied to
        # MSM left MSM who inject unable to acquire HIV from outside the cohort,
        # which is the reservoir RIDOH's MSM/IDU category describes. A zero in
        # either field means "use the scalar", so a scenario without the block
        # runs exactly as before.
        override = self._background_override(bg)
        if override is not None:
            prev_override = float(getattr(override, "partner_hiv_prevalence", 0.0) or 0.0)
            rate_override = float(getattr(override, "rate_per_timestep", 0.0) or 0.0)
            if prev_override > 0.0:
                prev_nonoud = prev_override
            if rate_override > 0.0:
                rate_nonoud = rate_override

        # Behavior multiplier for people with injecting OUD (if you use drug_type == "Inj")
        #
        # MSM/MSMW who inject take calibration.background.msm_inj_mult instead
        # when it is set (>= 0), added 2026-09-29: RIDOH's MSM/IDU category is
        # most PWID diagnoses in RI, and it is sexual acquisition from the MSM
        # community by men who also inject -- an outside exposure the all-agent
        # inj_multiplier does not describe.
        if getattr(self.agent, "drug_type", None) == "Inj":
            inj_mult = bg.inj_multiplier
            if getattr(self.agent, "sex_type", None) in ("MSM", "MSMW"):
                cal_bg = getattr(getattr(model, "calibration", None), "background", None)
                msm_inj = getattr(cal_bg, "msm_inj_mult", -1.0) if cal_bg is not None else -1.0
                if msm_inj is not None and msm_inj >= 0.0:
                    inj_mult = msm_inj
            rate_nonoud *= inj_mult

        # Expected number of acts with HIV+ non-OUD partners in this timestep
        expected_acts_with_hiv_pos = rate_nonoud * prev_nonoud

        if expected_acts_with_hiv_pos <= 0.0:
            return 0.0

        sex_role = getattr(self.agent, "sex_role", "versatile")
        acquisition = self.agent.location.params.partnership.sex.acquisition
        p_per_act = acquisition[self.agent.sex_type][sex_role]

        if p_per_act <= 0.0:
            return 0.0

        # Force-of-infection style: probability of at least one transmission event
        # assuming Poisson number of effective acts
        lambda_eff = expected_acts_with_hiv_pos * p_per_act
        p_background = 1.0 - math.exp(-lambda_eff)

        # Feature-specific acquisition risk adjustment (e.g. PrEP, vaccine) -
        # mirrors the relationship-pathway loop in get_transmission_probability.
        # There's no tracked source agent for a background/non-network partner,
        # so only the susceptible agent's own acquisition-side multipliers
        # apply; use model.time rather than self.time since self.time is only
        # set once this (currently HIV-) agent converts.
        for feature in model.features:
            if skip_prep and feature.name == "prep":
                continue  # the counterfactual without PrEP (expected-infection bookkeeping)
            agent_feature = getattr(self.agent, feature.name)
            p_background *= agent_feature.get_acquisition_risk_multiplier(
                model.time, "sex"
            )

        # Optionally scale by acquisition calibration (to keep in line with other transmission)
        p_background *= model.calibration.acquisition

        # Clip to [0,1] for safety
        if p_background < 0.0:
            p_background = 0.0
        elif p_background > 1.0:
            p_background = 1.0

        return p_background
