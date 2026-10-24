from . import base_feature
from .sdh import SDH
import logging
import math


class OUDState(base_feature.BaseFeature):
    """
    Natural history of Opioid Use Disorder (OUD) and injection behavior.

    Responsibilities:
    - OUD incidence, remission, relapse
    - Injection ↔ non-injection transitions
    - PWID population bookkeeping
    - Exposes OUD stats for reporting

    Interoperability:
    - Reads (but does NOT control): agent.on_moud
    - Exposes: agent.has_oud, agent.is_in_remission, agent.drug_type
    - Maintains: pop.pwid_agents
    """

    name = "oud_state"
    stats = [
        "oud_active",
        "oud_remission",
        "oud_none",
        # Look-back PWID denominator -- see set_stats. Reported, not scored.
        "pwid_recent",
        "pwid_recent_hiv",
    ]

    @classmethod
    def init_class(cls, params):
        """
        Warn if a non-injection bond's partner counts differ across drug_type
        blocks.

        `Population.sync_injection_partner_targets` only re-derives
        injection-capable bonds when drug_type changes, so a sex- or PCA-only
        bond configured differently per drug_type keeps its creation-time value
        for the agent's whole life. That is a deliberate trade-off (see that
        method's docstring), but it would silently ignore configured values, so
        surface it once here.
        """
        cls.audited_partner_targets = False

        cls._check_deprecated_window_keys(params)

        try:
            inj_bonds = {
                bond
                for bond, bond_def in params.classes.bond_types.items()
                if "injection" in bond_def.acts_allowed
            }
            other_bonds = [
                b for b in params.classes.bond_types if b not in inj_bonds
            ]

            for race, race_params in params.demographics.items():
                for sex_type, st_params in race_params.sex_type.items():
                    blocks = st_params.drug_type
                    for bond in other_bonds:
                        sigs = {
                            dt: cls._num_partners_signature(blk.num_partners[bond])
                            for dt, blk in blocks.items()
                            if bond in blk.num_partners
                        }
                        if len(set(sigs.values())) > 1:
                            logging.warning(
                                "demographics.%s.sex_type.%s specifies different "
                                "num_partners for the non-injection bond '%s' "
                                "across drug_types (%s). oud_state only "
                                "re-derives injection-capable bonds on a "
                                "drug_type transition, so agents who start or "
                                "stop injecting will keep the value drawn at "
                                "creation for this bond.",
                                race,
                                sex_type,
                                bond,
                                ", ".join(sorted(sigs)),
                            )
                            return  # one warning is enough
        except (AttributeError, KeyError, TypeError):
            # purely diagnostic - never block a run over it
            pass

    @staticmethod
    def _check_deprecated_window_keys(params):
        """
        Refuse to run on the raw-timestep window keys.

        Both were given weekly-step values while the model runs monthly, so a
        params file that still sets them would silently apply a ~4x-too-long
        window. Same treatment as `hiv_testing.moud_early_window`.
        """
        transitions = getattr(getattr(params, "oud_state", None), "transitions", None)
        if transitions is None:
            return

        for legacy, replacement, effect in (
            (
                "early_relapse_window",
                "early_relapse_window_months",
                "an elevated-relapse window covering ~2.2 years of remission "
                "instead of ~6 months, inflating the realized relapse hazard "
                "by ~1.95x",
            ),
            (
                "relapse_risk_window",
                "relapse_risk_window_months",
                "an ~8-month post-relapse tolerance-loss window instead of "
                "~8 weeks",
            ),
        ):
            found = getattr(transitions, legacy, -1)
            if found is not None and found >= 0:
                raise ValueError(
                    f"oud_state.transitions.{legacy} is deprecated: it was "
                    "expressed in raw timesteps with a weekly-step default, so "
                    f"on this model's monthly steps it produced {effect}. Use "
                    f"'{replacement}' instead and refit. "
                    f"(found {legacy}: {found})"
                )

    @staticmethod
    def _months_to_steps(transitions, key, steps_per_year, default_months):
        """
        Read a MONTHS-denominated window and return it in timesteps.
        """
        months = getattr(transitions, key, default_months)
        return float(months) * steps_per_year / 12.0

    @staticmethod
    def _num_partners_signature(dist_info):
        """
        Hashable identity of a partner-count distribution definition.

        Definitions that always yield zero collapse to one signature: the
        shipped params write "no partners of this type" as both `set_value 0`
        and `poisson` with mean 0, and that is not a meaningful difference.
        """
        values = tuple(
            (key, dist_info.vars[key].value, dist_info.vars[key].value_type)
            for key in sorted(dist_info.vars.keys())
        )

        def is_zero(value):
            try:
                return float(value) == 0.0
            except (TypeError, ValueError):
                return False

        if values and all(is_zero(value) for _, value, _ in values):
            return ("always-zero",)

        return (dist_info.dist_type, values)

    @classmethod
    def update_pop(cls, model):
        """
        Advance and expire the recent-relapse window for every agent.

        Runs here rather than in this feature's own `update_agent` so the
        decrement happens before any feature's per-agent update this step
        (`update_pop` for every feature runs before `update_agent` for any
        feature). moud/hiv_testing are declared earlier than oud_state in
        `titan/features/__init__.py`, so decrementing in `update_agent`
        left the flag visible for one step longer than
        `relapse_risk_window` intends.

        args:
            model: the instance of TITAN currently being run
        """
        if not getattr(cls, "audited_partner_targets", False):
            cls.audited_partner_targets = True
            cls._audit_partner_targets(model)

        steps_per_year = model.params.model.time.steps_per_year
        for a in model.pop.all_agents:
            agent_feature = getattr(a, cls.name)
            if not agent_feature.active:
                continue
            if not getattr(a, "recent_relapse", False):
                continue

            oud_params = getattr(a.location.params, "oud_state", None)
            if oud_params is None or not hasattr(oud_params, "transitions"):
                continue

            a.relapse_timer += 1
            # configured in months; relapse_timer counts timesteps
            relapse_window = cls._months_to_steps(
                oud_params.transitions,
                "relapse_risk_window_months",
                steps_per_year,
                2.0,  # ~8 weeks
            )
            if a.relapse_timer > relapse_window:
                a.recent_relapse = False
                a.relapse_timer = 0

    @classmethod
    def _audit_partner_targets(cls, model):
        """
        Warn once if any non-injecting agent holds nonzero injection-bond
        partner targets.

        `Population.create_agent` asserts this cannot happen at creation, and
        `sync_injection_partner_targets` maintains it across transitions, so a
        violation means the population was built by an older revision -- most
        likely baseline OUD-remission agents that were drawn as `Inj`, given
        `Inj` partner targets, and only then overwritten to `NonInj`. Those
        agents hold injection targets they can never fill.

        Deliberately warn-only. Re-deriving would overwrite restored state with
        fresh random draws, silently changing a saved population that callers
        expect to be reproducible; the population needs regenerating instead.

        Only the non-injecting direction is checked: an `Inj` agent with all-zero
        injection targets is not necessarily wrong, since the configured
        partner-count distributions can legitimately draw zero.
        """
        try:
            inj_bonds = [
                bond
                for bond, bond_def in model.params.classes.bond_types.items()
                if "injection" in bond_def.acts_allowed
            ]
            if not inj_bonds:
                return

            stale = [
                a
                for a in model.pop.all_agents
                if a.drug_type != "Inj"
                and any(a.mean_num_partners.get(b, 0) > 0 for b in inj_bonds)
            ]
            if stale:
                logging.warning(
                    "%d non-injecting agents hold nonzero injection-bond "
                    "partner targets, which Population.create_agent forbids at "
                    "creation. This population was almost certainly saved by a "
                    "revision that overwrote drug_type after deriving partner "
                    "targets; those targets can never be filled. Regenerate the "
                    "saved population rather than running scenarios off it.",
                    len(stale),
                )
        except (AttributeError, KeyError, TypeError):
            # purely diagnostic - never block a run over it
            pass

    def __init__(self, agent):
        super().__init__(agent)
        self.active = True
        # drug_type this agent's injection-bond partner targets were last
        # derived from; see _enforce_state_consistency.
        self.synced_drug_type = getattr(agent, "drug_type", None)

    # =========================================================
    # Initialization
    # =========================================================
    def init_agent(self, pop, time):
        """
        Initialize OUD-related memory.
        Baseline has_oud and drug_type are set upstream.
        """
        self.active = True  # REQUIRED

        if not hasattr(self.agent, "is_in_remission"):
            self.agent.is_in_remission = False

        if not hasattr(self.agent, "ever_injected"):
            self.agent.ever_injected = (self.agent.drug_type == "Inj")

        self.agent.time_in_remission = 0
        self.agent.time_injecting = 0

        # Seed the look-back stamp so baseline injectors are already inside the
        # window at t=0 rather than taking a year to enter it.
        if not hasattr(self.agent, "last_injection_time"):
            self.agent.last_injection_time = (
                time if self.agent.drug_type == "Inj" else None
            )

        # Relapse tracking (loss of tolerance)
        self.agent.recent_relapse = False
        self.agent.relapse_timer = 0

        # Reconcile has_oud / drug_type at baseline.
        # population.py independently samples has_oud (init_active/_remission)
        # and assigns drug_type from demographics (Inj/NonInj, never None).
        # We keep those independent — an agent can be a PWID without an OUD
        # diagnosis — but enforce two invariants:
        #   * remission agents must have has_oud=False (population sets True)
        #   * an OUD agent with no drug_type (shouldn't happen, but defensive)
        #     defaults to NonInj
        drug = getattr(self.agent, "drug_type", None)
        in_remission = getattr(self.agent, "is_in_remission", False)
        has_oud = getattr(self.agent, "has_oud", False)

        if in_remission:
            self.agent.has_oud = False
        elif has_oud and drug is None:
            self.agent.drug_type = "NonInj"

        # Population.create_agent has already derived partner targets from the
        # agent's final drug_type by this point (partnerships are initialized
        # before features), so record it rather than re-deriving.
        self.synced_drug_type = self.agent.drug_type

        self._enforce_state_consistency(pop)

    # =========================================================
    # Per-step update
    # =========================================================
    def update_agent(self, model):
        self._update_agent(model)

        # Stamp the last step on which the agent was injecting. `time_injecting`
        # cannot answer this -- it resets to 0 on every stop -- and drug_type is
        # instantaneous, flipping to NonInj the moment an agent enters remission.
        # A look-back is what the APCD target's denominator actually is: an
        # injection-related code anywhere in claims year Y puts a person in the
        # year-Y PWID denominator, and nothing takes them out again until the
        # next year. See set_stats.
        if self.agent.drug_type == "Inj":
            self.agent.last_injection_time = model.time

    def _update_agent(self, model):
        oud_params = getattr(self.agent.location.params, "oud_state", None)
        if oud_params is None or not hasattr(oud_params, "transitions"):
            return

        transitions = oud_params.transitions
        pop = model.pop

        steps_per_year = model.params.model.time.steps_per_year
        dt_years = 1.0 / steps_per_year

        def hazard_to_prob(h):
            return 0.0 if h <= 0 else 1.0 - math.exp(-h * dt_years)

        # ---------- Read shared states ----------
        has_oud = getattr(self.agent, "has_oud", False)
        in_remission = getattr(self.agent, "is_in_remission", False)
        drug_type = getattr(self.agent, "drug_type", None)
        on_moud = getattr(self.agent, "on_moud", False)

        # ---------- Rates ----------
        p_incidence = hazard_to_prob(
            getattr(transitions, "incidence_rate_per_year", 0.0)
        )

        remission_rate = transitions.remission_rate_per_year
        if on_moud:
            remission_rate *= getattr(transitions, "moud_remission_rr", 1.0)
        p_remission = hazard_to_prob(remission_rate)

        relapse_rate = getattr(transitions, "relapse_rate_per_year", 0.0)
        # configured in months; time_in_remission counts timesteps
        early_window = self._months_to_steps(
            transitions, "early_relapse_window_months", steps_per_year, 0.0
        )
        if self.agent.time_in_remission < early_window:
            relapse_rate *= getattr(transitions, "early_relapse_rr", 1.0)
        p_relapse = hazard_to_prob(relapse_rate)

        p_inject_start = hazard_to_prob(transitions.inject_start_rate_per_year)
        p_inject_stop = hazard_to_prob(transitions.inject_stop_rate_per_year)

        # =====================================================
        # CASE 0: No OUD (incidence)
        # =====================================================
        if not has_oud and not in_remission:
            if model.run_random.random() < p_incidence:
                self.agent.has_oud = True
                self.agent.is_in_remission = False
                self.agent.drug_type = "NonInj"

            self._enforce_state_consistency(pop)
            return

        # =====================================================
        # CASE 1: In remission → possible relapse
        # =====================================================
        if in_remission:
            self.agent.time_in_remission += 1

            if model.run_random.random() < p_relapse:
                self.agent.has_oud = True
                self.agent.is_in_remission = False
                self.agent.time_in_remission = 0

                # Mark recent relapse (loss of tolerance)
                self.agent.recent_relapse = True
                self.agent.relapse_timer = 0

                # Relapse may return to injecting, if the agent injected recently
                # enough to still be eligible. See _inject_relapse_eligible.
                if (
                    self._inject_relapse_eligible(model, transitions)
                    and model.run_random.random()
                    < getattr(transitions, "relapse_to_inj_prob", 0.0)
                ):
                    self.agent.drug_type = "Inj"
                else:
                    self.agent.drug_type = "NonInj"

            self._enforce_state_consistency(pop)
            return

        # =====================================================
        # CASE 2: Active OUD → possible remission
        # =====================================================
        if has_oud and model.run_random.random() < p_remission:
            self.agent.has_oud = False
            self.agent.is_in_remission = True
            self.agent.time_in_remission = 0
            self.agent.drug_type = "NonInj"

            # Clear relapse risk on remission
            self.agent.recent_relapse = False
            self.agent.relapse_timer = 0

            self._enforce_state_consistency(pop)
            return

        # =====================================================
        # CASE 3: Injection dynamics (active OUD only)
        # =====================================================
        if has_oud:
            if drug_type == "Inj":
                self.agent.time_injecting += 1
                if model.run_random.random() < p_inject_stop:
                    self.agent.drug_type = "NonInj"
            else:
                self.agent.time_injecting = 0
                # hardship raises the chance of starting to inject (SDH feature, 2026-10-24; relative, mean-preserving)
                if model.run_random.random() < p_inject_start * SDH.risk_mult(self.agent, model.params, "inject_start"):
                    self.agent.drug_type = "Inj"
                    self.agent.ever_injected = True

        self._enforce_state_consistency(pop)

    # =========================================================
    # Helpers
    # =========================================================
    def _inject_relapse_eligible(self, model, transitions):
        """
        May this agent relapse from remission back to INJECTING use?

        Eligibility is by RECENCY of injection, not by ever having injected.
        `oud_state.transitions.inject_relapse_window_years` sets how long
        eligibility survives after the last injecting timestep; the default of 999
        years exceeds any run length and so reproduces the old `ever_injected`
        latch exactly. See that parameter for why the latch was a problem.

        `last_injection_time` is stamped at the end of update_agent on every step
        the agent's drug_type is Inj -- so it is maintained by the model loop, not
        by reporting -- and seeded at baseline in init_agent so that agents who
        start as injectors are inside the window at t=0 rather than taking a year
        to enter it. It is None only for an agent that has never injected, which
        is the case `ever_injected` also excluded.

        Note the stamp is written AFTER _update_agent runs, so within a single
        step this reads the previous step's stamp. That is the intended reading:
        an agent relapsing at time t is judged on when they last injected before
        t, and an agent still injecting at t-1 is trivially inside any window.

        Falls back to `ever_injected` when the stamp is missing, so a population
        restored from an older savefile (population_io) degrades to the previous
        behaviour rather than losing every injector's eligibility at once.
        """
        window_yr = getattr(transitions, "inject_relapse_window_years", None)
        ever = bool(getattr(self.agent, "ever_injected", False))
        last_inj = getattr(self.agent, "last_injection_time", None)

        if window_yr is None or last_inj is None:
            # No window configured, or no stamp to compare it against. The second
            # case is either an agent who has never injected (ever_injected False,
            # so this correctly denies) or a population restored from a
            # pre-recency savefile (ever_injected True, so this falls back to the
            # old latch rather than silently zeroing every injector).
            return ever

        steps_per_year = model.params.model.time.steps_per_year
        return (model.time - last_inj) <= window_yr * steps_per_year

    def _enforce_state_consistency(self, pop):
        """
        Enforce invariants and PWID bookkeeping.

        drug_type is left intact for non-OUD agents — population.py guarantees
        it is always Inj or NonInj (never None), and downstream output stats
        stratify on drug_type with no 'None' key. Allowing non-OUD agents to
        retain their demographic drug_type also lets recreational PWID
        contribute to injection-related HIV transmission without being
        treated as OUD (MOUD eligibility gates on has_oud).
        """
        if self.agent.is_in_remission:
            self.agent.has_oud = False
            self.agent.drug_type = "NonInj"

        if self.agent.drug_type == "Inj":
            pop.pwid_agents.add_agent(self.agent)
        else:
            if self.agent in pop.pwid_agents.members:
                pop.pwid_agents.remove_agent(self.agent)

        # pwid_agents membership alone is not enough: injection-bond partner
        # targets are configured per drug_type and derived once at creation, so
        # a transition leaves them describing the agent's *previous* drug_type.
        # Re-derive here -- the single place every transition path funnels
        # through -- so starting/stopping injection actually changes who the
        # agent can form injection partnerships with.
        if self.agent.drug_type != self.synced_drug_type:
            self.synced_drug_type = self.agent.drug_type
            pop.sync_injection_partner_targets(self.agent)

    # =========================================================
    # Reporting
    # =========================================================
    def set_stats(self, stats_dict: dict, time: int):
        has_oud = getattr(self.agent, "has_oud", False)
        in_remission = getattr(self.agent, "is_in_remission", False)

        if has_oud:
            stats_dict["oud_active"] += 1
        elif in_remission:
            stats_dict["oud_remission"] += 1
        else:
            stats_dict["oud_none"] += 1

        # ---- look-back PWID denominator (reported, not scored) ----
        #
        # The HIV objectives read prevalence and incidence over agents whose
        # drug_type == "Inj" at the instant of measurement. The APCD targets
        # (#25 prevalence, #26 incidence) are built from a claims cohort, where
        # membership is granted for a whole claims year, so the two denominators
        # are not the same population. Measured on the post-rebuild run
        # (N=3000, seed 17, final timestep), the gap is a factor of two:
        #
        #     drug_type == Inj now   n= 322   prev 0.0186   <- what is scored
        #     injected in last 12mo  n= 486   prev 0.0288
        #     ever injected          n=1645   prev 0.0377
        #     whole population       n=3000   prev 0.0223
        #
        # The look-back denominator is the one that matches how the target was
        # constructed, and it moves the model TOWARD the 0.062 target rather than
        # away. It is reported alongside the scored quantity rather than swapped
        # in, because the same change cuts measured incidence by the same ~1.5x
        # (the incident-case numerator is identical under both denominators --
        # 7 cases -- while the denominator grows), so it makes the prevalence fit
        # easier and the incidence fit harder. Which denominator to score is a
        # target-definition question for the APCD extraction, not something to
        # settle by picking whichever flatters the model.
        #
        # See the 2026-07-30 calibration-design writeup in docs/calibration_log/.
        last_inj = getattr(self.agent, "last_injection_time", None)
        if last_inj is not None:
            steps_per_year = self.agent.location.params.model.time.steps_per_year
            if time - last_inj < steps_per_year:
                stats_dict["pwid_recent"] += 1
                hiv = getattr(self.agent, "hiv", None)
                if hiv is not None and getattr(hiv, "active", False):
                    stats_dict["pwid_recent_hiv"] += 1
