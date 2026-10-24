from typing import Dict, ClassVar
import logging

from . import base_feature
from .insurance import Insurance
from .. import agent
from .. import population
from .. import model
from ..parse_params import ObjMap


class MOUD(base_feature.BaseFeature):
    """
    Medication for Opioid Use Disorder (MOUD) feature.

    Represents pharmacologic treatment for OUD (e.g., buprenorphine, methadone).
    Agents can initiate, discontinue, and re-initiate MOUD over time.
    """

    name = "moud"
    stats = ["moud", "moud_active",
             # Initiations by an overdose survivor within one step of the event
             # (2026-10-01): with od_survivors_off_moud from the overdose feature
             # this gives the model's 30-day post-overdose linkage, the quantity
             # APCD job 2b measures and the Phase 7 linkage arms act on.
             "moud_start_postod_1step"]
    """
        MOUD collects the following stats:

        * moud - number of agents currently on MOUD
    """

    counts: ClassVar[Dict] = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)

        self.active = False  # currently on MOUD
        self.ever = False    # ever been on MOUD
        self.time_on_moud = 0  # consecutive steps on current MOUD episode
        self.started_postod_1step = False  # set at a restart within one step of an overdose

    # ================= CLASS-LEVEL METHODS =================

    @classmethod
    def init_class(cls, params: "ObjMap"):
        """
        Initialize the counts dictionary for the races and sex_types in the model.

        args:
            params: the population params
        """
        cls.counts = {
            race: {sex_type: 0 for sex_type in params.classes.sex_types}
            for race in params.classes.races
        }

        # Eligibility (_is_oud) reads has_oud / is_in_remission, both of which
        # are only ever set by the oud_state feature. With oud_state off, no
        # agent is ever eligible and MOUD silently never initiates.
        if not params.features.get("oud_state", False):
            logging.warning(
                "features.moud is enabled but features.oud_state is not. MOUD "
                "eligibility depends on has_oud / is_in_remission, which only "
                "oud_state sets, so no agent will ever initiate MOUD."
            )

    @classmethod
    def add_agent(cls, agent: "agent.Agent"):
        """
        Increment MOUD counts for the given agent's race and sex_type.
        """
        cls.counts[agent.race][agent.sex_type] += 1

    @classmethod
    def remove_agent(cls, agent: "agent.Agent"):
        """
        Decrement MOUD counts for the given agent's race and sex_type.
        """
        cls.counts[agent.race][agent.sex_type] -= 1

    # ================= AGENT INITIALIZATION =================

    def init_agent(self, pop: "population.Population", time: int):
        """
        Initialize MOUD status during population creation.

        Only OUD agents (drug_type in {Inj, NonInj}) are eligible. They are
        placed on MOUD at baseline with probability `moud.init` from the
        demographic params.

        args:
            pop: the population this agent is a part of
            time: the current time step
        """
        # Always publish the agent-level flag so downstream features can read it.
        self.agent.on_moud = False

        if not self._is_oud():
            return

        # ---- passive hooks only ----
        if not hasattr(self.agent, "recent_overdose"):
            self.agent.recent_overdose = False

        if not hasattr(self.agent, "recent_relapse"):
            self.agent.recent_relapse = False

        # Raised by STEP 5 when an agent stops MOUD; read by the overdose
        # feature (overdose.post_moud_stop_rr) and expired by its update_pop.
        if not hasattr(self.agent, "recent_moud_stop"):
            self.agent.recent_moud_stop = False
            self.agent.recent_moud_stop_timer = 0

        moud_params = (
            self.agent.location.params.demographics[self.agent.race]
            .sex_type[self.agent.sex_type]
            .drug_type[self.agent.drug_type]
            .moud
        )

        # t=0 placement probability. Precedence: init_at_equilibrium, then the
        # top-level moud.init_coverage calibration override, then the
        # per-demographic moud.init. Accessed via location.params (same pattern as
        # the effect-scale readers below), since init_agent runs during population
        # creation and has no model handle. Mirrors the discontinue_per_month
        # override in update_agent.
        moud_global = getattr(self.agent.location.params, "moud", None)
        equilibrium = self._equilibrium_coverage(moud_global)
        if equilibrium is not None:
            init_prob = equilibrium
        elif moud_global is not None and getattr(moud_global, "init_coverage", 0) > 0:
            init_prob = float(moud_global.init_coverage)
        else:
            init_prob = moud_params.init

        if pop.pop_random.random() < init_prob:
            self.initiate(pop.pop_random, moud_params, "init")

    @staticmethod
    def _equilibrium_coverage(moud_global):
        """
        Steady-state MOUD coverage implied by the reinit/discontinue dynamics, or
        None if `init_at_equilibrium` is off or the inputs are unusable.

        Once an agent has been on MOUD, STEP 2 of `update_agent` restarts them
        with probability `reinit_prob` per step and STEP 5 stops them with
        probability `discontinue_per_month`, so the ever-on pool settles at
        `reinit / (reinit + discontinue)` regardless of the t=0 placement. Placing
        agents there at creation removes the coverage transient rather than making
        the burn-in absorb it.
        """
        if moud_global is None:
            return None
        if not getattr(moud_global, "init_at_equilibrium", False):
            return None
        if not getattr(moud_global, "use_reinit", False):
            return None

        reinit = float(getattr(moud_global, "reinit_prob", 0.0) or 0.0)
        discontinue = float(getattr(moud_global, "discontinue_per_month", 0.0) or 0.0)
        if reinit <= 0.0 or discontinue <= 0.0:
            return None

        return reinit / (reinit + discontinue)

    # ================= PER-TIME-STEP UPDATE =================

    def update_agent(self, model: "model.TITAN"):

        # ===============================
        # STEP 1: Eligibility
        # ===============================
        if not self._is_oud():
            if self.active:
                self.active = False
                self.remove_agent(self.agent)
            self.time_on_moud = 0
            self.agent.on_moud = False
            return

        moud_global = getattr(model.params, "moud", None)
        if moud_global is not None:
            start_time = getattr(moud_global, "start_time", 0)
            use_reinit = getattr(moud_global, "use_reinit", True)
        else:
            start_time = 0
            use_reinit = True

        if model.time < start_time:
            return

        moud_params = (
            self.agent.location.params.demographics[self.agent.race]
            .sex_type[self.agent.sex_type]
            .drug_type[self.agent.drug_type]
            .moud
        )

        # ======================================================
        # NOT CURRENTLY ON MOUD
        # ======================================================
        if not self.active:

            # ===============================
            # STEP 2a: Care-seeking pathway
            # ===============================
            # When care settings are modelled, MOUD starts because a provider
            # offered it at a visit and the agent accepted. The post-relapse and
            # post-overdose multipliers below are skipped in that case: they
            # stood in for the emergency department pathway. That pathway is an
            # ED encounter here only because the care layer rolls one extra ED
            # visit in the step after a nonfatal overdose
            # (care_seeking.visit_prob_after_overdose, 2026-10-02); before that
            # nothing linked an overdose to a visit and post-overdose linkage
            # was unmodelled with this layer on.
            if model.params.features.care_seeking:
                if getattr(self.agent, "care_offer", {}).get("moud", False):
                    acceptance = (
                        getattr(moud_global, "acceptance", 1.0)
                        if moud_global is not None
                        else 1.0
                    )
                    acceptance = min(
                        1.0,
                        acceptance * getattr(self.agent, "care_offer_mult", {}).get("moud", 1.0),
                    )
                    # insurance as agent state (2026-10-24): uninsured agents accept less often
                    acceptance = min(1.0, acceptance * Insurance.acceptance_mult(self.agent, model.params, "moud"))
                    if model.run_random.random() < acceptance:
                        # Same 30-day linkage diagnostic as STEP 4 below; until
                        # 2026-10-02 only the care-off path set it, so
                        # survivor_any_moud_30d_share was undercounted to the
                        # on-MOUD-at-event share whenever the layer was on.
                        last_od = getattr(self.agent, "last_overdose_t", None)
                        if last_od is not None and 0 <= model.time - last_od <= 1:
                            self.started_postod_1step = True
                        self.initiate(model.run_random, moud_params, "prob")
                return

            # ===============================
            # STEP 2: Baseline vs re-initiation
            # ===============================
            if self.ever and use_reinit:
                # Previously on MOUD → re-initiation probability.
                # Top-level moud.reinit_prob (calibration override) takes precedence;
                # else fall back to demographic-level moud_params.reinit.prob. This is
                # the equilibrium-coverage ceiling lever (steady-state coverage among
                # ever-on agents ≈ reinit_prob / (reinit_prob + discontinue_per_month)).
                if moud_global is not None and getattr(moud_global, "reinit_prob", 0) > 0:
                    p_start = float(moud_global.reinit_prob)
                else:
                    p_start = moud_params.reinit.prob
            else:
                # Never on MOUD → baseline enrollment
                p_start = self._get_enroll_prob(model.time, moud_params)

            # ===============================
            # STEP 3: Post-relapse uptake
            # Top-level moud.post_relapse_rr (calibration override) takes
            # precedence; else fall back to demographic-level (legacy)
            # moud_params.post_relapse_rr. Mirrors the post_overdose_rr
            # precedence below - demographics.yml doesn't define a
            # post_relapse_rr field, so without this override the demographic
            # fallback silently no-ops to the getattr default of 1.0.
            # ===============================
            if getattr(self.agent, "recent_relapse", False):
                if moud_global is not None and hasattr(moud_global, "post_relapse_rr"):
                    p_start *= float(moud_global.post_relapse_rr)
                else:
                    p_start *= getattr(moud_params, "post_relapse_rr", 1.0)

            # ===============================
            # STEP 4: Post-overdose uptake
            # ===============================
            # Top-level moud.post_overdose_rr (calibration override) takes precedence;
            # else fall back to demographic-level (legacy) moud_params.post_overdose_rr.
            if getattr(self.agent, "recent_overdose", False):
                if moud_global is not None and hasattr(moud_global, "post_overdose_rr"):
                    p_start *= float(moud_global.post_overdose_rr)
                else:
                    p_start *= getattr(moud_params, "post_overdose_rr", 1.0)

            # Safety clamp
            p_start = min(1.0, max(0.0, p_start))

            if model.run_random.random() < p_start:
                last_od = getattr(self.agent, "last_overdose_t", None)
                if last_od is not None and 0 <= model.time - last_od <= 1:
                    self.started_postod_1step = True
                self.initiate(model.run_random, moud_params, "prob")

        # ======================================================
        # STEP 5: Persistence / discontinuation
        # Top-level moud.discontinue_per_month (calibration override) takes
        # precedence; else fall back to demographic-level (legacy) moud_params.discontinue.
        # ======================================================
        else:
            if moud_global is not None and getattr(moud_global, "discontinue_per_month", 0) > 0:
                discontinue_prob = float(moud_global.discontinue_per_month)
            else:
                discontinue_prob = moud_params.discontinue

            if model.run_random.random() < discontinue_prob:
                self.active = False
                self.time_on_moud = 0
                self.agent.on_moud = False
                self.remove_agent(self.agent)
                # Loss-of-tolerance window. The overdose feature runs after
                # this one within the step (it is declared later), so the
                # elevated risk applies from the month of stopping; the
                # overdose feature's update_pop expires the flag after
                # overdose.post_moud_stop_window steps.
                self.agent.recent_moud_stop = True
                self.agent.recent_moud_stop_timer = 0
            else:
                self.time_on_moud += 1
                self.agent.on_moud = True

    # ================= STATS & SCALING =================

    def set_stats(self, stats: Dict[str, int], time: int):
        """
        Add MOUD counts to the global stats accumulator.
        """
        if self.active:
            stats["moud"] += 1
            # MOUD among ACTIVE OUD only. This is the correct numerator for a MOUD
            # coverage whose denominator is oud_active, matching the APCD target
            # ("MOUD fills among F11-diagnosed OUD in year Y"). Remission agents are
            # MOUD-eligible and counted in `moud`, but must be excluded here so the
            # coverage ratio's numerator and denominator share the same population.
            if getattr(self.agent, "has_oud", False):
                stats["moud_active"] += 1
        if getattr(self, "started_postod_1step", False):
            stats["moud_start_postod_1step"] += 1
            self.started_postod_1step = False

    def get_overdose_risk_multiplier(self) -> float:
        """
        Multiplier for overdose risk when on MOUD.

        This allows the Overdose feature to query the MOUD effect.
        By default, it delegates to params.moud.overdose_scale.active/inactive.

        If you haven't set those yet, you can just return a fixed value
        (e.g., 0.4 when active, 1.0 when not active).
        """
        params = self.agent.location.params
        # If you add a calibration block for MOUD, you can use that here.
        moud_params = getattr(params, "moud", None)
        if moud_params is not None:
            if self.active:
                return moud_params.overdose_scale.active
            else:
                return moud_params.overdose_scale.inactive
        # Default: 60% reduction in overdose risk when on MOUD
        return 0.4 if self.active else 1.0

    def get_injection_risk_multiplier(self) -> float:
        """
        Return a multiplier for injection-related HIV/HCV risk.

        Example (from YAML):
        moud.injection_scale.active   ~ 0.7 (30% reduction)
        moud.injection_scale.inactive ~ 1.0
        """
        params = self.agent.location.params
        moud_global = getattr(params, "moud", None)

        if moud_global is not None:
            if self.active:
                return moud_global.injection_scale.active
            else:
                return moud_global.injection_scale.inactive

        # Fallback: 30% reduction if active, no change if not
        return 0.7 if self.active else 1.0

    

    # ================= HELPER METHODS =================

    def _is_oud(self) -> bool:
        """
        MOUD eligibility: active OUD or stable remission. Gating on has_oud
        (rather than drug_type) so that non-OUD recreational PWID — who keep
        their demographic drug_type but were never diagnosed with OUD — are
        not made MOUD-eligible.
        """
        return (
            getattr(self.agent, "has_oud", False)
            or getattr(self.agent, "is_in_remission", False)
        )

    def _get_enroll_prob(self, time: int, moud_params: ObjMap) -> float:
        """
        Time-varying enrollment probability based on `moud.enroll` windows.

        moud:
          enroll:
            1:
              prob: ...
              start: ...
              stop: ...
            2:
              prob: ...
              start: ...
              stop: ...
        """
        enroll_prob = 0.0
        for window in moud_params.enroll.values():
            if window.start <= time < window.stop:
                enroll_prob = window.prob
                break

        return max(0.0, min(1.0, enroll_prob))

    def initiate(self, rand_gen, moud_params: ObjMap, init_or_prob: str):
        """
        Initiate an agent on MOUD and update class-level counts.

        args:
            rand_gen: random number generator (pop_random or run_random)
            moud_params: demographic-specific MOUD params
            init_or_prob: "init" vs "prob" to index adherence, etc. if needed
        """
        # For now we don't distinguish 'adherent' vs 'non-adherent' MOUD;
        # you can extend this later if you want subtypes.
        self.active = True
        self.ever = True
        self.time_on_moud = 0
        self.agent.on_moud = True
        # Back on treatment: the post-stop loss-of-tolerance window no longer
        # applies, whatever is left of it.
        self.agent.recent_moud_stop = False
        self.agent.recent_moud_stop_timer = 0
        self.add_agent(self.agent)
