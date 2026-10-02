from typing import Dict, ClassVar
import logging

import numpy as np

from . import base_feature
from .. import agent, population, model


class Overdose(base_feature.BaseFeature):
    """
    Overdose feature.

    Key properties:
    - Eligible ONLY if agent.has_oud == True
    - Annual probabilities converted to per-step probabilities
    - Injection vs non-injection baseline risk
    - MOUD effects applied multiplicatively
    - Elevated risk after relapse (loss of tolerance)
    - High-risk phenotype
    - Syringe services / naloxone protection
    - Nonfatal overdoses create short-term memory
    - Fatal overdoses trigger exit via exit_enter
    """

    name = "overdose"
    stats = ["overdose_events", "overdose_deaths",
             # Nonfatal events split by the survivor's MOUD status at the event,
             # added 2026-10-01 for the post-overdose linkage diagnostic (APCD
             # job 2b: 13-20% of RI survivors have any MOUD within 30 days).
             "od_survivors_on_moud", "od_survivors_off_moud"]

    # Class-level cumulative counts
    event_counts: ClassVar[Dict] = {}
    death_counts: ClassVar[Dict] = {}

    # ================= INITIALIZATION =================

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        self.events_this_step = 0
        self.deaths_this_step = 0

    @classmethod
    def init_class(cls, params):
        cls.event_counts = {
            race: {sex: 0 for sex in params.classes.sex_types}
            for race in params.classes.races
        }
        cls.death_counts = {
            race: {sex: 0 for sex in params.classes.sex_types}
            for race in params.classes.races
        }

        # Fatal overdoses remove the agent by setting agent.exit_class, which is
        # only acted on by TITAN.exit() -- and that only runs when exit_enter is
        # enabled. Without it, fatal_frac would silently be a pure counter.
        fatal_frac = getattr(getattr(params, "overdose", None), "fatal_frac", 0.0)
        if fatal_frac > 0 and not params.features.exit_enter:
            logging.warning(
                "overdose.fatal_frac is %.3f but features.exit_enter is false: "
                "fatal overdoses will be counted but agents will NOT be removed "
                "from the model.",
                fatal_frac,
            )

        # Event generation is gated on agent.has_oud, which is only ever set by
        # the oud_state feature. With oud_state off, this feature produces
        # exactly zero events and zero deaths with no other symptom.
        if not params.features.get("oud_state", False):
            logging.warning(
                "features.overdose is enabled but features.oud_state is not. "
                "Overdose events are gated on has_oud, which only oud_state "
                "sets, so no overdose events or deaths will be generated."
            )

    def init_agent(self, pop: "population.Population", time: int):
        self.active = True  # REQUIRED
        self.events_this_step = 0
        self.deaths_this_step = 0

        # Nonfatal overdose memory (used by MOUD)
        self.agent.recent_overdose = False
        self.agent.recent_overdose_timer = 0

        # Post-MOUD-stop memory (raised by the moud feature, read here)
        if not hasattr(self.agent, "recent_moud_stop"):
            self.agent.recent_moud_stop = False
            self.agent.recent_moud_stop_timer = 0

    # ================= POPULATION-LEVEL UPDATE =================

    @classmethod
    def update_pop(cls, model: "model.TITAN"):
        """
        Advance and expire the recent-overdose window for every agent.

        This runs here, ahead of any feature's per-agent `update_agent`
        (`update_pop` for every feature runs before `update_agent` for any
        feature - see `TITAN.update_all_agents`), so that consumers like
        `moud` and `hiv_testing` read this step's expiry rather than the
        previous step's. Doing the decrement inside this feature's own
        `update_agent` instead left the flag visible for one step longer
        than `post_overdose_window` intends, since moud/hiv_testing are
        declared earlier than overdose and so read the stale value first.

        args:
            model: the instance of TITAN currently being run
        """
        for a in model.pop.all_agents:
            agent_feature = getattr(a, cls.name)
            if not agent_feature.active:
                continue
            flagged_od = getattr(a, "recent_overdose", False)
            flagged_stop = getattr(a, "recent_moud_stop", False)
            if not (flagged_od or flagged_stop):
                continue

            params = getattr(a.location.params, "overdose", None)
            if params is None:
                continue

            if flagged_od:
                a.recent_overdose_timer = getattr(a, "recent_overdose_timer", 0) + 1
                steps_per_year = model.params.model.time.steps_per_year
                post_od_window = getattr(
                    params, "post_overdose_window", steps_per_year // 2
                )
                if a.recent_overdose_timer > post_od_window:
                    a.recent_overdose = False
                    a.recent_overdose_timer = 0

            # Post-MOUD-stop window. The count includes the step the agent
            # stopped on -- the moud feature raises the flag before this
            # feature's update_agent applies it within that step -- so a window
            # of 1 is exactly the month of stopping, and the next step's
            # update_pop clears it.
            if flagged_stop:
                a.recent_moud_stop_timer = getattr(a, "recent_moud_stop_timer", 0) + 1
                if a.recent_moud_stop_timer >= getattr(params, "post_moud_stop_window", 1):
                    a.recent_moud_stop = False
                    a.recent_moud_stop_timer = 0

    # ================= PER-STEP UPDATE =================

    def update_agent(self, model: "model.TITAN"):
        if not self.active:
            return

        self.events_this_step = 0
        self.deaths_this_step = 0
        self.survivors_on_moud_this_step = 0
        self.survivors_off_moud_this_step = 0

        params = getattr(self.agent.location.params, "overdose", None)
        if params is None:
            return

        # Strict eligibility for generating new events
        if not getattr(self.agent, "has_oud", False):
            return

        # ---------- Base annual probability by drug type ----------
        if self.agent.drug_type == "Inj":
            annual_prob = getattr(params, "base_prob_inj", 0.0)
        else:
            annual_prob = getattr(params, "base_prob_noninj", 0.0)

        annual_prob = max(0.0, min(1.0, annual_prob))

        # ---------- Annual → per-step conversion ----------
        steps_per_year = model.params.model.time.steps_per_year
        dt = 1.0 / steps_per_year
        hazard = -np.log(1.0 - annual_prob) if annual_prob > 0 else 0.0
        prob = 1.0 - np.exp(-hazard * dt)

        # ==================================================
        # Multiplicative risk modifiers (preserve original)
        # ==================================================

        # ---------- MOUD effect (FIXED) ----------
        params_moud = getattr(self.agent.location.params, "moud", None)
        if params_moud is not None:
            if getattr(self.agent, "on_moud", False):
                prob *= params_moud.overdose_scale.active
            else:
                prob *= params_moud.overdose_scale.inactive

        # ---------- Elevated risk after relapse ----------
        if getattr(self.agent, "recent_relapse", False):
            prob *= getattr(params, "post_relapse_od_rr", 1.0)

        # ---------- Elevated risk right after stopping MOUD ----------
        # Loss of tolerance in the weeks after leaving treatment (Sordo 2017,
        # Santo 2021: 2.4-3.3x the rest of time out of treatment). Applied on
        # top of the off-MOUD baseline the block above already restored, and
        # never while on MOUD, whose own scale applies instead.
        if (
            getattr(self.agent, "recent_moud_stop", False)
            and not getattr(self.agent, "on_moud", False)
        ):
            prob *= getattr(params, "post_moud_stop_rr", 1.0)

        # ---------- High-risk phenotype ----------
        high_risk_feat = getattr(self.agent, "high_risk", None)
        if high_risk_feat is not None and getattr(high_risk_feat, "active", False):
            prob *= getattr(params, "high_risk_mult", 1.0)

        # ---------- Syringe services / naloxone ----------
        if (
            hasattr(self.agent, "syringe_services")
            and self.agent.syringe_services.active
        ):
            prob *= getattr(params, "ssp_rr", 1.0)

        prob = max(0.0, min(1.0, prob))

        # ==================================================
        # Determine overdose event
        # ==================================================
        if model.run_random.random() >= prob:
            return

        # ---------- Overdose occurred ----------
        self.events_this_step += 1
        self.event_counts[self.agent.race][self.agent.sex_type] += 1

        # Flag nonfatal overdose for downstream linkage
        self.agent.recent_overdose = True
        self.agent.recent_overdose_timer = 0
        # The step of the event, so moud can tell a restart within one step
        # (the 30-day linkage the APCD measures) from any later restart in the
        # post_overdose_window, whatever the feature update order does to the
        # timer above.
        self.agent.last_overdose_t = getattr(model, "time", None)

        # ---------- Fatality ----------
        fatal_frac = getattr(params, "fatal_frac", 0.0)
        if model.run_random.random() < fatal_frac:
            self.deaths_this_step += 1
            self.death_counts[self.agent.race][self.agent.sex_type] += 1

            self.agent.exit_class = "death"
            self.agent.death_reason = "overdose"
        else:
            if getattr(self.agent, "on_moud", False):
                self.survivors_on_moud_this_step += 1
            else:
                self.survivors_off_moud_this_step += 1

    # ================= STATS =================

    def set_stats(self, stats: Dict[str, int], time: int):
        if self.events_this_step:
            stats["overdose_events"] += self.events_this_step
        if self.deaths_this_step:
            stats["overdose_deaths"] += self.deaths_this_step
        if getattr(self, "survivors_on_moud_this_step", 0):
            stats["od_survivors_on_moud"] += self.survivors_on_moud_this_step
        if getattr(self, "survivors_off_moud_this_step", 0):
            stats["od_survivors_off_moud"] += self.survivors_off_moud_this_step
