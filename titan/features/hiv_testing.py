from typing import Dict, ClassVar
import logging
import math

from . import base_feature
from .. import agent, population, model
from ..parse_params import ObjMap


class HIVTesting(base_feature.BaseFeature):
    """
    HIV testing feature.

    Conceptual model:
    - Background (routine) testing
    - Elevated testing for PWID
    - Care-linked testing during MOUD
    - Event-triggered testing following relapse or nonfatal overdose

    Design principles:
    - Annualized probabilities converted to per-timestep hazards
    - Testing only affects HIV diagnosis (no causal feedback)
    - Diagnosis is delegated to HIV.exposure via HIV.diagnose()
    """

    name = "hiv_testing"
    stats = ["hiv_tests"]

    # Optional class-level counts by race / sex_type
    test_counts: ClassVar[Dict] = {}

    # =====================================================
    # Initialization
    # =====================================================
    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        self.tests_this_step = 0

    @classmethod
    def init_class(cls, params: ObjMap):
        cls.test_counts = {
            race: {sex: 0 for sex in params.classes.sex_types}
            for race in params.classes.races
        }

        testing_params = getattr(params, "hiv_testing", None)
        if testing_params is None:
            return

        # base_prob was silently redefined from per-timestep to annual. Refuse to
        # run on a params file still using the old key rather than quietly
        # applying a stale fitted value under the new meaning.
        legacy_base_prob = getattr(testing_params, "base_prob", -1.0)
        if legacy_base_prob >= 0:
            raise ValueError(
                "hiv_testing.base_prob is deprecated: it was a per-timestep "
                "probability and is now annual. Rename it to "
                "'base_prob_annual' and refit -- reusing the old value "
                "unchanged understates testing by roughly steps_per_year. "
                f"(found base_prob: {legacy_base_prob})"
            )

        legacy_early_window = getattr(testing_params, "moud_early_window", -1)
        if legacy_early_window >= 0:
            raise ValueError(
                "hiv_testing.moud_early_window is deprecated: it was expressed "
                "in raw timesteps with a weekly-step default, so on this "
                "model's monthly steps it applied a ~4x-too-long early-MOUD "
                "window and moud_stable_mult never took effect. Use "
                "'moud_early_window_months' instead. "
                f"(found moud_early_window: {legacy_early_window})"
            )

        # Testing that cannot run during burn-in leaves every agent infected
        # during burn-in permanently undiagnosed and therefore off HAART.
        burn_steps = params.model.time.burn_steps
        start_time = getattr(testing_params, "start_time", -9999)
        if burn_steps > 0 and start_time > -burn_steps:
            logging.warning(
                "hiv_testing.start_time=%d is later than the start of burn-in "
                "(-%d): agents who acquire HIV before then can never be "
                "diagnosed or enroll in HAART, so diagnosed prevalence at t=0 "
                "will fall below demographics' hiv.dx.init.",
                start_time,
                burn_steps,
            )

    def init_agent(self, pop: "population.Population", time: int):
        self.active = True  # REQUIRED
        self.tests_this_step = 0

    # =====================================================
    # Per-step update
    # =====================================================
    def update_agent(self, model: "model.TITAN"):
        if not self.active:
            return

        # Agent must have HIV exposure
        if not hasattr(self.agent, "hiv"):
            return

        hiv_feat = self.agent.hiv  # type: ignore[attr-defined]

        testing_params = getattr(model.params, "hiv_testing", None)
        if testing_params is None:
            return

        # Testing availability start time (negative values allow burn-in testing)
        start_time = getattr(testing_params, "start_time", -9999)
        if model.time < start_time:
            return

        steps_per_year = model.params.model.time.steps_per_year

        if model.params.features.care_seeking:
            prob = self.encounter_prob(testing_params, steps_per_year)
        else:
            prob = self.state_proxy_prob(testing_params, steps_per_year)

        # Clamp
        prob = max(0.0, min(1.0, prob))

        # =================================================
        # Apply test
        # =================================================
        if model.run_random.random() >= prob:
            return

        # Test occurred
        self.tests_this_step += 1
        self.test_counts[self.agent.race][self.agent.sex_type] += 1

        # =================================================
        # Diagnosis logic (delegated to HIV exposure)
        # =================================================
        if hiv_feat.active and not hiv_feat.dx:
            hiv_feat.diagnose(model)
        elif not hiv_feat.active and model.params.features.care_seeking:
            # a negative result: the test-and-offer bundle (2026-10-09)
            self.bundle_prep_after_negative_test(model)

    def bundle_prep_after_negative_test(self, model: "model.TITAN"):
        """
        PrEP raised at the visit where the test came back negative (the
        test-and-offer bundle, 2026-10-09; referral path, gating and dated
        levers 2026-10-12). The setting is the one that offered the test this
        step. At a provider who prescribes PrEP the bundle is raised with
        prep_bundle_prob and delivered the same day or at a result visit one
        step later. At a provider who does not prescribe, a referral is raised
        with prep_referral_prob; it is never same-day, and the patient attends
        elsewhere and starts with prep_referral_return_prob. A programme that
        reached the agent in this setting (care_seeking.reached[setting]) applies
        its floors and overrides; an agent reached nowhere, or only in another
        setting, gets the base values. From prep_bundle_lever_start_step the
        dated lever values (prep_bundle_prob_lever, prep_referral_prob_lever)
        replace the base probabilities where set (>= 0). The acceptance
        multiplier is the setting's acceptance_mult_prep, times
        reach_acceptance_mult_prep when reached. One bundle per step.
        """
        offers_by_setting = getattr(self.agent, "care_offer_by_setting", None) or {}
        care = getattr(model.params, "care_seeking", None)
        if care is None:
            return
        prep = getattr(self.agent, "prep", None)
        # no conversation, and no draw, for an agent who cannot start: already on
        # PrEP, no indication, or stopped PrEP this very step (a monitoring-visit
        # test must not re-enrol the agent it was ordered for; 2026-10-13)
        if prep is None or not prep.eligible(model.time) or getattr(prep, "stopped_this_step", False):
            return
        cs = getattr(self.agent, "care_seeking", None)
        active_by_setting = getattr(self.agent, "care_active", None) or {}
        for setting_name, offers in offers_by_setting.items():
            if not offers.get("hiv_test"):
                continue
            try:
                setting = care.setting[setting_name]
            except (KeyError, AttributeError, TypeError):
                continue
            # the programme's floors and overrides apply at an ACTIVE programme
            # visit in this setting (reach rolled, per-visit engagement rolled),
            # the same gate resolve_offers uses for the offer floors (2026-10-13;
            # was: any agent ever reached in the setting)
            reached = bool(active_by_setting.get(setting_name, False))
            willing = False
            if cs is not None:
                idx = (getattr(cs, "provider_id", None) or {}).get(setting_name)
                pool = (getattr(type(cs), "providers", None) or {}).get(setting_name, [])
                if idx is not None and idx < len(pool):
                    willing = bool(pool[idx]["prep"]["willing"])
            lever_on = model.time >= int(getattr(setting, "prep_bundle_lever_start_step", -999))
            if willing:
                p = float(getattr(setting, "prep_bundle_prob", 0.0))
                lever = float(getattr(setting, "prep_bundle_prob_lever", -1.0))
                if lever_on and lever >= 0.0:
                    p = lever
                if reached:
                    p = max(p, float(getattr(setting, "reach_prep_bundle_prob", -1.0)))
            else:
                p = float(getattr(setting, "prep_referral_prob", 0.0))
                lever = float(getattr(setting, "prep_referral_prob_lever", -1.0))
                if lever_on and lever >= 0.0:
                    p = lever
                if reached:
                    p = max(p, float(getattr(setting, "reach_prep_referral_prob", -1.0)))
            if p <= 0.0:
                continue
            if model.run_random.random() >= p:
                continue      # not raised at this setting's visit; another setting's test may still raise it
            mult = float(getattr(setting, "acceptance_mult_prep", 1.0))
            if reached:
                mult *= float(getattr(setting, "reach_acceptance_mult_prep", 1.0))
            if willing:
                same_day = float(getattr(setting, "prep_same_day_prob", 1.0))
                ret = float(getattr(setting, "prep_deferred_return_prob", 1.0))
                if reached and float(getattr(setting, "reach_prep_same_day_prob", -1.0)) >= 0.0:
                    same_day = float(getattr(setting, "reach_prep_same_day_prob"))
                if reached and float(getattr(setting, "reach_prep_deferred_return_prob", -1.0)) >= 0.0:
                    ret = float(getattr(setting, "reach_prep_deferred_return_prob"))
                prep.offer_after_negative_test(model, same_day, ret, acceptance_mult=mult, setting_name=setting_name)
            else:
                ret = float(getattr(setting, "prep_referral_return_prob", 0.0))
                if reached and float(getattr(setting, "reach_prep_referral_return_prob", -1.0)) >= 0.0:
                    ret = float(getattr(setting, "reach_prep_referral_return_prob"))
                prep.offer_after_negative_test(model, 0.0, ret, referral=True, acceptance_mult=mult, setting_name=setting_name)
            return

    # =====================================================
    # Testing probability, two pathways
    # =====================================================
    def encounter_prob(self, testing_params: ObjMap, steps_per_year: int) -> float:
        """
        Testing probability when the care-seeking feature is enabled.

        A test happens because a provider offered one at a visit and the agent
        accepted. Nothing here depends on the agent's state directly - who gets
        tested follows from where they seek care and which provider they are
        attached to.

        The multipliers used by the state-proxy pathway are deliberately not
        applied. They stood in for care settings, and the care settings now
        exist; applying both would count the same pathway twice.

        args:
            testing_params: the hiv_testing params
            steps_per_year: number of time steps per year

        returns:
            probability of being tested this time step
        """
        offered = getattr(self.agent, "care_offer", {}).get("hiv_test", False)
        if offered:
            mult = getattr(self.agent, "care_offer_mult", {}).get("hiv_test", 1.0)
            return min(1.0, getattr(testing_params, "acceptance", 1.0) * mult)

        # testing that happens outside any clinical setting, e.g. community or
        # syringe services testing; zero unless the parameter is set. Since
        # 2026-10-12 people who do not inject have their own rate
        # (nonclinical_annual_noninj) when it is set (>= 0): the pinned PWID
        # rate comes from a PWID survey and does not describe them.
        nonclinical_annual = getattr(testing_params, "nonclinical_annual", 0.0)
        if self.agent.drug_type == "NonInj":
            noninj = getattr(testing_params, "nonclinical_annual_noninj", -1.0)
            if noninj >= 0.0:
                nonclinical_annual = noninj
        return self.annual_to_step(nonclinical_annual, steps_per_year)

    def state_proxy_prob(self, testing_params: ObjMap, steps_per_year: int) -> float:
        """
        Testing probability from the original state-proxy pathway, used when the
        care-seeking feature is disabled.

        A base annual rate is modified by agent state - injecting, on MOUD,
        recently relapsed, recently overdosed - each standing in for a care
        setting that the model did not otherwise represent.

        args:
            testing_params: the hiv_testing params
            steps_per_year: number of time steps per year

        returns:
            probability of being tested this time step
        """
        base_annual_prob = getattr(testing_params, "base_prob_annual", 0.0)
        prob = self.annual_to_step(base_annual_prob, steps_per_year)

        # PWID-specific testing
        if self.agent.drug_type == "Inj":
            prob *= getattr(testing_params, "pwid_mult", 1.0)

        # Non-injecting OUD (weaker)
        elif self.agent.drug_type == "NonInj":
            prob *= getattr(testing_params, "oud_mult", 1.0)

        # High-risk behavior (if present)
        high_risk_feat = getattr(self.agent, "high_risk", None)
        if high_risk_feat is not None and getattr(high_risk_feat, "active", False):
            prob *= getattr(testing_params, "high_risk_mult", 1.0)

        # Care-linked testing (MOUD)
        if getattr(self.agent, "on_moud", False):
            moud_feat = getattr(self.agent, "moud", None)
            if moud_feat is not None:
                # window is configured in months; time_on_moud is in timesteps
                early_months = getattr(
                    testing_params, "moud_early_window_months", 0.0
                )
                early_window = early_months * steps_per_year / 12.0
                if moud_feat.time_on_moud < early_window:
                    prob *= getattr(testing_params, "moud_early_mult", 1.0)
                else:
                    prob *= getattr(testing_params, "moud_stable_mult", 1.0)

        # Recent relapse from remission
        if getattr(self.agent, "recent_relapse", False):
            prob *= getattr(testing_params, "post_relapse_mult", 1.0)

        # Recent nonfatal overdose (ED pathway)
        if getattr(self.agent, "recent_overdose", False):
            prob *= getattr(testing_params, "post_overdose_mult", 1.0)

        return prob

    @staticmethod
    def annual_to_step(annual_prob: float, steps_per_year: int) -> float:
        """
        Convert an annual probability to a per-time-step probability via the
        hazard rate.

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

    # =====================================================
    # Stats
    # =====================================================
    def set_stats(self, stats: Dict[str, int], time: int):
        if self.tests_this_step > 0:
            stats["hiv_tests"] += self.tests_this_step
        self.tests_this_step = 0
