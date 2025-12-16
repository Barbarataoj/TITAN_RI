from . import base_interaction
from .. import features
from .. import model
from .. import agent
from ..distributions import poisson


class Injection(base_interaction.BaseInteraction):
    name = "injection"

    @classmethod
    def get_num_acts(cls, model: "model.TITAN", rel: "agent.Relationship") -> int:
        """
        Simulate random transmission of HIV between two PWID agents through injection.

        args:
            model: The currently running model
            rel: The relationship in which the interaction is happening
        """

        # --- New: guard against non-PWID partners -------------------------
        # Only PWID–PWID (drug_type == "Inj" for both) should generate
        # injection acts. If either partner is NonInj/None/etc., we return 0.
        if rel.agent1.drug_type != "Inj" or rel.agent2.drug_type != "Inj":
            return 0
        # ------------------------------------------------------------------

        # original assumptions (kept as internal sanity)
        # assert rel.agent1.drug_type == "Inj"
        # assert rel.agent2.drug_type == "Inj"

        agent_params = (
            rel.agent1.location.params.demographics[rel.agent1.race]
            .sex_type[rel.agent1.sex_type]
            .injection
        )
        partner_params = (
            rel.agent2.location.params.demographics[rel.agent2.race]
            .sex_type[rel.agent2.sex_type]
            .injection
        )

        mean_num_acts = (
            min(agent_params.num_acts, partner_params.num_acts)
            * model.calibration.injection.act
        )
        share_acts = poisson(model.np_random, mean_num_acts)

        if share_acts < 1:
            return 0

        # syringe services program risk
        if (
            rel.agent1.syringe_services.active  # type: ignore[attr-defined]
            or rel.agent2.syringe_services.active  # type: ignore[attr-defined]
        ):
            p_unsafe_injection = features.SyringeServices.enrolled_risk
        else:
            p_unsafe_injection = agent_params.unsafe_prob

        # ===== MOUD risk reduction for injection-related HIV/HCV =====
        # We assume each agent has a MOUD feature (titan.features.moud.MOUD)
        # with a get_injection_risk_multiplier() method returning
        #   <1.0 if on MOUD (e.g., 0.7), 1.0 if not on MOUD.
        moud_mult = 1.0

        moud1 = getattr(rel.agent1, "moud", None)
        if moud1 is not None:
            moud_mult = min(moud_mult, moud1.get_injection_risk_multiplier())

        moud2 = getattr(rel.agent2, "moud", None)
        if moud2 is not None:
            moud_mult = min(moud_mult, moud2.get_injection_risk_multiplier())

        p_unsafe_injection *= moud_mult
        # ===== END MOUD block =====

        # diagnosis risk reduction
        if rel.agent1.hiv.dx or rel.agent2.hiv.dx:  # type: ignore[attr-defined]
            p_unsafe_injection *= 1 - model.params.hiv.dx.risk_reduction.injection

        for _ in range(share_acts):
            if model.run_random.random() > p_unsafe_injection:
                share_acts -= 1

        return share_acts
