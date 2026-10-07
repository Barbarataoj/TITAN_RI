from typing import Dict, ClassVar, Optional

import numpy as np  # type: ignore

from . import base_feature
from .. import agent
from .. import population
from .. import model
from ..parse_params import ObjMap
from .. import exposures


class Prep(base_feature.BaseFeature):
    name = "prep"
    stats = ["prep", "prep_new", "prep_injectable", "prep_oral", "prep_injections",
             "prep_injection_visits"]
    """
        PrEP collects the following stats:

        * prep - number of agents with active PrEP
        * prep_new - number of agents who became active PrEP this time step
        * prep_injectable - number of agents on injectable PrEP
        * prep_oral - number of agents on oral PrEP
    """

    # class level attributes to track all Prep agents
    counts: ClassVar[Dict[str, int]] = {}

    def __init__(self, agent: "agent.Agent"):
        super().__init__(agent)
        # agent level attributes
        self.active = False
        self.adherent = False
        self.ever = False  # ever enrolled (for programme-driven restarts)
        self.type = ""
        self.time = None
        self.last_dose_time: Optional[int] = None
        # per-product injectable state (prep.use_products)
        self.product = ""
        self.injections = 0
        self.next_due: Optional[int] = None
        self.overdue = False
        self.injections_this_step = 0

    @classmethod
    def init_class(cls, params: "ObjMap"):
        """
        Initialize the counts dictionary for the races in the model.

        args:
            params: the population params
        """
        cls.counts = {race: 0 for race in params.classes.races}
        # per-product stocks and injection counts, when the product table is in use
        prep_params = getattr(params, "prep", None)
        if prep_params is not None and getattr(prep_params, "use_products", False):
            for name in prep_params.products:
                for stat in (f"prep_{name}", f"prep_injections_{name}"):
                    if stat not in cls.stats:
                        cls.stats.append(stat)

    def init_agent(self, pop: "population.Population", time: int):
        """
        Initialize the agent for this feature during population initialization (`Population.create_agent`).  Called on only features that are enabled per the params.

        If an agent does not have HIV, is PrEP eligible, and time is at least the prep start time, they are randomly asigned to enroll in PrEP.

        args:
            pop: the population this agent is a part of
            time: the current time step
        """
        params = self.agent.location.params
        if self.eligible(time):
            if "Racial" in params.prep.target_model:
                if (
                    pop.pop_random.random()
                    < params.demographics[self.agent.race]
                    .sex_type[self.agent.sex_type]
                    .prep.init
                ):
                    self.enroll(pop.pop_random, time)
            elif pop.pop_random.random() < params.prep.init:
                self.enroll(pop.pop_random, time)

    def update_agent(self, model: "model.TITAN"):
        """
        Update the agent for this feature for a time step.  Called once per time step in `TITAN.update_all_agents`. Agent level updates are done after population level updates.   Called on only features that are enabled per the params.

        If the agent is not hiv and time is at least the prep start time, if the agent is already on PrEP update their PrEP attributes, if the agent isn't on PrEP and is eleigible, initiate PrEP.

        args:
            model: the instance of TITAN currently being run
        """
        self.injections_this_step = 0
        self.injection_visit_this_step = False
        if (
            not self.agent.hiv.active  # type: ignore[attr-defined]
            and model.time >= self.agent.location.params.prep.start_time
        ):
            if self.active:
                self.progress(model)
            elif self.eligible(model.time):
                self.initiate(model)

    @classmethod
    def add_agent(cls, agent: "agent.Agent"):
        """
        Add an agent to the class (not instance).

        Add agent to the PrEP counts by race, and add the agent to the set of new agents.

        args:
            agent: the agent to add to the class attributes
        """
        # set up if this is the first time being called
        cls.counts[agent.race] += 1

    @classmethod
    def remove_agent(cls, agent):
        """
        Remove an agent from the class (not instance).

        Decrement the prep counts by race.

        args:
            agent: the agent to remove from the class attributes
        """
        cls.counts[agent.race] -= 1

    def set_stats(self, stats: Dict[str, int], time: int):
        if self.active:
            stats["prep"] += 1
            if self.time == time:
                stats["prep_new"] += 1
            if self.type == "Inj":
                stats["prep_injectable"] += 1
            elif self.type == "Oral":
                stats["prep_oral"] += 1
            # The product named "oral" IS the Oral type counted above; counting
            # it again under prep_<product> doubled prep_oral and, through the
            # cash-flow layer that prices oral PrEP from it, doubled oral drug
            # cost and 340B revenue (found 2026-10-06, fixed 2026-10-07).
            if self.product and f"prep_{self.product}" != "prep_oral":
                stats[f"prep_{self.product}"] += 1
        if self.injections_this_step:
            stats["prep_injections"] += self.injections_this_step
            if self.product:
                stats[f"prep_injections_{self.product}"] += self.injections_this_step
            if getattr(self, "injection_visit_this_step", False):
                stats["prep_injection_visits"] += 1

    def get_acquisition_risk_multiplier(self, time: int, interaction_type: str):
        """
        Get a multiplier for how prep reduces risk of HIV acquisition.

        By default, returns 1.0

        args:
            time: the current model time step
            interaction_type: The type of interaction where the agent could acquire HIV (e.g. 'sex', 'injection' - from [params.classes.interaction_types])
        """
        if self.active and self.last_dose_time is not None:
            params = self.agent.location.params
            if self.type == "Inj" and self.product:
                spec = params.prep.products[self.product]
                eff = spec.efficacy
                if interaction_type == "injection" and getattr(spec, "efficacy_injection", -1.0) >= 0:
                    eff = spec.efficacy_injection
                due = self.next_due if self.next_due is not None else time
                past = time - due - getattr(spec, "tail_full_steps", 0)
                if past <= 0:
                    return 1.0 - eff
                half = spec.tail_half_life_steps
                decay = 0.0 if half <= 0 else 0.5 ** (past / half)
                return 1.0 - eff * decay
            if self.type == "Oral":
                adherence = "adherent" if self.adherent else "non_adherent"
                return 1.0 - params.prep.efficacy[adherence]
            elif self.type == "Inj":
                annualized_last_dose_time = (
                    time - self.last_dose_time
                ) / params.model.time.steps_per_year
                annualized_half_life = params.prep.half_life / 365
                load = params.prep.peak_load * (
                    (0.5) ** (annualized_last_dose_time / annualized_half_life)
                )
                return np.exp(-5.528636721 * load)

        return 1.0

    # =============== HELPER METHODS ===================

    def initiate(self, model: "model.TITAN", force: bool = False):
        """
        Place agents onto PrEP treatment. PrEP treatment assumes that the agent knows their HIV status is negative.

        args:
            model : instance of TITAN being run
            force : whether to force the agent to enroll instead of using the appropriate algorithm per the prep params
        """
        # Prep only valid for agents not on prep and are HIV negative
        if self.active or self.agent.hiv.active:  # type: ignore[attr-defined]
            return

        params = self.agent.location.params

        if params.features.care_seeking and not force:
            self.initiate_from_encounter(model)
            return

        if force:
            self.enroll(model.run_random, model.time)
        elif params.prep.cap_as_prob:
            if "Racial" in params.prep.target_model:
                if (
                    model.run_random.random()
                    <= params.demographics[self.agent.race]
                    .sex_type[self.agent.sex_type]
                    .prep.cap
                ):
                    self.enroll(model.run_random, model.time)
            else:
                if model.run_random.random() <= params.prep.cap:
                    self.enroll(model.run_random, model.time)
        else:
            if "Racial" in params.prep.target_model:
                num_prep_agents = self.counts[self.agent.race]
                all_hiv_agents = exposures.HIV.agents
                all_race = {
                    a for a in model.pop.all_agents if a.race == self.agent.race
                }

                num_hiv_agents = len(all_hiv_agents & all_race)
                target_prep = (len(all_race) - num_hiv_agents) * params.demographics[
                    self.agent.race
                ].sex_type[self.agent.sex_type].prep.cap
            else:
                num_prep_agents = sum(self.counts.values())
                target_prep = int(
                    (model.pop.all_agents.num_members() - len(exposures.HIV.agents))
                    * params.prep.cap
                )

            if num_prep_agents < target_prep:
                self.enroll(model.run_random, model.time)

    def initiate_from_encounter(self, model: "model.TITAN"):
        """
        Place an agent on PrEP because a provider offered it at a visit and the
        agent accepted.

        Used instead of the cap and target_model machinery when the care-seeking
        feature is enabled. Those mechanisms allocate PrEP to a target share of
        the population by rule; this routes it through the encounters that
        actually deliver it, so coverage becomes a consequence of where agents
        seek care and which providers they are attached to.

        This is also where PrEP becomes coupled to MOUD, without a term for it.
        Agents on MOUD attend substance use treatment often, and providers in
        that setting have their own willingness to prescribe PrEP, so the
        coupling follows from attendance rather than from a multiplier. Adding an
        explicit MOUD term on top would count the same pathway twice.

        args:
            model: instance of TITAN being run
        """
        if not getattr(self.agent, "care_offer", {}).get("prep", False):
            # a reached agent who stopped PrEP may restart without a new offer
            # (post-initiation support: care_seeking reach_prep_restart_prob)
            restart = getattr(self.agent, "care_reach", {}).get("prep_restart_prob", 0.0)
            if getattr(self, "ever", False) and restart > 0 and model.run_random.random() < restart:
                self.enroll(model.run_random, model.time)
            return
        acceptance = getattr(self.agent.location.params.prep, "acceptance", 1.0)
        acceptance = min(1.0, acceptance * getattr(self.agent, "care_offer_mult", {}).get("prep", 1.0))
        if model.run_random.random() < acceptance:
            self.enroll(model.run_random, model.time)

    def enroll(self, rand_gen, time):
        """
        Enroll an agent in PrEP

        args:
            rand_gen: random number generator
        """
        params = self.agent.location.params

        self.active = True
        self.ever = True
        self.time = time
        self.last_dose_time = time

        adherence = self.demographic_value("adherence")
        adherence = min(
            1.0, adherence * getattr(self.agent, "care_reach", {}).get("prep_adherence_mult", 1.0)
        )
        self.adherent = rand_gen.random() < adherence

        self.product = ""
        self.injections = 0
        self.next_due = None
        self.overdue = False
        if getattr(params.prep, "use_products", False):
            self.product = self.draw_product(params.prep.products, rand_gen, time)
            spec = params.prep.products[self.product]
            self.type = spec.type
            if self.type == "Inj":
                # the first injection is given at the enrolment visit
                self.give_injection(spec, time, visit=False)
        elif "Inj" in params.prep.type and "Oral" in params.prep.type:
            if rand_gen.random() < params.prep.lai.prob:
                self.type = "Inj"
            else:
                self.type = "Oral"
        else:
            self.type = params.prep.type[0]

        self.add_agent(self.agent)

    @staticmethod
    def product_share(spec, time) -> float:
        """A product's share of starts at `time` (availability and linear ramp); -1 = remainder."""
        if time is not None and time < getattr(spec, "available_from_step", -9999):
            return 0.0
        share = float(spec.share_of_starts)
        if share < 0:
            return -1.0
        start = getattr(spec, "share_ramp_start_step", -1)
        end = getattr(spec, "share_ramp_end_step", -1)
        if time is not None and start >= 0 and end > start and time > start:
            frac = min(1.0, (time - start) / (end - start))
            share = share + frac * (float(getattr(spec, "share_end", share)) - share)
        return share

    @classmethod
    def draw_product(cls, products, rand_gen, time=None) -> str:
        """
        Draw a product name by its share of starts at `time`. Products with
        share -1 split the remainder (1 minus the others); first product if all
        shares are 0.
        """
        names = list(products.keys())
        raw = [cls.product_share(products[n], time) for n in names]
        fixed = sum(s for s in raw if s >= 0)
        n_rem = sum(1 for s in raw if s < 0)
        rem = max(0.0, 1.0 - fixed) / n_rem if n_rem else 0.0
        shares = [s if s >= 0 else rem for s in raw]
        total = sum(shares)
        if total <= 0:
            return names[0]
        draw = rand_gen.random() * total
        acc = 0.0
        for n, s in zip(names, shares):
            acc += s
            if draw < acc:
                return n
        return names[-1]

    def give_injection(self, spec, time: int, visit: bool = True):
        """
        Record an injection at `time`: protection restarts from now, the next
        injection falls due one step later during initiation and interval_steps
        later after it. A scheduled injection (visit=True) is a care encounter in
        prep.monitoring_setting with an HIV test accepted with probability 1 when
        test_at_injection; the enrolment injection rides on the enrolment visit.
        """
        self.injections += 1
        self.injections_this_step += 1
        self.last_dose_time = time
        self.overdue = False
        step = 1 if self.injections < spec.initiation_injections else spec.interval_steps
        self.next_due = time + step
        if not visit:
            return
        self.injection_visit_this_step = True
        params = self.agent.location.params
        setting = getattr(params.prep, "monitoring_setting", "primary_care")
        enc = getattr(self.agent, "care_encounter", None)
        if enc is not None:
            enc[setting] = enc.get(setting, 0) + 1
        if spec.test_at_injection:
            offers = getattr(self.agent, "care_offer", None)
            if offers is not None:
                offers["hiv_test"] = True
                mult = getattr(self.agent, "care_offer_mult", None)
                if mult is not None:
                    mult["hiv_test"] = max(mult.get("hiv_test", 1.0), 1e6)

    def progress(self, model: "model.TITAN", force: bool = False):
        """
        Update agent's PrEP status and discontinue stochastically or if `force` is True

        args:
            model: instance of the TITAN being run
            force: whether to force discontinuation of PrEP
        """
        if force:
            self.discontinue()  # TO_REVIEW should this just remove the agent from counts, or discontinue? does it depend on type?
            return

        if self.type == "Inj" and self.product:
            self.progress_injectable(model)
            return

        if self.type == "Oral":
            discontinue = self.demographic_value("discontinue")
            prep_params = self.agent.location.params.prep
            late_after = getattr(prep_params, "discontinue_late_after_steps", 0)
            if late_after > 0 and self.time is not None and model.time - self.time >= late_after:
                discontinue *= getattr(prep_params, "discontinue_late_mult", 1.0)
            discontinue *= getattr(self.agent, "care_reach", {}).get("prep_discontinue_mult", 1.0)
            if model.run_random.random() < discontinue:
                self.discontinue()
            else:
                self.last_dose_time = model.time

        # TO_REVIEW should inj prep have a way to continue at the year mark (besides maybe getting prep again through the normal channels of enrollment)?
        if (
            self.type == "Inj"
            and self.last_dose_time
            + self.agent.location.params.model.time.steps_per_year
            == model.time
        ):
            self.discontinue()

    def progress_injectable(self, model: "model.TITAN"):
        """
        Injection schedule for a product injectable: at the due step the agent
        attends with probability `attendance` (x a reached agent's
        reach_prep_monitoring_mult); after a miss it returns with probability
        `return_prob` per step (x the same multiplier), and stops if more than
        max_gap_steps late.
        """
        spec = self.agent.location.params.prep.products[self.product]
        if self.next_due is None or model.time < self.next_due:
            return
        mult = getattr(self.agent, "care_reach", {}).get("prep_monitoring_mult", 1.0)
        if not self.overdue:
            if model.run_random.random() < min(1.0, spec.attendance * mult):
                self.give_injection(spec, model.time)
                return
            self.overdue = True
        elif model.run_random.random() < min(1.0, spec.return_prob * mult):
            self.give_injection(spec, model.time)
            return
        if model.time - self.next_due >= spec.max_gap_steps:
            self.discontinue()

    def demographic_value(self, field: str) -> float:
        """
        The agent's PrEP `discontinue` or `adherence`: the drug-type-level value
        (demographics.<race>.sex_type.<sex>.drug_type.<drug>.prep.<field>) when
        the scenario sets one, else the sex-type-level value the engine has
        always read. Lets a scenario give people who inject their own
        persistence without changing the others (2026-10-02).
        """
        sex = self.agent.location.params.demographics[self.agent.race].sex_type[self.agent.sex_type]
        try:
            drug_prep = sex.drug_type[self.agent.drug_type].prep
            value = drug_prep[field] if isinstance(drug_prep, dict) else getattr(drug_prep, field)
            if value is not None and float(value) >= 0:   # -1 (schema default) = not set
                return float(value)
        except (KeyError, AttributeError, TypeError):
            pass
        return float(getattr(sex.prep, field))

    def discontinue(self):
        """
        Discontinue PrEP usage
        """
        self.active = False
        self.type = ""
        self.time = None
        self.last_dose_time = None
        self.product = ""
        self.next_due = None
        self.overdue = False
        self.remove_agent(self.agent)

    def eligible(self, time) -> bool:
        """
        Determine if an agent is eligible for PrEP

        returns:
            whether the agent is eligible
        """
        params = self.agent.location.params
        if self.agent.hiv.active or time < params.prep.start_time:  # type: ignore[attr-defined]
            return False

        target_model = params.prep.target_model
        gender = params.classes.sex_types[self.agent.sex_type].gender

        if (
            self.active
            or self.agent.vaccine.active  # type: ignore[attr-defined]
            or params.features.random_trial
        ):
            return False

        all_eligible_models = {"Allcomers", "Racial"}

        if all_eligible_models.intersection(target_model):
            return True

        if "cdc_women" in target_model:
            if gender == "F":
                if self.cdc_eligible():
                    return True

        if "cdc_msm" in target_model:
            if gender == "M" and self.cdc_eligible():
                return True

        if "pwid_sex" in target_model:
            if self.agent.drug_type == "Inj" and self.cdc_eligible():
                return True

        if "pwid" in target_model:
            if self.agent.drug_type == "Inj":
                return True

        if "ssp_sex" in target_model:
            if self.agent.syringe_services.active and self.cdc_eligible():  # type: ignore[attr-defined]
                return True

        if "ssp" in target_model:
            if self.agent.syringe_services.active:  # type: ignore[attr-defined]
                return True

        return False

    def cdc_eligible(self) -> bool:
        """
        Determine agent eligibility for PrEP under CDC criteria

        returns:
            cdc eligibility
        """
        if self.agent.is_msm():
            return True

        ongoing_duration = self.agent.location.params.partnership.ongoing_duration
        for rel in self.agent.relationships:
            partner = rel.get_partner(self.agent)
            if rel.duration > ongoing_duration and partner.hiv.dx:  # type: ignore[attr-defined]
                return True

            if partner.drug_type == "Inj" or partner.is_msm():
                return True

        return False
