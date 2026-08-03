#!/usr/bin/env python
# encoding: utf-8

import random
from collections import deque
from copy import copy
from math import ceil
from typing import List, Dict, Set, Optional, Tuple
import logging

import numpy as np  # type: ignore
import networkx as nx  # type: ignore
import nanoid  # type: ignore

from . import parse_params
from . import agent as ag
from . import location
from . import partnering
from . import utils
from . import features
from . import exposures
from .distributions import poisson


class Population:
    def __init__(self, params: "parse_params.ObjMap", id: Optional[str] = None):
        """
        Initialize Population object.

        args:
            params : Model parameters
            id: 8 character identifier for a model
        """
        if id is None:
            self.id = nanoid.generate(size=8)
        else:
            self.id = id

        utils.set_up_logging(params)

        logging.info(f"Population ID: {self.id}")

        self.pop_seed = utils.get_check_rand_int(params.model.seed.ppl)

        # Init RNG for population creation to pop_seed
        self.pop_random = random.Random(self.pop_seed)
        self.np_random = np.random.default_rng(self.pop_seed)

        self.enable_graph = params.model.network.enable
        self.components: List = []

        if self.enable_graph:
            self.graph = nx.Graph()
        else:
            self.graph = None

        self.params = params

        # set up the in-scope exposures
        self.exposures = [
            exposure
            for exposure in exposures.BaseExposure.__subclasses__()
            if self.params.exposures[exposure.name]
        ]
        # initialize the class level items
        for exposure in self.exposures:
            exposure.init_class(params)

        # set up the in-scope features
        self.features = [
            feature
            for feature in features.BaseFeature.__subclasses__()
            if self.params.features[feature.name]
        ]
        # initialize the class level items
        for feature in self.features:
            feature.init_class(params)

        # set up the population's locations and edges
        self.geography = location.Geography(params)

        # All agent set list
        self.all_agents = ag.AgentSet("AllAgents")

        # pwid agents (performance for partnering)
        self.pwid_agents = ag.AgentSet("PWID", parent=self.all_agents)

        # agents who can take on a partner
        self.partnerable_agents: Dict[str, Set["ag.Agent"]] = {}
        for bond_type in self.params.classes.bond_types.keys():
            self.partnerable_agents[bond_type] = set()

        # who can sleep with whom
        self.sex_partners: Dict[str, Set["ag.Agent"]] = {}
        for sex_type in self.params.classes.sex_types.keys():
            self.sex_partners[sex_type] = set()

        self.relationships: Set["ag.Relationship"] = set()

        # find average partnership durations
        self.mean_rel_duration: Dict[str, Dict] = partnering.get_mean_rel_duration(
            self.params
        )

        logging.info("  Creating agents")
        # for each location in the population, create agents per that location's demographics
        init_time = -1 * self.params.model.time.burn_steps
        for loc in self.geography.locations.values():
            for race in params.classes.races:
                for i in range(
                    round(
                        params.model.num_pop
                        * loc.ppl
                        * loc.params.demographics[race].ppl
                    )
                ):
                    if self.all_agents.num_members() >= self.params.model.num_pop:
                        logging.warning(
                            "WARNING: not adding agent to population - too many agents"
                        )
                        break
                    agent = self.create_agent(loc, race, init_time)
                    self.add_agent(agent)

        # initialize relationships
        logging.info("  Creating Relationships")
        self.update_partner_assignments(0)

    def create_agent(
        self,
        loc: "location.Location",
        race: str,
        time: int,
        sex_type: Optional[str] = None,
        drug_type: Optional[str] = None,
        age: Optional[int] = None,
    ) -> "ag.Agent":
        """
        Create a new agent with randomly assigned attributes according to population
        demographics [params.demographics].
        """

        # -------------------------
        # Assign sex type
        # -------------------------
        if not sex_type:
            sex_type = utils.safe_random_choice(
                loc.pop_weights[race]["values"],
                self.pop_random,
                weights=loc.pop_weights[race]["weights"],
            )
            if sex_type is None:
                raise ValueError("Agent must have sex type")

        # -------------------------
        # Assign OUD state
        #
        # Drawn BEFORE drug_type because OUD remission implies the agent is not
        # currently injecting. Conditioning the drug_type draw on that (rather
        # than overwriting drug_type after the fact) keeps three things correct
        # that used to be wrong:
        #   * the realized injection share still matches what demographics
        #     specifies (see conditional_drug_weights)
        #   * agent_params / mean_num_partners / partnerable_agents are derived
        #     from the agent's final drug_type, not a value overwritten later
        #   * ever_injected reflects real injection history rather than the
        #     post-override drug_type
        # -------------------------
        has_oud = False
        in_remission = False
        oud_enabled = self.params.features.get("oud_state", False)

        if oud_enabled:
            init_oud = getattr(self.params, "oud_state", None)
            if init_oud is None:
                raise ValueError(
                    "features.oud_state is true, but no oud_state: section found in YAML"
                )

            r = self.pop_random.random()
            if r < init_oud.init_active:
                has_oud = True
            elif r < init_oud.init_active + init_oud.init_remission:
                in_remission = True
            # else: no OUD

        # -------------------------
        # Assign drug type
        # -------------------------
        inj_share = self.injection_share(loc, race, sex_type)

        if not drug_type:
            if in_remission:
                # In OUD remission -> not currently injecting.
                drug_type = "NonInj"
            else:
                values, weights = self.conditional_drug_weights(
                    loc, race, sex_type, oud_enabled
                )
                drug_type = utils.safe_random_choice(
                    values, self.pop_random, weights=weights
                )
            if drug_type is None:
                raise ValueError("Agent must have drug type")

        # -------------------------
        # Assign age
        # -------------------------
        if not age:
            age = self.get_age(loc, race)

        # -------------------------
        # Create agent
        # -------------------------
        agent = ag.Agent(sex_type, age, race, drug_type, loc)

        # -------------------------
        # Apply OUD state
        #
        # Set before exposure/feature init so MOUD.init_agent and
        # OUDState.init_agent observe the correct state.
        # -------------------------
        agent.has_oud = has_oud
        agent.is_in_remission = in_remission

        # Injection history. Remission agents are not currently injecting, but
        # many of them injected before entering remission. Drawing history from
        # the unconditional demographic injection share keeps `ever_injected`
        # -- which gates relapse-to-injection in oud_state -- from being
        # systematically False for the entire baseline remission pool.
        if in_remission:
            agent.ever_injected = self.pop_random.random() < inj_share
        else:
            agent.ever_injected = drug_type == "Inj"

        # -------------------------
        # Assign sexual role
        # -------------------------
        sex_role = utils.safe_random_choice(
            loc.role_weights[race][sex_type]["values"],
            self.pop_random,
            weights=loc.role_weights[race][sex_type]["weights"],
        )
        if sex_role is None:
            raise ValueError("Agent must have sex role")
        agent.sex_role = sex_role

        agent_params = (
            agent.location.params.demographics[race]
            .sex_type[sex_type]
            .drug_type[drug_type]
        )

        # -------------------------
        # Initialize exposures
        # -------------------------
        for exposure in self.exposures:
            getattr(agent, exposure.name).init_agent(self, time)

        # -------------------------
        # Initialize partnerships
        # -------------------------
        for bond, bond_def in loc.params.classes.bond_types.items():
            agent.partners[bond] = set()

            agent.mean_num_partners[bond] = self.draw_mean_num_partners(
                agent, bond, agent_params.num_partners[bond]
            )

            agent.target_partners[bond] = agent.mean_num_partners[bond]

            if "injection" in bond_def.acts_allowed:
                assert agent.drug_type == "Inj" or agent.mean_num_partners[bond] == 0

            if agent.target_partners[bond] > 0:
                self.partnerable_agents[bond].add(agent)

        # -------------------------
        # Initialize feature-level attributes
        # (OUD state was applied above, so MOUD.init_agent sees has_oud /
        # is_in_remission and OUDState.init_agent sees ever_injected)
        # -------------------------
        for feature in self.features:
            getattr(agent, feature.name).init_agent(self, time)

        return agent

    def draw_mean_num_partners(self, agent: "ag.Agent", bond: str, dist_info) -> int:
        """
        Draw an agent's mean number of partners for `bond` from the partner-count
        distribution in its demographics block, scaled to the bond's mean
        relationship duration.

        args:
            agent: the agent to draw for (supplies race and location params)
            bond: the bond type being derived
            dist_info: the `num_partners[bond]` distribution definition
        """
        return ceil(
            utils.safe_dist(dist_info, self.np_random)
            * utils.safe_divide(
                agent.location.params.calibration.sex.partner,
                self.mean_rel_duration[bond][agent.race],
            )
        )

    def sync_injection_partner_targets(self, agent: "ag.Agent") -> None:
        """
        Re-derive injection-bond partner targets after `agent.drug_type` changed.

        `create_agent` derives `mean_num_partners` once, from the demographics
        block matching the agent's drug_type *at creation*. `oud_state` can flip
        drug_type at any step (OUD incidence, remission, relapse,
        inject_start, inject_stop), and injection partner counts are configured
        per drug_type -- `NonInj` blocks set them to 0.

        Without re-deriving, drug_type transitions are inert for transmission:
        an agent who starts injecting keeps `mean_num_partners == 0` for every
        injection-capable bond and never enters `partnerable_agents`, so it can
        neither seek nor be selected as an injection partner; an agent who stops
        injecting keeps nonzero targets it can never fill, because
        `partnering.select_partner` intersects injection candidates with
        `pop.pwid_agents`, and so is retried every step until it hits the
        break_point.

        Only injection-capable bonds are re-derived. Sex- and PCA-only bonds are
        deliberately left alone: `mean_num_partners` is a persistent per-agent
        trait, and re-drawing it on every drug-type flip would destabilize
        individual sexual activity levels -- and so the core-group structure
        that drives sexual transmission -- for no modelling gain. `OUDState`
        warns at `init_class` if a non-injection bond's partner counts actually
        differ across drug_type blocks, which is the only case where that
        restriction would lose information.

        args:
            agent: the agent whose drug_type just changed
        """
        agent_params = (
            agent.location.params.demographics[agent.race]
            .sex_type[agent.sex_type]
            .drug_type[agent.drug_type]
        )

        for bond, bond_def in agent.location.params.classes.bond_types.items():
            if "injection" not in bond_def.acts_allowed:
                continue

            agent.mean_num_partners[bond] = self.draw_mean_num_partners(
                agent, bond, agent_params.num_partners[bond]
            )
            agent.target_partners[bond] = poisson(
                self.np_random, agent.mean_num_partners[bond]
            )

        self.update_partnerability(agent)

    def injection_share(
        self, loc: "location.Location", race: str, sex_type: str
    ) -> float:
        """
        Unconditional share of this demographic cell that injects, per
        [params.demographics].
        """
        drug_weights = loc.drug_weights[race][sex_type]
        total = sum(drug_weights["weights"])
        if total <= 0:
            return 0.0

        for value, weight in zip(drug_weights["values"], drug_weights["weights"]):
            if value == "Inj":
                return weight / total
        return 0.0

    def conditional_drug_weights(
        self,
        loc: "location.Location",
        race: str,
        sex_type: str,
        oud_enabled: bool,
    ) -> Tuple[List[str], List[float]]:
        """
        Drug-type weights for an agent known NOT to be in OUD remission.

        Remission agents are assigned NonInj, so drawing everyone else straight
        from [params.demographics] would realize an injection share of
        `p * (1 - init_remission)` rather than the specified `p`. Rescaling the
        Inj weight by `1 / (1 - init_remission)` restores the marginal, so
        `demographics.*.drug_type.Inj.ppl` keeps its documented meaning as a
        population share.
        """
        drug_weights = loc.drug_weights[race][sex_type]
        values = list(drug_weights["values"])
        total = sum(drug_weights["weights"])

        if total <= 0:
            return values, list(drug_weights["weights"])

        normalized = [w / total for w in drug_weights["weights"]]

        remission = 0.0
        if oud_enabled:
            remission = getattr(self.params.oud_state, "init_remission", 0.0)

        if remission <= 0.0:
            return values, normalized

        p_inj = self.injection_share(loc, race, sex_type)
        if p_inj <= 0.0:
            return values, normalized

        p_inj_active = p_inj / (1.0 - remission)
        if p_inj_active > 1.0:
            logging.warning(
                "Injection share %.3f for %s/%s is unattainable with "
                "oud_state.init_remission=%.3f (would need %.3f of "
                "non-remission agents to inject). Clamping: realized injection "
                "share will be %.3f, not %.3f.",
                p_inj,
                race,
                sex_type,
                remission,
                p_inj_active,
                1.0 - remission,
                p_inj,
            )
            p_inj_active = 1.0

        non_inj = 1.0 - p_inj
        adjusted = []
        for value, weight in zip(values, normalized):
            if value == "Inj":
                adjusted.append(p_inj_active)
            elif non_inj <= 0:
                adjusted.append(0.0)
            else:
                # spread the remainder over the non-injection types in
                # proportion to their original weights
                adjusted.append((1.0 - p_inj_active) * (weight / non_inj))

        return values, adjusted


    def add_agent(self, agent: "ag.Agent"):
        """
        Adds an agent to the population

        args:
            agent : The agent to be added
        """
        # Add to all agent set
        self.all_agents.add_agent(agent)

        if agent.drug_type == "Inj":
            self.pwid_agents.add_agent(agent)

        # who can sleep with this agent
        for sex_type in self.params.classes.sex_types[agent.sex_type].sleeps_with:
            self.sex_partners[sex_type].add(agent)

        if self.enable_graph:
            self.graph.add_node(agent)

    def add_relationship(self, rel: "ag.Relationship"):
        """
        Add a new relationship to the population.

        args:
            rel : The Relationship to be added
        """
        self.relationships.add(rel)

        if self.enable_graph:
            self.graph.add_edge(rel.agent1, rel.agent2, type=rel.bond_type)

    def remove_agent(self, agent: "ag.Agent"):
        """
        Remove an agent from the population.

        args:
            agent : Agent to remove
        """
        for rel in copy(agent.relationships):
            rel.progress(force=True)
            self.remove_relationship(rel)

        self.all_agents.remove_agent(agent)

        for partner_type in self.sex_partners:
            if agent in self.sex_partners[partner_type]:
                self.sex_partners[partner_type].remove(agent)

        for exposure in self.exposures:
            agent_attr = getattr(agent, exposure.name)
            if agent_attr.active:
                exposure.remove_agent(agent)

        for feature in self.features:
            agent_attr = getattr(agent, feature.name)
            if agent_attr.active:
                feature.remove_agent(agent)

        if self.enable_graph:
            self.graph.remove_node(agent)

        for bond in self.partnerable_agents.values():
            if agent in bond:
                bond.remove(agent)

        # mark agent component as -1 (no component)
        agent.component = "-1"

    def remove_relationship(self, rel: "ag.Relationship"):
        """
        Remove a relationship from the population.

        args:
            rel : Relationship to remove
        """
        self.relationships.remove(rel)

        # without this relationship, are agents partnerable again?
        self.update_partnerability(rel.agent1)
        self.update_partnerability(rel.agent2)

        if self.enable_graph:
            self.graph.remove_edge(rel.agent1, rel.agent2)

    def get_age(self, loc: "location.Location", race: str) -> int:
        """
        Given the population characteristics, get a random age to assign to an agent given the race of that agent

        args:
            race : race of the agent whose age is being generated

        returns:
            age and the bin the age came from
        """
        bins = loc.params.demographics[race].age
        i = utils.get_independent_bin(self.pop_random, bins)
        age = self.pop_random.randrange(bins[i].min, bins[i].max)
        return age

    def update_agent_partners(
        self, agent: "ag.Agent", bond_type: str, components: List
    ) -> bool:
        """
        Finds and bonds new partner. Creates relationship object for partnership,
            calcs partnership duration, adds it to the population, and adds to networkX graph if self.enable_graph
            is set True.

        args:
            agent: Agent that is seeking a new partner
            bond_type: What type of bond the agent is seeking to make

        returns:
            True if no match was found for agent (used for retries)
        """
        partnerable_agents = self.partnerable_agents[bond_type]
        if (
            self.pop_random.random()
            < self.params.partnership.network.same_component.prob
            and agent.has_partners()
        ):
            # find agent's component
            agent_component: Set["ag.Agent"] = set()
            for comp in components:
                if agent in comp:
                    agent_component = comp
                    break

            partnerable_agents = partnerable_agents & agent_component

        partner = partnering.select_partner(
            agent,
            partnerable_agents,
            self.sex_partners,
            self.pwid_agents,
            self.params,
            self.pop_random,
            bond_type,
        )
        no_match = True

        if partner:
            race = utils.safe_random_choice([agent.race, partner.race], self.pop_random)
            duration = partnering.get_partnership_duration(
                agent.location.params, self.np_random, bond_type, race
            )
            relationship = ag.Relationship(
                agent, partner, duration, bond_type=bond_type
            )
            self.add_relationship(relationship)
            # can partner still partner?
            if len(partner.partners[bond_type]) > (
                partner.target_partners[bond_type]
                * self.params.calibration.partnership.buffer
            ):
                self.partnerable_agents[bond_type].remove(partner)
            no_match = False
        return no_match

    def update_partner_assignments(self, t: int):
        """
        Determines which agents will seek new partners from All_agentSet.
            Calls update_agent_partners for any agents that desire partners.

        args:
            t: current time step of the model
        """
        # update agent targets annually
        if t % self.params.model.time.steps_per_year == 0:
            self.update_partner_targets()

        if self.enable_graph:
            network_components = [set(g.nodes()) for g in self.components]
        else:
            network_components = []

        # Now create partnerships until available partnerships are out
        for bond in self.params.classes.bond_types:
            eligible_agents = deque(
                [
                    a
                    for a in self.all_agents
                    if len(a.partners[bond]) < a.target_partners[bond]
                ]
            )
            attempts = {a: 0 for a in eligible_agents}

            while eligible_agents:
                agent = eligible_agents.popleft()
                if len(agent.partners[bond]) < agent.target_partners[bond]:
                    # no match
                    if self.update_agent_partners(agent, bond, network_components):
                        attempts[agent] += 1

                    # add agent back to eligible pool
                    if (
                        len(agent.partners[bond]) < agent.target_partners[bond]
                        and attempts[agent]
                        < self.params.calibration.partnership.break_point
                    ):
                        eligible_agents.append(agent)

        if self.enable_graph:
            self.trim_graph()

        self.update_agent_components()

    def update_partner_targets(self):
        """
        Update the target number of partners for each agent and bond type
        """
        for a in self.all_agents:
            for bond in self.params.classes.bond_types:
                a.target_partners[bond] = poisson(
                    self.np_random, a.mean_num_partners[bond]
                )
            self.update_partnerability(a)

    def update_partnerability(self, a):
        """
        Update whether each agent in the population is currently able to form new relationships for each bond type
        """
        for bond in self.params.classes.bond_types.keys():
            if a in self.partnerable_agents[bond]:
                if len(a.partners[bond]) > (
                    a.target_partners[bond] * self.params.calibration.partnership.buffer
                ):
                    self.partnerable_agents[bond].remove(a)
            elif len(a.partners[bond]) < (
                a.target_partners[bond] * self.params.calibration.partnership.buffer
            ):
                self.partnerable_agents[bond].add(a)

    def update_agent_components(self):
        """
        Update the component IDs associated with each agent based on the current state of the graph
        """
        if self.enable_graph:
            self.components = utils.connected_components(self.graph)
            for id, component in enumerate(self.components):
                for agent in component.nodes:
                    agent.component = str(id)

            self.params.classes.components = list(
                map(str, range(-1, len(self.components)))
            )

    def trim_graph(self):
        """
        Initialize network with graph-based algorithm for relationship
            adding/pruning
        """
        if self.params.model.network.type == "comp_size":

            def trim_component(component, max_size):
                for agent in component.nodes:
                    if (
                        self.pop_random.random()
                        < self.params.calibration.network.trim.prob
                    ):
                        for rel in copy(agent.relationships):
                            if len(agent.relationships) == 1:
                                break  # Make sure that agents stay part of the
                                # network by keeping one bond
                            rel.progress(force=True)
                            self.remove_relationship(rel)

                # recurse on new sub-components
                sub_comps = utils.connected_components(component)
                for sub_comp in sub_comps:
                    if sub_comp.number_of_nodes() > max_size:
                        trim_component(component, max_size)
                    else:
                        break

            components = self.connected_components()
            for comp in components:
                if (
                    comp.number_of_nodes()
                    > self.params.model.network.component_size.max
                ):
                    logging.info(f"TOO BIG {comp} {comp.number_of_nodes()}")
                    trim_component(comp, self.params.model.network.component_size.max)

        logging.info(f"  Total agents in graph: {self.graph.number_of_nodes()}")

    def connected_components(self) -> List:
        """
        Get connected components in graph (if enabled)

        returns:
            list of connected components
        """
        if self.enable_graph:
            return self.components
        else:
            raise ValueError(
                "Can't get connected_components, population doesn't have graph enabled."
            )

    def migrate(self):
        """
        Have agents migrate between locations with probabilities defined in `location.migration.matrix_file`.
        """
        m_attr = self.params.location.migration.attribute
        for a in self.all_agents:
            m_param = a.location.migration_weights
            if self.pop_random.random() < m_param["prob"]:
                new_loc = utils.safe_random_choice(
                    m_param["values"],
                    self.pop_random,
                    weights=m_param["weights"],
                )
                if m_attr == "name":
                    a.location = self.geography.locations[new_loc]
                elif m_attr == "category":
                    a.location = utils.safe_random_choice(
                        self.geography.categories[new_loc], self.pop_random
                    )