import pytest
import os
from types import SimpleNamespace

from titan.population import *
from titan.parse_params import create_params, ObjMap

from tests.conftest import FakeRandom


@pytest.mark.unit
def test_create_agent(make_population, params):
    pop = make_population(n=100)

    a1 = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0)
    assert a1.race == "white_nonhispanic"
    assert a1.knowledge.opinion in range(
        5
    ), f"Agents opinion of injectible PrEP is out of bounds {a1.knowledge.opinion}"

    a2 = pop.create_agent(pop.geography.locations["world"], "black_nonhispanic", 0)
    assert a2.race == "black_nonhispanic"

    a3 = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="HM")
    assert a3.sex_type == "HM"
    assert a3.race == "white_nonhispanic"

    # check PWID and HIV and high risk
    pop.pop_random = FakeRandom(-0.1)
    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["HM"] = ObjMap(
        {"values": ["Inj"], "weights": [1.0]}
    )
    a4 = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="HM")
    assert a4.drug_type == "Inj"
    assert a4.hiv.active
    assert a4.hiv.aids
    assert a4.hiv.dx
    assert a4.haart.active
    assert a4.haart.adherent
    assert a4.high_risk.active
    assert a4.high_risk.ever

    # check not PWID and HIV
    pop.pop_random = FakeRandom(0.999)
    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["HM"] = ObjMap(
        {"values": ["NonInj"], "weights": [1.0]}
    )
    a4 = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="HM")
    assert a4.drug_type == "NonInj"
    assert a4.hiv.active is False
    assert a4.prep.active is False
    assert a4.random_trial.treated is False


@pytest.mark.unit
def test_create_agent_proportions(make_population, params):
    pop = make_population(n=100)

    n = 1000
    race = "white_nonhispanic"
    # oud_state's remission roll in create_agent can independently override
    # drug_type to NonInj after the weighted draw below; disable it here so
    # this test isolates the drug_type-weight draw it's actually checking.
    pop.params.features.oud_state = False
    # check proportions
    pop.geography.locations["world"].pop_weights[race] = {
        "values": ["HM", "HF"],
        "weights": [0.1, 0.9],
    }
    prop_idu = round(params.demographics[race].sex_type["HF"].drug_type["Inj"].ppl * n)
    num_HM = 0
    num_HF = 0
    num_PWID = 0
    for i in range(n):
        a = pop.create_agent(pop.geography.locations["world"], race, 0)
        if a.drug_type == "Inj":
            num_PWID += 1

        if a.sex_type == "HF":
            num_HF += 1
        elif a.sex_type == "HM":
            num_HM += 1
        else:
            assert False

    assert num_HM > 70 and num_HM < 130
    assert num_HF > 830 and num_HF < 930
    assert num_PWID > prop_idu - 50 and num_PWID < prop_idu + 50


@pytest.mark.unit
def test_add_remove_agent_to_pop(make_population):
    pop = make_population(n=100)
    agent = pop.create_agent(
        pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="HM"
    )
    agent.drug_type = "Inj"
    agent.hiv.active = True
    agent.hiv.aids = True
    agent.hiv.dx = True
    agent.hiv.add_agent(agent)
    agent.monkeypox.active = True
    agent.monkeypox.dx = True
    agent.monkeypox.add_agent(agent)

    pop.add_agent(agent)

    assert agent in pop.all_agents.members

    assert pop.graph.has_node(agent)

    pop.remove_agent(agent)

    assert agent not in pop.all_agents.members

    assert not pop.graph.has_node(agent)


@pytest.mark.unit
def test_get_age(make_population, params):
    pop = make_population(n=100)

    race = "white_nonhispanic"

    expected_ages = [15, 25, 35, 45, 55]
    for i in range(1, 6):
        # make sure rand is less than the setting
        pop.pop_random = FakeRandom(params.demographics[race].age[i].prob - 0.001)
        age = pop.get_age(pop.geography.locations["world"], race)
        assert age == expected_ages[i - 1]


@pytest.mark.unit
def test_update_agent_partners_one_agent(make_population, params):
    pop = make_population(n=1)
    params.model.num_pop = 0

    agent = next(iter(pop.all_agents))  # the only agent in the pop

    for bond in params.classes.bond_types:
        pop.update_agent_partners(agent, bond, [])  # noMatch == True
    assert agent in pop.graph.nodes()
    assert len(pop.graph.edges()) == 0


@pytest.mark.unit
def test_update_agent_partners_PWID_no_match(make_population, params):
    pop = make_population(n=0)

    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["MSM"]["values"] = ["Inj"]
    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["MSM"]["weights"] = [1.0]

    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["HF"]["values"] = ["NonInj"]
    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["HF"]["weights"] = [1.0]

    # FakeRandom(1.1) must be set before agent creation, not just before
    # add_agent: create_agent's oud_state remission roll also consumes
    # pop_random, and with a real (unseeded) generator it can probabilistically
    # override the drug_type just forced above via drug_weights.
    pop.pop_random = FakeRandom(1.1)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="HF")
    pop.add_agent(a)
    pop.add_agent(p)
    assert a.drug_type == "Inj"
    assert p.drug_type == "NonInj"

    for bond in params.classes.bond_types.keys():
        assert pop.update_agent_partners(a, bond, [])
        assert a in pop.graph.nodes()
        assert p in pop.graph.nodes()
        assert not a.partners[bond]
        assert len(pop.graph.edges()) == 0

    p.sex_type = "MSM"
    p.drug_type = "NonInj"
    for bond in params.classes.bond_types.keys():
        assert pop.update_agent_partners(a, bond, [])
        assert a in pop.graph.nodes()
        assert p in pop.graph.nodes()
        assert not a.partners[bond]
        assert len(pop.graph.edges()) == 0


@pytest.mark.unit
def test_update_agent_partners_MSM_no_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="HF")
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "NonInj"
    p.drug_type = "NonInj"
    pop.add_agent(a)
    pop.add_agent(p)

    assert pop.update_agent_partners(a, "Sex", [])
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert not a.partners["Sex"]
    assert len(pop.graph.edges()) == 0


@pytest.mark.unit
def test_update_agent_partners_PWID_match(make_population, params):
    pop = make_population(n=0)

    # This test forces every agent to draw Inj via drug_weights. oud_state's
    # remission branch legitimately overrides that (remission implies not
    # currently injecting), so disable it -- this test is about injection
    # partnering, not OUD state.
    pop.params.features.oud_state = False

    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["MSM"]["values"] = ["Inj"]
    pop.geography.locations["world"].drug_weights["white_nonhispanic"]["MSM"]["weights"] = [1.0]

    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    assert (
        pop.geography.locations["world"]
        .params.demographics.white_nonhispanic.sex_type.MSM.drug_type.Inj.num_partners.Inj.vars[1]
        .value
    )
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    assert p.drug_type == "Inj"
    assert p.mean_num_partners["Inj"]
    assert p.target_partners["Inj"]
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "Inj"
    pop.add_agent(a)
    p.target_partners["Inj"] = 10
    pop.add_agent(p)
    assert pop.partnerable_agents["Inj"]

    no_match = pop.update_agent_partners(a, "Inj", [])
    assert no_match is False
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert a.partners["Inj"]
    assert len(pop.graph.edges()) == 1


@pytest.mark.unit
def test_update_agent_partners_MSM_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "NonInj"
    p.drug_type = "NonInj"
    a.target_partners["Sex"] = 25
    p.target_partners["Sex"] = 25
    pop.add_agent(a)
    pop.add_agent(p)

    no_match = pop.update_agent_partners(a, "Sex", [])
    assert no_match is False
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert a.partners
    assert len(pop.graph.edges()) == 1


@pytest.mark.unit
def test_update_agent_partners_NDU_PWID_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "black_nonhispanic", 0, sex_type="MSM")
    # ensure random sex partner no assorting
    pop.np_random = FakeRandom(0.99)
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "NonInj"
    p.drug_type = "Inj"

    a.target_partners["Sex"] = 100
    p.target_partners["Sex"] = 100
    pop.add_agent(a)
    pop.add_agent(p)

    no_match = pop.update_agent_partners(a, "Sex", [])
    assert no_match is False
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert a.partners
    assert len(pop.graph.edges()) == 1
    for (
        rel
    ) in (
        a.relationships
    ):  # check that duration uses "randomly selected" (second) partner
        assert rel.duration != partnering.get_partnership_duration(
            a.location.params, pop.np_random, "Sex", a.race
        )
        assert rel.duration == partnering.get_partnership_duration(
            a.location.params, pop.np_random, "Sex", p.race
        )


@pytest.mark.unit
def test_update_partner_assignments_MSM_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "NonInj"
    p.drug_type = "NonInj"

    pop.add_agent(a)
    pop.add_agent(p)
    a.target_partners["Sex"] = 100
    p.target_partners["Sex"] = 100
    assert params.model.network.enable is True
    assert pop.enable_graph

    pop.update_partner_assignments(1)
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert a.partners
    assert len(pop.graph.edges()) == 1


@pytest.mark.unit
def test_update_partner_assignments_PWID_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "Inj"
    p.drug_type = "Inj"
    pop.add_agent(a)
    pop.add_agent(p)

    a.target_partners["Inj"] = 100
    p.target_partners["Inj"] = 100
    assert params.model.network.enable is True
    assert pop.enable_graph

    pop.update_partner_assignments(1)
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert a.partners
    assert pop.relationships
    assert len(pop.graph.edges()) == 1


@pytest.mark.unit
def test_update_partner_assignments_NDU_PWID_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "NonInj"
    p.drug_type = "Inj"
    a.target_partners["Sex"] = 100
    p.target_partners["Sex"] = 100
    pop.add_agent(a)
    pop.add_agent(p)
    a.mean_num_partners["Sex"] = 100
    p.mean_num_partners["Sex"] = 100
    assert params.model.network.enable == True
    assert pop.enable_graph

    pop.update_partner_assignments(1)
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert a.partners
    assert pop.relationships
    assert len(pop.graph.edges()) == 1


@pytest.mark.unit
def test_update_partner_assignments_no_match(make_population, params):
    pop = make_population(n=0)
    # pass drug_type at creation rather than overwriting it afterwards -- an
    # agent created as Inj keeps its injection partner targets and its
    # partnerable_agents["Inj"] membership, so relabelling it NonInj later left
    # the agent internally inconsistent and made this test seed-dependent
    a = pop.create_agent(
        pop.geography.locations["world"],
        "white_nonhispanic",
        0,
        sex_type="MSM",
        drug_type="NonInj",
    )
    a.id = 1
    p = pop.create_agent(
        pop.geography.locations["world"],
        "white_nonhispanic",
        0,
        sex_type="HM",
        drug_type="NonInj",
    )
    p.id = 2
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.target_partners["Sex"] = 50
    p.target_partners["Sex"] = 50
    pop.add_agent(a)
    pop.add_agent(p)

    params.model.num_pop = 0
    assert p not in pop.sex_partners[a.sex_type]
    assert a not in pop.sex_partners[p.sex_type]

    pop.update_partner_assignments(1)
    assert not a.partners["Social"]
    assert not p.partners["Social"]
    assert not p.partners["Sex"]
    assert not a.partners["Sex"]
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert pop.all_agents.num_members() == 2
    assert len(pop.graph.edges()) == 0


@pytest.mark.unit
def test_network_init_scale_free(params):
    """Test if all Inj,NonInj,None drug use agents are in the population"""
    net = Population(params)
    assert params.model.num_pop == net.all_agents.num_members()

    for agent in net.all_agents:
        assert agent in net.graph.nodes()

    for agent in net.all_agents:
        assert agent.drug_type in ["Inj", "NonInj"]
        assert agent.sex_type in params.classes.sex_types


@pytest.mark.unit
def test_network_init_max_k(params):
    """Test if all Inj,NonInj,None drug use agents are in the population"""
    params.model.network.type = "comp_size"
    net = Population(params)
    assert params.model.num_pop == net.all_agents.num_members()

    for agent in net.all_agents:
        assert agent in net.graph.nodes()

    for agent in net.all_agents:
        assert agent.drug_type in ["Inj", "NonInj"]
        assert agent.sex_type in params.classes.sex_types


@pytest.mark.unit
def test_population_consistency(params):
    """Test if Drug users add up"""
    net = Population(params)
    assert net.graph.number_of_edges() == len(net.relationships)
    assert net.graph.number_of_nodes() == net.all_agents.num_members()
    assert params.model.num_pop == net.all_agents.num_members()


@pytest.mark.unit
def test_partnering_same_component_singleton_match(make_population, params):
    pop = make_population(n=0)
    a = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    p = pop.create_agent(pop.geography.locations["world"], "white_nonhispanic", 0, sex_type="MSM")
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)
    a.drug_type = "NonInj"
    p.drug_type = "NonInj"

    pop.add_agent(a)
    pop.add_agent(p)
    a.target_partners["Sex"] = 100
    p.target_partners["Sex"] = 100
    pop.params.partnership.network.same_component.prob = 2
    assert params.model.network.enable is True
    assert pop.enable_graph

    pop.update_partner_assignments(1)
    assert a in pop.graph.nodes()
    assert p in pop.graph.nodes()
    assert len(a.partners["Sex"]) == 1


@pytest.mark.unit
def test_partnering_cross_component(make_population, make_relationship, params):
    pop = make_population(n=0)

    # create as NonInj rather than relabelling after the fact -- see the note in
    # test_update_partner_assignments_no_match
    def make_noninj_msm():
        return pop.create_agent(
            pop.geography.locations["world"],
            "white_nonhispanic",
            0,
            sex_type="MSM",
            drug_type="NonInj",
        )

    a = make_noninj_msm()
    b = make_noninj_msm()
    c = make_noninj_msm()
    d = make_noninj_msm()
    e = make_noninj_msm()
    # ensure random sex partner no assorting
    pop.pop_random = FakeRandom(1.1)

    pop.add_agent(a)
    pop.add_agent(b)
    pop.add_agent(c)
    pop.add_agent(d)
    pop.add_agent(e)
    a.target_partners["Sex"] = 100
    b.target_partners["Sex"] = 100
    c.target_partners["Sex"] = 100
    d.target_partners["Sex"] = 100
    e.target_partners["Sex"] = 100

    # make a, b, and c a component, not d
    r1 = make_relationship(a, b)
    r2 = make_relationship(b, c)
    r3 = make_relationship(d, e)
    pop.add_relationship(r1)
    pop.add_relationship(r2)
    pop.add_relationship(r3)
    pop.update_agent_components()

    pop.params.partnership.network.same_component.prob = 2
    assert params.model.network.enable is True
    assert pop.enable_graph

    pop.update_partner_assignments(1)
    assert a in c.partners["Sex"]
    assert d not in c.partners["Sex"]
    assert d not in b.partners["Sex"]
    assert d not in a.partners["Sex"]


@pytest.mark.unit
def test_trim_components(make_population):
    n = 200
    pop = make_population(n=20)
    pop.params.model.network.type = "comp_size"
    orig_num_components = len(pop.connected_components())
    assert orig_num_components < n

    pop.params.model.network.component_size.max = n
    pop.trim_graph()
    pop.update_agent_components()
    assert len(pop.components) == orig_num_components

    pop.params.model.network.component_size.max = 2
    pop.trim_graph()
    pop.update_agent_components()
    assert len(pop.components) > orig_num_components
    assert len(pop.components) != n
    assert max(map(len, pop.components)) == 2


@pytest.mark.unit
def test_location_migration_name(make_population, tmpdir):
    param_file = "tests/params/multi_location.yml"
    params = create_params(None, param_file, tmpdir)
    pop = make_population(p=params, n=20)

    pop.pop_random = FakeRandom(0)  # highest weight always picked

    assert len(set(a.location.name for a in pop.all_agents)) == 4
    west_agents = [a for a in pop.all_agents if a.location.name == "west"]
    north_agents = [a for a in pop.all_agents if a.location.name == "north"]

    pop.migrate()

    # all west agents should be east now
    assert all(a.location.name == "east" for a in west_agents)

    # north agents should be un-changed
    assert all(a.location.name == "north" for a in north_agents)


@pytest.mark.unit
def test_location_migration_category(make_population, tmpdir):
    param_file = "tests/params/multi_location.yml"
    params = create_params(None, param_file, tmpdir)
    params.location.migration.attribute = "category"
    params.location.migration.probs_file = "tests/params/migration_cat.csv"
    pop = make_population(p=params, n=20)

    pop.pop_random = FakeRandom(0)  # highest weight always picked

    assert len(set(a.location.name for a in pop.all_agents)) == 4
    rightleft_agents = [a for a in pop.all_agents if a.location.category == "rightleft"]
    updown_agents = [a for a in pop.all_agents if a.location.category == "updown"]

    pop.migrate()

    # all rightleft agents should be updown now
    assert all(a.location.category == "updown" for a in rightleft_agents)

    # north agents should be un-changed
    assert all(a.location.category == "updown" for a in updown_agents)


@pytest.mark.unit
def test_sync_injection_partner_targets_start_injecting(make_population, params):
    """
    An agent that starts injecting must get injection-bond partner targets
    derived from its new drug_type, and become partnerable on those bonds.

    Without this, `oud_state.transitions.inject_start_rate_per_year` is inert
    for transmission: partner targets are derived once in `create_agent`, and
    `NonInj` demographics blocks set injection partner counts to 0, so the agent
    can neither seek nor be selected as an injection partner.
    """
    # these exercise Population.sync_injection_partner_targets directly;
    # oud_state would independently force remission agents to NonInj and
    # perturb the drug_type this test sets up
    params.features.oud_state = False
    pop = make_population(n=10, p=params)
    inj_bonds = [
        bond
        for bond, bond_def in params.classes.bond_types.items()
        if "injection" in bond_def.acts_allowed
    ]
    assert inj_bonds, "basic.yml should define injection-capable bonds"

    agent = pop.create_agent(
        pop.geography.locations["world"], "black_nonhispanic", 0, drug_type="NonInj"
    )
    # NonInj blocks specify set_value 0 for every injection bond
    assert all(agent.mean_num_partners[bond] == 0 for bond in inj_bonds)
    assert all(agent not in pop.partnerable_agents[bond] for bond in inj_bonds)

    agent.drug_type = "Inj"
    pop.sync_injection_partner_targets(agent)

    assert all(agent.mean_num_partners[bond] > 0 for bond in inj_bonds)
    assert all(agent in pop.partnerable_agents[bond] for bond in inj_bonds)


@pytest.mark.unit
def test_sync_injection_partner_targets_stop_injecting(make_population, params):
    """
    An agent that stops injecting must drop its injection targets, so it is no
    longer treated as seeking injection partners it can never match with
    (`select_partner` intersects injection candidates with `pwid_agents`).
    """
    # these exercise Population.sync_injection_partner_targets directly;
    # oud_state would independently force remission agents to NonInj and
    # perturb the drug_type this test sets up
    params.features.oud_state = False
    pop = make_population(n=10, p=params)
    inj_bonds = [
        bond
        for bond, bond_def in params.classes.bond_types.items()
        if "injection" in bond_def.acts_allowed
    ]

    agent = pop.create_agent(
        pop.geography.locations["world"], "black_nonhispanic", 0, drug_type="Inj"
    )
    assert any(agent.mean_num_partners[bond] > 0 for bond in inj_bonds)

    agent.drug_type = "NonInj"
    pop.sync_injection_partner_targets(agent)

    assert all(agent.mean_num_partners[bond] == 0 for bond in inj_bonds)
    assert all(agent.target_partners[bond] == 0 for bond in inj_bonds)


@pytest.mark.unit
def test_sync_injection_partner_targets_leaves_other_bonds_alone(
    make_population, params
):
    """
    Only injection-capable bonds are re-derived. `mean_num_partners` is a
    persistent per-agent trait, so re-drawing sex/PCA bonds on every drug-type
    flip would destabilize individual sexual activity levels.
    """
    # these exercise Population.sync_injection_partner_targets directly;
    # oud_state would independently force remission agents to NonInj and
    # perturb the drug_type this test sets up
    params.features.oud_state = False
    pop = make_population(n=10, p=params)
    other_bonds = [
        bond
        for bond, bond_def in params.classes.bond_types.items()
        if "injection" not in bond_def.acts_allowed
    ]
    assert other_bonds, "basic.yml should define non-injection bonds"

    agent = pop.create_agent(
        pop.geography.locations["world"], "black_nonhispanic", 0, drug_type="NonInj"
    )
    before = {bond: agent.mean_num_partners[bond] for bond in other_bonds}

    agent.drug_type = "Inj"
    pop.sync_injection_partner_targets(agent)

    assert {bond: agent.mean_num_partners[bond] for bond in other_bonds} == before


@pytest.mark.unit
def test_oud_state_transition_syncs_injection_targets(make_population, params):
    """
    Regression test for the whole path: an `oud_state` inject_start transition
    must leave the agent able to form injection partnerships, not just present
    in `pwid_agents`.
    """
    params.features.oud_state = True
    # force the inject_start branch: no remission, certain start
    params.oud_state.transitions.remission_rate_per_year = 0.0
    params.oud_state.transitions.inject_start_rate_per_year = 1e6

    pop = make_population(n=10, p=params)
    inj_bonds = [
        bond
        for bond, bond_def in params.classes.bond_types.items()
        if "injection" in bond_def.acts_allowed
    ]

    agent = pop.create_agent(
        pop.geography.locations["world"], "black_nonhispanic", 0, drug_type="NonInj"
    )
    agent.has_oud = True
    agent.is_in_remission = False
    agent.oud_state.synced_drug_type = "NonInj"
    assert all(agent.mean_num_partners[bond] == 0 for bond in inj_bonds)

    model = SimpleNamespace(
        pop=pop,
        params=params,
        run_random=FakeRandom(0.0),
        time=1,
    )
    agent.oud_state.update_agent(model)

    assert agent.drug_type == "Inj"
    assert agent in pop.pwid_agents.members
    assert all(agent.mean_num_partners[bond] > 0 for bond in inj_bonds), (
        "drug_type flipped to Inj but injection partner targets were not "
        "re-derived, so the agent can never form an injection partnership"
    )
    assert all(agent in pop.partnerable_agents[bond] for bond in inj_bonds)
