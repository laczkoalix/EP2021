"""Functions to load and filter the network"""

import pandas as pd
import numpy as np

# notice that  I use all caps for module level strings
# it may be nicer to export this to a csv, but life is sometimes too short to be tidy
COLUMNS_STR = """    user_id
    public
    completion_percentage
    gender
    region
    last_login
    registration
    AGE
    body
    I_am_working_in_field
    spoken_languages
    hobbies
    I_most_enjoy_good_food
    pets
    body_type
    my_eyesight
    eye_color
    hair_color
    hair_type
    completed_level_of_education
    favourite_color
    relation_to_smoking
    relation_to_alcohol
    sign_in_zodiac
    on_pokec_i_am_looking_for
    love_is_for_me
    relation_to_casual_sex
    my_partner_should_be
    marital_status
    children
    relation_to_children
    I_like_movies
    I_like_watching_movie
    I_like_music
    I_mostly_like_listening_to_music
    the_idea_of_good_evening
    I_like_specialties_from_kitchen
    fun
    I_am_going_to_concerts
    my_active_sports
    my_passive_sports
    profession
    I_like_books
    life_style
    music
    cars
    politics
    relationships
    art_culture
    hobbies_interests
    science_technologies
    computers_internet
    education
    sport
    movies
    travelling
    health
    companies_brands
    more"""
COLUMNS_LIST = [col.strip() for col in COLUMNS_STR.split("\n")]

# reproducibility matters!
np.random.seed(42)


def select_relevant_profiles(all_profiles):
    """Select relevant profiles
    criteria:
    * is public
    * region is selected region
    * AGE specified
    * GENDER SPECIFIED
    """
    public_condition = all_profiles["public"] == 1
    age_condition = all_profiles["AGE"] > 14
    gender_condition = all_profiles["gender"].isin([0, 1])
    return all_profiles.loc[public_condition & age_condition & gender_condition]


def select_relevant_edges(all_edges, selected_ids):
    """Select relevant edges for those profiles that are relevant"""
    source_condition = all_edges["source"].isin(selected_ids)
    sink_condition = all_edges["sink"].isin(selected_ids)
    return all_edges.loc[source_condition & sink_condition]


def convert_edges_to_undirected(edges):
    """Convert edges to undirected, and keep only mutual connections"""
    undirected_edges = (
        edges.assign(
            smaller_id=lambda df: df[["source", "sink"]].min(axis=1),
            greater_id=lambda df: df[["source", "sink"]].max(axis=1),
        )
        .groupby(["smaller_id", "greater_id"])
        .agg({"source": "count"})
    )
    print(undirected_edges["source"].value_counts())
    return (
        undirected_edges.loc[undirected_edges["source"] == 2]
        .drop("source", axis=1)
        .reset_index()
    )


def load_and_select_profiles_and_edges():
    """load and select relevant profiles, then filter and undirect edges"""
    print("loading profiles")
    # TODO: Add some functionality to only read a subset of the data!
    profiles = pd.read_csv(
        "soc-pokec-profiles.txt",
        sep="\t",
        names=COLUMNS_LIST,
        index_col=False,
        usecols=["user_id", "public", "gender", "region", "AGE"],
    )
    print("loading edges")
    edges = pd.read_csv(
        "soc-pokec-relationships.txt", sep="\t", names=["source", "sink"]
    )
    selected_profiles = select_relevant_profiles(profiles)
    selected_ids = selected_profiles["user_id"].unique()
    selected_edges = select_relevant_edges(edges, selected_ids)

    undirected_edges = convert_edges_to_undirected(selected_edges)
    nodes_with_edges = set(undirected_edges["smaller_id"].unique()).union(
        undirected_edges["greater_id"].unique()
    )
    print(f"Selected profiles: {len(selected_profiles)}")
    print(f"Nodes with edges: {len(nodes_with_edges)}")
    selected_profiles = selected_profiles[
        selected_profiles["user_id"].isin(nodes_with_edges)
    ]
    selected_profiles["AGE"] = selected_profiles["AGE"].clip(upper=50)
    selected_profiles = remove_test_set_gender_and_age(selected_profiles)
    return selected_profiles, undirected_edges


def remove_test_set_gender_and_age(nodes):
    """Remove the gender feature from a subset of the nodes for estimation"""
    # todo: the 40k  random can be adjusted if youre working with a subset
    test_profiles = np.random.choice(nodes["user_id"].unique(), 40000, replace=False)
    nodes["TRAIN_TEST"] = "TRAIN"
    test_condition = nodes["user_id"].isin(test_profiles)
    nodes.loc[test_condition, ["AGE", "gender"]] = np.nan
    nodes.loc[test_condition, ["TRAIN_TEST"]] = "TEST"

    return nodes

"Visualization function examples for the homework project"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_degree_distribution(G):
    """Plot a degree distribution of a graph
    TODO: log-log binning! To understand this better, check out networksciencebook.com
    """
    plot_df = (
        pd.Series(dict(G.degree)).value_counts().sort_index().to_frame().reset_index()
    )
    plot_df.columns = ["k", "count"]
    plot_df["log_k"] = np.log(plot_df["k"])
    plot_df["log_count"] = np.log(plot_df["count"])
    fig, ax = plt.subplots()

    ax.scatter(plot_df["k"], plot_df["count"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.suptitle("Mutual Degree Distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("count_k")


def plot_age_distribution_by_gender(nodes):
    """Plot a histogram where the color represents gender"""
    plot_df = nodes[["AGE", "gender"]].copy(deep=True).astype(float)
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.histplot(data=plot_df, x="AGE", hue="gender", bins=np.arange(0, 45, 5) + 15)


def plot_node_degree_by_gender(nodes, G):
    """Plot the average of node degree across age and gender"""
    # TODO: this could be generalized for different node level statistics as well!
    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    plot_df = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg({"degree": "mean"}).reset_index()
    )
    sns.lineplot(data=plot_df, x="AGE", y="degree", hue="gender")


def plot_age_relations_heatmap(edges_w_features):
    """Plot a heatmap that represents the distribution of edges"""
    # TODO: check what happpens without logging
    # TODO: instead of logging check what happens if you normalize with the row sum
    #  make sure you figure out an interpretation of that as well!
    # TODO: separate these charts by gender as well
    # TODO: column names could be nicer
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged)


"""Functions to manipulate graphs"""

import pandas as pd
import networkx as nx
import numpy as np


def create_graph_from_nodes_and_edges(nodes, edges):
    """Create a networkx graph object with all relevant features"""
    node_attributes = nodes.set_index("user_id").to_dict(orient="index")
    node_attributes_list = [
        (index, attr_dict) for index, attr_dict in node_attributes.items()
    ]
    G = nx.Graph()
    G.add_nodes_from(node_attributes_list)
    G.add_edges_from(edges.values.tolist())
    return G


def get_nbrs_for_node(node_id, G):
    """Return ids of nbrs of node"""
    return list(dict(G[node_id]).keys())


def get_features_of_node_list(node_list, node_df):
    """Return the features of a subset of nodes"""
    return node_df.loc[node_list, ["AGE", "gender"]].values.tolist()


def add_node_features_to_edges(nodes, edges):
    """Add features of nodes to edges in order to create heatmaps"""
    # TODO: column names could be nicer!
    edges_w_features = edges.merge(
        nodes[["user_id", "AGE", "gender"]].set_index("user_id"),
        how="left",
        left_on="smaller_id",
        right_index=True,
    )
    edges_w_features = edges_w_features.merge(
        nodes[["user_id", "AGE", "gender"]].set_index("user_id"),
        how="left",
        left_on="greater_id",
        right_index=True,
    )
    return edges_w_features