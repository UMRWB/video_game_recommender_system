# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import time
import os
import ast
from urllib.parse import quote  # Needed for _row_html tag links
import scipy.stats as stats  # For the Z-score

# Ensure pyarrow is installed for parquet: pip install pyarrow
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from sentence_transformers import SentenceTransformer

torch.classes.__path__ = [] # remove the error

# --- Streamlit Page Configuration & ** UPDATED: CSS ** ---
st.set_page_config(
    page_title="Game Recommender", layout="wide", initial_sidebar_state="expanded"
)
st.components.v1.html(
    """
    <script>
    function toggleTags(containerId) {
        const container = document.getElementById(containerId);
        const toggle = container.previousElementSibling;
        
        if (container.style.display === "inline" || container.style.display === "") {
            container.style.display = "none";
            toggle.textContent = "...";
        } else {
            container.style.display = "inline";
            toggle.textContent = "Hide";
        }
    }
    </script>
    """,
    height=0
)
st.markdown(  # Enhanced CSS with corrected title color and tag hover effects
    """<style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .stMultiSelect div[data-baseweb="select"] > div { padding-top: 0; }

        /* --- Game Row Styling --- */
        /* ... (Game Row, Seed Row styles remain the same) ... */
        .game-row { display: flex; align-items: flex-start; margin-bottom: 1rem; padding: 0.75rem; border: 1px solid #3a3a3a; border-radius: 6px; background-color: #2a2a2e; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: background-color 0.2s ease; }
        .game-row:hover { background-color: #333337; }
        .seed-row { border-color: #4CAF50; background-color: rgba(76, 175, 80, 0.1); }
        .seed-row:hover { background-color: rgba(76, 175, 80, 0.15); }

        /* --- Thumbnail --- */
        /* ... (Thumbnail styles remain the same) ... */
        .thumb-link { display: block; flex-shrink: 0; margin-right: 1.25rem; border-radius: 4px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
        .thumb { width: 184px; height: 69px; display: block; object-fit: cover; background-color: #111; transition: transform 0.2s ease; }
        .thumb-link:hover .thumb { transform: scale(1.05); }

        /* --- Metadata Area --- */
        .meta { flex-grow: 1; overflow: hidden; }

        /* --- Title --- */
        .title {
            font-weight: 600; font-size: 1.1em;
            /* MODIFIED: Force white color */
            color: #ffffff !important;
            display: block; margin-bottom: 0.4rem; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis;
            text-decoration: none !important; /* Force no underline */
            transition: color 0.2s ease, transform 0.2s ease;
            transform-origin: left center;
        }
         .title:hover {
            /* MODIFIED: Force red color */
            color: #e53e3e !important;
            text-decoration: none !important; /* Force no underline on hover */
            transform: scale(1.03);
         }

        /* --- Tags --- */
        .tags { margin-bottom: 0.6rem; line-height: 1.1; }
        .tags span { /* Style for the tag text box */
            display: inline-block; /* Needed for transform */
            background-color: #5c6b7a;
            color: #dcdedf;
            padding: 0.15rem 0.5rem; margin-right: 0.4rem; margin-bottom: 0.4rem;
            border-radius: 3px; font-size: 0.75em; white-space: nowrap;
            /* MODIFIED: Add transitions for background and transform */
            transition: background-color 0.2s ease, transform 0.15s ease;
            transform-origin: center center; /* Scale from center */
        }
        .tags a { /* The link wrapping the tag */
             text-decoration: none;
             color: inherit;
             display: inline-block; /* Ensures link takes up space */
        }
        /* MODIFIED: Target span within hovered link */
        .tags a:hover span {
            background-color: #3687c8; /* Brighter blue on hover */
            color: #ffffff; /* White text on hover */
            transform: scale(1.08); /* Make tag slightly larger */
        }
        .tags .more-tags {
            display: none; /* Hidden initially */
        }
        .tags .toggle-tags {
            cursor: pointer;
            color: #8bacd9;
            background-color: #3a4b5d;
            padding: 0.15rem 0.5rem;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            border-radius: 3px;
            font-size: 0.75em;
            transition: background-color 0.2s ease;
        }
        .tags .toggle-tags:hover {
            background-color: #526c8c;
        }

        /* --- Details Section --- */
        /* ... (Details styles remain the same) ... */
        .details { font-size: 0.85em; color: #a0a0a5; line-height: 1.5; display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; }
        .detail-item { white-space: nowrap; }
        .detail-item strong { color: #babcbf; margin-right: 0.3rem; font-weight: 500; }
        .score-value { font-weight: bold; color: #ffc107; }
        .rating-positive { color: #66C0F4; } .rating-mixed { color: #a89262; } .rating-negative { color: #c74a4a; }
        .price-free { color: #a1d95a; font-weight: bold; } .price-tag { color: #c6d4df; }
        .conf-score-value { color: #bdbdbd; }

    </style>""",
    unsafe_allow_html=True,
)


# --- Global Constants & Cache File Paths (Unchanged) ---
DATA_FILE = "games_data_cleaned.csv"
N_CLUSTERS_DEFAULT = 11
CACHE_DIR = "streamlit_cache"
PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data_listtags_wscore.parquet")
FEATURES_TITLE_PATH = os.path.join(CACHE_DIR, "features_embeddings_title.npy")
FEATURES_DESC_PATH = os.path.join(CACHE_DIR, "features_embeddings_desc.npy")
FEATURES_TAGS_PATH = os.path.join(CACHE_DIR, "features_multihot_tags.npy")
FEATURES_TAG_VOCAB_PATH = os.path.join(CACHE_DIR, "features_tag_vocab.npy")
CLUSTERED_DATA_PATH = os.path.join(CACHE_DIR, "clustered_data_listtags_wscore.parquet")
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Preprocessing Functions (Unchanged) ---
def preprocess_game_title(title):
    if not title or pd.isna(title):
        return ""
    title = str(title)
    title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    )
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[Â®â„¢Â©]", "", title)
    return title


def preprocess_game_description(description):
    if not description or pd.isna(description):
        return ""
    description = str(description)
    description = (
        unicodedata.normalize("NFKD", description)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    description = description.lower()
    description = re.sub(r"<[^>]+>", "", description)
    description = re.sub(r"http\S+|www\S+", "", description)
    description = re.sub(r"[\r\n\t]+", " ", description)
    description = re.sub(r"\s+", " ", description).strip()
    return description


def parse_tag_list(tag_string):
    if pd.isna(tag_string) or not isinstance(tag_string, str):
        return []
    try:
        tag_list = ast.literal_eval(tag_string)
        return (
            [str(tag).lower().strip() for tag in tag_list]
            if isinstance(tag_list, list)
            else []
        )
    except (ValueError, SyntaxError, TypeError):
        return []


# --- ** NEW: Wilson Score Calculation Function ** ---
def wilson_lower_bound(positive_ratio_pct, n_reviews, confidence=0.95):
    """Calculates the lower bound of the Wilson score interval."""
    if (
        n_reviews is None
        or n_reviews <= 0
        or pd.isna(n_reviews)
        or positive_ratio_pct is None
        or pd.isna(positive_ratio_pct)
    ):
        return 0.0
    n = int(n_reviews)
    p_hat = float(positive_ratio_pct) / 100.0
    if not (0 <= p_hat <= 1):
        p_hat = max(0, min(1, p_hat))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    numerator = (
        p_hat
        + (z**2 / (2 * n))
        - z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    )
    denominator = 1 + (z**2 / n)
    return numerator / denominator


# --- Caching Functions (Unchanged - with Disk Persistence) ---
@st.cache_resource
def load_sentence_transformer():
    print("Loading Sentence Transformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Model loaded.")
    return model


# --- ** UPDATED: load_and_preprocess_data ** ---
@st.cache_data(ttl=None)
def load_and_preprocess_data(filepath, cache_path):
    """Loads raw data, preprocesses it, adds Wilson score, caches to disk."""
    if os.path.exists(cache_path):
        print(f"Loading PREPROCESSED data (w/ Wilson) from: {cache_path}")
        df = pd.read_parquet(cache_path)
        print("Preprocessed data loaded.")
        return df

    print(f"Cache not found. Loading/Preprocessing RAW data from {filepath}...")
    # ... (Existing loading, date parsing, text preprocessing, tag parsing) ...
    try:
        df = pd.read_csv(filepath)
        df["date_release"] = pd.to_datetime(df["date_release"], errors="coerce")
        df.dropna(subset=["date_release"], inplace=True)
        df["proc_title"] = df["title"].apply(preprocess_game_title)
        df["proc_description"] = df["description"].apply(preprocess_game_description)
        print("Parsing 'tags' column...")
        df["tags_list"] = df["tags"].apply(parse_tag_list)
        print("Finished parsing tags.")

        # --- Add Wilson Score Calculation ---
        print("Calculating Wilson score lower bound...")
        df["positive_ratio"] = pd.to_numeric(df["positive_ratio"], errors="coerce")
        df["user_reviews"] = pd.to_numeric(df["user_reviews"], errors="coerce")
        df["wilson_score"] = df.apply(
            lambda row: wilson_lower_bound(row["positive_ratio"], row["user_reviews"]),
            axis=1,
        )
        print("Wilson score calculated.")
        # --- End Wilson Score Calculation ---

        print("Saving preprocessed data (w/ Wilson) to disk cache...")
        df.to_parquet(cache_path, index=False)
        print("Saved.")
        return df
    except FileNotFoundError:
        st.error(f"Error: RAW data file '{filepath}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading/preprocessing raw data: {e}")
        return None


@st.cache_data(ttl=None)
def generate_features(_df, _model, title_path, desc_path, tags_path, vocab_path):
    if (
        os.path.exists(title_path)
        and os.path.exists(desc_path)
        and os.path.exists(tags_path)
        and os.path.exists(vocab_path)
    ):
        print("Loading FEATURES from disk cache...")
        title_features = np.load(title_path)
        desc_features = np.load(desc_path)
        tag_features = np.load(tags_path)
        tag_vocab = np.load(vocab_path, allow_pickle=True).tolist()
        if (
            title_features.shape[0] == len(_df)
            and desc_features.shape[0] == len(_df)
            and tag_features.shape[0] == len(_df)
            and tag_features.shape[1] == len(tag_vocab)
        ):
            print("Features loaded.")
            return {
                "title": title_features,
                "description": desc_features,
                "tags": tag_features,
                "tag_names": tag_vocab,
            }
        else:
            print("Shape/Vocab mismatch. Recalculating...")
    print("Generating FEATURES...")
    start_time = time.time()
    title_features = _model.encode(
        _df["proc_title"].fillna("").tolist(), batch_size=64, show_progress_bar=False
    )
    desc_features = _model.encode(
        _df["proc_description"].fillna("").tolist(),
        batch_size=64,
        show_progress_bar=False,
    )
    print("Generating multi-hot tags...")
    mlb = MultiLabelBinarizer()
    tag_features = mlb.fit_transform(_df["tags_list"]).astype(np.float32)
    tag_vocab = mlb.classes_.tolist()
    features_dict = {
        "title": title_features,
        "description": desc_features,
        "tags": tag_features,
        "tag_names": tag_vocab,
    }
    print(f"Features generated. Time: {time.time() - start_time:.2f}s")
    print("Saving features and vocab...")
    np.save(title_path, title_features)
    np.save(desc_path, desc_features)
    np.save(tags_path, tag_features)
    np.save(vocab_path, np.array(tag_vocab, dtype=object))
    print("Saved.")
    return features_dict


# --- perform_clustering (Needs updated cache path) ---
@st.cache_data(ttl=None)
def perform_clustering(_df, _features, cache_path):  # Pass CLUSTERED_DATA_PATH here
    if os.path.exists(cache_path):
        print(f"Loading CLUSTERED data from disk cache: {cache_path}")
        df_clustered = pd.read_parquet(cache_path)
        print("Clustered data loaded.")
        return df_clustered
    print("Performing CLUSTERING...")
    start_time = time.time()
    df_clustered = _df.copy()
    features_to_stack = []
    if "title" in _features:
        features_to_stack.append(_features["title"])
    if "description" in _features:
        features_to_stack.append(_features["description"])
    if "tags" in _features:
        features_to_stack.append(_features["tags"])
    if not features_to_stack:
        st.warning("No features for clustering.")
        df_clustered["cluster"] = -1
    else:
        combined = np.hstack(features_to_stack)
        n_clusters = max(5, min(N_CLUSTERS_DEFAULT, int(len(df_clustered) ** 0.4)))
        print(f"Using {n_clusters} clusters.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_clustered["cluster"] = kmeans.fit_predict(combined)
        print(f"Clustering complete. Time: {time.time() - start_time:.2f}s")
    print("Saving clustered data...")
    df_clustered.to_parquet(cache_path, index=False)
    print("Saved.")
    return df_clustered


# --- Filtering Function (Unchanged) ---
def apply_filters(df, selected_tags, date_range, max_price):
    print(
        f"Applying filters: Tags={len(selected_tags)}, Date={date_range}, Price<={max_price}"
    )
    filtered_df = df.copy()
    if date_range and len(date_range) == 2:
        start_date, end_date = (
            pd.to_datetime(date_range[0]),
            pd.to_datetime(date_range[1]),
        )
        filtered_df = filtered_df[
            (filtered_df["date_release"] >= start_date)
            & (filtered_df["date_release"] <= end_date)
        ]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df["price_final"].fillna(0) <= max_price]
    if selected_tags:
        lowercase_selected_tags = {tag.lower() for tag in selected_tags}
        tag_filter_mask = filtered_df["tags_list"].apply(
            lambda game_tags: lowercase_selected_tags.issubset(set(game_tags))
        )
        filtered_df = filtered_df[tag_filter_mask]
    print(f"Filter results: {len(filtered_df)} rows")
    return filtered_df


# --- Recommendation Functions (Corrected idx assignment) ---
def get_recommendations_approach1(*args, **kwargs):
    (
        game_title,
        full_df,
        filtered_df,
        features,
        n_recommendations,
        w_title,
        w_desc,
        w_tag,
        w_popularity,
    ) = args
    print(f"\n--- Running A1 for '{game_title}' ---")
    status_messages = []
    processed_input_title = preprocess_game_title(game_title)

    # --- ** CORRECTED Block to find idx ** ---
    idx = None  # Initialize idx
    matches = full_df[full_df["proc_title"] == processed_input_title]  # Try exact match

    if not matches.empty:
        idx = matches.index[0]  # Assign idx on exact match
    else:
        # Try approximate match if exact failed
        matches_approx = full_df[
            full_df["proc_title"].str.contains(
                processed_input_title, case=False, na=False
            )
        ]
        if not matches_approx.empty:
            idx = matches_approx.index[0]  # Assign idx on approximate match
            matches = matches_approx  # Use the approx matches DataFrame
        else:
            # Neither exact nor approximate match found
            return pd.DataFrame(), status_messages + [
                f"Err(A1): Input '{game_title}' not found (exact or approx)."
            ]

    # If we reach here, idx MUST have been assigned
    found_title = full_df.loc[idx, "title"]
    status_messages.append(f"Info(A1): Input '{found_title}' found (Index: {idx}).")
    # --- ** End CORRECTED Block ** ---

    # --- Now safe to use idx for similarity calcs ---
    sim_scores_all = np.zeros(len(full_df))
    tag_sim_comp = np.zeros(len(full_df))
    if w_tag > 0 and "tags" in features and features["tags"].shape[1] > 0:
        try:
            tag_sim_comp = (
                w_tag
                * cosine_similarity(
                    features["tags"][idx].reshape(1, -1), features["tags"]
                )[0]
            )
        except Exception as e:  # Catch potential errors during similarity calc
            status_messages.append(f"Warn(A1): Tag sim error: {e}")
            # Continue without tag similarity if it fails

    if w_title > 0 and "title" in features:
        try:
            sim_scores_all += (
                w_title
                * cosine_similarity(
                    features["title"][idx].reshape(1, -1), features["title"]
                )[0]
            )
        except Exception as e:
            status_messages.append(f"Warn(A1): Title sim error: {e}")

    if w_desc > 0 and "description" in features:
        try:
            sim_scores_all += (
                w_desc
                * cosine_similarity(
                    features["description"][idx].reshape(1, -1),
                    features["description"],
                )[0]
            )
        except Exception as e:
            status_messages.append(f"Warn(A1): Desc sim error: {e}")

    sim_scores_all += tag_sim_comp  # Add tag similarity (might be zeros if failed)

    # --- ** Use Wilson Score for Popularity ** ---
    filtered_indices = filtered_df.index
    sim_scores_filtered = pd.Series(sim_scores_all, index=full_df.index).reindex(
        filtered_indices, fill_value=-np.inf
    )
    pop_filtered = filtered_df["wilson_score"].fillna(0.0).values  # Use wilson_score
    # --- End Wilson Score Usage ---

    total_w = w_title + w_desc + w_tag
    norm = max(1e-6, total_w if not np.isclose(total_w, 1.0) else 1.0)
    w_c = 1.0 - w_popularity
    final_scores = (w_c * (sim_scores_filtered / norm)) + (w_popularity * pop_filtered)

    # Use loc for safe assignment, check if idx is in the filtered scores index
    if idx in final_scores.index:
        final_scores.loc[idx] = -np.inf

    if final_scores.empty or final_scores.isna().all():
        return pd.DataFrame(), status_messages + ["Info(A1): No valid scores."]

    top_indices = final_scores.nlargest(n_recommendations).index
    valid_indices = top_indices[top_indices.isin(filtered_df.index)]
    if valid_indices.empty:
        return pd.DataFrame(), status_messages + ["Info(A1): No valid recs."]
    # Include wilson_score in output
    recs = filtered_df.loc[valid_indices][
        [
            "title",
            "positive_ratio",
            "user_reviews",
            "wilson_score",
            "date_release",
            "price_final",
        ]
    ].copy()
    recs["score"] = final_scores.loc[valid_indices]
    return recs, status_messages


def get_recommendations_approach2(*args, **kwargs):
    (
        game_title,
        full_df,
        filtered_df,
        features,
        n_recommendations,
        w_title,
        w_desc,
        w_tag,
        w_popularity_cluster,
    ) = args
    print(f"\n--- Running A2 for '{game_title}' ---")
    status_messages = []
    if "cluster" not in full_df.columns:
        return pd.DataFrame(), status_messages + ["Err(A2): No cluster data."]
    processed_input_title = preprocess_game_title(game_title)

    # --- ** CORRECTED Block to find idx ** ---
    idx = None  # Initialize idx
    matches = full_df[full_df["proc_title"] == processed_input_title]  # Try exact match

    if not matches.empty:
        idx = matches.index[0]  # Assign idx on exact match
    else:
        # Try approximate match if exact failed
        matches_approx = full_df[
            full_df["proc_title"].str.contains(
                processed_input_title, case=False, na=False
            )
        ]
        if not matches_approx.empty:
            idx = matches_approx.index[0]  # Assign idx on approximate match
            matches = matches_approx  # Use the approx matches DataFrame
        else:
            # Neither exact nor approximate match found
            return pd.DataFrame(), status_messages + [
                f"Err(A2): Input '{game_title}' not found (exact or approx)."
            ]

    # If we reach here, idx MUST have been assigned
    found_title = full_df.loc[idx, "title"]
    cluster_id = full_df.loc[idx, "cluster"]
    status_messages.append(
        f"Info(A2): Input '{found_title}' in Cluster {cluster_id} (Index: {idx})."
    )
    # --- ** End CORRECTED Block ** ---

    # --- Weight validation ---
    content_weights = {"title": w_title, "desc": w_desc, "tag": w_tag}
    total_content_weight = sum(content_weights.values())
    if not np.isclose(total_content_weight, 1.0):
        print(
            f"Warning (A2): Content weights normalized (Sum={total_content_weight:.2f})."
        )
    if total_content_weight > 1e-6:
        for k in content_weights:
            content_weights[k] /= total_content_weight
    else:
        for k in content_weights:
            content_weights[k] = 1.0 / len(content_weights)
    w_title_norm, w_desc_norm, w_tag_norm = (
        content_weights["title"],
        content_weights["desc"],
        content_weights["tag"],
    )
    if not (0 <= w_popularity_cluster <= 1):
        w_popularity_cluster = max(0, min(1, w_popularity_cluster))
        print("Warning (A2): Pop weight clamped.")
    w_content_cluster = 1.0 - w_popularity_cluster

    cluster_filtered_df = filtered_df[
        filtered_df["cluster"] == cluster_id
    ].copy()  # Use copy
    if len(cluster_filtered_df) <= 1:
        return pd.DataFrame(), status_messages + [
            f"Warn(A2): <=1 game in cluster {cluster_id}."
        ]
    subset_indices = cluster_filtered_df.index
    subset_map = {orig_idx: i for i, orig_idx in enumerate(subset_indices)}
    idx_in_subset = subset_map.get(idx, -1)
    if idx_in_subset == -1:
        return pd.DataFrame(), status_messages + [
            "Info(A2): Input excluded."
        ]  # If input excluded, can't proceed

    subset_features = {
        "title": features["title"][subset_indices],
        "description": features["description"][subset_indices],
        "tags": features["tags"][subset_indices],
    }
    content_sim_scores = np.zeros(len(subset_indices))  # Content similarity score
    try:
        t_s, d_s, g_s = (
            np.zeros_like(content_sim_scores),
            np.zeros_like(content_sim_scores),
            np.zeros_like(content_sim_scores),
        )
        if w_title_norm > 0:
            t_s = (
                w_title_norm
                * cosine_similarity(
                    subset_features["title"][idx_in_subset].reshape(1, -1),
                    subset_features["title"],
                )[0]
            )
        if w_desc_norm > 0:
            d_s = (
                w_desc_norm
                * cosine_similarity(
                    subset_features["description"][idx_in_subset].reshape(1, -1),
                    subset_features["description"],
                )[0]
            )
        if w_tag_norm > 0:
            g_s = (
                w_tag_norm
                * cosine_similarity(
                    subset_features["tags"][idx_in_subset].reshape(1, -1),
                    subset_features["tags"],
                )[0]
            )
            content_sim_scores = t_s + d_s + g_s
    except Exception as e:
        status_messages.append(f"Warn(A2): Sim error: {e}")
        content_sim_scores = np.full(
            len(subset_indices), -np.inf
        )  # Mark as invalid on error

    # --- ** Use Wilson Score for Popularity ** ---
    pop_subset = cluster_filtered_df["wilson_score"].fillna(0.0).values
    final_scores = (w_content_cluster * content_sim_scores) + (
        w_popularity_cluster * pop_subset
    )
    # --- End Wilson Score Usage ---

    final_scores[idx_in_subset] = -np.inf  # Exclude self
    final_series = pd.Series(final_scores, index=subset_indices)
    if final_series.empty or np.isneginf(final_series).all():
        return pd.DataFrame(), status_messages + ["Info(A2): No valid scores."]
    top_indices = final_series.nlargest(n_recommendations).index
    valid_indices = top_indices[top_indices.isin(cluster_filtered_df.index)]
    if valid_indices.empty:
        return pd.DataFrame(), status_messages + ["Info(A2): Top indices invalid."]
    # Include wilson_score in output
    recs = cluster_filtered_df.loc[valid_indices][
        [
            "title",
            "positive_ratio",
            "user_reviews",
            "wilson_score",
            "cluster",
            "date_release",
            "price_final",
        ]
    ].copy()
    recs["score"] = final_series.loc[valid_indices]
    return recs, status_messages


# --- HTML Row Rendering Function (Visually Appealing) ---
def _row_html(row: pd.Series, *, seed=False) -> str:
    """Renders a pandas Series as a visually appealing HTML row."""
    if not isinstance(row, pd.Series):
        return "<div>Error: Invalid row data</div>"

    try:
        appid = int(row.get("app_id", 0))
    except (ValueError, TypeError):
        appid = 0
    thumb_url = (
        f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
        # f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"
        if appid
        else "about:blank"
    )
    store_url = f"https://store.steampowered.com/app/{appid}/" if appid else "#"
    title = str(row.get("title", "N/A")).replace("<", "&lt;").replace(">", "&gt;")

    # --- Tags ---
    tags_html = ""
    tags_list = row.get("tags_list", [])  # Expects tags_list column
    if isinstance(tags_list, list) or isinstance(tags_list, np.ndarray):
        for t in tags_list[:10]:
            encoded_tag = quote(t)
            tag_link = f"?filter_tag={encoded_tag}"
            tags_html += f'<a href="{tag_link}" target="_self" title="Filter by {t}"><span>{t}</span></a>'
        if len(tags_list) > 10:
            tags_html += "<span>...</span>"

    # --- Details ---
    details_items = []  # Collect detail items as strings

    # 1. Score (if not seed)
    if not seed:
        score_val = row.get("score", None)
        if score_val is not None:
            try:
                details_items.append(
                    f"<span class='detail-item'><strong>Score:</strong> <span class='score-value'>{score_val:.3f}</span></span>"
                )
            except:
                pass

    # 2. Price
    price = row.get("price_final", None)
    if price is not None:
        try:
            price_val = float(price)
            price_str = (
                "<span class='price-free'>Free</span>"
                if price_val == 0
                else f"<span class='price-tag'>${price_val:.2f}</span>"
            )
            details_items.append(
                f"<span class='detail-item'><strong>Price:</strong> {price_str}</span>"
            )
        except:
            pass

    # 3. Rating & Reviews (Combine)
    rating = row.get("rating", "N/A")
    pos_ratio = row.get("positive_ratio", None)
    reviews = row.get("user_reviews", None)
    rating_class = "rating-mixed"
    review_details = ""
    if isinstance(rating, str):
        if "positive" in rating.lower():
            rating_class = "rating-positive"
        elif "negative" in rating.lower():
            rating_class = "rating-negative"
    rating_str = (
        f"<span class='{rating_class}'>{rating}</span>" if rating != "N/A" else "N/A"
    )
    review_details = rating_str
    if pos_ratio is not None:
        try:
            review_details += f"&nbsp;({int(pos_ratio)}% Pos)"
        except:
            pass  # Use &nbsp; for non-breaking space
    if reviews is not None:
        try:
            review_details += f"&nbsp;of&nbsp;{int(reviews):,} reviews"
        except:
            pass
    details_items.append(
        f"<span class='detail-item'><strong>Rating:</strong> {review_details}</span>"
    )

    # 4. Confidence Score
    wilson_score_val = row.get("wilson_score", None)
    if wilson_score_val is not None:
        try:
            details_items.append(
                f"<span class='detail-item'><strong>Conf.&nbsp;Score:</strong> <span class='conf-score-value'>{wilson_score_val:.3f}</span></span>"
            )
        except:
            pass

    # Join detail items and wrap in the details div
    details_html = f"<div class='details'>{''.join(details_items)}</div>"

    # --- Assemble Row ---
    css_class = "game-row seed-row" if seed else "game-row"
    return f"""
    <div class="{css_class}">
        <a href="{store_url}" target="_blank" class="thumb-link"><img src="{thumb_url}" class="thumb" alt="{title} thumbnail"></a>
        <div class="meta">
            <a href="{store_url}" target="_blank" class="title" title="{title}">{title}</a>
            <div class="tags">{tags_html}</div>
            {details_html}
        </div>
    </div>
    """


# --- Main Streamlit App ---
# ... (Initialization logic remains the same, ensure it uses updated cache paths) ...
st.title("Game Recommender Engine")
st.markdown("Compare recommendation approaches with filtering & keyword search.")
init_msg = st.empty()
with init_msg.status("Initializing data...", expanded=True):
    st.write("ML model...")
    model = load_sentence_transformer()
    st.write("Data...")
    games_df_processed = load_and_preprocess_data(
        DATA_FILE, PROCESSED_DATA_PATH
    )  # Use updated path
    st.write("Features...")
    features_dict = (
        generate_features(
            games_df_processed,
            model,
            FEATURES_TITLE_PATH,
            FEATURES_DESC_PATH,
            FEATURES_TAGS_PATH,
            FEATURES_TAG_VOCAB_PATH,
        )
        if games_df_processed is not None
        else None
    )
    st.write("Clustering...")
    games_df_clustered = (
        perform_clustering(games_df_processed, features_dict, CLUSTERED_DATA_PATH)
        if games_df_processed is not None and features_dict is not None
        else None
    )  # Use updated path

# --- Check Initialization Success (Unchanged) ---
if games_df_clustered is not None and features_dict is not None:
    init_msg.success("Initialization complete!")
    try:
        unique_tags_for_ui = sorted(
            list(
                set(
                    tag
                    for sublist in games_df_clustered["tags_list"]
                    for tag in sublist
                )
            )
        )
    except Exception:
        unique_tags_for_ui = []
    query_params = st.query_params
    filter_tag_query = query_params.get("filter_tag")
    default_tags_selected_ui = []
    tag_message_placeholder = st.sidebar.empty()
    if filter_tag_query:
        cleaned_query_tag = filter_tag_query.lower().strip()
        if cleaned_query_tag in unique_tags_for_ui:
            default_tags_selected_ui.append(cleaned_query_tag)
        with tag_message_placeholder:
            st.success(f"Filter: {filter_tag_query}")
            if st.button("Clear URL filter"):
                st.query_params.clear()
                st.rerun()
    elif filter_tag_query is None:
        pass
    else:
        with tag_message_placeholder:
            st.warning(f"URL Tag '{filter_tag_query}' not found.")

    # --- Sidebar Setup ---
    st.sidebar.header("1. Select Input Game")

    # ** Keyword Search **
    keyword = st.sidebar.text_input("Search game title by keyword:")
    if keyword:
        processed_keyword = preprocess_game_title(keyword)
        match_df = games_df_clustered[
            games_df_clustered["proc_title"].str.contains(
                processed_keyword, case=False, na=False
            )
        ]
    else:
        match_df = pd.DataFrame(
            columns=games_df_clustered.columns
        )  # Empty if no keyword

    # ** MODIFIED: Create AND Sort options list **
    # Get titles from matches and sort them alphabetically
    selectbox_options = sorted(
        match_df["title"].unique().tolist()
    )  # Added unique() and sorted()

    # Update placeholder logic based on whether options exist
    placeholder_text = (
        "Pick from keyword matches..."
        if keyword and selectbox_options
        else ("No matches found..." if keyword else "Enter keyword above...")
    )

    # Use the sorted list in the selectbox
    selected_title_from_keyword = st.sidebar.selectbox(
        "Select matching game:",
        options=selectbox_options,  # Use the sorted list
        index=None,
        placeholder=placeholder_text,
    )
    input_app_id_str = st.sidebar.text_input("Or enter Steam App ID:")
    selected_game_title = None
    input_error = False
    if selected_title_from_keyword:
        selected_game_title = selected_title_from_keyword
        st.sidebar.caption(f"Using: {selected_game_title}")
    elif input_app_id_str:
        try:
            input_app_id = int(input_app_id_str)
            title_match = games_df_clustered[
                games_df_clustered["app_id"] == input_app_id
            ]
            if not title_match.empty:
                selected_game_title = title_match["title"].iloc[0]
                st.sidebar.caption(f"Using AppID {input_app_id}: {selected_game_title}")
            else:
                st.sidebar.error(f"App ID {input_app_id} not found.")
                input_error = True
        except ValueError:
            st.sidebar.error("App ID must be integer.")
            input_error = True

    st.sidebar.header("2. Apply Filters")
    selected_tags = st.sidebar.multiselect(
        "Filter Tags (AND):",
        options=unique_tags_for_ui,
        default=default_tags_selected_ui,
    )
    min_date = games_df_clustered["date_release"].min().date()
    max_date = games_df_clustered["date_release"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Release Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    max_p = games_df_clustered["price_final"].max()
    selected_max_price = st.sidebar.number_input(
        "Max Price ($):",
        min_value=0.0,
        max_value=float(max_p),
        value=float(max_p),
        step=1.0,
        format="%.2f",
    )

    # --- ** UPDATED: Configure Recommendations - Add A2 Pop Weight ** ---
    st.sidebar.header("3. Configure Recommendations")
    num_recs = st.sidebar.number_input("Num Recs:", 1, 20, 10)
    st.sidebar.markdown("---")

    # default weights
    a1_w_title = 0.2
    a1_w_desc = 0.3
    a1_w_tag = 0.5
    a1_w_pop = 0.3
    a2_w_title = 0.2
    a2_w_desc = 0.3
    a2_w_tag = 0.5
    a2_w_pop = 0.3

    adv_options = st.sidebar.toggle("Advanced Options", False, key="debug_mode")
    if adv_options:
        st.sidebar.subheader("A1: Hybrid Weights")
        a1_w_title = st.sidebar.slider("Title (A1)", 0.0, 1.0, 0.2, 0.05, key="a1_t")
        a1_w_desc = st.sidebar.slider("Desc (A1)", 0.0, 1.0, 0.3, 0.05, key="a1_d")
        a1_w_tag = st.sidebar.slider("Tag (A1)", 0.0, 1.0, 0.5, 0.05, key="a1_g")
        a1_use_pop = st.sidebar.toggle("Wilson Boost (A1)?", True, key="a1_p")
        a1_w_pop = (
            st.sidebar.slider("Boost Weight (A1)", 0.0, 1.0, 0.3, 0.05, key="a1_pw")
            if a1_use_pop
            else 0.0
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("A2: Cluster Weights")
        a2_w_title = st.sidebar.slider("Title (A2)", 0.0, 1.0, 0.2, 0.05, key="a2_t")
        a2_w_desc = st.sidebar.slider("Desc (A2)", 0.0, 1.0, 0.3, 0.05, key="a2_d")
        a2_w_tag = st.sidebar.slider("Tag (A2)", 0.0, 1.0, 0.5, 0.05, key="a2_g")
        # ** NEW Slider for A2 Popularity **
        a2_use_pop = st.sidebar.toggle("Wilson Boost (A2)?", True, key="a2_p")
        a2_w_pop = (
            st.sidebar.slider("Boost Weight (A2)", 0.0, 1.0, 0.3, 0.05, key="a2_pw")
            if a2_use_pop
            else 0.0
        )
        st.sidebar.markdown("---")

    # --- Execute Button and Display Results using ** UPDATED HTML ** ---
else:
    init_msg.error("Initialization failed.")
    st.stop()

if st.sidebar.button(
    "Get Recommendations", use_container_width=True, disabled=input_error
):
    if not selected_game_title:
        st.error("Please select a game.")
    else:
        # ... (Filtering logic and header display) ...
        with st.spinner("Applying filters..."):
            filtered_games_df = apply_filters(
                games_df_clustered,
                selected_tags,
                selected_date_range,
                selected_max_price,
            )
        st.header(f"Recommendations for: {selected_game_title}")
        st.info(
            f"Found {len(filtered_games_df)} games matching filters (pool for recommendations)."
        )

        # --- Find and Display Seed Game Details ---
        # ... (Seed game display remains the same) ...
        seed_game_rows = games_df_clustered[
            games_df_clustered["title"] == selected_game_title
        ]
        if not seed_game_rows.empty:
            seed_game_row = seed_game_rows.iloc[0]
            st.markdown("#### Input Game:")
            st.markdown(_row_html(seed_game_row, seed=True), unsafe_allow_html=True)
            st.markdown("---")
        else:
            st.warning("Could not find input game details.")

        if filtered_games_df.empty or len(filtered_games_df) == 0:
            st.warning("No games match filters.")
        else:
            all_status_messages = []
            col1, col2 = st.columns(2)

            # --- Run Recs & Display Approach 1 ---
            with col1:
                st.subheader("Approach 1: Hybrid Similarity")
                with st.spinner("Calculating Hybrid Recs..."):
                    recs1, msg1 = get_recommendations_approach1(  # Pass a1_w_pop
                        selected_game_title,
                        games_df_clustered,
                        filtered_games_df,
                        features_dict,
                        num_recs,
                        a1_w_title,
                        a1_w_desc,
                        a1_w_tag,
                        a1_w_pop,
                    )
                all_status_messages.extend(msg1)
                if not recs1.empty:
                    # ** FIX: Only join missing columns needed for HTML **
                    cols_to_join = ["app_id", "tags_list", "rating"]
                    recs1_display = recs1.join(games_df_clustered[cols_to_join])

                    html_output = ""
                    for rec_index in (
                        recs1_display.index
                    ):  # Iterate through the combined frame index
                        try:
                            # Pass the row from the *combined* frame to _row_html
                            rec_row_full = recs1_display.loc[rec_index]
                            html_output += _row_html(rec_row_full, seed=False)
                        except KeyError:
                            st.warning(f"Index error A1: {rec_index}")
                        except Exception as e:
                            st.error(f"HTML error A1: {e}")
                    st.markdown(html_output, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found by Approach 1.")

            # --- Run Recs & Display Approach 2 ---
            with col2:
                st.subheader("Approach 2: Clustered Similarity")
                with st.spinner("Calculating Clustered Recs..."):
                    recs2, msg2 = get_recommendations_approach2(  # Pass a2_w_pop
                        selected_game_title,
                        games_df_clustered,
                        filtered_games_df,
                        features_dict,
                        num_recs,
                        a2_w_title,
                        a2_w_desc,
                        a2_w_tag,
                        a2_w_pop,  # Add a2_w_pop here
                    )
                all_status_messages.extend(msg2)
                if not recs2.empty:
                    # ** FIX: Only join missing columns needed for HTML **
                    cols_to_join = ["app_id", "tags_list", "rating"]
                    recs2_display = recs2.join(games_df_clustered[cols_to_join])

                    html_output = ""
                    for rec_index in (
                        recs2_display.index
                    ):  # Iterate through the combined frame index
                        try:
                            # Pass the row from the *combined* frame to _row_html
                            rec_row_full = recs2_display.loc[rec_index]
                            html_output += _row_html(rec_row_full, seed=False)
                        except KeyError:
                            st.warning(f"Index error A2: {rec_index}")
                        except Exception as e:
                            st.error(f"HTML error A2: {e}")
                    st.markdown(html_output, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found by Approach 2.")

            # --- Process Log (Unchanged) ---
            # ... (log display code) ...
            # st.markdown("---")
            # st.subheader("Process Log")
            # if all_status_messages:
            #     [
            #         st.error(m)
            #         if "Err" in m
            #         else st.warning(m)
            #         if "Warn" in m
            #         else st.info(m)
            #         for m in all_status_messages
            #     ]
            # else:
            #     st.info("Process completed.")


st.sidebar.markdown("---")
