import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from keras.datasets import mnist

st.set_page_config(page_title="LSH Image Explorer", page_icon="🔎", layout="wide")
DATA = Path(__file__).resolve().parent / "models"
BIT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]


@st.cache_data(show_spinner="Loading MNIST and saved LSH arrays...")
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # This matches the reversed training order used in the Colab preprocessing.
    train_images = train_images[::-1].copy()
    train_labels = train_labels[::-1].copy()
    names = ["train_embeddings", "test_embeddings", "train_hashes", "test_hashes"]
    saved = {name: np.load(DATA / f"{name}.npy", allow_pickle=False) for name in names}
    saved.update(train_images=train_images, test_images=test_images,
                 train_labels=train_labels, test_labels=test_labels)
    assert saved["train_hashes"].shape[1] >= 128
    assert saved["test_hashes"].shape[1] >= 128
    assert len(train_images) == len(saved["train_hashes"]) == len(saved["train_embeddings"])
    assert len(test_images) == len(saved["test_hashes"]) == len(saved["test_embeddings"])
    assert saved["train_embeddings"].shape[1] == saved["test_embeddings"].shape[1]
    return saved


try:
    d = load_data()
except (OSError, ValueError, AssertionError) as exc:
    st.error(f"Could not load MNIST or saved arrays: {exc}")
    st.info("Place the four embedding/hash .npy files in the models/ folder. The first run also needs internet to download MNIST.")
    st.stop()

train_images, test_images = d["train_images"], d["test_images"]
train_labels, test_labels = d["train_labels"], d["test_labels"]
train_features, test_features = d["train_embeddings"], d["test_embeddings"]
train_hashes, test_hashes = d["train_hashes"], d["test_hashes"]


@st.cache_data(show_spinner=False)
def bucket_index(bits):
    packed = np.packbits(train_hashes[:, :bits], axis=1)
    buckets = {}
    for i, row in enumerate(packed):
        buckets.setdefault(row.tobytes(), []).append(i)
    return {key: np.asarray(indices, dtype=np.int32) for key, indices in buckets.items()}


def candidates_for(query_idx, bits):
    key = np.packbits(test_hashes[query_idx, :bits]).tobytes()
    return bucket_index(bits).get(key, np.empty(0, dtype=np.int32))


def cosine_scores(query_idx, indices):
    a = np.asarray(train_features[indices], dtype=np.float32)
    b = np.asarray(test_features[query_idx], dtype=np.float32)
    numerator = a @ b
    denominator = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b), 1e-12)
    return numerator / denominator


def exact_topk(query_idx, k=10):
    indices = np.arange(len(train_features))
    scores = cosine_scores(query_idx, indices)
    return np.argsort(-scores, kind="stable")[:k]


def hamming_topk(query_idx, bits, k=10):
    distances = np.count_nonzero(train_hashes[:, :bits] != test_hashes[query_idx, :bits], axis=1)
    k = min(k, len(distances))
    threshold = np.partition(distances, k - 1)[k - 1]
    eligible = np.flatnonzero(distances <= threshold)
    # Exact cosine similarity breaks ties between equally distant binary codes.
    cosine = cosine_scores(query_idx, eligible)
    order = np.lexsort((-cosine, distances[eligible]))[:k]
    return eligible[order], distances[eligible[order]]


def bucket_topk(query_idx, bits, k=10):
    ids = candidates_for(query_idx, bits)
    if not len(ids):
        return ids, 0
    scores = cosine_scores(query_idx, ids)
    return ids[np.argsort(-scores, kind="stable")[:k]], len(ids)


def image_grid(indices, titles, query_idx, heading):
    fig, axes = plt.subplots(1, 11, figsize=(16, 2.2))
    axes[0].imshow(test_images[query_idx], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Query", fontsize=10)
    axes[0].axis("off")
    for j in range(10):
        ax = axes[j + 1]
        if j < len(indices):
            ax.imshow(train_images[indices[j]], cmap="gray", vmin=0, vmax=255)
            ax.set_title(titles[j], fontsize=9)
        ax.axis("off")
    fig.suptitle(heading, fontsize=13)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def code_text(hashrow, bits):
    s = "".join(map(str, np.asarray(hashrow[:bits])))
    return " ".join(s[i:i + 8] for i in range(0, len(s), 8))


@st.cache_data(show_spinner="Evaluating candidate recall for this query...")
def query_metrics(query_idx):
    true = set(map(int, exact_topk(query_idx, 10)))
    counts, recalls = [], []
    for b in BIT_LENGTHS:
        candidates = candidates_for(query_idx, b)
        counts.append(len(candidates))
        recalls.append(sum(int(i) in true for i in candidates) / len(true))
    return counts, recalls


st.title("LSH Image Explorer")
st.caption("MNIST · CNN embeddings · Charikar random hyperplane hashing")

with st.container(border=True):
    st.subheader("Query selection")
    query_idx = st.number_input("Test image index", min_value=0, max_value=len(test_images) - 1, value=0, step=1)
    a, b = st.columns([1, 5])
    with a:
        st.image(test_images[query_idx], width=120, clamp=True)
    with b:
        st.write(f"**Query index:** {query_idx}  |  **Digit label:** {int(test_labels[query_idx])}")
        st.caption("The query comes from the test set; retrieval searches the training set.")

st.subheader("Part 1 · Hash evolution")
if "bits" not in st.session_state:
    st.session_state.bits = 16
bits = st.slider("Hash length (shared across Parts 1–3)", 1, 128, key="bits")
if st.button("▶ Play 1 → 128 bits", help="Animate the growing prefix of the fixed 128-bit code"):
    progress = st.empty()
    for v in range(1, 129):
        progress.markdown(f"**{v} bits**\n\n`{code_text(test_hashes[query_idx], v)}`")
        time.sleep(0.035)
    st.session_state.bits = 128
    st.rerun()
st.code(code_text(test_hashes[query_idx], bits), language=None, wrap_lines=True)
st.caption("Each added bit comes from one additional random hyperplane. Existing bits do not change.")

st.subheader("Part 2 · Compare similar-label and different-label images")
# Query belongs to the test set, so no training image is the same dataset record.
same_idx = int(np.flatnonzero(train_labels == test_labels[query_idx])[0])
diff_idx = int(np.flatnonzero(train_labels != test_labels[query_idx])[0])
col1, col2, col3 = st.columns(3)
for col, img, label, index in [
    (col1, test_images[query_idx], "Query (test)", query_idx),
    (col2, train_images[same_idx], "Same label (train)", same_idx),
    (col3, train_images[diff_idx], "Different label (train)", diff_idx),
]:
    with col:
        st.image(img, width=120, clamp=True)
        st.caption(f"{label} · index {index}")

qhash = test_hashes[query_idx, :bits]
shash = train_hashes[same_idx, :bits]
dhash = train_hashes[diff_idx, :bits]
same_matches = int(np.count_nonzero(qhash == shash))
diff_matches = int(np.count_nonzero(qhash == dhash))
m1, m2 = st.columns(2)
m1.metric("Same-label bit agreement", f"{same_matches}/{bits}", f"{100 * same_matches / bits:.1f}% match", delta_color="off")
m2.metric("Different-label bit agreement", f"{diff_matches}/{bits}", f"{100 * diff_matches / bits:.1f}% match", delta_color="off")
with st.expander("View binary codes"):
    st.markdown("**Query**")
    st.code(code_text(qhash, bits), language=None, wrap_lines=True)
    st.markdown("**Same label**")
    st.code(code_text(shash, bits), language=None, wrap_lines=True)
    st.markdown("**Different label**")
    st.code(code_text(dhash, bits), language=None, wrap_lines=True)
fig, ax = plt.subplots(figsize=(14, 2))
ax.imshow(np.vstack([qhash != shash, qhash != dhash]).astype(int), cmap="Reds", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
ax.set_yticks([0, 1], ["Same label", "Different label"])
ax.set_xlabel("Bit position (red = disagreement)")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Same/different digit labels do not guarantee high/low embedding similarity.")

st.subheader("Part 3 · Image retrieval")
st.markdown("**A. Hamming-distance search — scan all training codes**")
hi, hd = hamming_topk(query_idx, bits)
image_grid(hi, [f"d={int(v)}" for v in hd], query_idx, f"{bits}-bit Hamming nearest neighbors")
st.caption("Results are ranked by Hamming distance; cosine similarity breaks ties.")
st.markdown("**B. Exact hash-bucket search — indexed lookup + cosine reranking**")
bi, count = bucket_topk(query_idx, bits)
st.write(f"**{count:,}** candidates in the matching bucket")
image_grid(bi, [f"Match {j + 1}" for j in range(len(bi))], query_idx, f"{bits}-bit exact-bucket retrieval")
if count == 0:
    st.warning("No exact hash matches at this bit length. This illustrates the recall tradeoff.")

st.subheader("Part 4 · Candidate count vs. missed nearest neighbors")
counts, recalls = query_metrics(query_idx)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(BIT_LENGTHS, counts, marker="o")
axes[0].set_title("Exact-bucket candidate count")
axes[0].set_ylabel("Candidates")
axes[0].set_yscale("symlog", linthresh=1)
axes[1].plot(BIT_LENGTHS, recalls, marker="o")
axes[1].set_title("Candidate recall@10")
axes[1].set_ylabel("Fraction of true top-10 neighbors retained")
axes[1].set_ylim(0, 1.05)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(BIT_LENGTHS)
    ax.set_xticklabels(BIT_LENGTHS)
    ax.set_xlabel("Hash length (bits)")
    ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("Ground truth = 10 closest training embeddings by exact cosine similarity. Recall is measured before cosine reranking.")
st.success("Takeaway: longer exact-match hashes reduce candidate work but can exclude true neighbors. Hamming scans avoid empty buckets, but compare against every code.")
