
import time
import urllib.request
from pathlib import Path
from html import escape

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import numpy as np
import streamlit as st

import io

def show_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

st.set_page_config(page_title="LSH Image Retrieval - Demo", page_icon="🔎", layout="wide")

DATA = Path(__file__).resolve().parent / "models"
BIT_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


@st.cache_data(show_spinner="Loading MNIST and saved LSH arrays...")
def load_data():
    # Download MNIST directly without Keras or TensorFlow
    mnist_path = Path("/tmp/mnist.npz")

    if not mnist_path.exists():
        urllib.request.urlretrieve(MNIST_URL, mnist_path)

    # Load MNIST dataset
    with np.load(mnist_path, allow_pickle=False) as mnist_data:
        train_images = mnist_data["x_train"]
        train_labels = mnist_data["y_train"]
        test_images = mnist_data["x_test"]
        test_labels = mnist_data["y_test"]

    # Match the reversed training order used in Colab preprocessing
    train_images = train_images[::-1].copy()
    train_labels = train_labels[::-1].copy()

    # Load precomputed LSH arrays
    names = [
        "train_embeddings",
        "test_embeddings",
        "train_hashes",
        "test_hashes",
    ]

    saved = {
        name: np.load(DATA / f"{name}.npy", allow_pickle=False)
        for name in names
    }

    saved.update(
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
    )

    # Validate dataset and embedding dimensions
    assert saved["train_hashes"].shape[1] >= 128
    assert saved["test_hashes"].shape[1] >= 128

    assert (
        len(train_images)
        == len(saved["train_hashes"])
        == len(saved["train_embeddings"])
    )

    assert (
        len(test_images)
        == len(saved["test_hashes"])
        == len(saved["test_embeddings"])
    )

    assert (
        saved["train_embeddings"].shape[1]
        == saved["test_embeddings"].shape[1]
    )

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
    if len(indices) == 0:
        st.info("No matching candidates found.")
        return

    fig, axes = plt.subplots(1, 11, figsize=(18, 2.8), dpi=120, facecolor="none")
    fig.patch.set_alpha(0)

    images = [test_images[query_idx]] + [train_images[idx] for idx in indices[:10]]
    labels = ["Query"] + list(titles[:10])

    for j, ax in enumerate(axes):
        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])

        if j >= len(images):
            ax.imshow(np.zeros((28, 28)), cmap="gray", alpha=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        #ax.imshow(images[j], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        ax.imshow(images[j], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor="white", linewidth=2, clip_on=False))
        ax.set_title(labels[j], fontsize=11, color="white", fontweight="normal", pad=12)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(1.5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.08, wspace=0.25)

    show_plot(fig)


def code_text(hashrow, bits):
    s = "".join(map(str, np.asarray(hashrow[:bits])))
    return " ".join(s[i:i + 8] for i in range(0, len(s), 8))


@st.cache_data
def query_metrics():
    with np.load(DATA / "retrieval_metrics.npz") as metrics:
        return (
            metrics["avg_candidates"].copy(),
            metrics["avg_recall"].copy()
        )

def hash_box(hash_text, bytes_per_row=4):
    # Split hash into individual bytes
    bits = "".join(str(hash_text).split())
    bytes_list = [
        bits[i:i + 8]
        for i in range(0, len(bits), 8)
    ]

    # Group 4 bytes (32 bits) per row
    rows = [
        " ".join(bytes_list[i:i + bytes_per_row])
        for i in range(0, len(bytes_list), bytes_per_row)
    ]

    formatted_hash = "\n".join(rows)

    st.markdown(
        f"""
        <div style="
            background-color: #1b1e26;
            color: #00ff88;
            padding: 28px 28px;
            min-height: 100px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-family: monospace;
            font-size: 26px;
            font-weight: 550;
            line-height: 1.8;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            box-sizing: border-box;
        ">{escape(formatted_hash)}</div>
        """,
        unsafe_allow_html=True
    )

def colored_hash_box(hash_bits, reference_bits=None, bits_per_row=32):
    hash_bits = np.asarray(hash_bits).flatten()

    if reference_bits is not None:
        reference_bits = np.asarray(reference_bits).flatten()

    html_parts = [
        '<div style="'
        'background:#1b1e26;'
        'padding:24px;'
        'border-radius:12px;'
        'margin-bottom:20px;'
        'font-family:monospace;'
        'font-size:20px;'
        'font-weight:600;'
        'line-height:1.8;'
        'text-align:center;'
        '">'
    ]

    for start in range(0, len(hash_bits), bits_per_row):
        end = min(start + bits_per_row, len(hash_bits))

        html_parts.append(
            '<div style="white-space:normal; '
            'overflow-wrap:anywhere; '
            'text-align:center;">'
        )

        for i in range(start, end):
            bit = int(hash_bits[i])

            if reference_bits is None:
                color = "#FFFFFF"
            elif bit == int(reference_bits[i]):
                color = "#299764"
            else:
                color = "#D34A49"

            html_parts.append(
                f'<span title="Bit {i+1}" '
                f'style="color:{color};">'
                f'{bit}</span>'
            )

            # Add spacing after each byte
            if (i + 1) % 8 == 0 and i + 1 < end:
                html_parts.append(
                    '<span style="margin-right:14px;"></span>'
                )

        html_parts.append('</div>')

    html_parts.append('</div>')

    st.markdown(
        "".join(html_parts),
        unsafe_allow_html=True
    )

def section_spacer():
    st.markdown("<div style='height: 40px;'></div>",
            unsafe_allow_html=True)
    
from tensorflow import keras

@st.cache_resource
def load_encoder():
    model = keras.models.load_model("models/cnn_model.keras", compile=False)
    return keras.Model(inputs=model.inputs, outputs=model.get_layer("embedding").output)

encoder = load_encoder()
hyperplanes = np.load("models/hyperplanes.npy")

def encode_drawing(image):
    x = image.astype(np.float32)[None, ..., None] / 255.0
    embedding = encoder.predict(x, verbose=0)[0]
    embedding = embedding / max(np.linalg.norm(embedding), 1e-10)
    hash_code = (embedding @ hyperplanes >= 0).astype(np.uint8)
    return embedding, hash_code


def live_hamming_topk(query_hash, n_bits, k=10):
    distances = np.count_nonzero(train_hashes[:, :n_bits] != query_hash[:n_bits], axis=1)
    indices = np.argsort(distances, kind="stable")[:k]
    return indices, distances[indices]

st.title("LSH Image Explorer")
st.caption("MNIST · CNN embeddings · Charikar random hyperplane hashing")

# Initialize query index
if "query_idx" not in st.session_state:
    st.session_state.query_idx = 0

# Callback for random query selection
def random_query():
    st.session_state.query_idx = int(
        np.random.randint(0, len(test_images))
    )

with st.container(border=True):
    st.subheader("Query selection")

    # Number input and random button
    input_col, button_col = st.columns([5, 1], vertical_alignment="bottom")

    with input_col:
        query_idx = st.number_input(
            "Test image index",
            min_value=0,
            max_value=len(test_images) - 1,
            step=1,
            key="query_idx"
        )

    with button_col:
        st.button(
            "🎲 Random Query",
            on_click=random_query,
            width="stretch"
        )

    # Center image and description
    left_space, content, right_space = st.columns([1, 3, 1])

    with content:
        img_col, text_col = st.columns(
            [1, 1.3],
            gap="large",
            vertical_alignment="center"
        )

        with img_col:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

            #ax.imshow(test_images[query_idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            ax.imshow(test_images[query_idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor="white", linewidth=2, clip_on=False))

            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            show_plot(fig)

        with text_col:
            st.markdown(f"### Query index: {query_idx}")
            st.markdown(f"### Digit label: {int(test_labels[query_idx])}")

            st.caption(
                "The query comes from the test set; "
                "retrieval searches the training set."
            )

section_spacer()

# START PART 1

st.subheader("Part 1 · Hash evolution")

if "bits" not in st.session_state:
    st.session_state.bits = 16

# Slider
bits = st.slider(
    "Hash length (shared across Parts 1–3)",
    min_value=1,
    max_value=128,
    key="bits"
)

# Static hash display
hash_box(code_text(test_hashes[query_idx], bits), 16)

section_spacer()

st.caption(
    "Each added bit comes from one additional random hyperplane. "
    "Existing bits do not change."
)

# Separate animation section
st.markdown("#### Watch the hash grow")

if st.button("▶ Play 1 → 128 bits"):
    animation_display = st.empty()

    for v in range(1, 129):
        with animation_display.container():
            hash_box(code_text(test_hashes[query_idx], v))

        time.sleep(0.035)

section_spacer()

# END PART 1

# START PART 2

st.subheader("Part 2 · Compare similar-label and different-label images")

# Select training examples
similarities = cosine_scores(
    query_idx,
    np.arange(len(train_features))
)

same_idx = int(np.argmax(similarities))
diff_idx = int(np.argmin(similarities))

# Extract binary hashes
qhash = test_hashes[query_idx, :bits]
shash = train_hashes[same_idx, :bits]
dhash = train_hashes[diff_idx, :bits]

same_matches = int(np.count_nonzero(qhash == shash))
diff_matches = int(np.count_nonzero(qhash == dhash))


# Helper: display a sharp MNIST digit
def show_mnist(img):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

    #ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, edgecolor="white", linewidth=2, clip_on=False))

    ax.axis("off")
    fig.subplots_adjust(
        left=0, right=1,
        top=1, bottom=0
    )

    show_plot(fig)


# ----------------------------------------
# 1. Three aligned image columns
# ----------------------------------------

left_space, content, right_space = st.columns([1, 5, 1])

with content:
    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    examples = [
        (col1, test_images[query_idx],
         "Query (test)", query_idx, test_labels[query_idx]),
        (col2, train_images[same_idx],
         "Most Similar (train)", same_idx, train_labels[same_idx]),
        (col3, train_images[diff_idx],
         "Least Similar (train)", diff_idx, train_labels[diff_idx])
    ]

    for col, img, label, index, digit in examples:
        with col:
            st.markdown(
                f"<h4 style='text-align:center'>{label}</h4>",
                unsafe_allow_html=True
            )

            show_mnist(img)

            st.markdown(
                f"<p style='text-align:center'>"
                f"Index: {index} · Digit: {int(digit)}"
                f"</p>",
                unsafe_allow_html=True
            )


# ----------------------------------------
# 2. Matching statistics
# ----------------------------------------

left_space, content, right_space = st.columns([1, 5, 1])

with content:
    m1, m2, m3 = st.columns(
        3,
        gap="large"
    )

    with m2:
        st.markdown(
            f"""
            <div style='height: 20px;'></div>
            <div style="text-align: center;">
                <div style="font-size: 0.9rem;">Same-label agreement</div>
                <div style="font-size: 2rem; font-weight: 600;">{same_matches}/{bits}</div>
                <div style="color: gray;">{100 * same_matches / bits:.1f}% match</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div style='height: 20px;'></div>
            <div style="text-align: center;">
                <div style="font-size: 0.9rem;">Different-label agreement</div>
                <div style="font-size: 2rem; font-weight: 600;">{diff_matches}/{bits}</div>
                <div style="color: gray;">{100 * diff_matches / bits:.1f}% match</div>
            </div>
            """,
            unsafe_allow_html=True
    )

# ----------------------------------------
# 4. Intuitive bit matching visualization
# ----------------------------------------

def bit_circle_comparison(qhash, shash, dhash, bits):
    st.markdown("#### Bit-by-bit comparison")

    comparisons = [
        ("Same label", qhash == shash),
        ("Different label", qhash == dhash)
    ]

    html_parts = [
        '<div style="background:#1b1e26; padding:24px; '
        'border-radius:12px; margin-bottom:24px;">'
    ]

    for label, matches in comparisons:
        match_count = int(np.count_nonzero(matches))
        percentage = 100 * match_count / len(matches)

        html_parts.append(
            f'<div style="margin-bottom:24px;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'color:white; font-size:17px; font-weight:600; '
            f'margin-bottom:14px;">'
            f'<span>{escape(label)}</span>'
            f'<span>{match_count}/{len(matches)} '
            f'({percentage:.1f}%)</span>'
            f'</div>'
        )

        # Display 32 circles per row
        for start in range(0, len(matches), 32):
            chunk = matches[start:start + 32]

            html_parts.append(
                '<div style="display:flex; gap:12px; '
                'margin-bottom:8px; flex-wrap:nowrap;">'
            )

            for offset, match in enumerate(chunk):
                color = "#299764" if match else "#D34A49"
                position = start + offset + 1
                status = "Match" if match else "Mismatch"

                html_parts.append(
                    f'<span title="Bit {position}: {status}" '
                    f'style="display:inline-block; width:18px; '
                    f'height:18px; min-width:18px; '
                    f'border-radius:50%; '
                    f'background-color:{color};"></span>'
                )

            html_parts.append('</div>')

        html_parts.append('</div>')

    # Legend
    html_parts.append(
        '<div style="display:flex; gap:24px; '
        'color:#aaaaaa; font-size:14px;">'
        '<span>🟢 Matching bit</span>'
        '<span>🔴 Different bit</span>'
        '</div>'
    )

    html_parts.append('</div>')

    # Join without Markdown-indented HTML
    st.markdown(
        "".join(html_parts),
        unsafe_allow_html=True
    )


bit_circle_comparison(qhash, shash, dhash, bits)

st.caption(
    "Green indicates matching hash bits; red indicates differing bits. "
    "Same/different digit labels do not guarantee high/low embedding similarity."
)

# ----------------------------------------
# 3. Binary codes (expanded by default)
# ----------------------------------------

with st.expander("View binary codes"):

    st.markdown("**Query**")
    colored_hash_box(qhash, bits_per_row=128)
    section_spacer()

    st.markdown("**Same label**")
    colored_hash_box(shash, reference_bits=qhash, bits_per_row=128)
    section_spacer()

    st.markdown("**Different label**")
    colored_hash_box(dhash, reference_bits=qhash, bits_per_row=128)
    section_spacer()

# END PART 2

# START PART 3

st.subheader("Part 3.1 · Image retrieval")

HAMMING_RETRIEVAL_BITS = [8, 32, 122]

# ----------------------------------------
# A. Hamming-distance retrieval
# ----------------------------------------

st.markdown(
    "### A. Hamming-distance search"
)

st.caption(
    "Scan all training hashes and rank by Hamming distance. "
    "Cosine similarity breaks ties."
)

for n_bits in HAMMING_RETRIEVAL_BITS:
    st.markdown(f"#### {n_bits}-bit Hamming retrieval")

    hi, hd = hamming_topk(query_idx, n_bits)

    image_grid(hi, [f"d={int(v)}" for v in hd], query_idx, f"{n_bits}-bit Hamming nearest neighbors")

# ----------------------------------------
# B. Exact hash-bucket retrieval
# ----------------------------------------

EXACT_RETRIEVAL_BITS = [16, 32, 64, 122]

st.markdown(
    "### B. Exact hash-bucket search"
)

st.caption(
    "Retrieve training images with identical hash prefixes, "
    "then rerank candidates using cosine similarity."
)

for n_bits in EXACT_RETRIEVAL_BITS:
    st.markdown(f"#### {n_bits}-bit Exact-bucket retrieval")

    bi, count = bucket_topk(query_idx, n_bits)

    st.write(f"**{count:,} candidates** in the matching bucket")

    if count > 0:
        image_grid(bi, [f"Match {j + 1}" for j in range(len(bi))], query_idx, f"{n_bits}-bit exact-bucket retrieval")
    else:
        st.warning(f"No exact hash matches at {n_bits} bits. " "This illustrates the recall tradeoff.")

# ----------------------------------------
# C. Live retrieval
# ----------------------------------------

from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.subheader("Part 3 · Live digit retrieval")
st.caption("Draw a handwritten digit and retrieve the 10 closest MNIST training images using LSH.")

left, center, right = st.columns([1, 3, 1])

with center:
    st.markdown("#### Draw a digit")

    st.markdown("""
    <style>
    div[data-testid="stCustomComponentV1"] {
        display: flex;
        justify-content: center;
    }

    div[data-testid="stCustomComponentV1"] iframe {
        width: 460px !important;
        min-width: 460px !important;
        height: 460px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=24,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="digit_canvas"
    )

    live_bits = st.select_slider("Hash length", options=[8, 16, 32, 64, 128], value=32)
    search = st.button("🔎 Find nearest neighbors", type="primary", width="stretch")

def preprocess_drawing(image_data):
    image = Image.fromarray(image_data.astype(np.uint8)).convert("RGBA")
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    image = np.asarray(Image.alpha_composite(background, image).convert("L"))

    # Find the bounding box of the handwritten digit
    mask = image > 20
    if not mask.any():
        return np.zeros((28, 28), dtype=np.uint8)

    ys, xs = np.where(mask)
    digit = Image.fromarray(image[ys.min():ys.max() + 1, xs.min():xs.max() + 1])

    # Resize while preserving aspect ratio
    w, h = digit.size
    scale = 20 / max(w, h)
    digit = digit.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)

    # Center on a 28x28 black canvas
    result = Image.new("L", (28, 28), 0)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    result.paste(digit, (x, y))

    return np.asarray(result, dtype=np.uint8)

def live_results_grid(indices, distances):
    fig, axes = plt.subplots(1, 10, figsize=(18, 2.8), dpi=120, facecolor="none")

    for j, ax in enumerate(axes):
        idx = indices[j]
        ax.imshow(train_images[idx], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Index: {idx}\nDigit: {train_labels[idx]}\nHamming: {distances[j]}", fontsize=9, color="white", labelpad=12)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(1.5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.3, wspace=0.3)
    show_plot(fig)


if search and canvas.image_data is not None:
    drawing = preprocess_drawing(canvas.image_data)
    query_embedding, query_hash = encode_drawing(drawing)
    indices, distances = live_hamming_topk(query_hash, live_bits)

    st.markdown("#### Top 10 retrieved images")
    st.caption(f"Nearest neighbors using {live_bits}-bit Hamming distance")
    live_results_grid(indices, distances)

section_spacer()

# END PART 3
    
# START PART 4

st.subheader('Part 4 · Average candidate count vs. missed nearest neighbors')

counts, recalls = query_metrics()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="none")

axes[0].plot(BIT_LENGTHS, counts, marker="o")
axes[0].set_xlabel("Number of hash bits")
axes[0].set_ylabel("Candidate count")
axes[0].set_title("Average candidate count vs. hash length")
axes[0].grid(alpha=0.3)

axes[1].plot(BIT_LENGTHS, recalls, marker="o")
axes[1].set_xlabel("Number of hash bits")
axes[1].set_ylabel("Recall@10")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Average candidate recall vs. hash length")
axes[1].grid(alpha=0.3)

for ax in axes:
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#888888")

fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

st.caption(
    'Results averaged across the first 100 MNIST test queries. '
    'Ground truth = 10 nearest training embeddings by exact cosine similarity. '
    'Recall measures the fraction of those neighbors retained by exact-bucket filtering.'
)

st.success(
    "Takeaway: longer exact-match hashes reduce candidate work but can exclude "
    "true neighbors. Hamming scans avoid empty buckets, but compare against every code."
)

# END PART 4
