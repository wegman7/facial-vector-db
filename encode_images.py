# %% [markdown]
# # 🖼️ Simple Image Vector Database with FAISS
# This notebook shows how to:
# 1. Pull a **CLIP image embedding** model from **Hugging Face**
# 2. Create a few **dummy images**
# 3. **Embed** them and store in a **FAISS** index
# 4. Run **inference**: image→image and text→image
# 5. Save and reload the mini database
#
# ---
# ### 🔧 Requirements
# - sentence-transformers
# - faiss-cpu
# - Pillow
# - torch, torchvision, torchaudio (CPU is fine)
#
# Uncomment the next line to install:
# !pip install -q sentence-transformers faiss-cpu Pillow torch torchvision torchaudio transformers

# %% [markdown]
## 1️⃣ Setup

# %%
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw
import faiss, pickle, json, numpy as np
from sentence_transformers import SentenceTransformer

# %% [markdown]
## 2️⃣ Load a CLIP Image Embedding Model

# %%
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
model = SentenceTransformer(MODEL_NAME)
print(f"Loaded model: {MODEL_NAME}")

# %% [markdown]
## 3️⃣ Create a Few Dummy Images

# %%
data_dir = Path("./data_images")
data_dir.mkdir(parents=True, exist_ok=True)

def make_dummy_image(path: Path, bg: Tuple[int,int,int], shape: str, text: str):
    img = Image.new("RGB", (256, 256), color=bg)
    draw = ImageDraw.Draw(img)
    if shape == "square":
        draw.rectangle([48, 48, 208, 208], outline=(255,255,255), width=6)
    elif shape == "circle":
        draw.ellipse([48, 48, 208, 208], outline=(255,255,255), width=6)
    elif shape == "triangle":
        draw.polygon([(128, 48), (48, 208), (208, 208)], outline=(255,255,255))
    draw.text((10, 10), text, fill=(255,255,255))
    img.save(path)

samples = [
    ("red_square.png",   (200,40,40),   "square",   "red square"),
    ("blue_circle.png",  (40,80,200),   "circle",   "blue circle"),
    ("green_triangle.png",(40,170,100), "triangle", "green triangle"),
    ("yellow_square.png",(220,200,40),  "square",   "yellow square"),
    ("purple_circle.png",(130,60,160),  "circle",   "purple circle"),
]

image_paths, labels = [], []
for name, color, shape, label in samples:
    p = data_dir / name
    make_dummy_image(p, color, shape, label)
    image_paths.append(str(p))
    labels.append(label)

print(f"✅ Created {len(image_paths)} images in {data_dir.resolve()}")

# %% [markdown]
## 4️⃣ Build Embeddings and a FAISS Vector Index

# %%
pil_images: List[Image.Image] = [Image.open(p).convert("RGB") for p in image_paths]
embeddings = model.encode(
    pil_images,
    batch_size=8,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

print("Embeddings shape:", embeddings.shape)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"Indexed {index.ntotal} images.")

id_to_path = {i: image_paths[i] for i in range(len(image_paths))}
id_to_label = {i: labels[i] for i in range(len(image_paths))}

# %% [markdown]
## 5️⃣ Define Search Helpers

# %%
def search_image_to_image(query_img: Image.Image, k: int = 3):
    q_emb = model.encode([query_img.convert("RGB")],
                         convert_to_numpy=True,
                         normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    return [
        {"path": id_to_path[int(i)], "label": id_to_label[int(i)], "score": float(s)}
        for s, i in zip(scores[0], idxs[0])
    ]

def search_text_to_image(query_text: str, k: int = 3):
    txt_emb = model.encode([query_text],
                           convert_to_numpy=True,
                           normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(txt_emb, k)
    return [
        {"path": id_to_path[int(i)], "label": id_to_label[int(i)], "score": float(s)}
        for s, i in zip(scores[0], idxs[0])
    ]

# %% [markdown]
## 6️⃣ Try a Few Example Queries

# %%
# Image → Image
query_path = Path("./data_images/reddish_square_query.png")
make_dummy_image(query_path, (210,60,60), "square", "reddish square")
q_img = Image.open(query_path)
print("🔎 Image→Image results for a reddish square query:")
for r in search_image_to_image(q_img, k=3):
    print(f"  - {r['label']:>16} | score={r['score']:.3f}")

# Text → Image
for q in ["a red square", "a blue circle", "a green triangle"]:
    print(f"\n🔎 Text→Image for: '{q}'")
    for r in search_text_to_image(q, k=3):
        print(f"  - {r['label']:>16} | score={r['score']:.3f}")

# %% [markdown]
## 7️⃣ Save the Vector Database to Disk

# %%
faiss.write_index(index, "image_index.faiss")
with open("image_meta.pkl", "wb") as f:
    pickle.dump({"id_to_path": id_to_path, "id_to_label": id_to_label}, f)
print("✅ Saved FAISS index and metadata.")

# %% [markdown]
## 8️⃣ (Optional) Reload and Search Again

# %%
index2 = faiss.read_index("image_index.faiss")
with open("image_meta.pkl", "rb") as f:
    meta = pickle.load(f)
id_to_path2 = {int(k): v for k, v in meta["id_to_path"].items()}
id_to_label2 = {int(k): v for k, v in meta["id_to_label"].items()}
print(f"✅ Reloaded index with {index2.ntotal} items.")

query = "circle"
q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
scores, idxs = index2.search(q_emb, 3)
print(f"\n🔍 After reload, query: {query}")
for s, i in zip(scores[0], idxs[0]):
    print(f"  - {id_to_label2[int(i)]:>16} | score={float(s):.3f}")
