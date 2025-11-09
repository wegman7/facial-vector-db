import numpy as np
import faiss
import cv2
from insightface.app import FaceAnalysis

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    v = np.nan_to_num(v)                # guard against NaNs/Infs
    n = np.linalg.norm(v) + 1e-12
    return v / n

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)  # CPU on Mac is fine

# ---------- index known faces ----------
known_images = {
    "./facial_images/josh.jpeg":  "josh",
    "./facial_images/sarah.jpeg": "sarah",
}

known_embs = []
known_meta = []

for path, person in known_images.items():
    img = cv2.imread(path)
    if img is None:
        print(f"[warn] cannot read {path}")
        continue
    faces = app.get(img)
    if not faces:
        print(f"[info] no faces in {path}")
        continue

    # Usually one face per headshot; keep the largest if there are extras
    f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    e = l2_normalize(f.embedding)
    # Sanity check
    print(f"[{person}] min={e.min():.4f} max={e.max():.4f} norm={np.linalg.norm(e):.4f}")

    known_embs.append(e)
    known_meta.append({"person": person, "source": path, "bbox": f.bbox.tolist()})

if not known_embs:
    raise RuntimeError("No known faces indexed.")

known_embs = np.vstack(known_embs).astype("float32")

# Cosine index (inner product on unit vectors)
index = faiss.IndexFlatIP(known_embs.shape[1])
index.add(known_embs)
print(f"[index] vectors: {index.ntotal}")

COS_THRESH = 0.40  # tune 0.35–0.60

def match_query_image(path, k=3):
    q_img = cv2.imread(path)
    if q_img is None:
        raise FileNotFoundError(path)
    q_faces = app.get(q_img)
    results = []
    for f in q_faces:
        q = l2_normalize(f.embedding)[None, :]   # (1, 512), unit-norm
        sims, idxs = index.search(q.astype("float32"), k)  # sims in [-1,1]
        best_sim, best_i = float(sims[0][0]), int(idxs[0][0])
        is_match = best_sim >= COS_THRESH
        results.append({
            "query_bbox": f.bbox.tolist(),
            "pred_person": known_meta[best_i]["person"] if is_match else "unknown",
            "best_cosine_sim": best_sim,
            "nearest_examples": [
                {**known_meta[int(i)], "cosine_sim": float(s)}
                for s, i in zip(sims[0], idxs[0])
            ],
        })
    return q_img, results

# ---------- NEW: draw boxes on the query image ----------
def draw_query_results(img_bgr, results, save_path=None, title="matches"):
    vis = img_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = map(int, r["query_bbox"])
        is_known = r["pred_person"] != "unknown"
        color = (0, 200, 0) if is_known else (0, 0, 255)
        label = f"{r['pred_person']}  cos={r['best_cosine_sim']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"[saved] {save_path}")
    # Show (safe for macOS)
    try:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()
    except Exception:
        # Fallback to OpenCV window if matplotlib isn't available
        cv2.imshow(title, vis); cv2.waitKey(0); cv2.destroyAllWindows()

# ---------- OPTIONAL: visualize top matched known face side-by-side ----------
def show_top_match_pair(query_img_bgr, result, title="top match"):
    """
    Displays the query face box and the top matched known face box side-by-side.
    """
    if not result["nearest_examples"]:
        return
    top = result["nearest_examples"][0]
    # Draw on query
    q = query_img_bgr.copy()
    x1, y1, x2, y2 = map(int, result["query_bbox"])
    cv2.rectangle(q, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(q, f"{result['pred_person']} cos={result['best_cosine_sim']:.2f}",
                (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)

    # Draw on known
    known = cv2.imread(top["source"])
    if known is not None:
        kx1, ky1, kx2, ky2 = map(int, top["bbox"])
        cv2.rectangle(known, (kx1, ky1), (kx2, ky2), (0, 200, 0), 2)
        cv2.putText(known, f"{top['person']} cos={top['cosine_sim']:.2f}",
                    (kx1, max(0, ky1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)

    # Combine horizontally (resize to same height)
    h = max(q.shape[0], known.shape[0]) if known is not None else q.shape[0]
    def resize_h(img, h):
        r = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*r), h))
    q_res = resize_h(q, h)
    if known is not None:
        k_res = resize_h(known, h)
        side = np.hstack([q_res, k_res])
    else:
        side = q_res

    try:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()
    except Exception:
        cv2.imshow(title, side); cv2.waitKey(0); cv2.destroyAllWindows()

# ---------- Query on the group photo ----------
q_img, res = match_query_image("./facial_images/together.jpeg", k=2)

for i, r in enumerate(res):
    print(f"[face {i}] pred={r['pred_person']}  cos={r['best_cosine_sim']:.3f}")
    for nn in r["nearest_examples"]:
        print(f"  ↳ {nn['person']} from {nn['source']}  cos={nn['cosine_sim']:.3f}")

# Draw overlays on the query image
draw_query_results(q_img, res, save_path="together_labeled.jpg", title="together.jpeg matches")

# (Optional) Show top match pairs for each detected face
for i, r in enumerate(res):
    show_top_match_pair(q_img, r, title=f"face {i}: top match")
