# %% [markdown]
# # 01 EDA: LUNA16 data audit, shortcut checks and patient-level split
# ## 0. Setup

# %%
# one place for imports and repo-relative paths so the notebook runs on any machine
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir())
DATA = ROOT / "data"
IMG_DIR = DATA / "CT" / "CT" / "image"
VOI_IMG = DATA / "VOIs" / "VOIs" / "image"
VOI_MASK = DATA / "VOIs" / "VOIs" / "nodule_mask"
PROC = DATA / "processed"
FIGS = ROOT / "results" / "figures" / "eda"
PROC.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

scan_files = sorted(IMG_DIR.glob("*.nii.gz"))
voi_files = sorted(VOI_IMG.glob("*.nii.gz"))
mask_files = sorted(VOI_MASK.glob("*.nii.gz"))
print(len(scan_files), len(voi_files), len(mask_files))

# %%
# shared helpers so every image, label and coordinate goes through the same logic
def load_volume(path):
    img = nib.as_closest_canonical(nib.load(path))
    return img, np.asanyarray(img.dataobj).astype(np.float32)


def is_malignant(ratings):
    return int(sum(r > 3 for r in ratings) >= 2)


def offset_vox(img, x, y, z):
    ijk = (np.linalg.inv(img.affine) @ np.array([-x, -y, z, 1]))[:3]
    return ijk - (np.array(img.shape) - 1) / 2


assert is_malignant([5, 5, 5, 4]) == 1
assert is_malignant([2, 1, 1, 1]) == 0

# %% [markdown]
# ## 1. Image inventory

# %%
# read every file header to check orientation, spacing and shape consistency across the dataset
def describe_file(path, kind):
    h = nib.load(path)
    sx, sy, sz = h.header.get_zooms()[:3]
    nx, ny, nz = h.shape
    return {"kind": kind, "file": path.name, "nx": nx, "ny": ny, "nz": nz,
            "sx": round(sx, 3), "sy": round(sy, 3), "sz": round(sz, 3),
            "orientation": "".join(nib.aff2axcodes(h.affine))}


sources = [("full_ct", scan_files), ("voi_image", voi_files), ("voi_mask", mask_files)]
inv_all = pd.DataFrame([describe_file(p, kind) for kind, files in sources for p in files])
inv_all.to_csv(PROC / "image_inventory.csv", index=False)

vi = inv_all[inv_all["kind"] == "voi_image"].assign(
    mm_x=lambda d: d["nx"] * d["sx"], mm_y=lambda d: d["ny"] * d["sy"], mm_z=lambda d: d["nz"] * d["sz"])

print(inv_all.groupby("kind")["orientation"].value_counts(), "\n")
print(inv_all.groupby("kind")[["sx", "sy", "sz"]].describe().T)
print(vi["sz"].value_counts().sort_index())
print("non-square pixels:", (vi["sx"] != vi["sy"]).sum())
print(vi[["nx", "ny", "nz", "mm_x", "mm_y", "mm_z"]].describe())

# %%
# confirm the canonical loader puts every VOI in one orientation before any patch extraction
print({"".join(nib.aff2axcodes(nib.as_closest_canonical(nib.load(p)).affine)) for p in voi_files})

# %%
# check intensities are real HU and that every mask is binary and aligned with its image
rows = []
for p in voi_files:
    a = nib.load(p)
    vol = np.asanyarray(a.dataobj)
    m_path = VOI_MASK / p.name
    row = {"file": p.name, "dtype": str(vol.dtype), "hu_min": float(vol.min()),
           "hu_max": float(vol.max()), "hu_mean": float(vol.mean()), "has_mask": m_path.exists()}
    if row["has_mask"]:
        m = nib.load(m_path)
        md = np.asanyarray(m.dataobj)
        row.update(same_shape=vol.shape == m.shape, same_affine=bool(np.allclose(a.affine, m.affine)),
                   mask_values=tuple(np.unique(md).tolist()), mask_voxels=int((md > 0).sum()))
    rows.append(row)
qc = pd.DataFrame(rows)

print(qc["dtype"].value_counts())
print(qc[["hu_min", "hu_max", "hu_mean", "mask_voxels"]].describe())
print(qc[["has_mask", "same_shape", "same_affine"]].apply(pd.Series.value_counts))
print(qc["mask_values"].value_counts())
print("masks without an image:", sorted({p.name for p in mask_files} - {p.name for p in voi_files}))

# %%
# eyeball one full scan in three planes with a fixed lung window to confirm orientation and aspect
def show_three_views(path, window=(-1400, 200)):
    img, vol = load_volume(path)
    sx, sy, sz = img.header.get_zooms()[:3]
    x, y, z = (s // 2 for s in vol.shape)
    kw = dict(cmap="gray", vmin=window[0], vmax=window[1], origin="lower")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(vol[:, :, z].T, aspect=sy / sx, **kw)
    ax[0].set_title("Axial")
    ax[1].imshow(vol[:, y, :].T, aspect=sz / sx, **kw)
    ax[1].set_title("Coronal")
    ax[2].imshow(vol[x, :, :].T, aspect=sz / sy, **kw)
    ax[2].set_title("Sagittal")
    for a in ax:
        a.axis("off")
    fig.suptitle(path.name.replace(".nii.gz", ""))
    plt.show()


show_three_views(scan_files[0])

# %% [markdown]
# ## 2. Metadata and labels

# %%
# confirm what one row means (one annotation) before aggregating to nodules
df = pd.read_excel(DATA / "MetadatabyAnnotation.xlsx")
print(df.shape)
print(df.columns.tolist())
df.info()
print(df.isna().sum()[lambda s: s > 0])

per_nodule = df.groupby(["patient_id", "nodule_id"])
print(per_nodule["coordX"].nunique().value_counts())
print(per_nodule.size().value_counts())

# %%
# inspect IDs with more than 4 annotations or two locations, which plain grouping would merge wrongly
size = per_nodule.size()
n_loc = per_nodule["coordX"].nunique()
odd = sorted(set(size[size > 4].index) | set(n_loc[n_loc > 1].index))
show = ["patient_id", "nodule_id", "anotation_id", "Malignancy_value", "coordX", "coordY", "coordZ", "diameter_mm"]
for pid, nid in odd:
    print(df[(df["patient_id"] == pid) & (df["nodule_id"] == nid)][show], "\n")

# %%
# build one row per nodule, keeping coordinates in the key so two nodules sharing an ID stay separate
df = df.assign(bbox_x=df["bboxHighX"] - df["bboxLowX"], bbox_z=df["bboxHighZ"] - df["bboxLowZ"])
key = ["patient_id", "nodule_id", "coordX", "coordY", "coordZ"]
nod = (df.groupby(key, as_index=False)
         .agg(ratings=("Malignancy_value", list),
              n_annotations=("anotation_id", "nunique"),
              diameter_mm=("diameter_mm", "first"),
              bbox_x=("bbox_x", "first"),
              bbox_z=("bbox_z", "first")))
nod["malignant"] = nod["ratings"].apply(is_malignant)
nod["part"] = nod.groupby(["patient_id", "nodule_id"]).cumcount()
nod["voi_name"] = nod["patient_id"] + "_R_" + nod["nodule_id"].astype(str) + ".nii.gz"
nod["has_voi"] = nod["voi_name"].isin({p.name for p in voi_files})

print(len(nod), nod["malignant"].value_counts().to_dict())
print(nod[~nod["has_voi"]][["patient_id", "nodule_id", "part"]])

# %% [markdown]
# ## 3. Link nodules to images

# %%
# keep only nodules that own an image, resolving the 0332 ID clash by distance to the VOI centre
voi = nib.load(VOI_IMG / "LIDC-IDRI-0332_R_1.nii.gz")
cand = nod[(nod["patient_id"] == "LIDC-IDRI-0332") & (nod["nodule_id"] == 1)]
dist = {r["part"]: np.linalg.norm(offset_vox(voi, r["coordX"], r["coordY"], r["coordZ"])) for _, r in cand.iterrows()}
owner = min(dist, key=dist.get)
print(dist, "-> image belongs to part", owner)
nod.loc[cand.index[cand["part"] != owner], "has_voi"] = False

props = vi[["file", "nx", "ny", "nz", "sx", "sz", "orientation", "mm_x", "mm_z"]].rename(columns={"file": "voi_name"})
final = nod[nod["has_voi"]].merge(props, on="voi_name", how="left")
final["thickness"] = pd.cut(final["sz"], [0, 1.0, 1.5, 3.0], labels=["thin (<=1.0)", "1.25", "thick (>=1.8)"])
final["pixel_size"] = pd.qcut(final["sx"], 3, labels=["small", "medium", "large"])
final["size_bin"] = pd.qcut(final["diameter_mm"], 3, labels=["small", "medium", "large"])

assert final["voi_name"].is_unique
print(len(final), final["malignant"].value_counts().to_dict())

# %% [markdown]
# ## 4. Coordinate validation

# %%
# find the sign convention by testing all four flips on one VOI, since only the right one lands inside the box
r = final[final["voi_name"] == "LIDC-IDRI-0001_R_1.nii.gz"].iloc[0]
voi = nib.load(VOI_IMG / r["voi_name"])
print("VOI shape", voi.shape)
for fx, fy in itertools.product([1, -1], [1, -1]):
    ijk = (np.linalg.inv(voi.affine) @ np.array([fx * r["coordX"], fy * r["coordY"], r["coordZ"], 1]))[:3]
    print(f"flip x={fx:+d}, y={fy:+d}", np.round(ijk, 1))

# %%
# confirm the (-x, -y, z) convention on a random sample and on every RAS-stored VOI
def centre_offsets(rows):
    out = []
    for _, r in rows.iterrows():
        dx, dy, dz = offset_vox(nib.load(VOI_IMG / r["voi_name"]), r["coordX"], r["coordY"], r["coordZ"])
        out.append({"voi_name": r["voi_name"], "dx": dx, "dy": dy, "dz": dz})
    return pd.DataFrame(out)


ras_names = set(vi.loc[vi["orientation"] == "RAS", "file"])
checks = [("30 random VOIs", final.sample(30, random_state=0)),
          ("all RAS VOIs", final[final["voi_name"].isin(ras_names)])]
for name, rows in checks:
    print(name)
    print(centre_offsets(rows)[["dx", "dy", "dz"]].describe().loc[["mean", "std", "min", "max"]])

# %%
# prove a VOI is an exact crop of its CT, which validates the coordinate conversion end to end
ct = nib.load(IMG_DIR / "LIDC-IDRI-0001.nii.gz")
voi = nib.load(VOI_IMG / "LIDC-IDRI-0001_R_1.nii.gz")
corner = (np.linalg.inv(ct.affine) @ voi.affine @ np.array([0, 0, 0, 1]))[:3]
i, j, k = np.round(corner).astype(int)
nx, ny, nz = voi.shape
assert np.allclose(ct.affine[:3, :3], voi.affine[:3, :3])
assert np.allclose(np.asanyarray(ct.dataobj)[i:i + nx, j:j + ny, k:k + nz], np.asanyarray(voi.dataobj))
print("VOI corner in CT voxels:", np.round(corner, 2), "-> exact crop confirmed")

# %% [markdown]
# ## 5. Shortcut checks and baselines

# %%
# test whether scanner settings are linked to the label, since a model could exploit that instead of the nodule
tab = pd.crosstab(final["thickness"], final["malignant"])
print(tab, "\np-value:", round(chi2_contingency(tab)[1], 4))
print(final.groupby("thickness", observed=True)["malignant"].agg(["count", "mean"]).round(2))
print(final.groupby("pixel_size", observed=True)["malignant"].agg(["count", "mean"]).round(2))
print(final.groupby("thickness", observed=True)["diameter_mm"].describe().round(1))
print(final.pivot_table(index="size_bin", columns="thickness", values="malignant",
                        aggfunc=["count", "mean"], observed=True).round(2))

# %%
# set baselines with patient-grouped CV so the image model has a fair bar to beat
def cv_auc(cols):
    pred = cross_val_predict(LogisticRegression(max_iter=1000), final[cols].values, final["malignant"].values,
                             cv=GroupKFold(5), groups=final["patient_id"], method="predict_proba")[:, 1]
    return round(roc_auc_score(final["malignant"], pred), 3)


baselines = [("scanner only", ["sz", "sx"]), ("diameter only", ["diameter_mm"]),
             ("diameter + scanner", ["diameter_mm", "sz", "sx"]), ("VOI depth only", ["mm_z"]),
             ("VOI width only", ["mm_x"])]
for name, cols in baselines:
    print(f"{name:20s}", cv_auc(cols))

for b, g in final.groupby("size_bin", observed=True):
    auc = roc_auc_score(g["malignant"], g["diameter_mm"])
    print(f"{b:7s} n={len(g)} malignant={g['malignant'].mean():.2f} diameter AUC={auc:.3f}")

# %%
# test whether VOI extent leaks nodule size, which decides how patches must be built
print(final[["diameter_mm", "mm_x", "mm_z", "bbox_x", "bbox_z"]].corr(method="spearman").round(2))
for c in ["mm_x", "mm_z"]:
    print(c, "AUC alone:", round(roc_auc_score(final["malignant"], final[c]), 3))

# %% [markdown]
# ## 6. Label noise

# %%
# measure how many labels change under alternative consensus rules to quantify label noise
final["n_raters"] = final["ratings"].apply(len)
final["n_high"] = final["ratings"].apply(lambda r: sum(x > 3 for x in r))
final["rating_mean"] = final["ratings"].apply(np.mean)
final["label_majority"] = (final["n_high"] > final["n_raters"] / 2).astype(int)
final["label_mean"] = (final["rating_mean"] > 3).astype(int)

print(pd.crosstab(final["malignant"], final["label_majority"], margins=True))
print(pd.crosstab(final["malignant"], final["label_mean"], margins=True))
print(final.groupby(["n_raters", "n_high"]).size().unstack(fill_value=0))

# %%
# flag nodules whose label depends on the consensus rule and measure how they affect the diameter baseline
labels = ["malignant", "label_majority", "label_mean"]
pred = {lab: cross_val_predict(LogisticRegression(max_iter=1000), final[["diameter_mm"]].values, final[lab].values,
                               cv=GroupKFold(5), groups=final["patient_id"], method="predict_proba")[:, 1]
        for lab in labels}
for lab in labels:
    print(f"{lab:15s} diameter-only AUC", round(roc_auc_score(final[lab], pred[lab]), 3))

final["ambiguous"] = ((final["label_majority"] != final["malignant"])
                      | (final["label_mean"] != final["malignant"])).astype(int)
clear = final["ambiguous"].eq(0).values
print("ambiguous nodules:", final["ambiguous"].sum(), round(final["ambiguous"].mean(), 3))
print(final.groupby("size_bin", observed=True)["ambiguous"].agg(["sum", "mean"]).round(2))
print("PDF label, clear nodules only, diameter AUC:",
      round(roc_auc_score(final.loc[clear, "malignant"], pred["malignant"][clear]), 3))

# %% [markdown]
# ## 7. Figures

# %%
# produce the report figures for class balance, diameter by class, rater counts and rating spread
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
final["malignant"].value_counts().sort_index().plot.bar(ax=ax[0], rot=0)
ax[0].set_xticklabels(["benign", "malignant"])
ax[0].set_title("Class balance")
for c, name in [(0, "benign"), (1, "malignant")]:
    ax[1].hist(final.loc[final["malignant"] == c, "diameter_mm"], bins=30, alpha=0.6, label=name)
ax[1].set(xlabel="diameter (mm)", ylabel="nodules", title="Diameter by class")
ax[1].legend()
final["n_raters"].value_counts().sort_index().plot.bar(ax=ax[2], rot=0)
ax[2].set(xlabel="ratings per nodule", title="Raters per nodule")
ax[3].hist(final["rating_mean"], bins=40)
ax[3].axvline(3, color="k", linestyle="--")
ax[3].set(xlabel="mean malignancy rating", ylabel="nodules", title="Rating spread")
plt.tight_layout()
plt.savefig(FIGS / "overview.png", dpi=150)
plt.show()

# %%
# show how malignancy rate rises with diameter, the pattern behind the diameter-only AUC
edges = [0, 4, 6, 8, 10, 15, 20, 40]
rate = final.groupby(pd.cut(final["diameter_mm"], edges), observed=True)["malignant"].agg(["mean", "count"])
ax = rate["mean"].plot.bar(figsize=(7, 4), rot=45)
ax.set(xlabel="diameter bin (mm)", ylabel="share malignant", title="Malignant share by diameter")
plt.tight_layout()
plt.savefig(FIGS / "malignant_share_by_diameter.png", dpi=150)
plt.show()
print(rate.round(2))

# %%
# look at real benign and malignant VOIs with mask outlines to see what the model will be given
picks = {c: final[final["malignant"] == c].sample(6, random_state=0) for c in (0, 1)}
fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
for row, c in enumerate((0, 1)):
    for col, (_, r) in enumerate(picks[c].iterrows()):
        img, vol = load_volume(VOI_IMG / r["voi_name"])
        _, mask = load_volume(VOI_MASK / r["voi_name"])
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        sx, sy = img.header.get_zooms()[:2]
        ax = axes[row, col]
        ax.imshow(vol[:, :, z].T, cmap="gray", vmin=-1400, vmax=200, origin="lower", aspect=sy / sx)
        ax.contour(mask[:, :, z].T, levels=[0.5], colors="r", linewidths=0.8)
        ax.set_title(f"{'malignant' if c else 'benign'} {r['diameter_mm']:.1f} mm\nratings {r['ratings']}", fontsize=8)
        ax.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "example_vois.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Patient-level split and final table

# %%
# lock a patient-level test set and CV folds now, balanced by label and slice thickness, before any modelling
strata = final["malignant"].astype(str) + "_" + final["thickness"].astype(str)

outer = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
trainval_idx, test_idx = next(outer.split(final, strata, final["patient_id"]))
final["split"] = "trainval"
final.loc[final.index[test_idx], "split"] = "test"
final["fold"] = -1

tv = final.iloc[trainval_idx]
inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for k, (_, val_idx) in enumerate(inner.split(tv, strata.iloc[trainval_idx], tv["patient_id"])):
    final.loc[tv.index[val_idx], "fold"] = k

test_patients = set(final.loc[final["split"] == "test", "patient_id"])
assert test_patients.isdisjoint(set(final.loc[final["split"] == "trainval", "patient_id"]))
for k in range(5):
    val_patients = set(final.loc[final["fold"] == k, "patient_id"])
    train_patients = set(final.loc[(final["split"] == "trainval") & (final["fold"] != k), "patient_id"])
    assert val_patients.isdisjoint(train_patients)

final[["voi_name", "patient_id", "split", "fold"]].to_csv(PROC / "splits.csv", index=False)
print(final.groupby(["split", "fold"]).agg(n=("malignant", "size"), malignant=("malignant", "mean"),
                                           patients=("patient_id", "nunique")).round(2))
print(pd.crosstab(final["split"], final["thickness"], normalize="index").round(2))

# %%
# save the final analysis table, including split and fold, as the single input for later phases
final.to_csv(PROC / "nodules.csv", index=False)
