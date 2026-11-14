"""Stage 8: do better models change the answer?

Compares tuned candidate models with the untuned logistic regression of stage 4, on the same
trainval rows and the same repeated fold assignments. Hyperparameters are tuned INSIDE each
training fold, on patient-grouped inner splits, so the validation fold never influences them.
The sealed test set is not used.

Also tests an equal-weight AVERAGE of the members' probabilities ("ensemble_avg"). No weights
are fitted, so it needs no extra tuning and cannot leak; it is compared with each member.
Saves the repeat-averaged out-of-fold probabilities (oof_predictions.csv) for threshold analysis.

Writes results/tables/model_comparison.csv          mean / seed range / 95% CI per model
       results/tables/model_comparison_paired.csv   each candidate minus the untuned baseline
       results/tables/model_comparison_per_seed.csv
       results/tables/oof_predictions.csv           out-of-fold probabilities, trainval only
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lunarad import config as cfg
from lunarad.evaluation import patient_bootstrap_difference, patient_bootstrap_repeats
from lunarad.features import load_feature_table
from lunarad.models import ALL_MODELS, CANDIDATE_MODELS, build_table, columns_for, repeated_oof
from lunarad.splits import validate_split

BASELINE = "logreg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="original_spacing", choices=list(cfg.PROTOCOLS))
    parser.add_argument("--inputs", nargs="+", default=["diameter_texture", "radiomics"])
    parser.add_argument("--models", nargs="+", default=[BASELINE, *CANDIDATE_MODELS],
                        choices=list(ALL_MODELS))
    parser.add_argument("--ensemble-members", nargs="+", default=cfg.ENSEMBLE_MEMBERS,
                        choices=list(ALL_MODELS), help="models averaged into ensemble_avg")
    parser.add_argument("--repeats", type=int, default=3,
                        help="fold assignments per model (tuned models are slow; 3 is enough)")
    parser.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    seeds = [cfg.SEED + i for i in range(args.repeats)]

    nodules = pd.read_csv(cfg.NODULES_CSV)
    splits = pd.read_csv(cfg.SPLITS_CSV)
    validate_split(nodules, splits)
    features = load_feature_table(cfg.FEATURES_CSV)
    table = build_table(args.protocol, features, nodules, splits)
    trainval = table[table["split"] == "trainval"].reset_index(drop=True)
    y, patients = trainval["malignant"].to_numpy(), trainval["patient_id"]
    cfg.TABLES.mkdir(parents=True, exist_ok=True)

    members = list(args.ensemble_members)
    use_ensemble = set(members) <= set(args.models)
    if not use_ensemble:
        print(f"ensemble skipped: members {members} are not all in --models")
    summary, per_seed, paired, oof_rows = [], [], [], []
    for input_name in args.inputs:
        columns = columns_for(input_name, table)
        predictions = {}

        def record(model_name, probs):
            predictions[model_name] = probs
            aucs = [roc_auc_score(y, probs[:, j]) for j in range(len(seeds))]
            aps = [average_precision_score(y, probs[:, j]) for j in range(len(seeds))]
            for seed, auc, ap in zip(seeds, aucs, aps):
                per_seed.append({"input": input_name, "model": model_name, "seed": seed,
                                 "roc_auc": auc, "average_precision": ap})
            ci = patient_bootstrap_repeats(y, probs, patients, replicates=args.bootstrap)
            summary.append({"protocol": args.protocol, "input": input_name, "model": model_name,
                            "n_features": len(columns), "n_repeats": len(seeds),
                            "roc_auc_mean": np.mean(aucs), "roc_auc_min": np.min(aucs),
                            "roc_auc_max": np.max(aucs), "average_precision_mean": np.mean(aps),
                            "roc_auc_ci_low": ci["roc_auc_ci_low"],
                            "roc_auc_ci_high": ci["roc_auc_ci_high"]})
            oof_rows.append(pd.DataFrame({
                "voi_name": trainval["voi_name"], "patient_id": trainval["patient_id"],
                "malignant": y, "protocol": args.protocol, "input": input_name,
                "model": model_name, "probability_malignant": probs.mean(axis=1)}))
            print(f"    AUC {np.mean(aucs):.3f}  95% CI [{ci['roc_auc_ci_low']:.3f}, "
                  f"{ci['roc_auc_ci_high']:.3f}]", flush=True)

        for model_name in args.models:
            print(f"{input_name} / {model_name} ...", flush=True)
            record(model_name, repeated_oof(trainval, columns, ALL_MODELS[model_name], seeds))
        if use_ensemble:
            print(f"{input_name} / ensemble_avg of {members} ...", flush=True)
            record("ensemble_avg", np.mean([predictions[m] for m in members], axis=0))
        pairs = [(m, BASELINE) for m in predictions if m != BASELINE] if BASELINE in predictions else []
        if use_ensemble:
            pairs += [("ensemble_avg", m) for m in members]
        for a, b in pairs:
            diff = patient_bootstrap_difference(y, predictions[a], predictions[b], patients,
                                                replicates=args.bootstrap)
            paired.append({"input": input_name, "comparison": f"{a} - {b}", **diff})

    pd.concat(oof_rows).to_csv(cfg.TABLES / "oof_predictions.csv", index=False)
    pd.DataFrame(per_seed).to_csv(cfg.TABLES / "model_comparison_per_seed.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(cfg.TABLES / "model_comparison.csv", index=False)
    print("\n" + result.round(3).to_string(index=False))
    if paired:
        pair_table = pd.DataFrame(paired)
        pair_table.to_csv(cfg.TABLES / "model_comparison_paired.csv", index=False)
        print("\nPaired AUC differences (A - B), same inputs and folds:")
        print(pair_table.drop(columns="bootstrap_valid_replicates").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
