# Radiologist feature and agreement summary (trainval only)

## Method

The analysis used the 828 nodules in the shared trainval split (542 benign and 286 malignant). Annotation rows were assigned to the closest final nodule centre within the same patient and accepted only when the distance was at most 5 mm. The 3,050 matched annotation rows exactly reproduced the shared `n_raters` count for every trainval nodule; 13 rows were outside the tolerance and excluded. All eight non-malignancy radiologist-feature fields were complete for the matched rows.

For category distributions, each nodule contributed its within-nodule share of each category; these shares were then averaged within the consensus-label group. For reader agreement, unanimity means all available ratings for a feature matched exactly. This avoids treating multiple ratings of the same nodule as independent observations.

## Results

| Feature | Unanimous ratings, benign | Unanimous ratings, malignant | Descriptive observation |
|---|---:|---:|---|
| Calcification | 82.8% | 91.6% | High agreement in both groups |
| Internal structure | 98.9% | 99.0% | Nearly unanimous in both groups |
| Lobulation | 28.4% | 5.6% | Agreement was low, especially in malignant-labelled nodules |
| Margin | 17.2% | 7.0% | Agreement was low in both groups |
| Sphericity | 8.5% | 6.6% | Agreement was low in both groups |
| Spiculation | 42.6% | 14.0% | Agreement was lower in malignant-labelled nodules |
| Subtlety | 8.5% | 35.3% | Agreement was low overall and differed by label group |
| Texture | 55.2% | 47.6% | Moderate agreement in both groups |

The most common category shares also differed between the consensus-label groups for several features. For example, the mean within-nodule share for “No Spiculation” was 77.7% in benign-labelled nodules and 41.3% in malignant-labelled nodules; “Obvious” subtlety was 28.2% and 65.7%, respectively. These are descriptive distributions of radiologist opinions, not independent biological measurements.

## Limitations

This is descriptive EDA on trainval only. Malignancy labels and the other feature ratings come from the same annotation process, so observed differences do not establish independent predictive value or causation. No hypothesis tests, classifier, feature selection, or test-set analysis were performed. Raw annotation rows and per-nodule agreement values remain local.

## Aggregate outputs

- `annotator_feature_category_distribution_trainval.csv`
- `annotator_feature_agreement_trainval.csv`

The per-nodule table, `annotator_feature_per_nodule_trainval.csv`, contains nodule and patient identifiers and should remain local.
