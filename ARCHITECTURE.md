# Nexchem_AI — Dependency Diagram

## Module-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENTRY POINTS                                       │
│                                                                             │
│   main.py (script)                  streamlit_app.py (GUI)                 │
└────────────┬────────────────────────────────┬───────────────────────────────┘
             │                                │
             ▼                                ▼
┌────────────────────┐          ┌─────────────────────────────┐
│   loaders/         │          │   loaders/                  │
│   raman_loader.py  │          │   raman_loader.py           │
├────────────────────┤          ├─────────────────────────────┤
│ load_raman()       │          │ load_raman()                │
│ select_directory() │          └──────────────┬──────────────┘
│ select_file()      │                         │
└────────┬───────────┘                         │
         │                                     │
         ▼                                     ▼
┌────────────────────────────────────────────────────────┐
│                   preprocessors/                       │
├────────────────────────┬───────────────────────────────┤
│  raman_preprocess.py   │  aligner.py   │ labeling.py   │
├────────────────────────┼───────────────┼───────────────┤
│ preprocess_pipeline_1()│ align_xy()    │ bin_targets() │
│ preprocess_pipeline_2()│ align_group_  │               │
│ group_preprocess()     │   xy()        │               │
│ group_preprocess_2()   │               │               │
│ preprocess_pipeline_   │               │               │
│   AsLS_SNV()           │               │               │
│ preprocess_none()      │               │               │
│ avg_y_block()          │               │               │
└────────────────────────┴───────────────┴───────────────┘
         │                                     │
         ▼                                     ▼
┌────────────────────────────────────────────────────────┐
│                      models/                           │
├──────────────┬─────────────────┬──────────────────────┤
│ run_loops.py │  wrappers.py    │ cross_val.py          │
├──────────────┼─────────────────┼──────────────────────┤
│ run_          │ PLS_model()     │ KFold_CV()            │
│  regression_ │ MLPRegressor_   │ KFold_Gridsearch_     │
│  loop()      │  model()        │  CV()                 │
│ run_          │ MLPClassifier_  │                       │
│  classifi-   │  model()        │                       │
│  cation_     │ PCA_model()     │                       │
│  loop()      │                 │                       │
├──────────────┴─────────────────┴──────────────────────┤
│             prediction_eval.py                         │
├────────────────────────────────────────────────────────┤
│             evaluate_on_prediction_set()               │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│                      plotting/                         │
├──────────────────┬────────────────┬────────────────────┤
│ plot_regression  │plot_classifier │ plot_raw.py        │
│ .py              │.py             │ plot_PCA.py        │
├──────────────────┼────────────────┼────────────────────┤
│ plot_pred_vs_    │ plot_confusion  │plot_spectra_       │
│  actual()        │  _matrix()     │ colored_by_        │
│ plot_cv_         │ plot_roc_       │  analyte()         │
│  performance()   │  curve()       │                    │
│ plot_coefficients│ plot_decision_  │                    │
│  ()              │  boundary()    │                    │
│ plot_feature_    │                │                    │
│  importance()    │                │                    │
│ plot_vip_scores()│                │                    │
│ plot_t2_q_       │                │                    │
│  residuals()     │                │                    │
│ plot_pls_scores()│                │                    │
│ print_CV_table() │                │                    │
│ print_model_     │                │                    │
│  summary()       │                │                    │
└──────────────────┴────────────────┴────────────────────┘
```

---

## Function-Level Call Graph

### `main.py` calls:
```
main.py
├── load_raman()                          ← loaders/raman_loader.py
├── group_preprocess_2()                  ← preprocessors/raman_preprocess.py
├── avg_y_block()                         ← preprocessors/raman_preprocess.py
├── align_group_xy()                      ← preprocessors/aligner.py
├── run_regression_loop()                 ← models/run_loops.py
└── evaluate_on_prediction_set()          ← models/prediction_eval.py
```

### `models/run_loops.py` calls:
```
run_regression_loop()
├── PLS_model()                           ← models/wrappers.py
└── MLPRegressor_model()                  ← models/wrappers.py

run_classification_loop()
├── bin_targets()                         ← preprocessors/labeling.py
└── MLPClassifier_model()                 ← models/wrappers.py
```

### `models/wrappers.py` calls:
```
PLS_model()
├── KFold_CV()                            ← models/cross_val.py
├── plot_cv_performance()                 ← plotting/plot_regression.py
├── plot_pred_vs_actual()                 ← plotting/plot_regression.py
├── plot_coefficients()                   ← plotting/plot_regression.py
├── plot_vip_scores()                     ← plotting/plot_regression.py
├── plot_t2_q_residuals()                 ← plotting/plot_regression.py
├── plot_pls_scores()                     ← plotting/plot_regression.py
├── print_CV_table()                      ← plotting/plot_regression.py
└── print_model_summary()                 ← plotting/plot_regression.py

MLPRegressor_model()
├── KFold_Gridsearch_CV()                 ← models/cross_val.py
├── plot_feature_importance()             ← plotting/plot_regression.py
├── plot_pred_vs_actual()                 ← plotting/plot_regression.py
└── print_model_summary()                 ← plotting/plot_regression.py

MLPClassifier_model()
├── KFold_Gridsearch_CV()                 ← models/cross_val.py
├── plot_confusion_matrix()               ← plotting/plot_classifier.py
├── plot_roc_curve()                      ← plotting/plot_classifier.py
└── plot_decision_boundary()              ← plotting/plot_classifier.py

PCA_model()   [standalone, called from streamlit_app.py]
└── (no plotting/ calls — plots inline with matplotlib)
```

### `models/cross_val.py` calls:
```
KFold_CV()
├── plot_cv_performance()                 ← plotting/plot_regression.py
└── plot_pred_vs_actual()                 ← plotting/plot_regression.py

KFold_Gridsearch_CV()
└── plot_pred_vs_actual()                 ← plotting/plot_regression.py
```

### `models/prediction_eval.py` calls:
```
evaluate_on_prediction_set()
└── plot_pred_vs_actual()                 ← plotting/plot_regression.py
```

### `streamlit_app.py` calls (all tabs):
```
streamlit_app.py
├── [Data Loading tab]
│   └── load_raman()                      ← loaders/raman_loader.py
│
├── [Preprocessing tab]
│   ├── preprocess_pipeline_1()           ← preprocessors/raman_preprocess.py
│   ├── preprocess_pipeline_2()           ← preprocessors/raman_preprocess.py
│   ├── preprocess_pipeline_AsLS_SNV()    ← preprocessors/raman_preprocess.py
│   ├── preprocess_none()                 ← preprocessors/raman_preprocess.py
│   ├── group_preprocess()                ← preprocessors/raman_preprocess.py
│   ├── group_preprocess_2()              ← preprocessors/raman_preprocess.py
│   ├── avg_y_block()                     ← preprocessors/raman_preprocess.py
│   ├── align_xy()                        ← preprocessors/aligner.py
│   ├── align_group_xy()                  ← preprocessors/aligner.py
│   └── plot_spectra_colored_by_analyte() ← plotting/plot_raw.py
│
├── [Modeling tab]
│   └── run_regression_loop()             ← models/run_loops.py
│           ├── PLS_model()               ← models/wrappers.py
│           └── MLPRegressor_model()      ← models/wrappers.py
│
├── [Prediction tab]
│   ├── group_preprocess_2()              ← preprocessors/raman_preprocess.py
│   ├── align_group_xy()                  ← preprocessors/aligner.py
│   └── evaluate_on_prediction_set()      ← models/prediction_eval.py
│
└── [PCA tab]
    ├── PCA_model()                       ← models/wrappers.py
    └── align_group_xy()                  ← preprocessors/aligner.py
```

---

## Dependency Matrix (imports by file)

| File | loaders | preprocessors | models | plotting |
|---|---|---|---|---|
| `main.py` | `raman_loader` | `raman_preprocess`, `aligner` | `run_loops`, `prediction_eval` | — |
| `streamlit_app.py` | `raman_loader` | `raman_preprocess`, `aligner` | `run_loops`, `wrappers`, `prediction_eval` | `plot_raw` |
| `models/run_loops.py` | — | `labeling` | `wrappers` | — |
| `models/wrappers.py` | — | — | `cross_val` | `plot_regression`, `plot_classifier` |
| `models/cross_val.py` | — | — | — | `plot_regression` |
| `models/prediction_eval.py` | — | — | — | `plot_regression` |

> **Note:** `plotting/` modules have no upward dependencies — they only import from `sklearn`, `matplotlib`, `numpy`, and `scipy`. They are pure output sinks.
