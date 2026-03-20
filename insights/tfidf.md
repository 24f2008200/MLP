Great questions. Let me address all three systematically.

---

## 1) Yes, TF-IDF Runs Every Time — And Here's The Fix

You're right. Inside `StackingClassifier`, each base model's pipeline runs its own `fit` → `transform` independently. So TF-IDF is being **fitted and applied 5 separate times** (once per base model), plus again for the meta-learner's CV folds. With 190k rows this is very expensive.

**The fix: pre-transform before stacking.**

The problem is your base models use *different* preprocessors (`preprocessor_light` vs `preprocessor_heavy` vs `preprocessor_nb`), so you can't share one matrix trivially. But you can unify them:

```python
# Step 1: Build ONE shared preprocessor with everything
preprocessor_shared = ColumnTransformer(transformers=[
    ("text_word", Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english", max_features=TFIDF_WORD_MAX, ngram_range=(1, 3),
            min_df=2, max_df=0.7, sublinear_tf=True, preprocessor=clean_text
        )),
        ("chi2", SelectKBest(chi2, k=CHISQUARE_WORD))
    ]), "comment"),

    ("text_chard", Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer='char_wb', max_features=TFIDF_CHAR_MAX, ngram_range=(3, 5),
            min_df=3, max_df=0.8, sublinear_tf=True, preprocessor=clean_text
        )),
        ("chi2", SelectKBest(chi2, k=CHISQUARE_CHAR))
    ]), "comment"),

    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="drop")

# Step 2: Fit ONCE on training data
X_train_proc = preprocessor_shared.fit_transform(X_train, y_train)
X_test_proc  = preprocessor_shared.transform(X_test)

# Step 3: Each model is now just a classifier, no pipeline
lr_model  = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
                                n_jobs=-1, class_weight=custom_weights)
nb_model  = MultinomialNB(alpha=0.2)   # ⚠️ see note below
svm_model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000,
                                              class_weight=custom_weights), cv=3)
xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, eval_metric='mlogloss')
lgb_model = lgb.LGBMClassifier(n_estimators=400, max_depth=7, learning_rate=0.08,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1,
                                class_weight=custom_weights, verbose=-1)

# Step 4: Stack on pre-processed arrays
stacking_model = StackingClassifier(
    estimators=[("lr", lr_model), ("svm", svm_model),
                ("xgb", xgb_model), ("lgb", lgb_model)],
    final_estimator=LogisticRegression(C=2.0, max_iter=1000,
                                        class_weight=custom_weights),
    cv=STACKING_CV, n_jobs=STACKING_N_JOBS
)

stacking_model.fit(X_train_proc, y_train)
```

> ⚠️ **NaiveBayes caveat**: `MultinomialNB` requires non-negative input. With `StandardScaler` in the shared pipeline, numeric columns will have negative values. Either drop NB from the stack, use a `MaxAbsScaler` instead for numerics, or keep NB in its own separate pipeline outside the stack.

**Expected speedup**: 3–5x on fit time since TF-IDF (the heaviest step) runs once instead of 5+ times.

---

## 2) Reducing Overfitting

Your current setup has several overfitting risks with 190k rows and heavy class imbalance.

**On XGBoost:**
```python
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,          # ↓ from 6 — deeper trees memorize noise
    learning_rate=0.05,   # ↓ slower learning = better generalization
    subsample=0.7,        # ↓ row subsampling adds variance reduction
    colsample_bytree=0.6, # ↓ feature subsampling
    colsample_bylevel=0.7,# add this — subsampling per tree level
    min_child_weight=5,   # ↑ from 1 — prevents splits on tiny groups
    gamma=0.3,            # ↑ from 0.1 — higher minimum split gain
    reg_alpha=0.1,        # add L1 regularization
    reg_lambda=2.0,       # add L2 regularization
    random_state=42, n_jobs=-1, eval_metric='mlogloss'
)
```

**On LightGBM:**
```python
lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=6,
    num_leaves=31,          # ↓ this matters more than max_depth in LGBM
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_samples=30,   # ↑ from 10 — require more samples per leaf
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42, n_jobs=-1,
    class_weight=custom_weights, verbose=-1
)
```

**On Logistic Regression:**
```python
# Lower C = stronger regularization
lr_model = LogisticRegression(C=0.3, solver="saga",  # saga handles L1+L2
                               penalty="elasticnet", l1_ratio=0.5,
                               max_iter=3000, n_jobs=-1,
                               class_weight=custom_weights)
```

**On TF-IDF itself** — overfitting can happen at feature selection too:
```python
# Tighten chi2 selection — fewer features = less noise fitting
SelectKBest(chi2, k=min(CHISQUARE_WORD, 15000))  # experiment with this cap
```

---

## 3) Improving Score on Class 3 (the hard problem)

With 9k vs ~60k+ per other class, this needs more than just `class_weight`. Here's a layered strategy:

**A) Fix class weights more aggressively:**
```python
# Instead of guessing, compute from data
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
custom_weights = dict(zip(classes, weights))

# Then manually boost class 3 further on top
custom_weights[3] *= 2.0  # experiment: 1.5 → 3.0
```

**B) SMOTE on the pre-processed matrix** (oversample class 3 in feature space):
```python
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek  # SMOTE + clean boundary noise

sm = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5))
X_train_bal, y_train_bal = sm.fit_resample(X_train_proc, y_train)

# Then train on X_train_bal — this physically creates new class 3 samples
stacking_model.fit(X_train_bal, y_train_bal)
```

**C) Use macro F1 as your optimization target** — accuracy will hide class 3 failures:
```python
from sklearn.metrics import classification_report, f1_score

y_pred = stacking_model.predict(X_test_proc)
print(classification_report(y_test, y_pred, digits=4))

# Use this as your single tuning metric
macro_f1 = f1_score(y_test, y_pred, average="macro")
```

**D) Threshold tuning at prediction time** — shift decision boundary toward class 3:
```python
# Get probabilities from stacking model
y_proba = stacking_model.predict_proba(X_test_proc)

# Manually boost class 3 probability before argmax
boost = np.array([1.0, 1.0, 1.0, 2.0])  # tune this multiplier
y_proba_boosted = y_proba * boost
y_pred_tuned = np.argmax(y_proba_boosted, axis=1)

print(classification_report(y_test, y_pred_tuned))
```

**E) Consider a two-stage classifier** — sometimes works very well for severe imbalance:
```python
# Stage 1: binary — is this class 3 or not?
y_binary = (y_train == 3).astype(int)
binary_clf = lgb.LGBMClassifier(class_weight="balanced", ...)
binary_clf.fit(X_train_proc, y_binary)

# Stage 2: among non-class-3, classify 0/1/2
mask = y_train != 3
multi_clf = lgb.LGBMClassifier(...)
multi_clf.fit(X_train_proc[mask], y_train[mask])

# Combine at inference
is_c3 = binary_clf.predict(X_test_proc)
final_pred = np.where(is_c3, 3, multi_clf.predict(X_test_proc))
```

---

**Quick priority order for biggest gains:**

1. Pre-process once (saves time, lets you iterate faster)
2. SMOTE or aggressive class weight boost on class 3
3. Threshold tuning — free win, no retraining needed
4. Regularization tweaks on XGB/LGBM
5. Two-stage classifier if class 3 F1 is still below 0.5
6. 

Yes, absolutely. You can insert a custom transformer in the NB pipeline that clips negatives to zero (which is more correct for NB than `abs`).

`abs` is mathematically wrong for NB — a value of `-0.5` and `+0.5` are very different features, but `abs` makes them identical. **Clipping to zero** is the right approach because it says "this feature simply isn't present/active for this sample."

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import sparse

class ClipNegatives(BaseEstimator, TransformerMixin):
    """Clips negative values to 0. Handles both dense and sparse matrices."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if sparse.issparse(X):
            X = X.copy()
            X.data = np.clip(X.data, 0, None)
            return X
        return np.clip(X, 0, None)
```

Then just drop it into the NB pipeline:

```python
nb_model = Pipeline([
    ("clip", ClipNegatives()),   # ← handles the StandardScaler negatives
    ("clf",  MultinomialNB(alpha=0.2))
])
```

And in the stacking setup it slots in cleanly:

```python
stacking_model = StackingClassifier(
    estimators=[
        ("lr",  lr_model),
        ("nb",  nb_model),   # ← works fine now
        ("svm", svm_model),
        ("xgb", xgb_model),
        ("lgb", lgb_model),
    ],
    final_estimator=LogisticRegression(C=2.0, max_iter=1000,
                                        class_weight=custom_weights),
    cv=STACKING_CV,
    n_jobs=STACKING_N_JOBS
)
```

> One more thing worth knowing: NB on clipped TF-IDF + scaled numerics is a bit of an odd combination anyway. NB works best when its input genuinely represents counts or frequencies. If your numeric columns are things like upvote counts or post length, clipping works fine. But if they're ratios or z-scores that can legitimately be negative in meaning, NB may not learn much from them — it'll just ignore the clipped values. That's okay though, it'll still learn well from the TF-IDF portion.

