Great results — all 4 models completed. Let me first analyse what the tuning tells us, then give you the updated pipeline code.

---

## Analysis of the 5k Tuning Results

### Class 3 is the critical problem across all models

| Model | Weighted F1 | Macro F1 | Class 3 F1 | Class 3 Recall |
|---|---|---|---|---|
| LogisticRegression | 0.8798 | 0.745 | 0.459 | 0.52 |
| SVM | 0.8791 | 0.704 | 0.294 | **0.19** |
| XGBoost | 0.8783 | 0.706 | 0.324 | 0.22 |
| ComplementNB | 0.7072 | 0.565 | 0.292 | 0.26 |

**SVM and XGBoost have high precision on Class 3 but terrible recall** — SVM: P=0.71, R=0.19. This means they're very confident when they predict Class 3 but they almost never predict it. They're being conservative and defaulting to Class 0/2. LR is the only model actually catching Class 3 cases (recall 0.52), which makes it the most valuable base model for the stacker right now.

**ComplementNB vs old MultinomialNB** — essentially identical (0.292 vs 0.286 F1 on Class 3). The 5k dataset is simply too small to show the difference. On 130k this gap will widen in CNB's favour.

**XGBoost took 7285 seconds (2 hours) on 5k rows** — on 130k this will be roughly 26× longer, potentially 50+ hours. You need to reduce `n_iter` to 10–15 and narrow the param grid for the overnight run. Plan for LightGBM instead — it's typically 5–10× faster.

---

Now here's the updated model definitions using tuning results:

Every tuned parameter is now wired in with comments explaining *why* each value was chosen. Here's what the analysis found and what to watch for overnight:

---

**Key insights from the 5k tuning:**

**SVM's surprising best C=0.1** — more regularisation than the default won. On high-dimensional TF-IDF (30k+ word features), heavier regularisation prevents the model from latching onto noise words. Expect this to hold or go even lower on 130k. The `hinge` loss win is the classic SVM loss — `squared_hinge` is usually better only when there's very little margin ambiguity.

**XGBoost's shallow trees (max_depth=4)** — the algorithm found that deep trees overfit on sparse text features. `min_child_weight=5` is the most important finding here — it forces each leaf to have at least 5 samples, which directly prevents the minority class (Class 3 with only 107 training examples at 5k) from creating overfit leaf nodes.

**The Class 3 recall problem is structural at 5k** — SVM gets R=0.19, XGB gets R=0.22. Neither is actually learning Class 3 well because with only 107 training examples, even with class weights they're defaulting conservative. On 130k (~3,500 Class 3 examples) this should jump significantly — potentially to 0.5–0.7 recall.

---

**Overnight run priority order:**

Run `LightGBM` first — fastest of the slow models, most likely to beat XGBoost on Class 3. Then `Ridge` (very fast). Skip re-running LR/SVM/XGB on 130k unless you want to verify the params held — the 5k params are likely stable. If you do re-run XGB, set `n_iter=10` and narrow the grid to `max_depth=[4,6]`, `learning_rate=[0.05, 0.08]` only — otherwise it'll run all night.
