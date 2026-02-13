Great question — the intuition here is subtle but important.

## What max_df does

`max_df=0.85` tells TF-IDF: *"ignore any word that appears in more than 85% of all documents."*

So if "does" appears in 90% of your 158,400 comments, it gets dropped entirely from the feature space.

---

## Why remove the most common words?

Think about what TF-IDF is actually trying to do. The full name is **Term Frequency - Inverse Document Frequency**, and the *IDF* part is the key here.

IDF rewards words that are **rare across documents but present in a specific one.** The formula is essentially:

```
IDF(word) = log(Total Documents / Documents containing word)
```

If "does" appears in 90% of documents, that fraction is nearly 1, and log(1) ≈ 0. So TF-IDF *already* gives it a near-zero score theoretically. The problem is that with 158,000 noisy real-world comments, the math doesn't perfectly zero it out — it still leaks through with small positive scores, as you can see in your output.

---

## The core intuition

Ask yourself this: **if a word appears in almost every comment, across all 4 classes, what does it tell your classifier about which class a comment belongs to?**

Absolutely nothing.

Your classifier is trying to distinguish between classes 0, 1, 2, 3. A word that shows up equally in all of them provides zero discriminative signal. It's like trying to identify a criminal in a lineup by noting that all of them have two eyes — the feature is useless because it doesn't separate anyone.

Words that *do* carry signal are ones that appear **frequently in class 1 but rarely in class 0**, or vice versa. That contrast is what lets the model learn.

---

## A concrete example from your data

Consider two words:

| Word | Appears in X% of docs | Useful? |
|---|---|---|
| "does" | 88% of all comments | No — appears everywhere equally |
| "slur" (hypothetical) | 4% of comments, but 60% of class 3 | Yes — highly discriminative |

The second word screams *"this comment might be class 3"* to your model. The first word says nothing.

---

## So why does `max_df` exist if IDF should handle this?

Because in practice, with large noisy datasets, very common words still accumulate enough raw term frequency to produce non-zero TF-IDF scores that clutter your feature space. `max_df` is a hard cutoff that says *"don't even bother computing TF-IDF for these words"* — it keeps your 30,000 feature slots reserved for words that actually have a chance of being informative, rather than wasting slots on grammatical glue.
