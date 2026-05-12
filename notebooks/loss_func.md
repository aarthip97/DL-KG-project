# Debiased Listwise Loss function

**Popularity-Debiased Logit Adjustment**
To mitigate popularity bias and explicitly promote catalog coverage, we apply L2-normalization to the embeddings and introduce a log-popularity penalty to the predicted logits.

Let $\pi_i$ represent the global popularity probability of item $i$, defined as its interaction frequency over the total interactions in the dataset. We replace the standard inner product with a temperature-scaled Cosine Similarity, denoted by $\tau$. We then compute the adjusted logit $\tilde{y}_{ui}$ by penalizing the similarity score with the item's log-popularity, controlled by the coverage-regularization hyperparameter $\lambda$:

$$\tilde{y}_{ui} = \frac{1}{\tau} \left( \frac{\mathbf{e}_u^\top \mathbf{e}_i}{||\mathbf{e}_u|| \, ||\mathbf{e}_i||} \right) + \lambda \log(\pi_i)$$

*(Note: Because $\pi_i \in (0, 1]$, the term $\log(\pi_i)$ is strictly negative, acting as a direct penalty to highly popular items).*

>We control this with a hyperparameter $\lambda$ (Lambda).
>* If $\lambda = 0$, you get pure NDCG optimization (high popularity bias).
>* If $\lambda > 0$, the model mathematically suppresses popular tracks, forcing it to rank undiscovered, long-tail tracks higher to minimize the loss. This directly explodes your Catalog Coverage.   
>* By exposing `lambda_reg` as a parameter, we can now run a grid search (e.g., $\lambda \in [0.0, 0.1, 0.3, 0.5]$) and track our `Overall_Score` just like we did in our KNN baseline.

The adjusted predicted distribution $Q_u$ is then formulated using the Softmax over the debiased logits:

$$Q_u(i) = \frac{\exp(\tilde{y}_{ui})}{\sum_{j \in I} \exp(\tilde{y}_{uj})}$$

The final loss remains the cross-entropy between the ideal intensity-weighted distribution $P_u$ and the debiased prediction distribution $Q_u$:

$$\mathcal{L}_{debiased} = -\frac{1}{|B|} \sum_{u \in B} \sum_{i \in I} P_u(i) \log Q_u(i)$$

By integrating $\lambda \log(\pi_i)$ directly into the Listwise objective, the model is mathematically forced to rank high-quality, long-tail items prominently to minimize the divergence, thereby increasing overall catalog coverage without requiring post-hoc heuristic re-ranking.