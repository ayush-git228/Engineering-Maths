# Linear Algebra × Agent AI

Learn **Engineering Maths** by building tiny **Agent AI** components using pure **NumPy/SciPy**.  
No API keys, no cloud. Each math topic powers a concrete agent behavior (routing, ranking, memory projection, planning, safety, summarization).

---
🤝 Contributing

I welcome all kinds of contributions - bug fixes 🐛, new features ✨, docs 📖, tests ✅, and ideas 💡.
I will keep on adding other topics for ex. Calculus, Probability and Statistics etc. to this, 
You can also contribute to make this existing repository better.

---

## 📂 What’s inside

- **la_core/** – Linear Algebra implementations you’d expect for GATE-DA:
  - Gaussian elimination, LU, eigen-decomposition, SVD, projections, rank/nullity, special matrices (projection/orthogonal/idempotent/partition), quadratic forms.
- **agent_apps/** – Mini-projects that connect LA → Agent behaviors:
  - 01_vector_search/cosine_sim.py – Vector search (cosine similarity)

  - 02_pagerank_agent/pagerank_power_iteration.py – PageRank agent

  - 03_least_squares_planner/linear_regression_normal_eq.py – Least-squares planner

        cost_forecast.ipynb – cost forecasting demo

  - 04_projection_based_memory/context_projection.ipynb – Projection-based memory

  - 05_low_rank_summarizer/svd_summarizer.py – Low-rank summarizer

        text_summarize.ipynb – summarization demo

  - 06_embedding_router/task_routing.ipynb – Embedding router

  - 07_quadratic_forms_rl/safe_actions.ipynb – Quadratic-forms RL safety demo

  
- **docs/** – 
  - `TOPICS_MAP.md` (maps each GATE-DA LA topic → implemented code)
  - `CHEATSHEETS.md` (quick LA formulas + agent connections)

---

## ⚙️ Install

```bash
conda env create -f environment.yml
conda activate dsai-la-agent
```

🚀 Run Examples

```bash
# Vector search (cosine similarity)
python agent_apps/01_vector_search/cosine_sim.py

# PageRank-style action ranking
python agent_apps/02_pagerank_agent/pagerank_power_iteration.py

# Learn planner weights with least squares
python agent_apps/03_least_squares_planner/linear_regression_normal_eq.py

# Projection-based memory (notebook)
jupyter notebook agent_apps/04_projection_based_memory/context_projection.ipynb

# Low-rank summarizer & toy recommender
python agent_apps/05_low_rank_summarizer/svd_summarizer.py
jupyter notebook agent_apps/05_low_rank_summarizer/text_summarize.ipynb

# Embedding router (notebook)
jupyter notebook agent_apps/06_embedding_router/task_routing.ipynb

# Quadratic-forms RL safety demo (notebook)
jupyter notebook agent_apps/07_quadratic_forms_rl/safe_actions.ipynb
```

🧪 Testing

```bash
pytest -q
```
