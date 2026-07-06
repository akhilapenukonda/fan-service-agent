# Fan Services AI Agent + Evaluation Harness

A RAG-based fan services assistant grounded in real, publicly posted NFL/team
policies (refunds, bag policy, parking/conduct), paired with a rigorous
evaluation harness covering accuracy, safety, latency, and cost.

## Project Structure
- `data/` — knowledge base source docs (real policy-grounded, paraphrased)
- `notebooks/01_agent_build.ipynb` — agent build, retrieval testing, eval harness
- `eval/` — vector store, 25-question test set results, eval CSV
- `app.py` — Streamlit chat interface demo

## Results
- Accuracy: 94.1% (16/17 correct, 1 partial) on in-scope/ambiguous/edge-case questions
- Safety: 100% (8/8) correct refusal rate on out-of-scope and adversarial questions
- Avg latency: 4.06s | Avg cost: $0.00223/query (~$11/month at 5,000 queries)

## Setup
\`\`\`
pip install -r requirements.txt
# Add your ANTHROPIC_API_KEY to a .env file
streamlit run app.py
\`\`\`
