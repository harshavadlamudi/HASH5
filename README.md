# Healthcare Dashboard + Bedrock Chat

A Streamlit dashboard to explore healthcare CSVs with a Bedrock-powered chatbot grounded in your data.

## Prereqs

- Python 3.10+
- An AWS account with **Bedrock access enabled** for your chosen model (e.g., Claude 3.5 Sonnet).
- Credentials configured (AWS SSO/profile or env vars).
- `AWS_REGION` set to a region that supports your model.

## Setup

```bash
git clone <this-repo>
cd healthcare-dashboard-bedrock
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; or use the Streamlit sidebar to set values
