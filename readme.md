# Shepherd AI - Emergency Evacuation Simulator

## Setup
1. Create a virtual environment: `python -m venv env`
2. Activate it: `.\env\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`

## Running the Project
- **Train:** Run `python world_setup.py` to start the 500k step training.
- **Monitor:** Run `python -m tensorboard.main --logdir ./ppo_shepherd_logs/` to see live graphs.
- **Model:** The trained brain is saved as `shepherd_ai_model.zip`.