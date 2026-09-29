# FraudShield – Credit Card Fraud Detector

## Run it (VS Code terminal, inside this folder)

```bash
pip install -r requirements.txt
python train.py      # trains the model once, creates the model/ folder
python app.py        # starts the website
```

Open http://127.0.0.1:5000 in your browser. Press Ctrl+C in the terminal to stop.

## Files
- `train.py` – your notebook's pipeline (split, label-encode, scale, balance, models) and saves the model
- `app.py` – Flask server that loads the model and answers predictions
- `templates/index.html` – the colourful interface
- `credit_card_fraud_10k.csv` – your dataset (keep it next to train.py)

## Notes
- `transaction_id` is left out of the features because it is just a row number.
- Balancing uses SMOTE if `imbalanced-learn` is installed (it is in your notebook), otherwise random oversampling.
- The app flags fraud from 30% probability upwards (see THRESHOLD in train.py).
