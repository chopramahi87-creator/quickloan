# QuickLoan Risk Dashboard

## Run it (3 commands)
```bash
pip install -r requirements.txt
python train_models.py      # trains 10 models, writes leaderboard.json + .joblib files
streamlit run app.py        # opens http://localhost:8501
```

## Files
| File | Role |
|---|---|
| `loans.csv` | Data (6,000 loans, target = `default`) |
| `train_models.py` | Trains singles + Voting/Bagging/Boosting/Stacking, scores all on one test set, saves outputs |
| `leaderboard.json` | Accuracy, recall, caught/wrong counts and AED amounts per model (read by tab 1) |
| `all_models.joblib` | Every trained model, so the sidebar can switch models |
| `loan_model.joblib` | Champion only, in the `{'model','columns'}` format from the brief |
| `app.py` | Streamlit dashboard (4 tabs) |
| `requirements.txt` | Dependencies |

## Dashboard tabs
1. **Leaderboard**: accuracy vs money-saved charts, sortable table, interest slider.
2. **Loan Decision**: enter an applicant -> APPROVE / REJECT, default probability, expected profit.
3. **Portfolio Money**: money saved vs approve-all, defaulters caught, good borrowers rejected, money-vs-cut-off curve.
4. **Data Explorer**: default rate by past defaults / home ownership / debt-to-income, feature importance.

## Important
`train_models.py` and `app.py` use the same split (`test_size=0.25, random_state=1, stratify=y`).
If you change one, change the other, or the Portfolio tab will be scored on rows the model trained on.
