import optuna
from optuna.trial import TrialState

storage = "sqlite:///optuna_repeated.db"

studies = optuna.study.get_all_study_summaries(storage=storage)

for s in studies:
    print(f"\nStudy: {s.study_name}")

    study = optuna.load_study(study_name=s.study_name, storage=storage)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed:
        print("  No completed trials.")
        continue

    print("  Best value:", study.best_value)
    print("  Param keys:", list(study.best_params.keys()))
    print("  Full params:", study.best_params)
