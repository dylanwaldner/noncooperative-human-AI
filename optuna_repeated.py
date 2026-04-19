import copy
import random
from dataclasses import dataclass


import numpy as np
import optuna

from repeated_games import (
    RepeatedGameEnv,
    AIAgent,
    LearningHumanPTAgent,
    AwareHumanPTAgent,
    train_agents,
    get_all_games,
)

BASE_SEED = 42


DEFAULT_PT_PARAMS = {
    "alpha": 0.88,
    "gamma": 0.61,
    "delta": 0.69,
    "lambd": 2.25,
    "r": 0.0,
}
	

@dataclass
class StudyConfig:
    game_name: str
    agent1_type: str
    agent2_type: str
    target_slot: int                  # 1 or 2

    episodes: int = 500
    horizon: int = 100
    state_history: int = 2
    action_size: int = 2

    ref_setting: str = "Fixed"
    ref_lambda: float = 0.95

    exploration_decay: float = 0.99
    n_seeds: int = 3
    last_frac: float = 0.2

    pt_params1: dict = None
    pt_params2: dict = None

    fixed_kwargs_by_type: dict = None


def make_objective(cfg):
    games = get_all_games()
    payoff_matrix = games[cfg.game_name]

    def objective(trial):
        target_agent_type = cfg.agent1_type if cfg.target_slot == 1 else cfg.agent2_type
        target_hparams = suggest_hparams(trial, target_agent_type)

        seed_scores = []

        for seed_idx in range(cfg.n_seeds):
            seed = BASE_SEED + seed_idx
            np.random.seed(seed)
            random.seed(seed)

            env = RepeatedGameEnv(
                payoff_matrix,
                horizon=cfg.horizon,
                state_history=cfg.state_history,
            )

            hparams = copy.deepcopy(target_hparams)
            agent1, agent2 = build_agents_for_trial(
                env=env,
                payoff_matrix=payoff_matrix,
                cfg=cfg,
                target_hparams=hparams,
            )

            results = train_agents(
                agent1=agent1,
                agent2=agent2,
                env=env,
                episodes=cfg.episodes,
                exploration_decay=cfg.exploration_decay,
                verbose=False,
                game_name=cfg.game_name,
            )

            reward_key = "avg_rewards1" if cfg.target_slot == 1 else "avg_rewards2"
            curve = np.asarray(results[reward_key], dtype=float)
            tail_n = max(1, int(len(curve) * cfg.last_frac))
            score = float(curve[-tail_n:].mean())

            seed_scores.append(score)

            trial.report(np.mean(seed_scores), step=seed_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(seed_scores))

    return objective

def run_study(cfg: StudyConfig, n_trials: int = 50, study_name: str | None = None, storage: str | None = None):
    sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True if storage else False,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(make_objective(cfg), n_trials=n_trials)

    return study


def build_agent(
    agent_type,
    agent_id,
    env,
    payoff_matrix,
    action_size,
    pt_params_self,
    pt_params_opp,
    ref_setting,
    ref_lambda,
    ref_point_opp,
    opponent_type,
    agent_kwargs=None,
):
    agent_kwargs = agent_kwargs or {}

    if agent_type == 'LH':
        agent_kwargs = copy.deepcopy(agent_kwargs or {})
        lambda_ref = agent_kwargs.pop("lambda_ref", ref_lambda)

        return LearningHumanPTAgent(
            env.state_size,
            action_size,
            action_size,
            pt_params_self,
            agent_id=agent_id,
            ref_setting=ref_setting,
            lambda_ref=lambda_ref,
            payoff_matrix=payoff_matrix,
            **agent_kwargs,
        )
    elif agent_type == 'AI':
        return AIAgent(
            env.state_size,
            action_size,
            action_size,
            agent_id=agent_id,
            **agent_kwargs,
        )

    elif agent_type == 'AH1':
        opp_params = dict()
        opp_params['opponent_type'] = opponent_type
        opp_params['opponent_action_size'] = action_size
        opp_params['opp_ref'] = None

        if opponent_type != "AI":  # PT agent
            opp_params['opp_ref'] = ref_point_opp
            opp_params['opp_pt'] = pt_params_opp

        return AwareHumanPTAgent(
            payoff_matrix,
            pt_params_self,
            action_size,
            env.state_size,
            agent_id=agent_id,
            opp_params=opp_params,
            ref_setting=ref_setting,
        )

    elif agent_type == 'AH2':
        opp_params = dict()
        opp_params['opponent_type'] = opponent_type
        opp_params['opponent_action_size'] = action_size
        opp_params['opp_ref'] = None

        if opponent_type != "AI":  # PT agent
            opp_params['opp_ref'] = ref_point_opp
            opp_params['opp_pt'] = pt_params_opp

        return AwareHumanPTAgent(
            payoff_matrix,
            pt_params_self,
            action_size,
            env.state_size,
            agent_id=agent_id,
            opp_params=opp_params,
            ref_setting=ref_setting,
            tit_for_tat=True,
        )

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def build_agents_for_trial(
    env,
    payoff_matrix,
    cfg,
    target_hparams,
):
    pt_params1 = copy.deepcopy(cfg.pt_params1 or DEFAULT_PT_PARAMS)
    pt_params2 = copy.deepcopy(cfg.pt_params2 or DEFAULT_PT_PARAMS)

    fixed_agent1_kwargs = copy.deepcopy((cfg.fixed_kwargs_by_type or {}).get(cfg.agent1_type, {}))
    fixed_agent2_kwargs = copy.deepcopy((cfg.fixed_kwargs_by_type or {}).get(cfg.agent2_type, {}))

    # Apply Optuna params only to the target slot
    if cfg.target_slot == 1:
        agent1_kwargs = copy.deepcopy(target_hparams)
        agent2_kwargs = fixed_agent2_kwargs
    else:
        agent1_kwargs = fixed_agent1_kwargs
        agent2_kwargs = copy.deepcopy(target_hparams)

    # ref points passed into aware-human opponent models
    ref_point1 = pt_params1.get('r', 0.0)
    ref_point2 = pt_params2.get('r', 0.0)

    agent1 = build_agent(
        agent_type=cfg.agent1_type,
        agent_id=0,
        env=env,
        payoff_matrix=payoff_matrix,
        action_size=cfg.action_size,
        pt_params_self=pt_params1,
        pt_params_opp=pt_params2,
        ref_setting=cfg.ref_setting,
        ref_lambda=cfg.ref_lambda,
        ref_point_opp=ref_point2,
        opponent_type=cfg.agent2_type,
        agent_kwargs=agent1_kwargs,
    )

    agent2 = build_agent(
        agent_type=cfg.agent2_type,
        agent_id=1,
        env=env,
        payoff_matrix=payoff_matrix,
        action_size=cfg.action_size,
        pt_params_self=pt_params2,
        pt_params_opp=pt_params1,
        ref_setting=cfg.ref_setting,
        ref_lambda=cfg.ref_lambda,
        ref_point_opp=ref_point1,
        opponent_type=cfg.agent1_type,
        agent_kwargs=agent2_kwargs,
    )

    return agent1, agent2

def suggest_hparams(trial, agent_type):
    if agent_type == "AI":
        return {
            "epsilon": trial.suggest_float("epsilon", 0.05, 0.5),
            "alpha": trial.suggest_float("alpha", 1e-3, 0.3, log=True),
            "k": trial.suggest_float("k", 0.3, 1.0),
            "tau": trial.suggest_float("tau", 1e-3, 0.5, log=True),
            "temp": trial.suggest_float("temp", 0.3, 3.0),
            "epsilon_decay": trial.suggest_float("epsilon_decay", 0.95, 0.999),
        }

    elif agent_type == "LH":
        return {
            "epsilon": trial.suggest_float("epsilon", 0.05, 0.5),
            "alpha": trial.suggest_float("alpha", 1e-3, 0.3, log=True),
            "k": trial.suggest_float("k", 0.3, 1.0),
            "tau": trial.suggest_float("tau", 1e-3, 0.5, log=True),
            "temperature": trial.suggest_float("temperature", 0.3, 3.0),
            "epsilon_decay": trial.suggest_float("epsilon_decay", 0.95, 0.999),
            "lam_b": trial.suggest_float("lam_b", 0.7, 0.999),
            "lambda_ref": trial.suggest_float("lambda_ref", 0.7, 0.999),
        }

    elif agent_type in ["AH1", "AH2"]:
        raise ValueError(f"Optuna search space not implemented for {agent_type}")

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def get_matchups_for_game(game_name, state_history):
    if game_name in ["PrisonersDilemma", "StagHunt", "Chicken"]:
        matchups = [
            ("AH1", "AI"),
            ("AH2", "AI"),
            ("LH", "AI"),
            ("AH1", "LH"),
            ("AH2", "LH"),
            ("AH1", "AH1"),
            ("AH2", "AH2"),
            ("LH", "LH"),
            ("AI", "AI"),
        ]
    else:
        matchups = [
            ("AH1", "AI"),
            ("AI", "AH1"),
            ("AH2", "AI"),
            ("AI", "AH2"),
            ("LH", "AI"),
            ("AI", "LH"),
            ("AH1", "LH"),
            ("LH", "AH1"),
            ("AH2", "LH"),
            ("LH", "AH2"),
            ("AH1", "AH1"),
            ("AH2", "AH2"),
            ("LH", "LH"),
            ("AI", "AI"),
        ]

    if state_history == 0:
        matchups = [
            (a1, a2) for (a1, a2) in matchups
            if a1 != "AH2" and a2 != "AH2"
        ]

    return matchups

def run_all_studies(games_dict, base_cfg, n_trials=50, storage="sqlite:///optuna_repeated.db"):
    all_results = dict()

    for game_name, payoff_matrix in games_dict.items():
        matchups = get_matchups_for_game(game_name, base_cfg.state_history)

        for agent1_type, agent2_type in matchups:
            matchup_key = f"{game_name}__{agent1_type}_vs_{agent2_type}"
            all_results[matchup_key] = dict()

            # tune slot 1
            cfg1 = StudyConfig(
                game_name=game_name,
                agent1_type=agent1_type,
                agent2_type=agent2_type,
                target_slot=1,

                episodes=base_cfg.episodes,
                horizon=base_cfg.horizon,
                state_history=base_cfg.state_history,
                action_size=base_cfg.action_size,

                ref_setting=base_cfg.ref_setting,
                ref_lambda=base_cfg.ref_lambda,

                exploration_decay=base_cfg.exploration_decay,
                n_seeds=base_cfg.n_seeds,
                last_frac=base_cfg.last_frac,

                pt_params1=copy.deepcopy(base_cfg.pt_params1),
                pt_params2=copy.deepcopy(base_cfg.pt_params2),

                fixed_kwargs_by_type=copy.deepcopy(base_cfg.fixed_kwargs_by_type),
            )

            study_name_1 = f"{matchup_key}__tune_agent1"
            try:
                study1 = run_study(cfg1, n_trials=n_trials, study_name=study_name_1, storage=storage)
                all_results[matchup_key]["tune_agent1"] = {
                    "best_value": study1.best_value,
                    "best_params": study1.best_params,
                }
            except ValueError as e:
                print(f"Skipping {study_name_1}: {e}")

            # tune slot 2
            cfg2 = StudyConfig(
                game_name=game_name,
                agent1_type=agent1_type,
                agent2_type=agent2_type,
                target_slot=2,

                episodes=base_cfg.episodes,
                horizon=base_cfg.horizon,
                state_history=base_cfg.state_history,
                action_size=base_cfg.action_size,

                ref_setting=base_cfg.ref_setting,
                ref_lambda=base_cfg.ref_lambda,

                exploration_decay=base_cfg.exploration_decay,
                n_seeds=base_cfg.n_seeds,
                last_frac=base_cfg.last_frac,

                pt_params1=copy.deepcopy(base_cfg.pt_params1),
                pt_params2=copy.deepcopy(base_cfg.pt_params2),

                fixed_kwargs_by_type=copy.deepcopy(base_cfg.fixed_kwargs_by_type),
            )

            study_name_2 = f"{matchup_key}__tune_agent2"
            try:
                study2 = run_study(cfg2, n_trials=n_trials, study_name=study_name_2, storage=storage)
                all_results[matchup_key]["tune_agent2"] = {
                    "best_value": study2.best_value,
                    "best_params": study2.best_params,
                }
            except ValueError as e:
                print(f"Skipping {study_name_2}: {e}")

    return all_results

if __name__ == "__main__":
    base_cfg = StudyConfig(
        game_name="PrisonersDilemma",   # placeholder, overwritten in run_all_studies
        agent1_type="LH",               # placeholder
        agent2_type="AI",               # placeholder
        target_slot=1,                  # placeholder
        
        episodes=500,
        horizon=100,
        state_history=2,
        action_size=2,

        ref_setting="EMA",
        ref_lambda=0.95,

        exploration_decay=0.99,
        n_seeds=3,
        last_frac=0.2,

        pt_params1=copy.deepcopy(DEFAULT_PT_PARAMS),
        pt_params2=copy.deepcopy(DEFAULT_PT_PARAMS),

        fixed_kwargs_by_type={
            "AI": {
                "epsilon": 0.3,
                "alpha": 0.1,
                "k": 0.7,
                "tau": 0.1,
                "temp": 1.3,
            },
            "LH": {
                "epsilon": 0.3,
                "alpha": 0.1,
                "k": 0.7,
                "tau": 0.1,
                "temperature": 1.3,
                "lam_b": 0.95,
                "lambda_ref": 0.95,
                "B": 5,
            },
            "AH1": {},
            "AH2": {},
        },
    )
    games = get_all_games()
    results = run_all_studies(
        games_dict=games,
        base_cfg=base_cfg,
        n_trials=50,
        storage="sqlite:///optuna_repeated.db",
    )

    for matchup_key, result in results.items():
        print(matchup_key, result)
