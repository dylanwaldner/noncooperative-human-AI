import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from collections import Counter
from matplotlib.ticker import FuncFormatter
from .utils import smooth, convert
import os
from pathlib import Path

import json
import gzip

DIR_PATH = "/Users/dylanwaldner/Projects/RLNash/Experiments"

def analyze_matchup(results, agent1_type, agent2_type, game_name, games_dict, payoff_matrix, pt_params, ref_type, env):
    """
    I am not going through in line, and i went a little in depth on the read me. 
    I feel like this is pretty straightforward, but if its not please just send me an email,
    I'll be quick to respond
    """
    state_history = env.state_history    

    path = Path(DIR_PATH) / f"game_{game_name}" / f"sh_{state_history}" / f"_matchup{agent1_type}_{agent2_type}"
    os.makedirs(path, exist_ok=True)

    num_experiments = len(results.keys())

    fig = plt.figure(figsize=(15, 10))

    # 1. Avg rewards
    ax1 = plt.subplot(3, 3, 1)
    window = 20

    smoothed_p1 = []
    smoothed_p2 = []
    for idx in range(len(results.keys())):
        smoothed1 = np.convolve(results[f"{idx}"]['avg_rewards1'], np.ones(window)/window, mode='valid')
        smoothed2 = np.convolve(results[f"{idx}"]['avg_rewards2'], np.ones(window)/window, mode='valid')

        smoothed_p1.append(smoothed1)
        smoothed_p2.append(smoothed2)
        
    smoothed_p1 = np.stack(smoothed_p1)
    smoothed_p2 = np.stack(smoothed_p2)

    mean_p1, mean_p2 = np.mean(smoothed_p1, axis=0), np.mean(smoothed_p2, axis=0)
    std_p1, std_p2 = np.std(smoothed_p1, axis=0), np.std(smoothed_p2, axis=0)
    se_p1, se_p2 = std_p1 / np.sqrt(num_experiments), std_p2 / np.sqrt(num_experiments)

    ax1.plot(mean_p1, label=f'{agent1_type}', linewidth=2)
    ax1.plot(mean_p2, label=f'{agent2_type}', linewidth=2)

    x = np.arange(len(mean_p1))

    ax1.fill_between(x, mean_p1 + 1.96 * se_p1, mean_p1 - 1.96 * se_p1, alpha=0.3)
    ax1.fill_between(x, mean_p2 + 1.96 * se_p2, mean_p2 - 1.96 * se_p2, alpha=0.3)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (± std)')
    ax1.set_title(f'95% Conf. Interval Reward Over {len(results.keys())} Runs\n{agent1_type} vs {agent2_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final strategy patterns
    ax2 = plt.subplot(3, 3, 2)

    ref_points_p1 = []
    ref_points_p2 = []

    # Storing these just to make AH ref point explicit
    r_1 = None
    r_2 = None

    for idx in range(len(results.keys())):
        ref_points1 = results[f"{idx}"]['ref_points1']
        ref_points2 = results[f"{idx}"]['ref_points2']
        
        if r_1 is None:
            r_1 = ref_points1[-1]

        if r_2 is None:
            r_2 = ref_points2[-1]

        ref_points_p1.append(ref_points1)
        ref_points_p2.append(ref_points2)

    ref_points_p1 = np.stack(ref_points_p1)
    ref_points_p2 = np.stack(ref_points_p2)

    mean_p1, mean_p2 = np.mean(ref_points_p1, axis=0), np.mean(ref_points_p2, axis=0)
    std_p1, std_p2 = np.std(ref_points_p1, axis=0), np.std(ref_points_p2, axis=0)
    se_p1, se_p2 = std_p1 / np.sqrt(num_experiments), std_p2 / np.sqrt(num_experiments)
    
    if agent1_type.startswith("AH"):
        ax2.plot(mean_p1, label=f'{agent1_type} r={r_1}')
    else:
        ax2.plot(mean_p1, label=f'{agent1_type}')

    if agent2_type.startswith("AH"):
        ax2.plot(mean_p2, label=f'{agent2_type} r={r_2}')
    else:
        ax2.plot(mean_p2, label=f'{agent2_type}')

    x = np.arange(len(mean_p1))

    ax2.fill_between(x, mean_p1 + 1.96 * se_p1, mean_p1 - 1.96 * se_p1, alpha=0.3)

    x = np.arange(len(mean_p2))
    ax2.fill_between(x, mean_p2 + 1.96 * se_p2, mean_p2 - 1.96 * se_p2, alpha=0.3)

    ax2.set_xlabel(f'Step (Every 100)')
    ax2.set_ylabel('Reference Point')
    ax2.set_title(f'95% Conf. Interval of Ref. Points Over {len(results.keys())} Runs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Action distribution
    ax3 = plt.subplot(3, 3, 3)

    joint_actions = []
    for idx in range(len(results.keys())):
        joint_action = results[f"{idx}"]["joint_actions"]
        joint_actions.append(joint_action)
      
    joint_actions = np.stack(joint_actions)

    mean_joint_actions = np.mean(joint_actions, axis=0)
    im = ax3.imshow(mean_joint_actions, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax3, label='Count')
    MAX_ACTIONS = 6

    if mean_joint_actions.shape[0] <= MAX_ACTIONS and mean_joint_actions.shape[1] <= MAX_ACTIONS:
        for i in range(mean_joint_actions.shape[0]):
            for j in range(mean_joint_actions.shape[1]):
                ax3.text(
                    j, i,
                    int(mean_joint_actions[i, j]),
                    ha="center",
                    va="center",
                    color="white" if mean_joint_actions[i, j] > mean_joint_actions.max() / 2 else "black"
                )

    ax3.set_xticks(np.arange(mean_joint_actions.shape[1]))
    ax3.set_xticklabels(np.arange(1, mean_joint_actions.shape[1] + 1))

    ax3.set_yticks(np.arange(mean_joint_actions.shape[0]))
    ax3.set_yticklabels(np.arange(1, mean_joint_actions.shape[0] + 1))

    if game_name == 'Double Auction Game':
        ax3.set_xlabel(f'Seller ({agent2_type})')
        ax3.set_ylabel(f'Buyer ({agent1_type})')

    else:
        ax3.set_xlabel(f'{agent2_type}')
        ax3.set_ylabel(f'{agent1_type}')

    ax3.set_title(f"Mean Joint Action Heatmap Over {len(results.keys())} Runs")

    # 4. Q value convergence 
    # 4. Q value convergence - Player 1
    ax4 = plt.subplot(3, 3, 4)
    if len(results["0"]['q_values1']) > 0:
        q_values_p1 = []
        for idx in range(len(results.keys())):
            q_values1 = np.stack(results[f"{idx}"]['q_values1'])
            q_values_p1.append(q_values1)

        q_values_p1 = np.stack(q_values_p1)

        mean_q_p1 = np.mean(q_values_p1, axis=0)
        se_q_p1 = np.std(q_values_p1, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_q_p1))

        if agent1_type == "LH":
            ax4.plot(mean_q_p1[:, 0, 0], label='a=0, a_-i=0')
            ax4.plot(mean_q_p1[:, 1, 0], label='a=1, a_-i=0')
            ax4.plot(mean_q_p1[:, 0, 1], label='a=0, a_-i=1')
            ax4.plot(mean_q_p1[:, 1, 1], label='a=1, a_-i=1')

            ax4.fill_between(x, mean_q_p1[:, 0, 0] + 1.96 * se_q_p1[:, 0, 0], mean_q_p1[:, 0, 0] - 1.96 * se_q_p1[:, 0, 0], alpha=0.3)
            ax4.fill_between(x, mean_q_p1[:, 1, 0] + 1.96 * se_q_p1[:, 1, 0], mean_q_p1[:, 1, 0] - 1.96 * se_q_p1[:, 1, 0], alpha=0.3)
            ax4.fill_between(x, mean_q_p1[:, 0, 1] + 1.96 * se_q_p1[:, 0, 1], mean_q_p1[:, 0, 1] - 1.96 * se_q_p1[:, 0, 1], alpha=0.3)
            ax4.fill_between(x, mean_q_p1[:, 1, 1] + 1.96 * se_q_p1[:, 1, 1], mean_q_p1[:, 1, 1] - 1.96 * se_q_p1[:, 1, 1], alpha=0.3)

        else:
            ax4.plot(mean_q_p1[:, 0], label='a=0')
            ax4.plot(mean_q_p1[:, 1], label='a=1')

            ax4.fill_between(x, mean_q_p1[:, 0] + 1.96 * se_q_p1[:, 0], mean_q_p1[:, 0] - 1.96 * se_q_p1[:, 0], alpha=0.3)
            ax4.fill_between(x, mean_q_p1[:, 1] + 1.96 * se_q_p1[:, 1], mean_q_p1[:, 1] - 1.96 * se_q_p1[:, 1], alpha=0.3)


        '''
        opp_freq = mean_joint_actions.sum(axis=0) / mean_joint_actions.sum()
        expected_payoff_a0 = payoff_matrix[0, :, 0] @ opp_freq
        expected_payoff_a1 = payoff_matrix[1, :, 0] @ opp_freq
        ax4.axhline(expected_payoff_a0, linestyle="--", color='gray', alpha=0.5, label="Expected R(a0)")
        ax4.axhline(expected_payoff_a1, linestyle="--", color='black', alpha=0.5, label="Expected R(a1)")
        '''
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Q-value')
    ax4.set_title(f'95% Conf. Interval {agent1_type} Q-Values Over {len(results.keys())} Runs')
    ax4.legend()

    # 5. Q value convergence - Player 2
    ax5 = plt.subplot(3, 3, 5)

    if len(results["0"]['q_values2']) > 0:
        q_values_p2 = []
        for idx in range(len(results.keys())):
            q_values2 = np.stack(results[f"{idx}"]['q_values2'])
            q_values_p2.append(q_values2)
            
        q_values_p2 = np.stack(q_values_p2)

        mean_q_p2 = np.mean(q_values_p2, axis=0)
        se_q_p2 = np.std(q_values_p2, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_q_p2))

        if agent2_type == "LH":
            ax5.plot(mean_q_p2[:, 0, 0], label='a=0, a_-i=0')
            ax5.plot(mean_q_p2[:, 1, 0], label='a=1, a_-i=0')
            ax5.plot(mean_q_p2[:, 0, 1], label='a=0, a_-i=1')
            ax5.plot(mean_q_p2[:, 1, 1], label='a=1, a_-i=1')

            ax5.fill_between(x, mean_q_p2[:, 0, 0] + 1.96 * se_q_p2[:, 0, 0], mean_q_p2[:, 0, 0] - 1.96 * se_q_p2[:, 0, 0], alpha=0.3)
            ax5.fill_between(x, mean_q_p2[:, 1, 0] + 1.96 * se_q_p2[:, 1, 0], mean_q_p2[:, 1, 0] - 1.96 * se_q_p2[:, 1, 0], alpha=0.3)
            ax5.fill_between(x, mean_q_p2[:, 0, 1] + 1.96 * se_q_p2[:, 0, 1], mean_q_p2[:, 0, 1] - 1.96 * se_q_p2[:, 0, 1], alpha=0.3)
            ax5.fill_between(x, mean_q_p2[:, 1, 1] + 1.96 * se_q_p2[:, 1, 1], mean_q_p2[:, 1, 1] - 1.96 * se_q_p2[:, 1, 1], alpha=0.3)
        else:
            ax5.plot(mean_q_p2[:, 0], label='a=0')
            ax5.plot(mean_q_p2[:, 1], label='a=1')

            ax5.fill_between(x, mean_q_p2[:, 0] + 1.96 * se_q_p2[:, 0], mean_q_p2[:, 0] - 1.96 * se_q_p2[:, 0], alpha=0.3)
            ax5.fill_between(x, mean_q_p2[:, 1] + 1.96 * se_q_p2[:, 1], mean_q_p2[:, 1] - 1.96 * se_q_p2[:, 1], alpha=0.3)

        '''
        opp_freq = mean_joint_actions.sum(axis=0) / mean_joint_actions.sum()
        expected_payoff_a0 = payoff_matrix[:, 0, 1] @ opp_freq
        expected_payoff_a1 = payoff_matrix[:, 1, 1] @ opp_freq
        ax5.axhline(expected_payoff_a0, linestyle="--", color='gray', alpha=0.5, label="Expected R(a0)")
        ax5.axhline(expected_payoff_a1, linestyle="--", color='black', alpha=0.5, label="Expected R(a1)")
        '''

    ax5.set_xlabel('Step')
    ax5.set_ylabel('Q-value')
    ax5.set_title(f'95% Conf. Interval {agent2_type} Q-Values Over {len(results.keys())} Runs')
    ax5.legend()

    # 6. Learning convergence (how much are q values changing)
    ax6 = plt.subplot(3, 3, 6)

    if len(results["0"]['q_values1']) > 0:
        q_changes_p1 = []
        for idx in range(len(results.keys())):
            q_values1 = np.stack(results[f"{idx}"]['q_values1'])
            q_values1, q_values1_copy = q_values1[1:], q_values1[:-1]
            q_values1_diff = q_values1 - q_values1_copy
            q_change1 = np.max(np.abs(q_values1_diff), axis = tuple(range(1, q_values1_diff.ndim)))

            q_changes_p1.append(q_change1)
                
        q_changes_p1 = np.stack(q_changes_p1)
        
        mean_changes_p1 = np.mean(q_changes_p1, axis=0) 
        se_changes_p1 = np.std(q_changes_p1, axis=0, ddof=1) / np.sqrt(num_experiments)

        x = np.arange(len(mean_changes_p1))

        ax6.plot(mean_changes_p1, label=f'{agent1_type}')
        ax6.fill_between(x, mean_changes_p1 + 1.96 * se_changes_p1, mean_changes_p1 - 1.96 * se_changes_p1, alpha=0.3)

    if len(results["0"]['q_values2']) > 0:
        q_changes_p2 = []
        for idx in range(len(results.keys())):
            q_values2 = np.stack(results[f"{idx}"]['q_values2'])
            q_values2, q_values2_copy = q_values2[1:], q_values2[:-1]
            q_values2_diff = q_values2 - q_values2_copy
            q_change2 = np.max(np.abs(q_values2_diff), axis = tuple(range(1, q_values2_diff.ndim)))
            q_changes_p2.append(q_change2)
      

        q_changes_p2 = np.stack(q_changes_p2)
            
        mean_changes_p2 = np.mean(q_changes_p2, axis=0)
        se_changes_p2 = np.std(q_changes_p2, axis=0, ddof=1) / np.sqrt(num_experiments)

        x = np.arange(len(mean_changes_p2))

        ax6.plot(mean_changes_p2, label=f'{agent2_type}')
        ax6.fill_between(x, mean_changes_p2 + 1.96 * se_changes_p2, mean_changes_p2 - 1.96 * se_changes_p2, alpha=0.3)

    ax6.set_xlabel("Steps")
    ax6.set_ylabel('Q Value Diff')
    ax6.set_title(f'95% Conf. Interval of Q Values Changes Over {len(results.keys())} Runs')
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)

    num_experiments = len(results.keys())

    all_plot_1 = {}
    all_plot_2 = {}

    for exp_idx in range(num_experiments):
        exp = results[str(exp_idx)]
        actions_1, actions_2 = exp['actions1'], exp['actions2']
        assert len(actions_1) == len(actions_2)

        plot_1, plot_2 = {}, {}
        for idx in range(len(actions_1)):
            ep_actions_1, ep_actions_2 = Counter(actions_1[idx]), Counter(actions_2[idx])

            for key, v in plot_1.items():
                v.append(0)
            for key, v in plot_2.items():
                v.append(0)

            for action, count in ep_actions_1.items():
                prob = count / len(actions_1[idx])
                if str(action) not in plot_1:
                    plot_1[str(action)] = [0] * (idx + 1)
                plot_1[str(action)][idx] = prob

            for action, count in ep_actions_2.items():
                prob = count / len(actions_2[idx])
                if str(action) not in plot_2:
                    plot_2[str(action)] = [0] * (idx + 1)
                plot_2[str(action)][idx] = prob

        for action, curve in plot_1.items():
            if action not in all_plot_1:
                all_plot_1[action] = []
            all_plot_1[action].append(curve)

        for action, curve in plot_2.items():
            if action not in all_plot_2:
                all_plot_2[action] = []
            all_plot_2[action].append(curve)

    first_exp = results["0"]
    num_episodes = len(first_exp['actions1'])
    window = num_episodes // 20
    x_labels = range(window - 1, num_episodes)

    if '0' in all_plot_1:
        curves = all_plot_1['0']
        smoothed = np.array([smooth(c, window) for c in curves])
        mean_curve = smoothed.mean(axis=0)
        se = smoothed.std(axis=0) / np.sqrt(num_experiments)
        ax7.plot(x_labels, mean_curve, label=f'{agent1_type} Action 0')
        ax7.fill_between(x_labels, mean_curve - 1.96 * se, mean_curve + 1.96 * se, alpha=0.3)

    if '0' in all_plot_2:
        curves = all_plot_2['0']
        smoothed = np.array([smooth(c, window) for c in curves])
        mean_curve = smoothed.mean(axis=0)
        se = smoothed.std(axis=0) / np.sqrt(num_experiments)
        ax7.plot(x_labels, mean_curve, label=f'{agent2_type} Action 0')
        ax7.fill_between(x_labels, mean_curve - 1.96 * se, mean_curve + 1.96 * se, alpha=0.3)

    ax7.set_title(f"Action 0 Policy Over Time\n{agent1_type} vs {agent2_type}")
    ax7.set_xlabel("Episodes")
    ax7.set_ylabel("Action 0 Probability")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    ax9 = plt.subplot(3, 3, 9)

    smoothed_gap1 = []
    smoothed_gap2 = []

    for idx in range(len(results.keys())):
        gap1 = np.array(results[f"{idx}"]['best_rewards1']).flatten() - np.array(results[f"{idx}"]['raw_rewards1']).flatten()
        gap2 = np.array(results[f"{idx}"]['best_rewards2']).flatten() - np.array(results[f"{idx}"]['raw_rewards2']).flatten()

        smoothed_gap1.append(np.convolve(gap1, np.ones(window)/window, mode='valid'))
        smoothed_gap2.append(np.convolve(gap2, np.ones(window)/window, mode='valid'))

    smoothed_gap1 = np.stack(smoothed_gap1)
    smoothed_gap2 = np.stack(smoothed_gap2)

    mean_gap1, mean_gap2 = np.mean(smoothed_gap1, axis=0), np.mean(smoothed_gap2, axis=0)
    se_gap1 = np.std(smoothed_gap1, axis=0) / np.sqrt(num_experiments)
    se_gap2 = np.std(smoothed_gap2, axis=0) / np.sqrt(num_experiments)

    x = np.arange(len(mean_gap1))

    ax8.plot(mean_gap1, label=f'{agent1_type}', linewidth=2)
    ax8.fill_between(x, mean_gap1 + 1.96*se_gap1, mean_gap1 - 1.96*se_gap1, alpha=0.3)
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Step')
    ax8.set_ylabel('BR Reward - Actual Reward')
    ax8.set_title(f'{agent1_type} Best Reward Discrepancy\n{num_experiments} Runs')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    ax9.plot(mean_gap2, label=f'{agent2_type}', linewidth=2)
    ax9.fill_between(x, mean_gap2 + 1.96*se_gap2, mean_gap2 - 1.96*se_gap2, alpha=0.3)
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Step')
    ax9.set_ylabel('BR Reward - Actual Reward')
    ax9.set_title(f'{agent2_type} Best Reward Discrepancy\n{num_experiments} Runs')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle(f'{game_name}: {agent1_type} vs {agent2_type}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(path / f"{ref_type}.png")

    #plt.show()

def compare_all_results(all_results, game_name, state_history, num_experiments, ref_type, payoff_table):
    """Compare performance across all matchups using the last 50 episodes of each run.

    Assumes:
        all_results[matchup_key]['results']['avg_rewards1'] has shape (num_runs, num_episodes)
        all_results[matchup_key]['results']['avg_rewards2'] has shape (num_runs, num_episodes)

    Returns:
        comparison_data: list[dict]
    """

    print("\n" + "=" * 80)
    print(f"COMPARISON ACROSS ALL MATCHUPS: {game_name}")
    print("=" * 80)

    path = Path(DIR_PATH) / f"game_{game_name}" / f"sh_{state_history}"
    path.mkdir(parents=True, exist_ok=True)

    comparison_data = []
    last_n = 50

    for matchup_key, data in all_results.items():
        run_means1 = []
        run_means2 = []

        for run_id, run_results in data.items():

            rewards1 = np.asarray(run_results.get("avg_rewards1", []), dtype=float)
            rewards2 = np.asarray(run_results.get("avg_rewards2", []), dtype=float)

            if len(rewards1) < last_n or len(rewards2) < last_n:
                continue

            run_means1.append(np.mean(rewards1[-last_n:]))
            run_means2.append(np.mean(rewards2[-last_n:]))

        if not run_means1 or not run_means2:
            continue

        run_means1 = np.asarray(run_means1)
        run_means2 = np.asarray(run_means2)

        final_avg1 = run_means1.mean()
        final_avg2 = run_means2.mean()

        # 95% CI across runs
        if len(run_means1) > 1:
            ci1 = 1.96 * run_means1.std(ddof=1) / np.sqrt(len(run_means1))
        else:
            ci1 = 0.0

        if len(run_means2) > 1:
            ci2 = 1.96 * run_means2.std(ddof=1) / np.sqrt(len(run_means2))
        else:
            ci2 = 0.0

        comparison_data.append({
            "Matchup": matchup_key,
            "Agent1_Avg": final_avg1,
            "Agent1_CI": ci1,
            "Agent2_Avg": final_avg2,
            "Agent2_CI": ci2,
            "Total_Avg": (final_avg1 + final_avg2) / 2,
            "Difference": abs(final_avg1 - final_avg2),
            "Num_Runs": len(run_means1)
        })
    if not comparison_data:
        print("\nNo valid comparison data found.")
        return comparison_data

    # Create comparison table
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Total_Avg", ascending=False)

    print("\nPerformance Comparison (mean of last 50 episodes across runs):")
    print(df.to_string(index=False))

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 12))

    # ------------------------------------------------------------------
    # Bar plot of average rewards with 95% CI
    # ------------------------------------------------------------------
    matchups = df["Matchup"].tolist()
    agent1_avgs = df["Agent1_Avg"].to_numpy()
    agent2_avgs = df["Agent2_Avg"].to_numpy()
    agent1_cis = df["Agent1_CI"].to_numpy()
    agent2_cis = df["Agent2_CI"].to_numpy()

    x = np.arange(len(matchups))
    width = 0.35

    ax1.bar(
        x - width / 2,
        agent1_avgs,
        width,
        yerr=agent1_cis,
        label="Agent 1",
        alpha=0.7,
        capsize=4,
    )
    ax1.bar(
        x + width / 2,
        agent2_avgs,
        width,
        yerr=agent2_cis,
        label="Agent 2",
        alpha=0.7,
        capsize=4,
    )

    ax1.set_xlabel("Matchup")
    ax1.set_ylabel("Average Reward")
    ax1.set_title(f"Average Rewards by Matchup")
    ax1.set_xticks(x)
    ax1.set_xticklabels(matchups, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ######################################
    ######## Policy comparisons ##########
    ######################################

    last_n = 50

    policy_rows = []

    for matchup_key, data in all_results.items():
        run_probs = {
            "P1_A0": [],
            "P1_A1": [],
            "P2_A0": [],
            "P2_A1": [],
        }

        for run_id, exp in data.items():
            actions_1 = exp["actions1"]
            actions_2 = exp["actions2"]

            last_actions_1 = actions_1[-last_n:]
            last_actions_2 = actions_2[-last_n:]

            flat_1 = [a for ep in last_actions_1 for a in ep]
            flat_2 = [a for ep in last_actions_2 for a in ep]

            if len(flat_1) == 0 or len(flat_2) == 0:
                continue

            c1 = Counter(flat_1)
            c2 = Counter(flat_2)

            total1 = sum(c1.values())
            total2 = sum(c2.values())

            run_probs["P1_A0"].append(c1.get(0, 0) / total1)
            run_probs["P1_A1"].append(c1.get(1, 0) / total1)
            run_probs["P2_A0"].append(c2.get(0, 0) / total2)
            run_probs["P2_A1"].append(c2.get(1, 0) / total2)

        if len(run_probs["P1_A0"]) == 0:
            continue

        row = {"Matchup": matchup_key}
        for key, vals in run_probs.items():
            arr = np.asarray(vals, dtype=float)
            row[f"{key}_Mean"] = arr.mean()
            row[f"{key}_CI"] = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

        policy_rows.append(row)

    policy_df = pd.DataFrame(policy_rows)

    matchups = policy_df["Matchup"].tolist()
    x = np.arange(len(matchups))
    width = 0.18

    ax2.bar(
        x - 1.5 * width,
        policy_df["P1_A0_Mean"],
        width,
        yerr=policy_df["P1_A0_CI"],
        capsize=4,
        label="Player 1: Action 0",
        alpha=0.8,
    )
    ax2.bar(
        x - 0.5 * width,
        policy_df["P1_A1_Mean"],
        width,
        yerr=policy_df["P1_A1_CI"],
        capsize=4,
        label="Player 1: Action 1",
        alpha=0.8,
    )
    ax2.bar(
        x + 0.5 * width,
        policy_df["P2_A0_Mean"],
        width,
        yerr=policy_df["P2_A0_CI"],
        capsize=4,
        label="Player 2: Action 0",
        alpha=0.8,
    )
    ax2.bar(
        x + 1.5 * width,
        policy_df["P2_A1_Mean"],
        width,
        yerr=policy_df["P2_A1_CI"],
        capsize=4,
        label="Player 2: Action 1",
        alpha=0.8,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(matchups, rotation=45, ha="right")
    ax2.set_ylabel("Final Action Probability")
    ax2.set_xlabel("Matchup")
    ax2.set_title(f"Converged Policy Distribution by Matchup")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # CPT Transformation Action Change Rate

    action_change_rows = []

    for matchup_key, data in all_results.items():
        run_rates_1 = []
        run_rates_2 = []

        for run_id, exp in data.items():
            flags_1 = np.asarray(exp.get("action_changed_flags1", []), dtype=float)
            flags_2 = np.asarray(exp.get("action_changed_flags2", []), dtype=float)

            if flags_1.size > 0:
                run_rates_1.append(flags_1.mean())

            if flags_2.size > 0:
                run_rates_2.append(flags_2.mean())

        if len(run_rates_1) == 0 and len(run_rates_2) == 0:
            continue

        row = {"Matchup": matchup_key}

        if len(run_rates_1) > 0:
            arr1 = np.asarray(run_rates_1, dtype=float)
            row["P1_FlipRate_Mean"] = arr1.mean()
            row["P1_FlipRate_CI"] = 1.96 * arr1.std(ddof=1) / np.sqrt(len(arr1)) if len(arr1) > 1 else 0.0
        else:
            row["P1_FlipRate_Mean"] = 0.0
            row["P1_FlipRate_CI"] = 0.0

        if len(run_rates_2) > 0:
            arr2 = np.asarray(run_rates_2, dtype=float)
            row["P2_FlipRate_Mean"] = arr2.mean()
            row["P2_FlipRate_CI"] = 1.96 * arr2.std(ddof=1) / np.sqrt(len(arr2)) if len(arr2) > 1 else 0.0
        else:
            row["P2_FlipRate_Mean"] = 0.0
            row["P2_FlipRate_CI"] = 0.0

        action_change_rows.append(row)

    action_change_df = pd.DataFrame(action_change_rows)

    matchups = action_change_df["Matchup"].tolist()
    x = np.arange(len(matchups))
    width = 0.35

    ax3.bar(
        x - width / 2,
        action_change_df["P1_FlipRate_Mean"],
        width,
        yerr=action_change_df["P1_FlipRate_CI"],
        capsize=4,
        label="Player 1",
        alpha=0.8,
    )

    ax3.bar(
        x + width / 2,
        action_change_df["P2_FlipRate_Mean"],
        width,
        yerr=action_change_df["P2_FlipRate_CI"],
        capsize=4,
        label="Player 2",
        alpha=0.8,
    )

    '''
    for i, v in enumerate(action_change_df["P1_FlipRate_Mean"]):
        ax3.text(
        x[i] - width/2,
        v + max(action_change_df["P1_FlipRate_Mean"]) * 0.1,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
        )

    for i, v in enumerate(action_change_df["P2_FlipRate_Mean"]):
        ax3.text(
        x[i] + width/2,
        v + max(action_change_df["P1_FlipRate_Mean"]) * 0.1,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
        )
    '''
    ax3.set_xticks(x)
    ax3.set_xticklabels(matchups, rotation=45, ha="right")
    ax3.set_ylabel("CPT Decision Flip Rate")
    ax3.set_xlabel("Matchup")
    ax3.set_title("CPT Preference Reversal Rate by Matchup")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # CPT Transformation Magnitude

    l2_rows = []

    for matchup_key, data in all_results.items():
        run_l2_1 = []
        run_l2_2 = []

        for run_id, exp in data.items():
            dists_1 = np.asarray(exp.get("pt_l2_dists1", []), dtype=float)
            dists_2 = np.asarray(exp.get("pt_l2_dists2", []), dtype=float)

            if dists_1.size > 0:
                run_l2_1.append(dists_1.mean())

            if dists_2.size > 0:
                run_l2_2.append(dists_2.mean())

        if len(run_l2_1) == 0 and len(run_l2_2) == 0:
            continue

        row = {"Matchup": matchup_key}

        if len(run_l2_1) > 0:
            arr1 = np.asarray(run_l2_1, dtype=float)
            row["P1_L2_Mean"] = arr1.mean()
            row["P1_L2_CI"] = 1.96 * arr1.std(ddof=1) / np.sqrt(len(arr1)) if len(arr1) > 1 else 0.0
        else:
            row["P1_L2_Mean"] = 0.0
            row["P1_L2_CI"] = 0.0

        if len(run_l2_2) > 0:
            arr2 = np.asarray(run_l2_2, dtype=float)
            row["P2_L2_Mean"] = arr2.mean()
            row["P2_L2_CI"] = 1.96 * arr2.std(ddof=1) / np.sqrt(len(arr2)) if len(arr2) > 1 else 0.0
        else:
            row["P2_L2_Mean"] = 0.0
            row["P2_L2_CI"] = 0.0

        l2_rows.append(row)

    l2_df = pd.DataFrame(l2_rows)

    matchups = l2_df["Matchup"].tolist()
    x = np.arange(len(matchups))
    width = 0.35

    ax4.bar(
        x - width / 2,
        l2_df["P1_L2_Mean"],
        width,
        yerr=l2_df["P1_L2_CI"],
        capsize=4,
        label="Player 1",
        alpha=0.8,
    )

    ax4.bar(
        x + width / 2,
        l2_df["P2_L2_Mean"],
        width,
        yerr=l2_df["P2_L2_CI"],
        capsize=4,
        label="Player 2",
        alpha=0.8,
    )

    ax4.set_xticks(x)
    ax4.set_xticklabels(matchups, rotation=45, ha="right")
    ax4.set_ylabel("Mean CPT-EU L2 Distance")
    ax4.set_xlabel("Matchup")
    ax4.set_title("CPT Transformation Magnitude by Matchup")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # Reference point comparison

    ref_rows = []

    for matchup_key, data in all_results.items():
        run_final_refs_1 = []
        run_final_refs_2 = []

        for run_id, exp in data.items():
            refs_1 = np.asarray(exp.get("ref_points1", []), dtype=float)
            refs_2 = np.asarray(exp.get("ref_points2", []), dtype=float)

            if refs_1.size > 0:
                run_final_refs_1.append(refs_1[-1])

            if refs_2.size > 0:
                run_final_refs_2.append(refs_2[-1])

        row = {"Matchup": matchup_key}

        if len(run_final_refs_1) > 0:
            arr1 = np.asarray(run_final_refs_1, dtype=float)
            row["P1_Ref_Mean"] = arr1.mean()
            row["P1_Ref_CI"] = 1.96 * arr1.std(ddof=1) / np.sqrt(len(arr1)) if len(arr1) > 1 else 0.0
        else:
            row["P1_Ref_Mean"] = 0.0
            row["P1_Ref_CI"] = 0.0

        if len(run_final_refs_2) > 0:
            arr2 = np.asarray(run_final_refs_2, dtype=float)
            row["P2_Ref_Mean"] = arr2.mean()
            row["P2_Ref_CI"] = 1.96 * arr2.std(ddof=1) / np.sqrt(len(arr2)) if len(arr2) > 1 else 0.0
        else:
            row["P2_Ref_Mean"] = 0.0
            row["P2_Ref_CI"] = 0.0

        ref_rows.append(row)

    ref_df = pd.DataFrame(ref_rows)

    matchups = ref_df["Matchup"].tolist()
    x = np.arange(len(matchups))
    width = 0.35

    ax5.bar(
        x - width / 2,
        ref_df["P1_Ref_Mean"],
        width,
        yerr=ref_df["P1_Ref_CI"],
        capsize=4,
        label="Player 1",
        alpha=0.8,
    )

    ax5.bar(
        x + width / 2,
        ref_df["P2_Ref_Mean"],
        width,
        yerr=ref_df["P2_Ref_CI"],
        capsize=4,
        label="Player 2",
        alpha=0.8,
    )

    '''
    for i, v in enumerate(ref_df["P1_Ref_Mean"]):
        ax5.text(
            x[i] - width / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9
        )

    for i, v in enumerate(ref_df["P2_Ref_Mean"]):
        ax5.text(
            x[i] + width / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9
        )
    '''

    ax5.set_xticks(x)
    ax5.set_xticklabels(matchups, rotation=45, ha="right")
    ax5.set_ylabel("Final Reference Point")
    ax5.set_xlabel("Matchup")
    ax5.set_title("Final Reference Point by Matchup")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    last_n = 50
    exploit_data = []

    # 2x2x2 payoff array:
    # payoffs[a1, a2, 0] = player 1 reward
    # payoffs[a1, a2, 1] = player 2 reward
    payoffs = payoff_table[game_name]['payoffs']

    for matchup_key, data in all_results.items():
        run_exploit1 = []
        run_exploit2 = []

        for run_id, exp in data.items():
            actions_1 = exp["actions1"]
            actions_2 = exp["actions2"]

            last_actions_1 = actions_1[-last_n:]
            last_actions_2 = actions_2[-last_n:]

            flat_1 = [a for ep in last_actions_1 for a in ep]
            flat_2 = [a for ep in last_actions_2 for a in ep]

            print(len(flat_1), len(flat_2))

            if len(flat_1) == 0 or len(flat_2) == 0:
                continue

            c1 = Counter(flat_1)
            c2 = Counter(flat_2)

            total1 = sum(c1.values())
            total2 = sum(c2.values())

            # Empirical mixed strategies
            p1 = np.array([
                c1.get(0, 0) / total1,
                c1.get(1, 0) / total1
            ], dtype=float)

            p2 = np.array([
                c2.get(0, 0) / total2,
                c2.get(1, 0) / total2
            ], dtype=float)

            # Actual expected rewards under empirical mixed play
            actual_p1 = 0.0
            actual_p2 = 0.0
            for a1 in range(2):
                for a2 in range(2):
                    prob = p1[a1] * p2[a2]
                    actual_p1 += prob * payoffs[a1, a2, 0]
                    actual_p2 += prob * payoffs[a1, a2, 1]

            # Best response value for player 1 against p2
            br_vals_p1 = []
            for a1 in range(2):
                val = 0.0
                for a2 in range(2):
                    val += p2[a2] * payoffs[a1, a2, 0]
                br_vals_p1.append(val)
            br_p1 = max(br_vals_p1)

            # Best response value for player 2 against p1
            br_vals_p2 = []
            for a2 in range(2):
                val = 0.0
                for a1 in range(2):
                    val += p1[a1] * payoffs[a1, a2, 1]
                br_vals_p2.append(val)
            br_p2 = max(br_vals_p2)

            exploit1 = br_p1 - actual_p1
            exploit2 = br_p2 - actual_p2

            run_exploit1.append(exploit1)
            run_exploit2.append(exploit2)

        if len(run_exploit1) == 0 or len(run_exploit2) == 0:
            continue

        run_exploit1 = np.asarray(run_exploit1, dtype=float)
        run_exploit2 = np.asarray(run_exploit2, dtype=float)

        ci1 = 1.96 * run_exploit1.std(ddof=1) / np.sqrt(len(run_exploit1)) if len(run_exploit1) > 1 else 0.0
        ci2 = 1.96 * run_exploit2.std(ddof=1) / np.sqrt(len(run_exploit2)) if len(run_exploit2) > 1 else 0.0

        exploit_data.append({
            "Matchup": matchup_key,
            "Agent1_Exploit": run_exploit1.mean(),
            "Agent1_CI": ci1,
            "Agent2_Exploit": run_exploit2.mean(),
            "Agent2_CI": ci2,
        })

    exploit_df = pd.DataFrame(exploit_data)
    exploit_df = exploit_df.set_index("Matchup").loc[df["Matchup"]].reset_index()

    x = np.arange(len(exploit_df["Matchup"]))

    ax6.bar(
        x - width/2,
        exploit_df["Agent1_Exploit"],
        width,
        yerr=exploit_df["Agent1_CI"],
        label="Agent 1",
        alpha=0.7,
        capsize=4
    )

    ax6.bar(
        x + width/2,
        exploit_df["Agent2_Exploit"],
        width,
        yerr=exploit_df["Agent2_CI"],
        label="Agent 2",
        alpha=0.7,
        capsize=4
    )

    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel("Matchup")
    ax6.set_ylabel("Best-Response Value - Policy Value")
    ax6.set_title("Missed Exploitability by Matchup")
    ax6.set_xticks(x)
    ax6.set_xticklabels(exploit_df["Matchup"], rotation=45, ha="right")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")


    exploit_data = []

    for matchup_key, data in all_results.items():
        run_gaps1 = []
        run_gaps2 = []

        for run_id, run_results in data.items():
            best1 = np.asarray(run_results.get("best_rewards1", []), dtype=float).flatten()
            raw1 = np.asarray(run_results.get("raw_rewards1", []), dtype=float).flatten()
            best2 = np.asarray(run_results.get("best_rewards2", []), dtype=float).flatten()
            raw2 = np.asarray(run_results.get("raw_rewards2", []), dtype=float).flatten()

            if len(best1) < last_n or len(best2) < last_n:
                continue

            run_gaps1.append(np.mean((best1 - raw1)[-last_n:]))
            run_gaps2.append(np.mean((best2 - raw2)[-last_n:]))

        if not run_gaps1 or not run_gaps2:
            continue

        run_gaps1 = np.asarray(run_gaps1)
        run_gaps2 = np.asarray(run_gaps2)

        ci1 = 1.96 * run_gaps1.std(ddof=1) / np.sqrt(len(run_gaps1)) if len(run_gaps1) > 1 else 0.0
        ci2 = 1.96 * run_gaps2.std(ddof=1) / np.sqrt(len(run_gaps2)) if len(run_gaps2) > 1 else 0.0

        exploit_data.append({
            "Matchup": matchup_key,
            "Agent1_Gap": run_gaps1.mean(),
            "Agent1_CI": ci1,
            "Agent2_Gap": run_gaps2.mean(),
            "Agent2_CI": ci2,
        })

    exploit_df = pd.DataFrame(exploit_data)
    exploit_df = exploit_df.set_index("Matchup").loc[df["Matchup"]].reset_index()

    x = np.arange(len(exploit_df["Matchup"]))

    ax7.bar(x - width/2, exploit_df["Agent1_Gap"], width, yerr=exploit_df["Agent1_CI"],
            label="Agent 1", alpha=0.7, capsize=4)
    ax7.bar(x + width/2, exploit_df["Agent2_Gap"], width, yerr=exploit_df["Agent2_CI"],
            label="Agent 2", alpha=0.7, capsize=4)
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax7.set_xlabel("Matchup")
    ax7.set_ylabel("Best Response Reward - Actual Reward")
    ax7.set_title("Exploitability Gap by Matchup")
    ax7.set_xticks(x)
    ax7.set_xticklabels(exploit_df["Matchup"], rotation=45, ha="right")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")

    ax8.axis('off')


    fig.suptitle(f"{game_name} — Learning Results Across Matchups - Last {last_n} Episodes", fontsize=16)

    fig.subplots_adjust(
    top=0.90,
    bottom=0.15,
    hspace=0.45,
    wspace=0.30
    )   

    #plt.show()

    plt.savefig(path / f"{game_name}_{ref_type}_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    all_results_slice = {}

    for key, value in all_results.items():
        all_results_slice[key] = {}

        for k, v in value.items():
            all_results_slice[key][k] = {
            'state': v['states'],
            'actions1': v['actions1'],
            'actions2': v['actions2'],
            }
    all_results_slice = convert(all_results_slice)
   

    with gzip.open(path / f"{game_name}_{ref_type}_StateVisits.json.gz", "wt", encoding="utf-8") as f:
        json.dump(all_results_slice, f)

    return comparison_data
