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

DIR_PATH = "/Users/dylanwaldner/Projects/RLNash/Experiments"

def analyze_matchup_da(results, agent1_type, agent2_type, game_name, payoff_matrix, pt_params, ref_type, env):
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

    for idx in range(len(results.keys())):
        ref_points1 = results[f"{idx}"]['ref_points1']
        ref_points2 = results[f"{idx}"]['ref_points2']

        ref_points_p1.append(ref_points1)
        ref_points_p2.append(ref_points2)

    ref_points_p1 = np.stack(ref_points_p1)
    ref_points_p2 = np.stack(ref_points_p2)

    mean_p1, mean_p2 = np.mean(ref_points_p1, axis=0), np.mean(ref_points_p2, axis=0)
    std_p1, std_p2 = np.std(ref_points_p1, axis=0), np.std(ref_points_p2, axis=0)
    se_p1, se_p2 = std_p1 / np.sqrt(num_experiments), std_p2 / np.sqrt(num_experiments)
    
    ax2.plot(mean_p1, label=f'{agent1_type}')
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

        # mean over runs
        mean_q_p1 = np.mean(q_values_p1, axis=0)  # shape: [time, a, a_-i]

        # collapse time (use final step OR average over time)
        final_q = mean_q_p1[-1]   # shape: [a, a_-i]
        # alternatively:
        # final_q = np.mean(mean_q_p1, axis=0)

        if agent1_type == "LH":
            final_q = final_q.T
            im = ax4.imshow(final_q, aspect='auto')

            # annotate cells
            for i in range(final_q.shape[0]):
                for j in range(final_q.shape[1]):
                    ax4.text(j, i, f"{final_q[i, j]:.2f}",
                             ha='center', va='center', color='white')

            ax4.set_xlabel("LH Action")
            ax4.set_ylabel("Opponent Action")

            ax4.set_title(f"{agent1_type} Q-Value Heatmap (Final Step)")

            ax4.set_xticks(np.arange(5))
            ax4.set_yticks(np.arange(5))

            ax4.set_xticklabels(np.arange(1, 6))
            ax4.set_yticklabels(np.arange(1, 6))

            plt.colorbar(im, ax=ax4)

        else:
            final_q = final_q.reshape(1, -1)

            im = ax4.imshow(final_q, aspect='auto')

            for j in range(final_q.shape[1]):
                ax4.text(j, 0, f"{final_q[0, j]:.2f}",
                         ha='center', va='center', color='white')

            ax4.set_yticks([])
            ax4.set_xlabel("Action")
            ax4.set_title(f"{agent1_type} Q-Value Heatmap (Final Step)")

            ax4.set_xticks(np.arange(5))
            ax4.set_xticklabels(np.arange(1, 6))

            plt.colorbar(im, ax=ax4)

    ax4.invert_yaxis()

    # 5. Q value heatmap - Player 2
    ax5 = plt.subplot(3, 3, 5)

    if len(results["0"]['q_values2']) > 0:
        q_values_p2 = []
        for idx in range(len(results.keys())):
            q_values2 = np.stack(results[f"{idx}"]['q_values2'])
            q_values_p2.append(q_values2)

        q_values_p2 = np.stack(q_values_p2)

        # mean over runs
        mean_q_p2 = np.mean(q_values_p2, axis=0)  # shape: [time, ...]

        # collapse time (use final step OR average over time)
        final_q = mean_q_p2[-1]

        if agent2_type == "LH":
            im = ax5.imshow(final_q, aspect='auto')

            # annotate cells
            for i in range(final_q.shape[0]):
                for j in range(final_q.shape[1]):
                    ax5.text(j, i, f"{final_q[i, j]:.2f}",
                             ha='center', va='center', color='white')

            ax5.set_xlabel("Opponent Action (a_-i)")
            ax5.set_ylabel("Agent Action (a)")
            ax5.set_title(f"{agent2_type} Q-Value Heatmap (Final Step)")

            ax5.set_xticks(np.arange(5))
            ax5.set_yticks(np.arange(5))

            ax5.set_xticklabels(np.arange(1, 6))
            ax5.set_yticklabels(np.arange(1, 6))

            plt.colorbar(im, ax=ax5)

        else:
            final_q = final_q.reshape(1, -1)

            im = ax5.imshow(final_q, aspect='auto')

            for j in range(final_q.shape[1]):
                ax5.text(j, 0, f"{final_q[0, j]:.2f}",
                         ha='center', va='center', color='white')

            ax5.set_xticks(np.arange(5))
            ax5.set_xticklabels(np.arange(1, 6))

            ax5.set_yticks([])
            ax5.set_xlabel("Action")
            ax5.set_title(f"{agent2_type} Q-Value Heatmap (Final Step)")

            plt.colorbar(im, ax=ax5)

    ax5.invert_yaxis()

    # 6. Learning convergence (how much are q values changing)
    ax6 = plt.subplot(3, 3, 6)

    if len(results["0"]['q_values1']) > 0:
        q_changes_p1 = []
        for idx in range(len(results.keys())):
            q_values1 = np.stack(results[f"{idx}"]['q_values1'])
            q_values1, q_values1_copy = q_values1[1:], q_values1[:-1]
            q_values1_diff = q_values1 - q_values1_copy
            q_change1 = np.mean(np.abs(q_values1_diff), axis = tuple(range(1, q_values1_diff.ndim)))

            q_changes_p1.append(q_change1)
                
        q_changes_p1 = np.stack(q_changes_p1)
        
        mean_changes_p1 = np.mean(q_changes_p1, axis=0) 
        se_changes_p1 = np.std(q_changes_p1, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_changes_p1))

        ax6.plot(mean_changes_p1, label=f'{agent1_type}')
        ax6.fill_between(x, mean_changes_p1 + 1.96 * se_changes_p1, mean_changes_p1 - 1.96 * se_changes_p1, alpha=0.3)

    if len(results["0"]['q_values2']) > 0:
        q_changes_p2 = []
        for idx in range(len(results.keys())):
            q_values2 = np.stack(results[f"{idx}"]['q_values2'])
            q_values2, q_values2_copy = q_values2[1:], q_values2[:-1]
            q_values2_diff = q_values2 - q_values2_copy
            q_change2 = np.mean(np.abs(q_values2_diff), axis = tuple(range(1, q_values2_diff.ndim)))
            q_changes_p2.append(q_change2)
      

        q_changes_p2 = np.stack(q_changes_p2)
            
        mean_changes_p2 = np.mean(q_changes_p2, axis=0)
        se_changes_p2 = np.std(q_changes_p2, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_changes_p2))

        ax6.plot(mean_changes_p2, label=f'{agent2_type}')
        ax6.fill_between(x, mean_changes_p2 + 1.96 * se_changes_p2, mean_changes_p2 - 1.96 * se_changes_p2, alpha=0.3)

    ax6.set_xlabel("Steps")
    ax6.set_ylabel('Q Value Diff')
    ax6.set_title(f'95% Conf. Interval of Q Values Changes Over {len(results.keys())} Runs')
    #ax6.xaxis.set_major_formatter(
    #    FuncFormatter(lambda x, pos: f"{int(x*k)}")
    #)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)

    num_experiments = len(results.keys())

    bid_runs = []
    ask_runs = []
    realized_price_runs = []
    trade_rate_runs = []

    for exp_idx in range(num_experiments):
        exp = results[str(exp_idx)]

        bids_by_episode = exp['actions1']
        asks_by_episode = exp['actions2']

        num_episodes = min(len(bids_by_episode), len(asks_by_episode))

        episode_bid_means = []
        episode_ask_means = []
        episode_realized_means = []
        episode_trade_rates = []

        for ep_idx in range(num_episodes):
            ep_bids = np.array(bids_by_episode[ep_idx], dtype=float)
            ep_asks = np.array(asks_by_episode[ep_idx], dtype=float)

            num_steps = min(len(ep_bids), len(ep_asks))

            # mean bid / ask
            episode_bid_means.append(np.mean(ep_bids) if len(ep_bids) > 0 else np.nan)
            episode_ask_means.append(np.mean(ep_asks) if len(ep_asks) > 0 else np.nan)

            if num_steps > 0:
                bids = ep_bids[:num_steps]
                asks = ep_asks[:num_steps]

                trade_mask = bids >= asks

                # trade rate
                episode_trade_rates.append(np.mean(trade_mask))

                # realized price
                if np.any(trade_mask):
                    realized_prices = (bids[trade_mask] + asks[trade_mask]) / 2.0
                    episode_realized_means.append(np.mean(realized_prices))
                else:
                    episode_realized_means.append(np.nan)
            else:
                episode_trade_rates.append(np.nan)
                episode_realized_means.append(np.nan)

        bid_runs.append(episode_bid_means)
        ask_runs.append(episode_ask_means)
        realized_price_runs.append(episode_realized_means)
        trade_rate_runs.append(episode_trade_rates)

    bid_runs = np.array(bid_runs, dtype=float)
    ask_runs = np.array(ask_runs, dtype=float)
    realized_price_runs = np.array(realized_price_runs, dtype=float)
    trade_rate_runs = np.array(trade_rate_runs, dtype=float)

    window = max(1, bid_runs.shape[1] // 20)

    def smooth_nan(x, w):
        return np.array([
            np.nanmean(x[i - w + 1:i + 1]) if not np.all(np.isnan(x[i - w + 1:i + 1])) else np.nan
            for i in range(w - 1, len(x))
        ])

    smoothed_bids = np.array([smooth_nan(run, window) for run in bid_runs])
    smoothed_asks = np.array([smooth_nan(run, window) for run in ask_runs])
    smoothed_prices = np.array([smooth_nan(run, window) for run in realized_price_runs])
    smoothed_trade = np.array([smooth_nan(run, window) for run in trade_rate_runs])

    # means
    mean_bids = np.nanmean(smoothed_bids, axis=0)
    mean_asks = np.nanmean(smoothed_asks, axis=0)
    mean_prices = np.nanmean(smoothed_prices, axis=0)
    mean_trade = np.nanmean(smoothed_trade, axis=0)

    # standard errors
    se_bids = np.nanstd(smoothed_bids, axis=0) / np.sqrt(num_experiments)
    se_asks = np.nanstd(smoothed_asks, axis=0) / np.sqrt(num_experiments)
    se_prices = np.nanstd(smoothed_prices, axis=0) / np.sqrt(num_experiments)
    se_trade = np.nanstd(smoothed_trade, axis=0) / np.sqrt(num_experiments)

    x = np.arange(len(mean_bids))

    # --- PRICE AXIS ---
    ax7.plot(x, mean_bids, linewidth=2, label='Mean Bid')
    ax7.fill_between(x, mean_bids + 1.96 * se_bids, mean_bids - 1.96 * se_bids, alpha=0.25)

    ax7.plot(x, mean_asks, linewidth=2, label='Mean Ask')
    ax7.fill_between(x, mean_asks + 1.96 * se_asks, mean_asks - 1.96 * se_asks, alpha=0.25)

    ax7.plot(x, mean_prices, linewidth=2, label='Mean Price')
    ax7.fill_between(x, mean_prices + 1.96 * se_prices, mean_prices - 1.96 * se_prices, alpha=0.25)

    ax7.set_title("Bids, Asks, Realized Price, and Trade Rate Over Time")
    ax7.set_xlabel("Episodes")
    ax7.set_ylabel("Price")
    ax7.grid(True, alpha=0.3)

    # --- TRADE RATE AXIS ---
    ax7b = ax7.twinx()

    ax7b.plot(
        x,
        mean_trade,
        linestyle='--',
        linewidth=1.5,
        alpha=0.6,
        label='Trade Rate'
    )
    ax7b.fill_between(
        x,
        mean_trade + 1.96 * se_trade,
        mean_trade - 1.96 * se_trade,
        alpha=0.08  # ↓ lower opacity (less visual dominance)
    )

    ax7b.set_ylabel("Trade Rate")
    ax7b.set_ylim(0, 1)

    # --- COMBINED LEGEND ---
    lines, labels = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7b.get_legend_handles_labels()

    ax7.legend(
        lines + lines2,
        labels + labels2,
        loc='upper left',
        fontsize=8,
        framealpha=0.6,
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3
    )

    # Exploitation Graphs
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
    ax8.set_ylabel('Best Response Reward - Actual Reward')
    ax8.set_title(f'{agent1_type} Exploitability Gap\n{num_experiments} Runs')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    ax9.plot(mean_gap2, label=f'{agent2_type}', linewidth=2)
    ax9.fill_between(x, mean_gap2 + 1.96*se_gap2, mean_gap2 - 1.96*se_gap2, alpha=0.3)
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Best Response Reward - Actual Reward')
    ax9.set_title(f'{agent2_type} Exploitability Gap\n{num_experiments} Runs')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle(f'{game_name}: {agent1_type} vs {agent2_type}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(path / f"{ref_type}.png")

    #plt.show()

def compare_all_da_results(all_results, game_name, state_history, num_experiments, ref_type, payoff_table):
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

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 4,
    )


    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:3])   # spans two columns
    ax3 = fig.add_subplot(gs[0, 3])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])

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
    max_actions = 0

    for matchup_key, data in all_results.items():
        p1_runs = []
        p2_runs = []

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

            n1 = max(c1.keys()) + 1
            n2 = max(c2.keys()) + 1
            max_actions = max(max_actions, n1, n2)

            probs1 = np.array([c1.get(a, 0) / total1 for a in range(n1)])
            probs2 = np.array([c2.get(a, 0) / total2 for a in range(n2)])

            p1_runs.append(probs1)
            p2_runs.append(probs2)

        if len(p1_runs) == 0:
            continue

        policy_rows.append({
            "matchup": matchup_key,
            "p1_runs": p1_runs,
            "p2_runs": p2_runs
        })

    matchups = [row["matchup"] for row in policy_rows]
    num_matchups = len(matchups)

    # --- build combined matrix ---
    combined = np.zeros((max_actions, num_matchups * 2))

    for j, row in enumerate(policy_rows):
        padded_p1 = []
        for arr in row["p1_runs"]:
            tmp = np.zeros(max_actions)
            tmp[:len(arr)] = arr
            padded_p1.append(tmp)
        mean_p1 = np.mean(padded_p1, axis=0)

        padded_p2 = []
        for arr in row["p2_runs"]:
            tmp = np.zeros(max_actions)
            tmp[:len(arr)] = arr
            padded_p2.append(tmp)
        mean_p2 = np.mean(padded_p2, axis=0)

        combined[:, 2*j] = mean_p1   # buyer
        combined[:, 2*j + 1] = mean_p2  # seller

    

    im = ax2.imshow(
        combined,
        aspect='auto',
        origin='lower',
        interpolation='nearest'
    )

    # --- x labels ---
    xticks = []
    xticklabels = []

    for j, m in enumerate(matchups):
        left, right = m.split("_vs_")

        xticks.extend([2*j, 2*j + 1])
        xticklabels.extend([
            f"{left} (B)",
            f"{right} (S)"
        ])

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, rotation=45, ha="right")

    # --- y labels ---
    ax2.set_yticks(np.arange(max_actions))
    ax2.set_yticklabels(np.arange(1, max_actions + 1))

    ax2.set_title("Final Policies (Buyer vs Seller)")
    ax2.set_xlabel("Matchup / Agent")
    ax2.set_ylabel("Action")

    # --- vertical separators between matchups ---
    for j in range(num_matchups - 1):
        ax2.axvline(2*j + 1.5, color='white', linewidth=2)

    # --- annotate (optional) ---
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            val = combined[i, j]
            ax2.text(
                j,
                i,
                f"{val:.2f}",
                ha='center',
                va='center',
                fontsize=7,
                color='white' if val > 0.5 else 'black'
            )

    ax2.set_anchor('C')
    ax2.margins(0)
    # tighten image bounds
    ax2.set_xlim(-0.5, combined.shape[1] - 0.5)
    ax2.set_ylim(-0.5, combined.shape[0] - 0.5)

    # now attach colorbar relative to final axis
    # cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
    #cbar.set_label("Probability")

    # shrink axis width
    box = ax2.get_position()
    ax2.set_position([
        box.x0,
        box.y0,
        box.width * 0.5,   # slightly stronger shrink
        box.height
    ])

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

    # payoffs[a1, a2, 0] = buyer / agent 1 reward
    # payoffs[a1, a2, 1] = seller / agent 2 reward
    payoffs = payoff_table
    num_actions = payoffs.shape[0]

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

            if len(flat_1) == 0 or len(flat_2) == 0:
                continue

            c1 = Counter(flat_1)
            c2 = Counter(flat_2)

            total1 = sum(c1.values())
            total2 = sum(c2.values())

            # empirical mixed strategies over n actions
            p1 = np.array([c1.get(a, 0) / total1 for a in range(num_actions)], dtype=float)
            p2 = np.array([c2.get(a, 0) / total2 for a in range(num_actions)], dtype=float)

            # actual expected rewards under empirical mixed play
            # actual_p1 = sum_{a1,a2} p1[a1] p2[a2] payoffs[a1,a2,0]
            # actual_p2 = sum_{a1,a2} p1[a1] p2[a2] payoffs[a1,a2,1]
            actual_p1 = p1 @ payoffs[:, :, 0] @ p2
            actual_p2 = p1 @ payoffs[:, :, 1] @ p2

            # best-response value for player 1 against p2
            # for each buyer action a1:
            # br_val[a1] = sum_{a2} p2[a2] payoffs[a1,a2,0]
            br_vals_p1 = payoffs[:, :, 0] @ p2
            br_p1 = np.max(br_vals_p1)

            # best-response value for player 2 against p1
            # for each seller action a2:
            # br_val[a2] = sum_{a1} p1[a1] payoffs[a1,a2,1]
            br_vals_p2 = p1 @ payoffs[:, :, 1]
            br_p2 = np.max(br_vals_p2)

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
        x - width / 2,
        exploit_df["Agent1_Exploit"],
        width,
        yerr=exploit_df["Agent1_CI"],
        label="Agent 1",
        alpha=0.7,
        capsize=4
    )

    ax6.bar(
        x + width / 2,
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


    fig.suptitle(f"{game_name} — Learning Results Across Matchups - Last {last_n} Episodes", fontsize=16)

    fig.subplots_adjust(
    top=0.90,
    bottom=0.15,
    hspace=0.45,
    wspace=0.30
    )   

    # shrink axis width
    box = ax2.get_position()
    ax2.set_position([
        box.x0,
        box.y0,
        box.width * 0.85,   # slightly stronger shrink
        box.height
    ])

    # create colorbar axis based on final heatmap position
    box = ax2.get_position()
    cax = fig.add_axes([box.x1 + 0.01, box.y0, 0.012, box.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Probability")

    #plt.show()

    plt.savefig(path / f"{game_name}_{ref_type}_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    #all_results = convert(all_results)
   

    #with open(path / f"{game_name}_{ref_type}_Comparison.json", "w") as f:
    #    json.dump(all_results, f)

    return comparison_data
