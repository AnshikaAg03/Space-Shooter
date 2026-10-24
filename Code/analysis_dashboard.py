import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def analyze_training_session(session_dir):
    """
    Create comprehensive analysis of a training session.
    
    Parameters:
    -----------
    session_dir : str
        Path to the training session directory
    """
    print(f"Analyzing training session: {os.path.basename(session_dir)}")
    
    # Find all metric files
    data_dir = os.path.join(session_dir, "data")
    if not os.path.exists(data_dir):
        print(f"No data directory found in {session_dir}")
        return
    
    # Load metrics from all snapshots
    metrics_files = sorted(glob.glob(os.path.join(data_dir, "metrics_episode_*.json")))
    if not metrics_files:
        print("No metrics files found")
        return
    
    # Load the latest metrics file
    with open(metrics_files[-1], 'r') as f:
        metrics = json.load(f)
    
    # Create output directory for analysis
    analysis_dir = os.path.join(session_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate comprehensive analysis
    print("Generating performance analysis...")
    performance_analysis(metrics, analysis_dir)
    
    print("Generating agent analysis...")
    agent_analysis(metrics, analysis_dir)
    
    print("Generating environment analysis...")
    environment_analysis(metrics, analysis_dir)
    
    print("Generating genetic evolution analysis...")
    genetic_analysis(metrics, analysis_dir)
    
    print("Generating comprehensive dashboard...")
    create_comprehensive_dashboard(metrics, analysis_dir)
    
    print(f"Analysis complete. Results saved to {analysis_dir}")

def performance_analysis(metrics, output_dir):
    """Generate performance analysis plots and statistics"""
    plt.figure(figsize=(15, 12))
    
    # Episode scores
    plt.subplot(2, 2, 1)
    episodes = range(1, len(metrics["episode_scores"]) + 1)
    plt.plot(episodes, metrics["episode_scores"], 'b-')
    
    # Add trend line
    if len(episodes) > 10:
        z = np.polyfit(episodes, metrics["episode_scores"], 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), 'r--', label=f'Trend: {z[0]:.4f}x + {z[1]:.1f}')
        
    plt.title('Score Progression')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add moving average
    window = min(20, len(episodes) // 5) if len(episodes) > 20 else 5
    if len(episodes) > window:
        moving_avg = np.convolve(metrics["episode_scores"], np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episodes)), moving_avg, 'g-', label=f'Moving Avg (n={window})')
    
    # Max wave progression
    plt.subplot(2, 2, 2)
    plt.plot(episodes, metrics["max_waves"], 'g-')
    
    # Show wave level distribution
    wave_counts = {}
    for wave in metrics["max_waves"]:
        if wave in wave_counts:
            wave_counts[wave] += 1
        else:
            wave_counts[wave] = 1
    
    # Plot wave counts as horizontal bars
    ax2 = plt.twinx()
    waves = sorted(wave_counts.keys())
    counts = [wave_counts[w] for w in waves]
    ax2.barh(waves, [c/len(episodes) for c in counts], alpha=0.3, color='gray')
    ax2.set_ylabel('Frequency')
    
    plt.title('Max Wave Progression')
    plt.xlabel('Episode')
    plt.ylabel('Wave')
    plt.grid(True, alpha=0.3)
    
    # Episode duration analysis
    plt.subplot(2, 2, 3)
    plt.plot(episodes, metrics["episode_durations"], 'm-')
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Seconds')
    plt.grid(True, alpha=0.3)
    
    # Duration distribution
    # Create bins for histogram
    if metrics["episode_durations"]:
        bins = np.linspace(min(metrics["episode_durations"]), 
                         max(metrics["episode_durations"]), 20)
        plt.hist(metrics["episode_durations"], bins=bins, alpha=0.5, orientation='horizontal')
    
    # Reward analysis
    plt.subplot(2, 2, 4)
    plt.plot(episodes, metrics["total_rewards"], 'r-')
    
    # Add trend line for rewards
    if len(episodes) > 10:
        z = np.polyfit(episodes, metrics["total_rewards"], 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), 'b--', label=f'Trend: {z[0]:.4f}x + {z[1]:.1f}')
    
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_analysis.png"), dpi=150)
    plt.close()
    
    # Calculate statistics for summary
    stats = {
        "episodes": len(metrics["episode_scores"]),
        "max_score": max(metrics["episode_scores"]),
        "avg_score": np.mean(metrics["episode_scores"]),
        "max_wave": max(metrics["max_waves"]),
        "avg_duration": np.mean(metrics["episode_durations"]),
        "max_reward": max(metrics["total_rewards"]),
        "avg_reward": np.mean(metrics["total_rewards"])
    }
    
    # Save statistics
    with open(os.path.join(output_dir, "performance_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def agent_analysis(metrics, output_dir):
    """Generate agent-specific analysis plots and statistics"""
    plt.figure(figsize=(15, 12))
    
    # Epsilon decay
    plt.subplot(2, 2, 1)
    if metrics["epsilon_values"]:
        episodes = range(1, len(metrics["epsilon_values"]) + 1)
        plt.plot(episodes, metrics["epsilon_values"], 'b-')
        plt.title('Exploration Rate (Epsilon) Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        
        # Calculate half-life of epsilon
        if metrics["epsilon_values"][0] > metrics["epsilon_values"][-1]:
            half_val = metrics["epsilon_values"][0] / 2
            for i, eps in enumerate(metrics["epsilon_values"]):
                if eps <= half_val:
                    plt.axvline(x=i, color='r', linestyle='--', 
                              label=f'Half-life: Episode {i}')
                    break
            plt.legend()
    
    # Learning loss curve
    plt.subplot(2, 2, 2)
    if metrics["loss_values"]:
        steps = range(1, len(metrics["loss_values"]) + 1)
        plt.plot(steps, metrics["loss_values"], 'r-', alpha=0.5)
        
        # Add smoothed curve
        window = min(100, len(steps) // 10) if len(steps) > 100 else 10
        if len(steps) > window:
            smoothed = np.convolve(metrics["loss_values"], np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(steps)), smoothed, 'r-', label=f'Smoothed (n={window})')
        
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Use log scale if range is large
        if max(metrics["loss_values"]) / (min(metrics["loss_values"]) + 1e-10) > 100:
            plt.yscale('log')
    
    # Q-value progression
    plt.subplot(2, 2, 3)
    if metrics["q_values"]:
        steps = range(1, len(metrics["q_values"]) + 1)
        plt.plot(steps, metrics["q_values"], 'g-', alpha=0.5)
        
        # Add smoothed curve
        window = min(100, len(steps) // 10) if len(steps) > 100 else 10
        if len(steps) > window:
            smoothed = np.convolve(metrics["q_values"], np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(steps)), smoothed, 'g-', label=f'Smoothed (n={window})')
        
        plt.title('Q-Value Progression')
        plt.xlabel('Training Step')
        plt.ylabel('Average Q-Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Action distribution
    plt.subplot(2, 2, 4)
    if metrics["action_distribution"]:
        # Convert string keys back to integers
        action_dist = {int(k): v for k, v in metrics["action_distribution"].items()}
        actions = sorted(action_dist.keys())
        counts = [action_dist[a] for a in actions]
        
        # Create readable labels
        action_labels = [
            "Up", "UpRight", "Right", "DownRight", 
            "Down", "DownLeft", "Left", "UpLeft",
            "Shoot", "RotateLeft", "RotateRight", 
            "Weapon1", "Weapon2", "Weapon3"
        ]
        
        labels = [action_labels[a] if a < len(action_labels) else f"Action {a}" for a in actions]
        
        # Plot as horizontal bars for better readability
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, counts, align='center')
        plt.yticks(y_pos, labels)
        plt.xlabel('Count')
        plt.title('Action Distribution')
        
        # Add percentage labels
        total = sum(counts)
        for i, count in enumerate(counts):
            plt.text(count + (max(counts) * 0.01), i, f"{count/total*100:.1f}%", 
                   va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agent_analysis.png"), dpi=150)
    plt.close()
    
    # Plot state visitation heatmap separately
    if "state_visits_heatmap" in metrics and isinstance(metrics["state_visits_heatmap"], list):
        plt.figure(figsize=(10, 8))
        
        # Convert nested list to numpy array
        heatmap_data = np.array(metrics["state_visits_heatmap"])
        
        # Add small constant to avoid log(0)
        heatmap_data = heatmap_data + 1e-6
        
        # Plot log-scaled heatmap
        plt.imshow(np.log(heatmap_data), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Log(Visit Count)')
        plt.title('Agent State Visitation Heatmap')
        plt.xlabel('X Position (Discretized)')
        plt.ylabel('Y Position (Discretized)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "state_heatmap.png"), dpi=150)
        plt.close()
    
    # Calculate statistics
    agent_stats = {}
    if metrics["epsilon_values"]:
        agent_stats["initial_epsilon"] = metrics["epsilon_values"][0]
        agent_stats["final_epsilon"] = metrics["epsilon_values"][-1]
    
    if metrics["loss_values"]:
        agent_stats["initial_loss"] = metrics["loss_values"][0]
        agent_stats["final_loss"] = metrics["loss_values"][-1]
        agent_stats["avg_loss"] = np.mean(metrics["loss_values"])
    
    if metrics["q_values"]:
        agent_stats["initial_q"] = metrics["q_values"][0]
        agent_stats["final_q"] = metrics["q_values"][-1]
        agent_stats["avg_q"] = np.mean(metrics["q_values"])
    
    if metrics["action_distribution"]:
        agent_stats["most_common_action"] = max(action_dist.items(), key=lambda x: x[1])[0]
        agent_stats["action_entropy"] = calculate_entropy(list(action_dist.values()))
    
    # Save statistics
    with open(os.path.join(output_dir, "agent_stats.json"), 'w') as f:
        json.dump(agent_stats, f, indent=2)
    
    return agent_stats

def environment_analysis(metrics, output_dir):
    """Generate environment-specific analysis plots and statistics"""
    plt.figure(figsize=(15, 12))
    
    # Enemy types distribution
    plt.subplot(2, 2, 1)
    if metrics["enemy_types"]:
        # Handle conversion from string keys
        enemy_types = {k: v for k, v in metrics["enemy_types"].items()}
        types = list(enemy_types.keys())
        counts = list(enemy_types.values())
        
        # Sort by count
        sorted_data = sorted(zip(types, counts), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_data)
        
        # Plot as horizontal bars
        y_pos = np.arange(len(types))
        plt.barh(y_pos, counts, align='center')
        plt.yticks(y_pos, types)
        plt.xlabel('Count')
        plt.title('Enemy Type Distribution')
        
        # Add percentage labels
        total = sum(counts)
        for i, count in enumerate(counts):
            plt.text(count + (max(counts) * 0.01), i, f"{count/total*100:.1f}%", 
                   va='center', ha='left')
    
    # Enemy damage distribution
    plt.subplot(2, 2, 2)
    if metrics["enemy_damage_dealt"]:
        plt.hist(metrics["enemy_damage_dealt"], bins=20, alpha=0.7, color='red')
        plt.title('Enemy Damage Distribution')
        plt.xlabel('Damage Dealt')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add mean and median lines
        mean_damage = np.mean(metrics["enemy_damage_dealt"])
        median_damage = np.median(metrics["enemy_damage_dealt"])
        plt.axvline(mean_damage, color='black', linestyle='--', 
                  label=f'Mean: {mean_damage:.2f}')
        plt.axvline(median_damage, color='green', linestyle='--', 
                  label=f'Median: {median_damage:.2f}')
        plt.legend()
    
    # Enemy accuracy analysis
    plt.subplot(2, 2, 3)
    if metrics["enemy_accuracy"]:
        plt.hist(metrics["enemy_accuracy"], bins=10, range=(0, 1), alpha=0.7, color='blue')
        plt.title('Enemy Accuracy Distribution')
        plt.xlabel('Accuracy (0-1)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add mean accuracy line
        mean_accuracy = np.mean(metrics["enemy_accuracy"])
        plt.axvline(mean_accuracy, color='red', linestyle='--', 
                  label=f'Mean: {mean_accuracy:.2f}')
        plt.legend()
    
    # Wave difficulty analysis
    plt.subplot(2, 2, 4)
    if metrics["wave_difficulty"]:
        episodes = range(1, len(metrics["wave_difficulty"]) + 1)
        plt.plot(episodes, metrics["wave_difficulty"], 'purple')
        plt.title('Wave Difficulty Progression')
        plt.xlabel('Episode')
        plt.ylabel('Difficulty Multiplier')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "environment_analysis.png"), dpi=150)
    plt.close()
    
    # Additional plot for phase analysis
    if metrics["phase_transitions"] or metrics["phase_durations"]:
        plt.figure(figsize=(12, 8))
        
        # Phase scores
        plt.subplot(2, 1, 1)
        phase_colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for phase, scores in metrics["phase_scores"].items():
            if scores:
                phase_idx = int(phase) % len(phase_colors)
                episodes = range(1, len(scores) + 1)
                plt.plot(episodes, scores, color=phase_colors[phase_idx], 
                       label=f'Phase {phase}')
        
        plt.title('Scores by Training Phase')
        plt.xlabel('Episodes in Phase')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Phase durations
        plt.subplot(2, 1, 2)
        if metrics["phase_durations"]:
            # Parse phase durations
            phases = []
            durations = []
            for entry in metrics["phase_durations"]:
                if isinstance(entry, list) and len(entry) == 2:
                    phases.append(entry[0])
                    durations.append(entry[1])
            
            if phases and durations:
                plt.bar(phases, durations, color='teal')
                plt.title('Phase Durations')
                plt.xlabel('Phase')
                plt.ylabel('Duration (seconds)')
                plt.grid(True, alpha=0.3)
                
                # Add duration labels
                for i, duration in enumerate(durations):
                    plt.text(i, duration + (max(durations) * 0.01), 
                           f"{duration:.1f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_analysis.png"), dpi=150)
        plt.close()
    
    # Calculate environment statistics
    env_stats = {}
    
    if metrics["enemy_types"]:
        env_stats["total_enemy_types"] = len(enemy_types)
        env_stats["most_common_enemy"] = max(enemy_types.items(), key=lambda x: x[1])[0]
    
    if metrics["enemy_damage_dealt"]:
        env_stats["avg_enemy_damage"] = np.mean(metrics["enemy_damage_dealt"])
        env_stats["max_enemy_damage"] = max(metrics["enemy_damage_dealt"])
    
    if metrics["enemy_accuracy"]:
        env_stats["avg_enemy_accuracy"] = np.mean(metrics["enemy_accuracy"])
    
    if metrics["wave_difficulty"]:
        env_stats["initial_difficulty"] = metrics["wave_difficulty"][0]
        env_stats["final_difficulty"] = metrics["wave_difficulty"][-1]
        env_stats["max_difficulty"] = max(metrics["wave_difficulty"])
    
    # Save statistics
    with open(os.path.join(output_dir, "environment_stats.json"), 'w') as f:
        json.dump(env_stats, f, indent=2)
    
    return env_stats

def genetic_analysis(metrics, output_dir):
    """Generate genetic algorithm analysis plots and statistics"""
    if not metrics["genome_fitness"] and not metrics["genome_diversity"]:
        return {}  # No genetic data to analyze
    
    plt.figure(figsize=(12, 8))
    
    # Fitness progression
    plt.subplot(2, 1, 1)
    if metrics["genome_fitness"]:
        generations = range(1, len(metrics["genome_fitness"]) + 1)
        plt.plot(generations, metrics["genome_fitness"], 'r-', label='Max Fitness')
        
        # Add trend line
        if len(generations) > 5:
            z = np.polyfit(generations, metrics["genome_fitness"], 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), 'r--', 
                   label=f'Trend: {z[0]:.4f}x + {z[1]:.1f}')
        
        plt.title('Genetic Algorithm Fitness Progression')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Diversity progression
    plt.subplot(2, 1, 2)
    if metrics["genome_diversity"]:
        generations = range(1, len(metrics["genome_diversity"]) + 1)
        plt.plot(generations, metrics["genome_diversity"], 'b-')
        
        # Add trend line
        if len(generations) > 5:
            z = np.polyfit(generations, metrics["genome_diversity"], 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), 'b--', 
                   label=f'Trend: {z[0]:.4f}x + {z[1]:.1f}')
        
        plt.title('Genetic Population Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Measure')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "genetic_analysis.png"), dpi=150)
    plt.close()
    
    # Calculate genetic statistics
    genetic_stats = {}
    
    if metrics["genome_fitness"]:
        genetic_stats["generations"] = len(metrics["genome_fitness"])
        genetic_stats["initial_fitness"] = metrics["genome_fitness"][0]
        genetic_stats["final_fitness"] = metrics["genome_fitness"][-1]
        genetic_stats["max_fitness"] = max(metrics["genome_fitness"])
        genetic_stats["fitness_improvement"] = (metrics["genome_fitness"][-1] / 
                                             max(0.001, metrics["genome_fitness"][0]))
    
    if metrics["genome_diversity"]:
        genetic_stats["initial_diversity"] = metrics["genome_diversity"][0]
        genetic_stats["final_diversity"] = metrics["genome_diversity"][-1]
        genetic_stats["diversity_change"] = (metrics["genome_diversity"][-1] / 
                                          max(0.001, metrics["genome_diversity"][0]))
    
    # Save statistics
    with open(os.path.join(output_dir, "genetic_stats.json"), 'w') as f:
        json.dump(genetic_stats, f, indent=2)
    
    return genetic_stats

def create_comprehensive_dashboard(metrics, output_dir):
    """Create a comprehensive dashboard combining all key metrics"""
    plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 4, figure=plt.gcf())
    
    # 1. Performance Summary (2x2 grid)
    ax1 = plt.subplot(gs[0:2, 0:2])
    episodes = range(1, len(metrics["episode_scores"]) + 1)
    
    # Plot score, wave, and reward on same plot with different scales
    ax1.plot(episodes, metrics["episode_scores"], 'b-', label='Score')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(episodes, metrics["max_waves"], 'g-', label='Wave')
    ax1_twin.set_ylabel('Wave', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    
    # Add rewards on third y-axis if different scale needed
    if (metrics["episode_scores"] and metrics["total_rewards"] and 
        max(metrics["episode_scores"]) > 5 * max(metrics["total_rewards"])):
        ax1_twin2 = ax1.twinx()
        ax1_twin2.spines['right'].set_position(('outward', 60))
        ax1_twin2.plot(episodes, metrics["total_rewards"], 'r-', label='Reward')
        ax1_twin2.set_ylabel('Reward', color='r')
        ax1_twin2.tick_params(axis='y', labelcolor='r')
    
    ax1.set_title('Training Performance Overview')
    
    # Create custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. Learning Parameters (top right)
    ax2 = plt.subplot(gs[0, 2:])
    if metrics["epsilon_values"]:
        ax2.plot(episodes, metrics["epsilon_values"], 'b-', label='Epsilon')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Exploration Rate Decay')
        ax2.grid(True, alpha=0.3)
    
    # 3. Q-Values and Loss (right, second row)
    ax3 = plt.subplot(gs[1, 2:])
    if metrics["loss_values"] and metrics["q_values"]:
        # Normalize data for comparison
        max_loss = max(metrics["loss_values"])
        max_q = max(metrics["q_values"])
        
        loss_steps = range(1, len(metrics["loss_values"]) + 1)
        q_steps = range(1, len(metrics["q_values"]) + 1)
        
        # Smooth data
        window_size = min(50, len(metrics["loss_values"]) // 10)
        if window_size > 1:
            smooth_loss = np.convolve(metrics["loss_values"], 
                                    np.ones(window_size)/window_size, mode='valid')
            smooth_q = np.convolve(metrics["q_values"], 
                                  np.ones(window_size)/window_size, mode='valid')
            
            ax3.plot(loss_steps[:len(smooth_loss)], smooth_loss / max_loss, 'r-', label='Loss (norm)')
            ax3.plot(q_steps[:len(smooth_q)], smooth_q / max_q, 'g-', label='Q-Value (norm)')
        else:
            ax3.plot(loss_steps, [l/max_loss for l in metrics["loss_values"]], 'r-', label='Loss (norm)')
            ax3.plot(q_steps, [q/max_q for q in metrics["q_values"]], 'g-', label='Q-Value (norm)')
            
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Normalized Value')
        ax3.set_title('Learning Progress')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Action Distribution (bottom left)
    ax4 = plt.subplot(gs[2, 0:2])
    if metrics["action_distribution"]:
        action_dist = {int(k): v for k, v in metrics["action_distribution"].items()}
        actions = sorted(action_dist.keys())
        counts = [action_dist[a] for a in actions]
        
        action_labels = [
            "Up", "UpRight", "Right", "DownRight", 
            "Down", "DownLeft", "Left", "UpLeft",
            "Shoot", "RotateLeft", "RotateRight", 
            "Weapon1", "Weapon2", "Weapon3"
        ]
        
        labels = [action_labels[a] if a < len(action_labels) else f"Action {a}" for a in actions]
        
        # Plot as horizontal bars
        y_pos = np.arange(len(labels))
        ax4.barh(y_pos, counts, align='center')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.set_xlabel('Count')
        ax4.set_title('Action Distribution')
        
        # Add percentage labels
        total = sum(counts)
        for i, count in enumerate(counts):
            ax4.text(count + (max(counts) * 0.01), i, f"{count/total*100:.1f}%", 
                   va='center', ha='left')
    
    # 5. Enemy Analysis (right, third row)
    ax5 = plt.subplot(gs[2, 2:])
    if metrics["enemy_types"]:
        enemy_types = {k: v for k, v in metrics["enemy_types"].items()}
        types = list(enemy_types.keys())
        counts = list(enemy_types.values())
        
        # Sort by count
        sorted_data = sorted(zip(types, counts), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_data)
        
        # Limit to top 5 if too many types
        if len(types) > 5:
            types = types[:5]
            counts = counts[:5]
        
        # Plot as pie chart
        ax5.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax5.axis('equal')
        ax5.set_title('Enemy Type Distribution')
    
    # 6. State Heatmap (bottom)
    ax6 = plt.subplot(gs[3, 0:2])
    if "state_visits_heatmap" in metrics and isinstance(metrics["state_visits_heatmap"], list):
        # Convert nested list to numpy array
        heatmap_data = np.array(metrics["state_visits_heatmap"])
        
        # Add small constant to avoid log(0)
        heatmap_data = heatmap_data + 1e-6
        
        # Plot log-scaled heatmap
        im = ax6.imshow(np.log(heatmap_data), cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax6, label='Log(Visit Count)')
        ax6.set_title('Agent State Visitation Heatmap')
        ax6.set_xlabel('X Position')
        ax6.set_ylabel('Y Position')
    
    # 7. Genetic Evolution (bottom right)
    ax7 = plt.subplot(gs[3, 2:])
    if metrics["genome_fitness"]:
        generations = range(1, len(metrics["genome_fitness"]) + 1)
        ax7.plot(generations, metrics["genome_fitness"], 'r-', label='Fitness')
        
        if metrics["genome_diversity"]:
            ax7_twin = ax7.twinx()
            ax7_twin.plot(generations[:len(metrics["genome_diversity"])], 
                        metrics["genome_diversity"], 'b--', label='Diversity')
            ax7_twin.set_ylabel('Diversity', color='b')
            ax7_twin.tick_params(axis='y', labelcolor='b')
        
        ax7.set_xlabel('Generation')
        ax7.set_ylabel('Fitness', color='r')
        ax7.tick_params(axis='y', labelcolor='r')
        ax7.set_title('Genetic Evolution')
        ax7.grid(True, alpha=0.3)
        
        # Add legend
        lines7, labels7 = ax7.get_legend_handles_labels()
        if metrics["genome_diversity"]:
            lines7b, labels7b = ax7_twin.get_legend_handles_labels()
            ax7.legend(lines7 + lines7b, labels7 + labels7b)
        else:
            ax7.legend()
    
    # Add title and layout
    plt.suptitle('AI Training Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "comprehensive_dashboard.png"), dpi=200)
    plt.close()

def calculate_entropy(distribution):
    """Calculate entropy of a probability distribution"""
    if not distribution:
        return 0
    
    total = sum(distribution)
    if total == 0:
        return 0
    
    probabilities = [x / total for x in distribution]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def main():
    """Main function to run the analysis tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze AI Training Sessions")
    parser.add_argument("--session_dir", type=str, help="Directory of the training session to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all sessions in models directory")
    
    args = parser.parse_args()
    
    if args.session_dir:
        analyze_training_session(args.session_dir)
    elif args.all:
        # Find all session directories
        session_dirs = []
        for root, dirs, _ in os.walk("models"):
            for d in dirs:
                if d.startswith("session_"):
                    session_dirs.append(os.path.join(root, d))
        
        for session_dir in session_dirs:
            analyze_training_session(session_dir)
    else:
        print("Please specify a session directory or use --all to analyze all sessions")

if __name__ == "__main__":
    main()
