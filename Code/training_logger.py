import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from collections import deque, defaultdict

class TrainingLogger:
    """Comprehensive data logging for AI training analysis"""
    
    def __init__(self, base_dir="training_logs", session_id=None):
        # Create a unique session ID if none provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_id = session_id
        self.base_dir = base_dir
        self.log_dir = os.path.join(base_dir, f"session_{session_id}")
        self.plots_dir = os.path.join(self.log_dir, "plots")
        self.models_dir = os.path.join(self.log_dir, "models")
        self.data_dir = os.path.join(self.log_dir, "data")
        
        # Create directories
        for dir_path in [self.log_dir, self.plots_dir, self.models_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Metrics storage
        self.metrics = {
            # Episode metrics
            "episode_scores": [],
            "episode_durations": [],
            "max_waves": [],
            "avg_rewards": [],
            "total_rewards": [],
            
            # Agent metrics
            "epsilon_values": [],
            "loss_values": [],
            "q_values": [],
            "action_distribution": defaultdict(int),
            "state_visits_heatmap": np.zeros((100, 100)),  # Discretized 2D state space
            
            # Enemy metrics
            "enemies_spawned": [],
            "enemies_destroyed": [],
            "enemy_types": defaultdict(int),
            "enemy_damage_dealt": [],
            "enemy_accuracy": [],
            
            # Phase metrics
            "phase_transitions": [],
            "phase_durations": [],
            "phase_scores": defaultdict(list),
            
            # Genetic algorithm metrics
            "generation_number": [],
            "genome_fitness": [],
            "genome_diversity": [],
            "mutation_rates": [],
            
            # Performance metrics
            "frame_rates": [],
            "memory_usage": [],
            "processing_times": [],
            
            # Wave manager metrics
            "wave_difficulty": [],
            "adaptation_strategy": [],
            "spawn_patterns": [],
        }
        
        # Real-time tracking
        self.episode_start_time = time.time()
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.episode_counter = 0
        self.snapshot_interval = 10  # Save data every 10 episodes
        
        # Initialize session metadata
        self.metadata = {
            "session_id": session_id,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": "unknown",
            "training_config": {},
        }
        
        # Save initial metadata
        self._save_metadata()
    
    def log_episode_start(self, episode_num):
        """Log the start of a new episode"""
        self.episode_counter = episode_num
        self.episode_start_time = time.time()
    
    def log_episode_end(self, score, max_wave, drl_agent, game):
        """Log metrics at the end of an episode"""
        # Calculate episode duration
        duration = time.time() - self.episode_start_time
        
        # Basic episode metrics
        self.metrics["episode_scores"].append(score)
        self.metrics["episode_durations"].append(duration)
        self.metrics["max_waves"].append(max_wave)
        self.metrics["total_rewards"].append(getattr(game, "total_reward", 0))
        
        # Agent metrics
        self.metrics["epsilon_values"].append(drl_agent.epsilon)
        
        if hasattr(drl_agent, "training_losses") and drl_agent.training_losses:
            self.metrics["loss_values"].append(drl_agent.training_losses[-1])
        
        if hasattr(drl_agent, "q_values") and drl_agent.q_values:
            self.metrics["q_values"].append(drl_agent.q_values[-1])
        
        # Log action distribution
        if hasattr(drl_agent, "last_action"):
            self.metrics["action_distribution"][drl_agent.last_action] += 1
        
        # Log enemy metrics
        self.metrics["enemies_destroyed"].append(
            len(getattr(game, "enemy_damage_tracker", {}).keys())
        )
        
        # Genetic algorithm metrics
        if hasattr(game, "genetic_evolver"):
            self.metrics["generation_number"].append(
                game.genetic_evolver.generations
            )
            
            # Add fitness of top genomes if available
            if hasattr(game.genetic_evolver, "populations"):
                top_fitness = max(genome.fitness for genome in 
                                game.genetic_evolver.populations[0]) if game.genetic_evolver.populations else 0
                self.metrics["genome_fitness"].append(top_fitness)
                
                # Calculate diversity as average distance between genomes
                if game.genetic_evolver.populations and len(game.genetic_evolver.populations[0]) > 1:
                    diversity = self._calculate_genome_diversity(game.genetic_evolver.populations[0])
                    self.metrics["genome_diversity"].append(diversity)
        
        # Log wave manager metrics
        if hasattr(game.wave_manager, "difficulty_multiplier"):
            self.metrics["wave_difficulty"].append(game.wave_manager.difficulty_multiplier)
        
        if hasattr(game.wave_manager, "adaptation_strategy"):
            self.metrics["adaptation_strategy"].append(game.wave_manager.adaptation_strategy)
        
        # Check if we should save a snapshot
        if self.episode_counter % self.snapshot_interval == 0:
            self.save_snapshot()
            self.generate_plots()
    
    def log_phase_transition(self, old_phase, new_phase, score):
        """Log when training transitions between phases"""
        phase_duration = time.time() - self.phase_start_time
        
        self.metrics["phase_transitions"].append((old_phase, new_phase, self.episode_counter))
        self.metrics["phase_durations"].append((old_phase, phase_duration))
        self.metrics["phase_scores"][old_phase].append(score)
        
        self.current_phase = new_phase
        self.phase_start_time = time.time()
        
        # Create a phase transition plot
        self._plot_phase_performance()
    
    def log_action(self, action, state):
        """Log agent actions and state visits"""
        # Update action distribution
        self.metrics["action_distribution"][action] += 1
        
        # Update state visits heatmap (assuming first two dimensions are position)
        if len(state) >= 2:
            # Discretize player position for heatmap
            x_bin = min(int(state[0] * 100), 99)
            y_bin = min(int(state[1] * 100), 99)
            self.metrics["state_visits_heatmap"][y_bin, x_bin] += 1
    
    def log_enemy_stats(self, enemy_type, damage_dealt, shots_fired, shots_hit):
        """Log detailed enemy performance statistics"""
        self.metrics["enemy_types"][enemy_type] += 1
        self.metrics["enemy_damage_dealt"].append(damage_dealt)
        
        if shots_fired > 0:
            accuracy = shots_hit / shots_fired
        else:
            accuracy = 0
        
        self.metrics["enemy_accuracy"].append(accuracy)
    
    def log_performance(self, frame_rate, memory_usage=0, processing_time=0):
        """Log system performance metrics"""
        self.metrics["frame_rates"].append(frame_rate)
        self.metrics["memory_usage"].append(memory_usage)
        self.metrics["processing_times"].append(processing_time)
    
    def save_snapshot(self):
        """Save current metrics to disk"""
        # Save metrics as JSON
        metrics_file = os.path.join(self.data_dir, f"metrics_episode_{self.episode_counter}.json")
        
        # Convert defaultdicts and numpy arrays to serializable format
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, defaultdict):
                serializable_metrics[key] = dict(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Fix for DataFrame creation - ensure all columns are 1D
        df_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # Check if it's a 1D array with simple types
                if all(not isinstance(item, (list, tuple, dict, np.ndarray)) for item in value):
                    df_metrics[key] = value
        
        # In save_snapshot method:
        if df_metrics:
            try:
                # Find the most common length
                lengths = [len(v) for v in df_metrics.values()]
                if not lengths:
                    print("No metrics to save to CSV")
                    return
                    
                common_length = max(set(lengths), key=lengths.count)
                
                # Only keep arrays of the same length
                filtered_metrics = {k: v for k, v in df_metrics.items() if len(v) == common_length}
                
                df = pd.DataFrame(filtered_metrics)
                csv_file = os.path.join(self.data_dir, f"metrics_episode_{self.episode_counter}.csv")
                df.to_csv(csv_file, index=False)
            except Exception as e:
                print(f"Error saving CSV metrics: {str(e)}")
        
        # Save state visits heatmap separately
        if "state_visits_heatmap" in self.metrics:
            np.save(
                os.path.join(self.data_dir, f"state_heatmap_episode_{self.episode_counter}.npy"),
                np.array(self.metrics["state_visits_heatmap"])
            )
        
        print(f"Saved metrics snapshot at episode {self.episode_counter}")
    
    def generate_plots(self):
        """Generate and save visualization plots"""
        episode_numbers = list(range(1, len(self.metrics["episode_scores"]) + 1))
        
        # Basic performance plots
        self._create_performance_plots(episode_numbers)
        
        # Agent-specific plots
        self._create_agent_plots(episode_numbers)
        
        # Enemy and genetic algorithm plots
        self._create_enemy_plots()
        
        # State visitation heatmap
        self._create_state_heatmap()
        
        # Save combined dashboard
        self._create_dashboard()
        
        print(f"Generated plots at episode {self.episode_counter}")
    
    def _create_performance_plots(self, episode_numbers):
        """Create and save performance plots"""
        plt.figure(figsize=(12, 8))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(episode_numbers, self.metrics["episode_scores"], 'b-')
        plt.title('Score per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Plot max waves
        plt.subplot(2, 2, 2)
        plt.plot(episode_numbers, self.metrics["max_waves"], 'g-')
        plt.title('Max Wave per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Wave')
        
        # Plot durations
        plt.subplot(2, 2, 3)
        plt.plot(episode_numbers, self.metrics["episode_durations"], 'm-')
        plt.title('Episode Duration')
        plt.xlabel('Episode')
        plt.ylabel('Seconds')
        
        # Plot total rewards
        plt.subplot(2, 2, 4)
        plt.plot(episode_numbers, self.metrics["total_rewards"], 'r-')
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"performance_episode_{self.episode_counter}.png"))
        plt.close()
    
    def _create_agent_plots(self, episode_numbers):
        """Create and save agent-specific plots"""
        plt.figure(figsize=(12, 8))
        
        # Plot epsilon decay
        plt.subplot(2, 2, 1)
        plt.plot(episode_numbers, self.metrics["epsilon_values"], 'b-')
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        # Plot loss values
        if self.metrics["loss_values"]:
            plt.subplot(2, 2, 2)
            plt.plot(range(len(self.metrics["loss_values"])), self.metrics["loss_values"], 'r-')
            plt.title('Training Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
        
        # Plot Q-values
        if self.metrics["q_values"]:
            plt.subplot(2, 2, 3)
            plt.plot(range(len(self.metrics["q_values"])), self.metrics["q_values"], 'g-')
            plt.title('Average Q-Value')
            plt.xlabel('Training Step')
            plt.ylabel('Q-Value')
        
        # Plot action distribution
        if self.metrics["action_distribution"]:
            plt.subplot(2, 2, 4)
            actions = list(self.metrics["action_distribution"].keys())
            counts = list(self.metrics["action_distribution"].values())
            plt.bar(actions, counts)
            plt.title('Action Distribution')
            plt.xlabel('Action')
            plt.ylabel('Count')
            
            # Add action labels
            action_labels = [
                "Up", "UpRight", "Right", "DownRight", 
                "Down", "DownLeft", "Left", "UpLeft",
                "Shoot", "RotateLeft", "RotateRight", 
                "Weapon1", "Weapon2", "Weapon3"
            ]
            if max(actions) < len(action_labels):
                plt.xticks(actions, [action_labels[a] for a in actions], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"agent_metrics_episode_{self.episode_counter}.png"))
        plt.close()
    
    def _create_enemy_plots(self):
        """Create and save enemy-related plots"""
        plt.figure(figsize=(12, 8))
        
        # Plot enemy types distribution
        if self.metrics["enemy_types"] and len(self.metrics["enemy_types"]) > 0:
            plt.subplot(2, 2, 1)
            enemy_types = list(self.metrics["enemy_types"].keys())
            counts = list(self.metrics["enemy_types"].values())
            plt.bar(enemy_types, counts)
            plt.title('Enemy Type Distribution')
            plt.xlabel('Enemy Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        else:
            plt.subplot(2, 2, 1)
            plt.text(0.5, 0.5, "No enemy data collected yet", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Enemy Type Distribution (No Data)')
            
        # Plot enemy damage dealt
        if self.metrics["enemy_damage_dealt"]:
            plt.subplot(2, 2, 2)
            plt.hist(self.metrics["enemy_damage_dealt"], bins=10)
            plt.title('Enemy Damage Distribution')
            plt.xlabel('Damage Dealt')
            plt.ylabel('Frequency')
        
        # Plot enemy accuracy
        if self.metrics["enemy_accuracy"]:
            plt.subplot(2, 2, 3)
            plt.hist(self.metrics["enemy_accuracy"], bins=10, range=(0, 1))
            plt.title('Enemy Accuracy Distribution')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
        
        # Plot genetic algorithm metrics
        if self.metrics["genome_fitness"]:
            plt.subplot(2, 2, 4)
            generations = list(range(1, len(self.metrics["genome_fitness"]) + 1))
            plt.plot(generations, self.metrics["genome_fitness"], 'r-', label='Fitness')
            
            if self.metrics["genome_diversity"]:
                plt.plot(generations[:len(self.metrics["genome_diversity"])], 
                        self.metrics["genome_diversity"], 'b--', label='Diversity')
            
            plt.title('Genetic Algorithm Metrics')
            plt.xlabel('Generation')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"enemy_metrics_episode_{self.episode_counter}.png"))
        plt.close()
    
    def _create_state_heatmap(self):
        """Create and save state visitation heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Add small constant to avoid log(0)
        heatmap_data = self.metrics["state_visits_heatmap"] + 1e-6
        
        # Use log scale for better visualization
        plt.imshow(np.log(heatmap_data), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Log(Visit Count)')
        plt.title('Agent State Visitation Heatmap')
        plt.xlabel('X Position (Discretized)')
        plt.ylabel('Y Position (Discretized)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"state_heatmap_episode_{self.episode_counter}.png"))
        plt.close()
    
    def _plot_phase_performance(self):
        """Create and save phase-specific performance plots"""
        plt.figure(figsize=(12, 8))
        
        # Plot scores by phase
        plt.subplot(2, 1, 1)
        phase_colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for phase, scores in self.metrics["phase_scores"].items():
            if scores:
                phase_idx = int(phase) % len(phase_colors)
                episodes = list(range(1, len(scores) + 1))  # Fixed: create proper episode numbers
                plt.plot(episodes, scores, color=phase_colors[phase_idx], 
                    label=f'Phase {phase}')
        
        plt.title('Scores by Training Phase')
        plt.xlabel('Episodes in Phase')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Phase durations with different colors
        plt.subplot(2, 1, 2)
        if self.metrics["phase_durations"]:
            phases = []
            durations = []
            colors = []
            
            for i, entry in enumerate(self.metrics["phase_durations"]):
                if isinstance(entry, list) and len(entry) == 2:
                    phase_num = int(entry[0]) 
                    phases.append(phase_num)
                    durations.append(entry[1])
                    colors.append(phase_colors[phase_num % len(phase_colors)])
            
            if phases and durations:
                plt.bar(phases, durations, color=colors)  # Use the color list here
    def _create_dashboard(self):
        """Create a combined dashboard of key metrics"""
        plt.figure(figsize=(15, 10))
        
        # Episode numbers for x-axis
        episode_numbers = list(range(1, len(self.metrics["episode_scores"]) + 1))
        
        # Plot score and max wave
        plt.subplot(3, 2, 1)
        plt.plot(episode_numbers, self.metrics["episode_scores"], 'b-', label='Score')
        plt.title('Score Progress')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        ax2 = plt.twinx()
        ax2.plot(episode_numbers, self.metrics["max_waves"], 'r-', label='Wave')
        ax2.set_ylabel('Max Wave', color='r')
        
        # Plot epsilon and loss
        plt.subplot(3, 2, 2)
        plt.plot(episode_numbers, self.metrics["epsilon_values"], 'g-', label='Epsilon')
        plt.title('Learning Parameters')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        if self.metrics["loss_values"]:
            ax3 = plt.twinx()
            loss_episodes = list(range(1, len(self.metrics["loss_values"]) + 1))
            ax3.plot(loss_episodes, self.metrics["loss_values"], 'y-', label='Loss')
            ax3.set_ylabel('Loss', color='y')
        
        # Plot total rewards
        plt.subplot(3, 2, 3)
        plt.plot(episode_numbers, self.metrics["total_rewards"], 'purple')
        plt.title('Total Reward Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot action distribution as pie chart
        plt.subplot(3, 2, 4)
        if self.metrics["action_distribution"]:
            labels = []
            sizes = []
            for action, count in self.metrics["action_distribution"].items():
                if action < 14:  # Assuming we have 14 action types
                    action_labels = [
                        "Up", "UpRight", "Right", "DownRight", 
                        "Down", "DownLeft", "Left", "UpLeft",
                        "Shoot", "RotateLeft", "RotateRight", 
                        "Weapon1", "Weapon2", "Weapon3"
                    ]
                    labels.append(action_labels[action])
                    sizes.append(count)
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Action Distribution')
        
        # Plot state heatmap
        plt.subplot(3, 2, 5)
        heatmap_data = self.metrics["state_visits_heatmap"] + 1e-6
        plt.imshow(np.log(heatmap_data), cmap='hot', interpolation='nearest')
        plt.title('State Visitation Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Plot genetic metrics if available
        plt.subplot(3, 2, 6)
        if self.metrics["genome_fitness"]:
            generations = list(range(1, len(self.metrics["genome_fitness"]) + 1))
            plt.plot(generations, self.metrics["genome_fitness"], 'r-', label='Fitness')
            plt.title('Genetic Evolution')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            
            if self.metrics["genome_diversity"]:
                ax4 = plt.twinx()
                ax4.plot(generations[:len(self.metrics["genome_diversity"])], 
                        self.metrics["genome_diversity"], 'b--', label='Diversity')
                ax4.set_ylabel('Diversity', color='b')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"dashboard_episode_{self.episode_counter}.png"))
        plt.close()
    
    def _calculate_genome_diversity(self, population):
        """Calculate diversity as average distance between genomes"""
        if not population or len(population) <= 1:
            return 0
            
        # Simplified diversity calculation
        total_dist = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                # Calculate distance between genomes (assuming they have parameters attribute)
                if hasattr(population[i], 'parameters') and hasattr(population[j], 'parameters'):
                    params_i = population[i].parameters
                    params_j = population[j].parameters
                    
                    # Calculate Euclidean distance between parameter vectors
                    dist = sum((p_i - p_j) ** 2 for p_i, p_j in zip(params_i, params_j)) ** 0.5
                    total_dist += dist
                    count += 1
        
        return total_dist / max(1, count)  # Avoid division by zero
    
    def _save_metadata(self):
        """Save session metadata to disk"""
        metadata_file = os.path.join(self.log_dir, "session_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def set_training_config(self, config):
        """Update the training configuration metadata"""
        self.metadata["training_config"] = config
        self._save_metadata()
    
    def set_device(self, device):
        """Update the device information"""
        self.metadata["device"] = device
        self._save_metadata()
    
    def log_training_completed(self):
        """Log the completion of training"""
        self.metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["total_episodes"] = self.episode_counter
        self.metadata["final_score"] = self.metrics["episode_scores"][-1] if self.metrics["episode_scores"] else 0
        self.metadata["final_wave"] = self.metrics["max_waves"][-1] if self.metrics["max_waves"] else 0
        
        self._save_metadata()
        self.save_snapshot()
        self.generate_plots()
        
        print(f"Training session {self.session_id} completed and data saved.")

# Example usage:
# logger = TrainingLogger()
# logger.log_episode_start(1)
# ... (training happens)
# logger.log_episode_end(score, max_wave, drl_agent, game)
