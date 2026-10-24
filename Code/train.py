import argparse
import os
import time
import psutil
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from collections import deque
import pandas as pd
from main import Game
from ai.drl_agent import DRLAgent
from training_logger import TrainingLogger

def train_ai_system(episodes=250, batch_size=64, max_steps=10000, render=False, 
                   save_interval=10, load_models=False, models_dir="models",
                   wave_type="normal", modular_training=False,combat_only=False):
    """
    Train the complete AI system with modular training phases.
    Enhanced with comprehensive data logging.
    """
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a unique session directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(models_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    logger = TrainingLogger(base_dir=models_dir, session_id=timestamp)
    
    # Save training configuration
    config = {
        "episodes": episodes,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "wave_type": wave_type,
        "modular_training": modular_training,
        "save_interval": save_interval
    }
    logger.set_training_config(config)
    
    if modular_training:
        if combat_only:
            # Only allocate episodes to Combat phase
            phases = [
                {"name": "Full Combat", "phase": 4, "episodes": episodes}
            ]
            print(f"Combat-only training enabled (Phase 4)")
        else:
            # Normal phase distribution
            phase_episodes = episodes // 4
            phases = [
                {"name": "Meteor Dodging", "phase": 1, "episodes": phase_episodes},
                {"name": "Laser Dodging", "phase": 2, "episodes": phase_episodes},
                {"name": "Aim Training", "phase": 3, "episodes": phase_episodes}
            ]
            # Add Full Combat
            phases.append({"name": "Full Combat", "phase": 4, "episodes": phase_episodes})
            print(f"Modular training enabled with {len(phases)} phases")
    # Initialize game
    game = Game(ai_mode=True, wave_type=wave_type, phased_training=modular_training)
    game.drl_agent = DRLAgent(game, device=game._get_device())
    
    # Log device information
    logger.set_device(game._get_device())
    
    # Load models if requested
    if load_models:
        load_ai_models(game, models_dir)
        if combat_only:
            # Force Combat phase after loading model
            game.wave_manager.set_training_phase(4)
            print(f"Set training phase to Full Combat (4)")
    
    # Initialize tracking variables
    scores = []
    waves_reached = []
    durations = []
    avg_rewards = []
    
    # Recent performance for plotting (maintain the original deques)
    recent_scores = deque(maxlen=10)
    recent_waves = deque(maxlen=10)
    recent_rewards = deque(maxlen=10)
    

    process = psutil.Process()
    
    # For plotting during training
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training phase tracking
    current_phase_index = 0
    phase_episode_count = 0
    
    # Track time for FPS calculation
    last_time = time.time()
    fps_counter = 0
    fps_values = []
    
    # Main training loop
    for episode in range(1, episodes + 1):
        # Log episode start
        logger.log_episode_start(episode)
        
        # Handle phase transitions for modular training
        if modular_training:
            current_phase = phases[current_phase_index]
            
            # Check if we need to switch phases
            if phase_episode_count >= current_phase["episodes"] and current_phase_index < len(phases) - 1:
                # Save models before switching
                save_ai_models(game, session_dir, episode-1, f"phase_{current_phase_index}")
                
                # Log phase transition
                old_phase = current_phase["phase"]
                current_phase_index += 1
                new_phase = phases[current_phase_index]["phase"]
                logger.log_phase_transition(old_phase, new_phase, game.score)
                
                # Move to next phase
                phase_episode_count = 0
                new_phase_info = phases[current_phase_index]
                
                print(f"\n=== Switching to Phase {current_phase_index+1}: {new_phase_info['name']} ===")
                game.wave_manager.set_training_phase(new_phase_info["phase"])
            
            # Update current phase info for logging
            current_phase = phases[current_phase_index]
            print(f"\nEpisode {episode}/{episodes} - Phase: {current_phase['name']} ({phase_episode_count+1}/{current_phase['episodes']})")
            phase_episode_count += 1
        else:
            print(f"\nEpisode {episode}/{episodes}")
        
        # Reset game if needed
        if game.game_over or episode == 1:
            game.reset_game()
        
        # Track metrics
        total_reward = 0
        steps = 0
        max_wave = 0
        enemy_count = 0
        enemy_destroyed_count = 0
        episode_start_time = time.time()
        processing_times = []
        
        # Run episode
        while not game.game_over and steps < max_steps:
            step_start_time = time.time()
            
            # Get state and take action
            state = game.drl_agent.get_state()
            action = game.drl_agent.act(state)
            
            # Log action for distribution tracking
            logger.log_action(action, state)
            
            # Execute action
            game.drl_agent.execute_action(action)
            
            # Update game
            game.update()
            if render:
                game.draw()
            
            # Get new state and reward
            next_state = game.drl_agent.get_state()
            reward = game._calculate_reward()
            total_reward += reward
            
            # Store experience and train
            game.drl_agent.remember(state, action, reward, next_state, game.game_over)
            if steps % 4 == 0:
                game.drl_agent.replay()
            
            # Update metrics
            steps += 1
            max_wave = max(max_wave, game.wave_manager.current_wave)
            
            # Track enemy metrics
            if hasattr(game, 'enemies_destroyed_this_frame'):
                enemy_destroyed_count += game.enemies_destroyed_this_frame
            
            # Track spawned enemies
            current_enemy_count = len(game.enemies)
            if current_enemy_count > enemy_count:
                enemy_count += (current_enemy_count - enemy_count)
            
            # Performance tracking
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            processing_times.append(step_duration)
            
            # FPS calculation
            fps_counter += 1
            if step_end_time - last_time >= 1.0:  # Calculate FPS every second
                fps = fps_counter / (step_end_time - last_time)
                fps_values.append(fps)
                fps_counter = 0
                last_time = step_end_time
                
                # Log system performance
                memory_mb = process.memory_info().rss / (1024 * 1024)
                logger.log_performance(fps, memory_mb, np.mean(processing_times))
                processing_times = []
                
            # Track enemy data every 100 steps
            if steps % 100 == 0 and hasattr(game, 'genetic_evolver'):
                # Log data from genetic algorithm
                for enemy_id, damage in game.enemy_damage_tracker.items():
                    if enemy_id in game.enemy_shot_tracker:
                        shots_fired, shots_hit = game.enemy_shot_tracker[enemy_id]
                        # Get enemy type - add try/except in case the enemy is gone
                        enemy_type = "unknown"
                        for enemy in game.enemies:
                            if id(enemy) == enemy_id:
                                enemy_type = getattr(enemy, 'debug_type', 'normal')
                                break
                        
                        logger.log_enemy_stats(enemy_type, damage, shots_fired, shots_hit)
        
        # Episode finished - log stats
        episode_duration = time.time() - episode_start_time
        avg_reward = total_reward / max(1, steps)  # Calculate average reward per step
        
        print(f"  Score: {game.score}, Max Wave: {max_wave}, Steps: {steps}")
        print(f"  Duration: {episode_duration:.2f}s, Avg Reward: {avg_reward:.4f}")
        print(f"  Epsilon: {game.drl_agent.epsilon:.4f}")
        
        # Update original tracking arrays (important!)
        scores.append(game.score)
        waves_reached.append(max_wave)
        durations.append(episode_duration)
        avg_rewards.append(avg_reward)
        
        # Update recent deques (maintain original functionality)
        recent_scores.append(game.score)
        recent_waves.append(max_wave)
        recent_rewards.append(avg_reward)
        
        # Store the metrics in logger too
        logger.metrics["avg_rewards"] = avg_rewards  # Add average reward tracking
        
        # Log episode end
        logger.log_episode_end(game.score, max_wave, game.drl_agent, game)
        
        # Update genetic algorithm periodically (same as original code)
        if hasattr(game, 'wave_manager') and hasattr(game.wave_manager, 'wave_active'):
            if getattr(game, 'wave_completed', False) or (hasattr(game, 'prev_wave_active') and 
                                                        game.prev_wave_active and 
                                                        not game.wave_manager.wave_active):
                # Evolve enemies periodically - keep original code's logic
                if game.wave_manager.current_wave % 3 == 0 and hasattr(game, 'genetic_evolver'):
                    print("Evolving enemy population...")
                    game.genetic_evolver.evolve()
                    
                    # Add explicit logging for genetic evolution
                    if hasattr(game.genetic_evolver, 'generations'):
                        logger.metrics["generation_number"].append(game.genetic_evolver.generations)
                        
                        # Log top fitness if available
                        if hasattr(game.genetic_evolver, 'populations') and game.genetic_evolver.populations:
                            top_fitness = max(genome.fitness for genome in 
                                            game.genetic_evolver.populations[0]) if game.genetic_evolver.populations[0] else 0
                            logger.metrics["genome_fitness"].append(top_fitness)
        
        # Periodically save models
        if episode % save_interval == 0:
            model_path = os.path.join(session_dir, f"episode_{episode}")
            save_ai_models(game, model_path, episode)
        
        # Update target network periodically
        if episode % 5 == 0:
            game.drl_agent.target_net.load_state_dict(game.drl_agent.policy_net.state_dict())
        
        # Update plots - use the original function parameters to maintain compatibility
        update_training_plots(axes, episodes, scores, waves_reached, avg_rewards, durations, 
                             recent_scores, recent_waves, recent_rewards)
    
   # At the end of train_ai_system function
    try:
        print(f"Saving final models to {session_dir}")
        save_ai_models(game, session_dir, episodes, "final")
    except Exception as e:
        print(f"ERROR saving final models: {str(e)}")
    
    # Save all metrics to CSV
    save_training_metrics(session_dir, scores, waves_reached, durations, avg_rewards,
                        epsilon_values=logger.metrics["epsilon_values"],
                        loss_values=logger.metrics["loss_values"] if "loss_values" in logger.metrics else None,
                        action_dist=logger.metrics["action_distribution"] if "action_distribution" in logger.metrics else None)
    
    # Log training completion
    logger.log_training_completed()
    
    # Clean up
    plt.ioff()
    
    # Return the original arrays for compatibility
    return scores, waves_reached, avg_rewards


def run_evaluation(num_games=10, render=True, models_dir="models", wave_type="normal"):
    """
    Evaluate the trained AI system by running games and measuring performance
    
    Parameters:
    -----------
    num_games : int
        Number of games to run for evaluation
    render : bool
        Whether to render the games during evaluation
    models_dir : str
        Directory containing saved models
    wave_type : str
        Wave type to use ("normal" or "neural")
    """
    # Load game with AI mode
    game = Game(ai_mode=True, wave_type=wave_type)
    
    # Initialize the DRL agent
    game.drl_agent = DRLAgent(game, device=game._get_device())
    
    # Load the best models
    load_ai_models(game, models_dir)
    
    # Set evaluation mode (no exploration)
    game.drl_agent.epsilon = 0.0
    
    # Metrics to track
    scores = []
    max_waves = []
    durations = []
    
    # Create logger for detailed metrics
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(models_dir, f"evaluation_{eval_timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Log evaluation parameters
    with open(os.path.join(eval_dir, "eval_params.json"), 'w') as f:
        json.dump({
            "num_games": num_games,
            "wave_type": wave_type,
            "models_dir": models_dir,
            "timestamp": eval_timestamp
        }, f, indent=2)
    
    for i in range(num_games):
        print(f"\nEvaluation Game {i+1}/{num_games}")
        
        # Reset game
        game.reset_game()
        
        # Track metrics
        steps = 0
        max_wave = 0
        total_reward = 0
        start_time = time.time()
        
        # Run until game over
        while not game.game_over and steps < 10000:
            # Get state and choose action
            state = game.drl_agent.get_state()
            action = game.drl_agent.act(state)  # Will use best action due to epsilon=0
            
            # Execute action
            game.drl_agent.execute_action(action)
            
            # Update game
            game.update()
            
            # Track reward
            reward = game._calculate_reward()
            total_reward += reward
            
            # Render if requested
            if render:
                game.draw()
            
            # Track metrics
            steps += 1
            max_wave = max(max_wave, game.wave_manager.current_wave)
        
        # Game finished
        duration = time.time() - start_time
        
        # Record metrics
        scores.append(game.score)
        max_waves.append(max_wave)
        durations.append(duration)
        
        # Print results
        print(f"  Score: {game.score}, Max Wave: {max_wave}")
        print(f"  Duration: {duration:.2f}s, Total Reward: {total_reward:.2f}")
    
    # Save evaluation results
    results_df = pd.DataFrame({
        'game': range(1, num_games + 1),
        'score': scores,
        'max_wave': max_waves,
        'duration': durations
    })
    results_df.to_csv(os.path.join(eval_dir, "eval_results.csv"), index=False)
    
    # Generate summary statistics
    summary = {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_wave': np.mean(max_waves),
        'std_wave': np.std(max_waves),
        'avg_duration': np.mean(durations),
        'std_duration': np.std(durations),
        'min_score': min(scores),
        'max_score': max(scores),
        'min_wave': min(max_waves),
        'max_wave': max(max_waves)
    }
    
    # Save summary
    with open(os.path.join(eval_dir, "eval_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print overall statistics
    print("\nEvaluation Results:")
    print(f"  Average Score: {summary['avg_score']:.2f} Â± {summary['std_score']:.2f}")
    print(f"  Average Max Wave: {summary['avg_wave']:.2f} Â± {summary['std_wave']:.2f}")
    print(f"  Average Duration: {summary['avg_duration']:.2f}s Â± {summary['std_duration']:.2f}s")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(1, num_games + 1), scores)
    plt.axhline(y=summary['avg_score'], color='r', linestyle='--', label=f"Avg: {summary['avg_score']:.2f}")
    plt.title('Evaluation Scores')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.bar(range(1, num_games + 1), max_waves)
    plt.axhline(y=summary['avg_wave'], color='r', linestyle='--', label=f"Avg: {summary['avg_wave']:.2f}")
    plt.title('Max Waves Reached')
    plt.xlabel('Game')
    plt.ylabel('Wave')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.bar(range(1, num_games + 1), durations)
    plt.axhline(y=summary['avg_duration'], color='r', linestyle='--', label=f"Avg: {summary['avg_duration']:.2f}s")
    plt.title('Game Durations')
    plt.xlabel('Game')
    plt.ylabel('Duration (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "eval_results.png"))
    
    # Return results
    return scores, max_waves, durations


def update_training_plots(axes, episodes, scores, waves, rewards, durations, 
                        recent_scores, recent_waves, recent_rewards):
    """Update the training progress plots - maintain original functionality"""
    # Clear previous plots
    for ax in axes.flat:
        ax.clear()
    
    # Plot score
    axes[0, 0].plot(scores, 'b-', label='Score')
    axes[0, 0].plot(np.arange(len(scores)), 
                   [np.mean(recent_scores) for _ in range(len(scores))], 
                   'r-', label='Average (Last 10)')
    axes[0, 0].set_title('Score per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    
    # Plot max wave reached
    axes[0, 1].plot(waves, 'g-', label='Max Wave')
    axes[0, 1].plot(np.arange(len(waves)), 
                   [np.mean(recent_waves) for _ in range(len(waves))], 
                   'r-', label='Average (Last 10)')
    axes[0, 1].set_title('Max Wave per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Wave')
    axes[0, 1].legend()
    
    # Plot average reward
    axes[1, 0].plot(rewards, 'y-', label='Avg Reward')
    axes[1, 0].plot(np.arange(len(rewards)), 
                   [np.mean(recent_rewards) for _ in range(len(rewards))], 
                   'r-', label='Average (Last 10)')
    axes[1, 0].set_title('Average Reward per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    
    # Plot episode duration
    axes[1, 1].plot(durations, 'm-', label='Duration')
    axes[1, 1].set_title('Episode Duration')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Seconds')
    
    # Update the display
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def save_training_metrics(session_dir, scores, waves, durations, rewards, 
                        epsilon_values=None, loss_values=None, action_dist=None):
    """Save training metrics to multiple CSV files for detailed analysis"""
    # Create metrics directory
    metrics_dir = os.path.join(session_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create timestamp for this snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Basic performance metrics - use original dataframe approach
    metrics = pd.DataFrame({
        'episode': range(1, len(scores) + 1),
        'score': scores,
        'max_wave': waves,
        'duration': durations,
        'avg_reward': rewards
    })
    
    metrics_path = os.path.join(metrics_dir, f"training_metrics_{timestamp}.csv")
    metrics.to_csv(metrics_path, index=False)
    
    # Add detailed metrics if available
    if epsilon_values:
        # Save epsilon values to CSV
        epsilon_df = pd.DataFrame({
            'episode': range(1, len(epsilon_values) + 1),
            'epsilon': epsilon_values
        })
        epsilon_path = os.path.join(metrics_dir, f"epsilon_values_{timestamp}.csv")
        epsilon_df.to_csv(epsilon_path, index=False)
    
    if loss_values:
        # Save loss values to CSV
        loss_df = pd.DataFrame({
            'step': range(1, len(loss_values) + 1),
            'loss': loss_values
        })
        loss_path = os.path.join(metrics_dir, f"loss_values_{timestamp}.csv")
        loss_df.to_csv(loss_path, index=False)
    
    if action_dist:
        # Save action distribution to CSV
        action_df = pd.DataFrame(list(action_dist.items()), columns=['action', 'count'])
        action_path = os.path.join(metrics_dir, f"action_distribution_{timestamp}.csv")
        action_df.to_csv(action_path, index=False)
    
    print(f"Saved training metrics to {metrics_dir}")


def save_ai_models(game, models_dir, episode, suffix=""):
    """Save all AI models with episode number"""
    # Create episode directory
    if suffix:
        episode_dir = os.path.join(models_dir, f"episode_{episode}_{suffix}")
    else:
        episode_dir = os.path.join(models_dir, f"episode_{episode}")
    os.makedirs(episode_dir, exist_ok=True)
    
    print(f"Saving models to directory: {episode_dir}")
    
    # Try saving DRL model
    try:
        drl_path = os.path.join(episode_dir, "drl_agent.pt")
        print(f"Attempting to save DRL model to: {drl_path}")
        game.drl_agent.save_model(drl_path)
        print(f"DRL model saved successfully")
    except Exception as e:
        print(f"ERROR saving DRL model: {str(e)}")
    
    # Try saving genetic algorithm state
    try:
        ga_path = os.path.join(episode_dir, "genetic_evolver.pkl")
        print(f"Attempting to save genetic evolver to: {ga_path}")
        game.genetic_evolver.save_state(ga_path)
        print(f"Genetic evolver saved successfully")
    except Exception as e:
        print(f"ERROR saving genetic evolver: {str(e)}")
    
    # Save training phase info
    try:
        if hasattr(game.wave_manager, 'training_phase'):
            phase_path = os.path.join(episode_dir, "training_phase.txt")
            with open(phase_path, "w") as f:
                f.write(str(game.wave_manager.training_phase))
            print(f"Saved training phase: {game.wave_manager.training_phase}")
    except Exception as e:
        print(f"ERROR saving training phase: {str(e)}")

def load_ai_models(game, models_dir):
    """Load the most recent AI models if available with improved feedback"""
    print(f"Attempting to load AI models from: {models_dir}")
    
    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"ERROR: Directory {models_dir} does not exist!")
        return False
    
    # Check for model files directly in the specified directory
    drl_path = os.path.join(models_dir, "drl_agent.pt")
    ga_path = os.path.join(models_dir, "genetic_evolver.pkl")
    
    models_loaded = False

    # Check for training phase file
    phase_path = os.path.join(models_dir, "training_phase.txt")
    if os.path.exists(phase_path):
        try:
            with open(phase_path, "r") as f:
                phase = int(f.read().strip())
                if hasattr(game.wave_manager, 'set_training_phase'):
                    game.wave_manager.set_training_phase(phase)
                    print(f"âœ… Training phase {phase} loaded successfully")
        except Exception as e:
            print(f"âŒ Error loading training phase: {str(e)}")
    
    # Try to load DRL model
    if os.path.exists(drl_path):
        try:
            game.drl_agent.load_model(drl_path)
            print(f"âœ… DRL agent loaded successfully from {drl_path}")
            print(f"   Current epsilon: {game.drl_agent.epsilon:.4f}")
            models_loaded = True
        except Exception as e:
            print(f"âŒ Error loading DRL model: {str(e)}")
    else:
        print(f"âŒ DRL model not found at {drl_path}")
    
    # Try to load genetic algorithm state
    if os.path.exists(ga_path):
        try:
            game.genetic_evolver.load_state(ga_path)
            print(f"âœ… Genetic evolver loaded successfully from {ga_path}")
            if hasattr(game.genetic_evolver, "generations"):
                print(f"   Current generation: {game.genetic_evolver.generations}")
            models_loaded = True
        except Exception as e:
            print(f"âŒ Error loading genetic evolver: {str(e)}")
    else:
        print(f"âŒ Genetic evolver not found at {ga_path}")
    
    if models_loaded:
        print("âœ… Models loaded successfully - AI will use trained behavior")
        # Add visual indicator in the game
        if hasattr(game, "model_loaded"):
            game.model_loaded = True
    else:
        print("âš ï¸ No models were loaded - AI will use default behavior")
        
    return models_loaded
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate AI for Space Shooter Game")
    parser.add_argument("--mode", choices=["train", "eval", "play"], default="train",
                       help="Mode: train, eval, or play")
    parser.add_argument("--episodes", type=int, default=250, help="Number of training episodes")
    parser.add_argument("--render", action="store_true", help="Render game during training")
    parser.add_argument("--load", action="store_true", help="Load existing models")
    parser.add_argument("--save-interval", type=int, default=10, 
                       help="How often to save models (in episodes)")
    parser.add_argument("--eval-games", type=int, default=10, 
                       help="Number of games to run in evaluation mode")
    parser.add_argument("--wave-type", choices=["normal", "neural"], default="normal",
                       help="Wave type to use (normal or neural)")
    parser.add_argument("--modular-training", action="store_true", 
                       help="Enable modular training through phases")
    parser.add_argument("--models-dir", type=str, default="models",
                   help="Directory containing saved models")
    parser.add_argument("--model-path", type=str, default=None,
                    help="Path to model for continued training")
    parser.add_argument("--combat-only", action="store_true",
                   help="Only train on Combat phase (phase 4)")

    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ai_system(
            episodes=args.episodes,
            render=args.render,
            load_models=args.load,
            save_interval=args.save_interval,
            wave_type=args.wave_type,
            modular_training=args.modular_training,
            models_dir=args.models_dir,  # Pass the model directory
            combat_only=args.combat_only  # Pass the combat-only flag
        )
    elif args.mode == "eval":
        run_evaluation(num_games=args.eval_games)
    elif args.mode == "play":
        # Run the game with AI enabled but let player watch
        game = Game(ai_mode=True, wave_type=args.wave_type)
        if args.load:
            load_ai_models(game, args.models_dir)  # Use the parameter instead of hardcoded "models"
        game.run()
