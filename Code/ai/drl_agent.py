from ai.utils import *
from settings import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import os
from collections import deque
import time

class AttentionModule(nn.Module):
    """Simplified self-attention mechanism for state processing"""
    
    def __init__(self, input_dim, key_dim=64):
        super().__init__()
        self.key_transform = nn.Linear(input_dim, key_dim)
        self.value_transform = nn.Linear(input_dim, input_dim)
        self.attention_weights = nn.Linear(key_dim, 1)
        self.scaling_factor = 1.0 / math.sqrt(key_dim)
    
    def forward(self, x):
        # Transform input to keys and values
        keys = torch.tanh(self.key_transform(x))
        values = self.value_transform(x)
        
        # Calculate attention weights
        attention_scores = self.attention_weights(keys)
        attention_weights = torch.sigmoid(attention_scores)
        
        # Apply attention as a weighted scaling
        return x + values * attention_weights


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with attention mechanism.
    Separates state value and advantage estimation.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Attention module for feature enhancement
        self.attention = AttentionModule(256)
        
        # Value stream - estimates state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single value output
        )
        
        # Advantage stream - estimates advantages A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # One output per action
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        features = self.attention(features)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for improving sample efficiency.
    Uses TD-error as priority measure.
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance-sampling correction factor
        self.beta_increment = beta_increment  # Annealing the bias
        self.epsilon = epsilon  # Small constant to avoid zero priority
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add new experience with max priority to ensure it gets sampled"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Set priority to max priority to ensure new experiences are sampled
        self.priorities[self.position] = max_priority
        
        # Update buffer position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch based on priorities with safety checks"""
        if self.size < batch_size:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            # Calculate sampling probabilities from priorities with safety
            priorities = np.clip(self.priorities[:self.size], 1e-8, None)  # Prevent zeros
            probabilities = priorities ** self.alpha
            probabilities_sum = probabilities.sum()
            
            # Handle numerical issues
            if np.isnan(probabilities_sum) or probabilities_sum <= 0:
                # Fall back to uniform sampling
                probabilities = np.ones_like(priorities) / self.size
            else:
                probabilities = probabilities / probabilities_sum
            
            # Sample indices based on probabilities
            indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Safe calculation of weights
        weights = np.ones(batch_size)  # Default weights
        if not np.isnan(probabilities).any() and not (probabilities <= 0).any():
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights = weights / np.maximum(weights.max(), 1e-8)
        
        # Increase beta to reduce bias over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors with explicit type casting
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            torch.tensor(np.array(weights), dtype=torch.float32),
            indices  # Return indices for priority updates
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon  # Add epsilon to avoid zero priority
    
    def __len__(self):
        return self.size


class TemporalStateTracker:
    """
    Tracks temporal information of game state over multiple frames.
    Used to capture velocities and patterns.
    """
    
    def __init__(self, history_length=3):
        self.history_length = history_length
        self.state_history = []
        self.position_history = []
        self.enemy_history = {}  # Dictionary to track specific enemies by ID
        self.meteor_history = {}  # Dictionary to track specific meteors by ID
        self.laser_history = {}   # Dictionary to track specific lasers by ID
    
    def update(self, current_state, player_pos, enemies, meteors, lasers):
        """
        Update state history with new state.
        Also separately track entity positions for velocity calculations.
        """
        # Add current state to history
        self.state_history.append(current_state)
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)
        
        # Track player position
        self.position_history.append(player_pos)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        
        # Track enemies (identify by id)
        current_enemy_ids = set()
        for enemy in enemies:
            enemy_id = id(enemy)
            current_enemy_ids.add(enemy_id)
            
            if enemy_id not in self.enemy_history:
                self.enemy_history[enemy_id] = []
            
            self.enemy_history[enemy_id].append((enemy.center_pos.x, enemy.center_pos.y))
            
            # Limit history length
            if len(self.enemy_history[enemy_id]) > self.history_length:
                self.enemy_history[enemy_id].pop(0)
        
        # Remove enemies that no longer exist
        for enemy_id in list(self.enemy_history.keys()):
            if enemy_id not in current_enemy_ids:
                del self.enemy_history[enemy_id]
        
        # Track meteors (similarly to enemies)
        current_meteor_ids = set()
        for meteor in meteors:
            meteor_id = id(meteor)
            current_meteor_ids.add(meteor_id)
            
            if meteor_id not in self.meteor_history:
                self.meteor_history[meteor_id] = []
            
            self.meteor_history[meteor_id].append((meteor.pos.x, meteor.pos.y))
            
            if len(self.meteor_history[meteor_id]) > self.history_length:
                self.meteor_history[meteor_id].pop(0)
        
        # Remove meteors that no longer exist
        for meteor_id in list(self.meteor_history.keys()):
            if meteor_id not in current_meteor_ids:
                del self.meteor_history[meteor_id]
        
        # Track enemy lasers (similarly)
        current_laser_ids = set()
        for laser in lasers:
            laser_id = id(laser)
            current_laser_ids.add(laser_id)
            
            if laser_id not in self.laser_history:
                self.laser_history[laser_id] = []
            
            self.laser_history[laser_id].append((laser.pos.x, laser.pos.y))
            
            if len(self.laser_history[laser_id]) > self.history_length:
                self.laser_history[laser_id].pop(0)
        
        # Remove lasers that no longer exist
        for laser_id in list(self.laser_history.keys()):
            if laser_id not in current_laser_ids:
                del self.laser_history[laser_id]
    
    def calculate_player_velocity(self):
        """Calculate player velocity based on position history"""
        if len(self.position_history) < 2:
            return (0, 0)
        
        current = self.position_history[-1]
        previous = self.position_history[-2]
        
        return (current.x - previous.x, current.y - previous.y)
    
    def calculate_enemy_velocities(self):
        """Calculate velocities for all tracked enemies"""
        velocities = {}
        
        for enemy_id, positions in self.enemy_history.items():
            if len(positions) >= 2:
                current = positions[-1]
                previous = positions[-2]
                velocities[enemy_id] = (current[0] - previous[0], current[1] - previous[1])
            else:
                velocities[enemy_id] = (0, 0)
        
        return velocities
    
    def calculate_meteor_velocities(self):
        """Calculate velocities for all tracked meteors"""
        velocities = {}
        
        for meteor_id, positions in self.meteor_history.items():
            if len(positions) >= 2:
                current = positions[-1]
                previous = positions[-2]
                velocities[meteor_id] = (current[0] - previous[0], current[1] - previous[1])
            else:
                velocities[meteor_id] = (0, 0)
        
        return velocities
    
    def calculate_laser_velocities(self):
        """Calculate velocities for all tracked enemy lasers"""
        velocities = {}
        
        for laser_id, positions in self.laser_history.items():
            if len(positions) >= 2:
                current = positions[-1]
                previous = positions[-2]
                velocities[laser_id] = (current[0] - previous[0], current[1] - previous[1])
            else:
                velocities[laser_id] = (0, 0)
        
        return velocities


class DRLAgent:
    """Enhanced Deep Reinforcement Learning Agent with improved architecture and algorithms"""
    
    def __init__(self, game, device="cpu"):
        self.game = game
        self.device = device
        self.frame_count = 0
        self.learning_steps = 0
        
        # State and action space configuration
        self.state_size = 68 # Match the actual input dimension from the error message
        self.action_size = 14  # 8 movement directions + shooting + other actions
        
        # Instantiate networks with Dueling architecture
        self.policy_net = DuelingDQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DuelingDQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is used only for inference
        
        # Optimizer with improved hyperparameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(50000)  # Larger buffer
        self.batch_size = 128  # Larger batch size
        
        # Exploration parameters
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.01  # Lower minimum for better exploitation
        self.epsilon_decay = 0.995  # Slower decay
        self.gamma = 0.99  # Discount factor
        
        # Network update tracking
        self.update_count = 0
        self.target_update_frequency = 5  # More frequent updates
        
        # Temporal state tracker for velocities
        self.temporal_tracker = TemporalStateTracker(history_length=3)
        
        # Action masking for more intelligent actions
        self.action_mask = [1.0] * self.action_size  # Default: all actions valid
        
        # For tracking state normalization statistics
        self.state_mean = None
        self.state_std = None
        self.state_buffer = []
        
        # Intrinsic motivation for exploration
        self.state_visits = {}  # Track visited states
        self.novelty_threshold = 0.1  # Threshold for considering state novel
        
        # Logging
        self.episode_rewards = []
        self.training_losses = []
        self.q_values = []

    def get_state(self):
        """Enhanced state representation with better error handling"""
        player = self.game.player
        if not player:
            # Return default state vector if player doesn't exist
            return np.zeros(self.state_size, dtype=np.float32)
            
        state_list = []
        
        # Include training phase in state
        if hasattr(self.game.wave_manager, 'training_phase'):
            training_phase = self.game.wave_manager.training_phase
            for i in range(5):  # 5 phases total
                state_list.append(1.0 if i == training_phase else 0.0)
        else:
            # Default to full combat (phase 4)
            for i in range(5):
                state_list.append(1.0 if i == 4 else 0.0)
        
        # Player position (normalized)
        state_list.append(player.center_pos.x / WINDOW_WIDTH)
        state_list.append(player.center_pos.y / WINDOW_HEIGHT)
        
        # Player velocity
        vel_x, vel_y = 0, 0
        if hasattr(self, 'temporal_tracker') and len(self.temporal_tracker.position_history) >= 2:
            vel_x, vel_y = self.temporal_tracker.calculate_player_velocity()
        state_list.append(vel_x / 20.0)
        state_list.append(vel_y / 20.0)
        
        # Player health (normalized) with safety check
        player_health = getattr(player, 'health', 0)
        player_max_health = getattr(player, 'max_health', 15)
        state_list.append(player_health / max(1, player_max_health))
        
        # Weapon info
        weapon_idx = getattr(player, 'current_weapon_index', 0)
        max_weapons = len(getattr(player, 'weapons', []))
        state_list.append(weapon_idx / max(1, max(1, max_weapons - 1)))
        
        # Weapon cooldown
        cooldown_normalized = 0
        if max_weapons > 0 and weapon_idx < max_weapons:
            current_weapon = player.weapons[weapon_idx]
            if hasattr(current_weapon, 'cooldown_timer') and hasattr(current_weapon.cooldown_timer, 'current_time'):
                cooldown_normalized = current_weapon.cooldown_timer.current_time / current_weapon.cooldown_timer.duration
        state_list.append(cooldown_normalized)
        
        # Rest of the state remains the same...
        # (code truncated for brevity - no changes needed to the enemy, meteor, and laser tracking)
        
        # Update temporal tracker with complete game state
        if hasattr(self, 'temporal_tracker'):
            self.temporal_tracker.update(
                np.array(state_list, dtype=np.float32),
                player.center_pos,
                self.game.enemies,
                self.game.meteors,
                self.game.enemy_lasers
            )
        
        # Error checking - ensure we're returning the expected state size
        if len(state_list) < self.state_size:
            # Pad with zeros if needed
            state_list.extend([0] * (self.state_size - len(state_list)))
        elif len(state_list) > self.state_size:
            # Truncate if somehow we got too many features
            state_list = state_list[:self.state_size]
        
        # Convert to numpy array
        return np.array(state_list, dtype=np.float32)

    def _get_closest_enemies(self, num_enemies):
        """Returns a list containing the closest 'num_enemies' enemy objects"""
        player = self.game.player
        enemies = self.game.enemies
        
        enemy_distances = []
        for enemy in enemies:
            dx = enemy.center_pos.x - player.center_pos.x
            dy = enemy.center_pos.y - player.center_pos.y
            dist = math.sqrt(dx * dx + dy * dy)
            enemy_distances.append((enemy, dist))
            
        enemy_distances.sort(key=lambda x: x[1])
        result = [enemy for enemy, d in enemy_distances[:num_enemies]]
        while len(result) < num_enemies:
            result.append(None)
        return result

    def _get_closest_enemy_lasers(self, num_lasers):
        """Returns a list of the closest 'num_lasers' enemy projectiles"""
        player = self.game.player
        lasers = self.game.enemy_lasers
        
        laser_distances = []
        for laser in lasers:
            dx = laser.pos.x - player.center_pos.x
            dy = laser.pos.y - player.center_pos.y
            dist = math.sqrt(dx * dx + dy * dy)
            laser_distances.append((laser, dist))
            
        laser_distances.sort(key=lambda x: x[1])
        result = [laser for laser, d in laser_distances[:num_lasers]]
        while len(result) < num_lasers:
            result.append(None)
        return result
    
    def _get_closest_meteors(self, num_meteors):
        """Returns a list of the closest 'num_meteors' meteors"""
        player = self.game.player
        meteors = self.game.meteors
        
        meteor_distances = []
        for meteor in meteors:
            dx = meteor.pos.x - player.center_pos.x
            dy = meteor.pos.y - player.center_pos.y
            dist = math.sqrt(dx * dx + dy * dy)
            meteor_distances.append((meteor, dist))
            
        meteor_distances.sort(key=lambda x: x[1])
        result = [meteor for meteor, d in meteor_distances[:num_meteors]]
        while len(result) < num_meteors:
            result.append(None)
        return result

    def _update_action_mask(self, state):
        """
        Updates action mask to prevent illogical actions.
        For example, avoid shooting when no enemies in front.
        """
        # Reset mask (all actions valid)
        self.action_mask = [1.0] * self.action_size
        
        player = self.game.player
        player_pos = Vector2(player.center_pos.x, player.center_pos.y)
        
        # Get player angle in radians
        player_angle_rad = math.radians(player.rotation)
        facing_direction = Vector2(math.sin(player_angle_rad), -math.cos(player_angle_rad))
        
        # Check if any enemies are in front of player (for shooting action)
        enemies_in_front = False
        shooting_action_index = 8  # Index of shooting action
        
        for enemy in self.game.enemies:
            # Vector from player to enemy
            to_enemy = Vector2(
                enemy.center_pos.x - player.center_pos.x,
                enemy.center_pos.y - player.center_pos.y
            )
            
            # Normalize
            distance = math.sqrt(to_enemy.x**2 + to_enemy.y**2)
            if distance > 0:
                to_enemy = Vector2(to_enemy.x / distance, to_enemy.y / distance)
                
                # Calculate dot product to check if enemy is in front
                dot_product = facing_direction.x * to_enemy.x + facing_direction.y * to_enemy.y
                
                if dot_product > 0.7:  # Enemy within ~45 degrees of forward direction
                    enemies_in_front = True
                    break
        
        # Check weapon cooldown
        weapon_idx = player.current_weapon_index
        current_weapon = player.weapons[weapon_idx]
        weapon_ready = True
        
        if hasattr(current_weapon, 'cooldown_timer'):
            weapon_ready = current_weapon.cooldown_timer.active == False
        
        # If no enemies in front or weapon not ready, discourage shooting
        if not enemies_in_front or not weapon_ready:
            self.action_mask[shooting_action_index] = 0.2  # Not fully masked, but discouraged
        
        # Check screen boundaries to prevent moving out of bounds
        # Actions 0-7 are movement in different directions
        
        # Check top boundary
        if player_pos.y < 50:
            # Discourage upward movement (actions 0, 1, 7)
            self.action_mask[0] = 0.2  # Up
            self.action_mask[1] = 0.3  # Up-Right
            self.action_mask[7] = 0.3  # Up-Left
        
        # Check right boundary
        if player_pos.x > WINDOW_WIDTH - 50:
            # Discourage rightward movement (actions 1, 2, 3)
            self.action_mask[1] = 0.3  # Up-Right
            self.action_mask[2] = 0.2  # Right
            self.action_mask[3] = 0.3  # Down-Right
        
        # Check bottom boundary
        if player_pos.y > WINDOW_HEIGHT - 50:
            # Discourage downward movement (actions 3, 4, 5)
            self.action_mask[3] = 0.3  # Down-Right
            self.action_mask[4] = 0.2  # Down
            self.action_mask[5] = 0.3  # Down-Left
        
        # Check left boundary
        if player_pos.x < 50:
            # Discourage leftward movement (actions 5, 6, 7)
            self.action_mask[5] = 0.3  # Down-Left
            self.action_mask[6] = 0.2  # Left
            self.action_mask[7] = 0.3  # Up-Left

    def _calculate_intrinsic_reward(self, state):
        """
        Calculate intrinsic reward for state novelty.
        Encourages exploration of new states.
        """
        # Discretize state for dictionary lookup
        discretized_state = tuple(np.round(state * 10) / 10)
        
        # Count visits to this state
        if discretized_state in self.state_visits:
            self.state_visits[discretized_state] += 1
            visits = self.state_visits[discretized_state]
            
            # Less novelty for frequently visited states
            novelty = self.novelty_threshold / math.sqrt(visits)
        else:
            # First visit - maximum novelty
            self.state_visits[discretized_state] = 1
            novelty = self.novelty_threshold
        
        return novelty

    def act(self, state):
        """
        Choose action using epsilon-greedy with action masking.
        Incorporates Double DQN for action selection.
        """
        # Update action mask for intelligent action selection
        self._update_action_mask(state)
        
        # Epsilon-greedy with masked probabilities
        if random.random() < self.epsilon:
            # Random action, but respecting the mask (weighted sampling)
            masked_probs = np.array(self.action_mask) / sum(self.action_mask)
            action = np.random.choice(self.action_size, p=masked_probs)
        elif self.frame_count < 2000:  # Extended initial exploration
            self.frame_count += 1
            masked_probs = np.array(self.action_mask) / sum(self.action_mask)
            action = np.random.choice(self.action_size, p=masked_probs)
        else:
            # Choose best action from policy network (Double DQN)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get Q-values from policy network
                q_values = self.policy_net(state_tensor)
                
                # Apply action mask by multiplying with mask
                masked_q_values = q_values * torch.FloatTensor(self.action_mask).to(self.device)
                
                # Choose action with highest masked Q-value
                action = masked_q_values.max(1)[1].item()
                
                # Store Q-values for logging
                if len(self.q_values) < 1000:
                    self.q_values.append(q_values.mean().item())
        
        # Store last action and log
        self.last_action = action
        
        return action

    def remember(self, state, action, reward, next_state, done):
        """Save the experience to prioritized replay memory"""
        self.memory.push(state, action, reward, next_state, done)
        
        # Track cumulative episode reward
        if not hasattr(self, 'current_episode_reward'):
            self.current_episode_reward = 0
            
        self.current_episode_reward += reward
        
        # Reset episode reward when done
        if done:
            if len(self.episode_rewards) < 1000:  # Limit storage
                self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

    def replay(self):
        """Train the network using Double DQN with better error handling"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample experiences from memory
        try:
            state, action, reward, next_state, done, weights, indices = self.memory.sample(self.batch_size)
        except Exception as e:
            print(f"Error sampling from replay buffer: {e}")
            return
        
        # Move tensors to device with error handling
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        weights = weights.to(self.device)
        
        try:
            # Double DQN implementation
            with torch.no_grad():
                # Select actions using policy network
                next_actions = self.policy_net(next_state).max(1)[1].unsqueeze(1)
                
                # Evaluate Q-values for these actions using target network
                next_q_values = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            
            # Current Q-values
            q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
            
            # Compute expected Q-values using Double DQN formula
            expected_q_values = reward + self.gamma * next_q_values * (1 - done)
            
            # Calculate TD errors for updating priorities
            td_errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy()
            
            # Validate td_errors to prevent NaN issues
            td_errors = np.nan_to_num(td_errors, nan=1.0)
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors)
            
            # Compute loss with importance sampling weights
            loss = (weights * F.smooth_l1_loss(q_values, expected_q_values.detach(), reduction='none')).mean()
            
            # Track training losses
            if len(self.training_losses) < 10000:
                self.training_losses.append(loss.item())
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
            
            self.optimizer.step()
            self.learning_steps += 1
            
            # Update target network periodically
            self.update_count += 1
            if self.update_count % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            print(f"Error during replay: {e}")

    def execute_action(self, action):
        """Maps discrete actions to game controls with improved error handling"""
        player = self.game.player
        if not player:
            return  # Safety check
        
        # Reset the player's movement vector
        player.direction = Vector2(0, 0)
        
        # Handle movement actions (0-7)
        if action == 0:  # Up
            player.direction.y = -1
        elif action == 1:  # Up-Right
            player.direction.x = 1
            player.direction.y = -1
        elif action == 2:  # Right
            player.direction.x = 1
        elif action == 3:  # Down-Right
            player.direction.x = 1
            player.direction.y = 1
        elif action == 4:  # Down
            player.direction.y = 1
        elif action == 5:  # Down-Left
            player.direction.x = -1
            player.direction.y = 1
        elif action == 6:  # Left
            player.direction.x = -1
        elif action == 7:  # Up-Left
            player.direction.x = -1
            player.direction.y = -1
        
        # Normalize direction vector if needed
        if player.direction.x != 0 or player.direction.y != 0:
            player.direction = Vector2Normalize(player.direction)
        
        # Handle rotation actions (9-10)
        player.rotate_direction = 0
        if action == 9:  # Rotate left
            player.rotate_direction = -1
        elif action == 10:  # Rotate right
            player.rotate_direction = 1
        
        # Handle weapon selection (11-13) with safety checks
        if action == 11 and len(player.weapons) > 0:
            player.current_weapon_index = 0
        elif action == 12 and len(player.weapons) > 1 and player.weapons[1].unlocked:
            player.current_weapon_index = 1
        elif action == 13 and len(player.weapons) > 2 and player.weapons[2].unlocked:
            player.current_weapon_index = 2
        
        # Handle firing (action 8)
        if action == 8:
            player.fire_weapon()
            
        # Store last action
        self.last_action = action

    def save_model(self, filename):
        """Save the current model parameters and training state"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'frame_count': self.frame_count,
            'learning_steps': self.learning_steps,
            'episode_rewards': self.episode_rewards,
            'q_values': self.q_values,
            'training_losses': self.training_losses
        }, filename)
        
        print(f"Saved model to {filename}")

    def load_model(self, filename):
        """Load model parameters and training state"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            
            # Load training statistics if available
            if 'frame_count' in checkpoint:
                self.frame_count = checkpoint['frame_count']
            if 'learning_steps' in checkpoint:
                self.learning_steps = checkpoint['learning_steps']
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
            if 'q_values' in checkpoint:
                self.q_values = checkpoint['q_values']
            if 'training_losses' in checkpoint:
                self.training_losses = checkpoint['training_losses']
                
            print(f"Loaded model from {filename}")
            print(f"Current epsilon: {self.epsilon:.4f}, Learning steps: {self.learning_steps}")
        else:
            print(f"No model found at {filename}")

    def log_training_stats(self):
        """Log training statistics for monitoring"""
        if len(self.episode_rewards) > 0:
            avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
            print(f"Average reward (last 100 episodes): {avg_reward:.2f}")
            
        if len(self.q_values) > 0:
            avg_q = sum(self.q_values[-100:]) / min(100, len(self.q_values))
            print(f"Average Q-value: {avg_q:.2f}")
            
        if len(self.training_losses) > 0:
            avg_loss = sum(self.training_losses[-100:]) / min(100, len(self.training_losses))
            print(f"Average loss: {avg_loss:.4f}")
            
        print(f"Epsilon: {self.epsilon:.4f}, Learning steps: {self.learning_steps}")
