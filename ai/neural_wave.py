from ai.utils import *
from waves import WaveManager
from settings import *
class WaveManagerNN(nn.Module):
    """Neural network for predicting optimal enemy compositions"""
    
    def __init__(self, input_dim, output_dim):
        super(WaveManagerNN, self).__init__()
        # Input features: player stats, wave number, historical data
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output is probabilities for different enemy types
        return F.softmax(self.fc3(x), dim=1)


class NeuralWaveManager(WaveManager):
    """Enhanced wave manager with neural network-guided enemy generation"""
    
    def __init__(self, game, device="cpu"):
        super().__init__(game)
        self.device = device
        
        # Neural network for enemy composition
        self.input_dim = 14  # Player stats + game state + historical data
        self.output_dim = 5  # Probabilities for different enemy types
        self.nn_model = WaveManagerNN(self.input_dim, self.output_dim).to(device)
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=0.0005)
        self.movement_variance = 0.5  # Default mid value
        
        # Player behavior tracking
        self.max_history_length = 100
        self.player_positions = []
        self.player_actions = []
        self.player_weapon_usage = [0, 0, 0]  # Count for each weapon type
        self.player_avg_position = Vector2(WINDOW_WIDTH/2, WINDOW_HEIGHT/2)
        self.player_successful_evasions = 0
        self.player_hits_taken = 0
        
        # Wave performance metrics
        self.wave_completion_times = []
        self.enemy_survival_rates = []
        self.player_damage_per_wave = []
        
        # Current adaptation strategy
        self.adaptation_strategy = "balanced"  # balanced, aggressive, defensive, swarming
        self.difficulty_multiplier = 1.0
        self.current_enemy_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal starting weights
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.3  # Sometimes try random compositions
        
    def update_player_metrics(self, dt):
        """Track player behavior over time"""
        player = self.game.player
        
        # Track player position
        self.player_positions.append((player.center_pos.x, player.center_pos.y))
        if len(self.player_positions) > self.max_history_length:
            self.player_positions.pop(0)
        
        # Update average position
        if self.player_positions:
            avg_x = sum(pos[0] for pos in self.player_positions) / len(self.player_positions)
            avg_y = sum(pos[1] for pos in self.player_positions) / len(self.player_positions)
            self.player_avg_position = Vector2(avg_x, avg_y)
        
        # Track weapon usage
        if player.current_weapon_index < len(self.player_weapon_usage):
            self.player_weapon_usage[player.current_weapon_index] += dt
        
        # Calculate movement variance (how much the player moves around)
        if len(self.player_positions) >= 10:
            recent_positions = self.player_positions[-10:]
            x_values = [pos[0] for pos in recent_positions]
            y_values = [pos[1] for pos in recent_positions]
            self.movement_variance = (np.var(x_values) + np.var(y_values)) / (WINDOW_WIDTH * WINDOW_HEIGHT)
        else:
            self.movement_variance = 0.5  # Default mid value
        
    def analyze_player_strategy(self):
        """Analyze player behavior to identify their strategy"""
        # Analyze player positioning
        edge_preference = 0
        center_preference = 0
        
        for pos in self.player_positions:
            # Check if position is near edge
            dist_to_edge = min(pos[0], WINDOW_WIDTH - pos[0], pos[1], WINDOW_HEIGHT - pos[1])
            if dist_to_edge < 100:
                edge_preference += 1
            
            # Check if position is near center
            center_x, center_y = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
            dist_to_center = math.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)
            if dist_to_center < 200:
                center_preference += 1
        
        # Normalize preferences
        if self.player_positions:
            edge_preference /= len(self.player_positions)
            center_preference /= len(self.player_positions)
        
        # Calculate weapon preference
        total_weapon_usage = sum(self.player_weapon_usage)
        weapon_preference = [0, 0, 0]
        if total_weapon_usage > 0:
            weapon_preference = [usage / total_weapon_usage for usage in self.player_weapon_usage]
        
        # Calculate evasion skill
        if self.player_hits_taken + self.player_successful_evasions > 0:
            evasion_skill = self.player_successful_evasions / (self.player_hits_taken + self.player_successful_evasions)
        else:
            evasion_skill = 0.5  # Default
        
        # Determine player strategy
        if edge_preference > 0.6:
            player_strategy = "edge_hugger"
        elif center_preference > 0.6:
            player_strategy = "center_camper"
        elif self.movement_variance > 0.7:
            player_strategy = "erratic_mover"
        elif evasion_skill > 0.7:
            player_strategy = "defensive"
        elif weapon_preference[2] > 0.6:  # If using rapid fire a lot
            player_strategy = "aggressive"
        else:
            player_strategy = "balanced"
        
        return player_strategy
    
    def adapt_to_player(self, player_strategy):
        """Adapt wave generation strategy based on player behavior"""
        # Adjust enemy weights based on player strategy
        if player_strategy == "edge_hugger":
            # Use enemies that can track and follow the player
            self.current_enemy_weights = [0.1, 0.4, 0.1, 0.3, 0.1]  # More swarm and sniper
            self.adaptation_strategy = "flanking"
            
        elif player_strategy == "center_camper":
            # Use enemies that can attack from multiple directions
            self.current_enemy_weights = [0.1, 0.3, 0.2, 0.1, 0.3]  # More swarm and bomber
            self.adaptation_strategy = "surrounding"
            
        elif player_strategy == "erratic_mover":
            # Use enemies with predictive targeting
            self.current_enemy_weights = [0.2, 0.1, 0.1, 0.5, 0.1]  # More snipers
            self.adaptation_strategy = "predictive"
            
        elif player_strategy == "defensive":
            # Use enemies that can overwhelm defensive play
            self.current_enemy_weights = [0.1, 0.1, 0.5, 0.1, 0.2]  # More tanks
            self.adaptation_strategy = "overwhelming"
            
        elif player_strategy == "aggressive":
            # Use enemies that can counter aggressive play
            self.current_enemy_weights = [0.3, 0.1, 0.3, 0.2, 0.1]  # More normal and tank
            self.adaptation_strategy = "countering"
            
        else:  # balanced
            # Use a mix of enemies
            self.current_enemy_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
            self.adaptation_strategy = "balanced"
    
    def get_enemy_composition_features(self):
        """Generate input features for the neural network"""
        player = self.game.player
        features = []
        
        # Player stats
        features.append(player.health / 15)  # Normalized health
        features.append(player.current_weapon_index / max(1, len(player.weapons) - 1))
        
        # Player position
        features.append(player.center_pos.x / WINDOW_WIDTH)
        features.append(player.center_pos.y / WINDOW_HEIGHT)
        
        # Game state
        features.append(self.current_wave / 10)  # Normalize wave number
        features.append(self.enemies_remaining / max(1, self.total_enemies_to_spawn))
        features.append(len(self.game.enemy_lasers) / 20)  # Normalize projectile count
        
        # Player behavior metrics
        features.append(self.movement_variance)
        
        # Historical performance
        if self.wave_completion_times:
            features.append(min(1.0, self.wave_completion_times[-1] / 60))  # Normalize to 0-1
        else:
            features.append(0.5)  # Default
            
        if self.enemy_survival_rates:
            features.append(self.enemy_survival_rates[-1])
        else:
            features.append(0.5)  # Default
            
        if self.player_damage_per_wave:
            features.append(min(1.0, self.player_damage_per_wave[-1] / 10))
        else:
            features.append(0.5)  # Default
        
        # Weapon usage preferences
        for weapon_usage in self.player_weapon_usage:
            features.append(min(1.0, weapon_usage / 10))
        
        return np.array(features, dtype=np.float32)
    
    def predict_enemy_composition(self):
        """Use neural network to predict optimal enemy composition"""
        # Use exploration sometimes to try new compositions
        if random.random() < self.exploration_rate:
            # Random exploration with some bias toward current weights
            weights = [(w + random.random()) / 2 for w in self.current_enemy_weights]
            # Normalize
            total = sum(weights)
            return [w / total for w in weights]
        
        # Get features for the neural network
        features = self.get_enemy_composition_features()
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict enemy composition
        with torch.no_grad():
            enemy_probs = self.nn_model(features_tensor).squeeze(0).cpu().numpy()
        
        return enemy_probs
    
    def create_wave_enemy(self):
        """Create an enemy with neural network-guided attributes"""
        # Get enemy type probability distribution
        enemy_probs = self.predict_enemy_composition()
        
        # Choose enemy type based on probabilities
        enemy_types = ["normal", "swarm", "tank", "sniper", "bomber"]
        enemy_type = random.choices(enemy_types, weights=enemy_probs)[0]
        
        # Create the enemy using the factory
        enemy = self.game.enemy_factory.create_enemy(enemy_type)
        
        # Apply difficulty multiplier to enemy attributes
        if enemy:
            # Scale health with difficulty
            enemy.health = max(1, int(enemy.health * self.difficulty_multiplier))
            
            # Scale speed with difficulty (carefully to not make it impossible)
            speed_factor = 1.0 + (self.difficulty_multiplier - 1.0) * 0.3
            enemy.speed *= speed_factor
            
            # Adapt enemy behavior based on adaptation strategy
            if self.adaptation_strategy == "flanking":
                # Enemies try to get around to the sides of the player
                if enemy_type in ["swarm", "sniper"]:
                    enemy.movement_pattern = 1  # More zigzag movement
                
            elif self.adaptation_strategy == "surrounding":
                # Enemies try to surround the player
                if enemy_type == "bomber":
                    enemy.movement_pattern = 2  # More swooping movement
                
            elif self.adaptation_strategy == "predictive":
                # Enemies with better targeting
                if enemy_type == "sniper":
                    if hasattr(enemy, 'shoot_interval'):
                        enemy.shoot_interval *= 0.9  # Shoot more frequently
                
            elif self.adaptation_strategy == "overwhelming":
                # Tankier enemies
                if enemy_type == "tank":
                    enemy.health += 1  # Extra health
                
            elif self.adaptation_strategy == "countering":
                # Faster enemies
                enemy.speed *= 1.1  # Slightly faster
        
        return enemy
    
    def wave_completed(self):
        """Record metrics when a wave is completed"""
        current_time = get_time()
        if hasattr(self, 'wave_start_time'):
            completion_time = current_time - self.wave_start_time
            self.wave_completion_times.append(completion_time)
            
            # Calculate enemy survival rate (inverted, lower is better for player)
            if self.total_enemies_to_spawn > 0:
                survival_rate = 1.0 - (self.enemies_spawned - self.enemies_remaining) / self.total_enemies_to_spawn
                self.enemy_survival_rates.append(survival_rate)
            else:
                self.enemy_survival_rates.append(0.0)
                
            # Store player damage this wave (initialized in start_next_wave)
            self.player_damage_per_wave.append(self.player_damage_this_wave)
        
        # Use metrics to update neural network
        self.update_nn_model()
        
        # Adjust difficulty based on player performance
        self.adjust_difficulty()
    
    def start_next_wave(self):
        """Start the next wave with enhanced difficulty"""
        # Record wave start time
        self.wave_start_time = get_time()
        self.player_damage_this_wave = 0
        
        # Analyze player strategy and adapt
        player_strategy = self.analyze_player_strategy()
        self.adapt_to_player(player_strategy)
        
        # Continue with regular wave starting logic
        super().start_next_wave()
    
    def update_nn_model(self):
        """Update neural network based on wave performance"""
        # Only update if we have enough data
        if len(self.wave_completion_times) < 2:
            return
        
        # Calculate reward: faster completion time = better enemy composition
        last_time = self.wave_completion_times[-1]
        avg_time = sum(self.wave_completion_times[:-1]) / len(self.wave_completion_times[:-1])
        
        # If last wave was completed faster than average, penalize the model
        # (we want waves to last longer)
        reward = -1.0 if last_time < avg_time else 1.0
        
        # Also factor in damage dealt to player (more damage = better enemy composition)
        if len(self.player_damage_per_wave) >= 2:
            last_damage = self.player_damage_per_wave[-1]
            avg_damage = sum(self.player_damage_per_wave[:-1]) / len(self.player_damage_per_wave[:-1])
            damage_reward = 1.0 if last_damage > avg_damage else -0.5
            reward += damage_reward
        
        # Get the features from the last wave
        features = self.get_enemy_composition_features()
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Target is to adjust current weights based on reward
        target = torch.FloatTensor(self.current_enemy_weights).unsqueeze(0).to(self.device)
        
        # If reward is positive, encourage current distribution
        # If reward is negative, encourage exploration away from current distribution
        if reward < 0:
            # Create a more uniform distribution as target
            target = target * 0.7 + torch.ones_like(target) * 0.3 / self.output_dim
        
        # Train the model
        self.optimizer.zero_grad()
        output = self.nn_model(features_tensor)
        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
    
    def adjust_difficulty(self):
        """Adjust difficulty based on player performance"""
        # Calculate player success rate
        if len(self.wave_completion_times) >= 3:
            # Look at recent completion times trend
            recent_times = self.wave_completion_times[-3:]
            
            # If completion times are decreasing, increase difficulty
            if recent_times[0] > recent_times[1] > recent_times[2]:
                self.difficulty_multiplier = min(2.0, self.difficulty_multiplier * 1.1)
                print(f"Increasing difficulty to {self.difficulty_multiplier:.2f}")
            
            # If completion times are very long, decrease difficulty slightly
            elif recent_times[-1] > 60:  # If last wave took over a minute
                self.difficulty_multiplier = max(1.0, self.difficulty_multiplier * 0.95)
                print(f"Decreasing difficulty to {self.difficulty_multiplier:.2f}")
        
        # Also consider health lost
        if len(self.player_damage_per_wave) >= 2:
            recent_damage = self.player_damage_per_wave[-2:]
            
            # If player is taking too much damage, ease off slightly
            if sum(recent_damage) > 10:  # Lost more than 10 health in last 2 waves
                self.difficulty_multiplier = max(1.0, self.difficulty_multiplier * 0.9)
                print(f"Reducing difficulty due to high damage to {self.difficulty_multiplier:.2f}")
            
            # If player isn't taking damage, increase difficulty
            elif sum(recent_damage) == 0:
                self.difficulty_multiplier = min(2.0, self.difficulty_multiplier * 1.15)
                print(f"Increasing difficulty due to no damage to {self.difficulty_multiplier:.2f}")
    
    def update(self, dt):
        """Update the wave manager with enhanced tracking"""
        super().update(dt)
        
        # Update player metrics
        self.update_player_metrics(dt)
        
        # Debug information
        if hasattr(self, 'debug_timer'):
            self.debug_timer += dt
            if self.debug_timer >= 5.0:  # Print debug info every 5 seconds
                self.debug_timer = 0
                player_strategy = self.analyze_player_strategy()
                print(f"Current wave: {self.current_wave}, Strategy: {self.adaptation_strategy}")
                print(f"Player strategy: {player_strategy}, Movement variance: {self.movement_variance:.2f}")
                print(f"Difficulty: {self.difficulty_multiplier:.2f}")
                weights_str = ", ".join([f"{w:.2f}" for w in self.current_enemy_weights])
                print(f"Enemy weights: [{weights_str}]")
        else:
            self.debug_timer = 0
    
    def save_model(self, filename):
        """Save neural network model to file"""
        torch.save({
            'model': self.nn_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'weights': self.current_enemy_weights,
            'difficulty': self.difficulty_multiplier
        }, filename)
        
        # Also save historical data
        with open(f"{filename}_history.pkl", "wb") as f:
            pickle.dump({
                'wave_completion_times': self.wave_completion_times,
                'enemy_survival_rates': self.enemy_survival_rates,
                'player_damage_per_wave': self.player_damage_per_wave
            }, f)
    
    def load_model(self, filename):
        """Load neural network model from file"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.nn_model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_enemy_weights = checkpoint['weights']
            self.difficulty_multiplier = checkpoint['difficulty']
            
            # Load historical data if available
            history_file = f"{filename}_history.pkl"
            if os.path.isfile(history_file):
                with open(history_file, "rb") as f:
                    history = pickle.load(f)
                    self.wave_completion_times = history['wave_completion_times']
                    self.enemy_survival_rates = history['enemy_survival_rates']
                    self.player_damage_per_wave = history['player_damage_per_wave']
            
            print(f"Loaded wave manager model from {filename}")
        else:
            print(f"No wave manager model found at {filename}")
