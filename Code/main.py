from pyray import *
from raylib import *
from random import randint, uniform
from os.path import join
from math import sqrt, inf
import math
import time
from collections import deque
import os
from datetime import datetime
import torch
import numpy as np
from custom_timer import Timer
from waves import *
from ai.drl_agent import DRLAgent
from ai.neural_wave import NeuralWaveManager
from ai.genetic import GeneticEnemyEvolver

# Constants from your setting.py
WINDOW_WIDTH, WINDOW_HEIGHT = 1800, 980
ENEMY_SPEED = 150
ENEMY_MAX_HEALTH = 3
ENEMY_SPAWN_INTERVAL = 3
BG_COLOR = (15, 10, 25, 255)
PLAYER_SPEED = 500
LASER_SPEED = 600
METEOR_SPEED_RANGE = [300, 400]
METEOR_TIMER_DURATION = 0.4
FONT_SIZE = 120
SAVE_MODEL_DIR = "models"

# Create models directory if it doesn't exist
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

class Game:
    def __init__(self, ai_mode=False,wave_type="normal",phased_training=False):
        init_window(WINDOW_WIDTH, WINDOW_HEIGHT, 'Space shooter')
        init_audio_device()
        self.score = 0
        self.import_assets()

        # Initialize the use of AI
        self.ai_mode = ai_mode
        
        # Add game timer display
        self.game_timer = Timer(inf, True, True)  # inf-second timer that repeats
        
        # Add game state tracking
        self.game_over = False
        self.game_over_timer = 0
        self.wave_completed = False
        
        # Frame counter for periodic actions
        self.frame_count = 0
        
        # For training metrics
        self.enemies_destroyed_this_frame = 0
        self.player_took_damage_this_frame = False
        self.prev_wave_active = False

        self.assets['health'] = load_texture(join('images', 'spaceship.png'))  # Health icon

        # Game object collections
        self.lasers, self.meteors, self.explosions = [], [], []
        self.enemy_beams = []
        self.enemies, self.enemy_lasers = [], []  # Enemy collections
        
        # Timers
        self.meteor_timer = Timer(METEOR_TIMER_DURATION, True, True, self.create_meteor)
        
        # Player setup
        self.player = Player(
            self.assets['player'], 
            Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2), 
            self.shoot_laser
        )
        
        # Music
        play_music_stream(self.audio['music'])
        
        # Initialize weapons
        self.player.weapons = [
            Weapon(self),          # Standard laser
            TripleShot(self),      # Triple shot 
            RapidFire(self)        # Rapid fire
        ]
        self.current_reward = 0.0
        self.reward_history = deque(maxlen=60)  # Track last 60 frames (~1 second)
        self.total_reward = 0.0
        self.show_rewards = True  # Add a toggle option
        self.model_loaded = False # Flag to check if model is loaded


        # Initialize AI or standard components based on mode
        if self.ai_mode:
            # Initialize DRL agent for player control
            self.drl_agent = DRLAgent(self, device=self._get_device())
            
            # Initialize neural wave manager (instead of standard)
            self.wave_type = wave_type
            if wave_type == "normal":
                self.wave_manager = WaveManager(self,phased_training=phased_training)
            elif wave_type == "neural":
                self.wave_manager = NeuralWaveManager(self, device=self._get_device(),phased_training=phased_training)
            else:
                raise ValueError(f"Unknown wave type: {wave_type}")
                
            # Initialize genetic algorithm for enemy evolution
            self.genetic_evolver = GeneticEnemyEvolver(self)
            
            # For tracking damage for genetic algorithm
            self.enemy_damage_tracker = {}  # enemy_id -> damage dealt to player
            self.enemy_lifetime_tracker = {}  # enemy_id -> time alive
            self.enemy_shot_tracker = {}  # enemy_id -> [shots_fired, shots_hit]
            
             # Enemy factory depends on wave type
            if wave_type == "normal":
                self.enemy_factory = EnemyFactory(self)
            else:
                self.enemy_factory = EnhancedEnemyFactory(self)
        else:
            # Standard game components
            self.wave_manager = WaveManager(self)
            self.enemy_factory = EnemyFactory(self)
        
        # Setup enemy spawn timer (controlled by wave manager)
        self.enemy_timer = Timer(ENEMY_SPAWN_INTERVAL, True, False, self.spawn_wave_enemy)
        
        # Start the first wave
        self.wave_manager.start_next_wave()

        #Logging method
        self.previous_position = Vector2(self.player.center_pos.x, self.player.center_pos.y)  # For movement tracking
        self.previous_rotation = self.player.rotation  # For rotation tracking

    def _get_device(self):
        """Determine the best device for PyTorch to use"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return "mps"  # For Apple Silicon
        else:
            return "cpu"

    def import_assets(self):
        self.assets = {
        'player': load_texture(join('images', 'spaceship.png')),
        'star': load_texture(join('images', 'star.png')),
        'laser': load_texture(join('images', 'laser.png')),
        'meteor': load_texture(join('images', 'meteor.png')),
        'explosion': [load_texture(join('images', 'explosion', f'{i}.png')) for i in range(1,29)],
        'font': load_font_ex(join('font', 'Stormfaze.otf'), FONT_SIZE, ffi.NULL, 0),
        
        # Add different enemy textures with meaningful names
        'enemy': load_texture(join('images','Enemies','enemyBlack1.png')),  # Default enemy
        'enemy_tank': load_texture(join('images','Enemies','enemyBlack4.png')),  # Tank enemy
        'enemy_swarm': load_texture(join('images','Enemies','enemyBlack2.png')),  # Swarm enemy  
        'enemy_sniper': load_texture(join('images','Enemies','enemyRed3.png')),  # Sniper enemy
        'enemy_bomber': load_texture(join('images','Enemies','enemyGreen5.png')),  # Bomber enemy
        
        'enemy_laser': load_texture(join('images','Lasers','laserBlue01.png')),
        'cannon_laser': load_texture(join('images','Lasers','laserBlue11.png'))
        }

        self.audio = {
            'laser': load_sound(join('audio','laser.wav')),
            'explosion': load_sound(join('audio','explosion.wav')),
            'music': load_music_stream(join('audio', 'music.wav')),
            # Optionally add enemy sound effects
            'enemy_laser': load_sound(join('audio','laser.wav'))  # Can reuse or add new sound
        }

        self.star_data = [
            (
                Vector2(randint(0, WINDOW_WIDTH),randint(0, WINDOW_HEIGHT)), # pos
                uniform(0.5, 1.6) # size
            ) for i in range(30)
        ]
    
    def draw_stars(self):
        for star in self.star_data:
            draw_texture_ex(self.assets['star'],star[0], 0, star[1], WHITE)

    def draw_timer(self):
        # Check if timer is active and has a valid start time
        if self.game_timer.active and self.game_timer.start_time > 0:
            current_time = get_time()
            
            # Calculate elapsed time instead of remaining time
            if current_time >= self.game_timer.start_time:
                elapsed_time = current_time - self.game_timer.start_time
                
                # Format elapsed time as MM:SS
                total_seconds = int(elapsed_time)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                time_text = f"{minutes:02d}:{seconds:02d}"
                
                # Measure text size for positioning
                text_size = measure_text_ex(self.assets['font'], time_text, FONT_SIZE, 0)
                
                # Position in top right with some padding
                position = Vector2(WINDOW_WIDTH - text_size.x - 50, 20)
                
                # Draw the text
                draw_text_ex(self.assets['font'], time_text, position, FONT_SIZE, 0, WHITE)
            else:
                # Fallback display if timing is invalid
                draw_text_ex(self.assets['font'], "00:00", Vector2(WINDOW_WIDTH - 200, 20), FONT_SIZE, 0, WHITE)
        else:
            # Default display when timer is inactive
            draw_text_ex(self.assets['font'], "00:00", Vector2(WINDOW_WIDTH - 100, 20), FONT_SIZE, 0, WHITE)

    def check_weapon_unlocks(self):
        """Check if weapons should be unlocked based on wave progress"""
        current_wave = self.wave_manager.current_wave
        
        # Unlock Triple Shot at wave 2
        if current_wave >= 2 and not self.player.weapons[1].unlocked:
            self.player.weapons[1].unlocked = True
            print(f"Weapon Unlocked: {self.player.weapons[1].name}")
            
        # Unlock Rapid Fire at wave 3
        if current_wave >= 3 and not self.player.weapons[2].unlocked:
            self.player.weapons[2].unlocked = True
            print(f"Weapon Unlocked: {self.player.weapons[2].name}")
    
    def create_laser(self, pos, direction):
        """Helper method to create a laser for weapons system"""
        # Create a laser with the provided direction
        return Laser(self.assets['laser'], pos, direction)

    def shoot_laser(self, pos, direction=Vector2(0,-1)):
        # Modified to accept a direction parameter with default
        self.lasers.append(Laser(self.assets['laser'], pos, direction))
        play_sound(self.audio['laser'])

    def shoot_enemy_laser(self, texture, pos, direction, speed_multiplier=1.0, scale=1.0, damage=1):
        """Enhanced method to handle special laser properties"""
        # Create the laser with the additional parameters
        laser = EnemyLaser(texture, pos, direction, speed_multiplier, scale, damage)
        
        # For AI mode, track which enemy fired this laser
        if self.ai_mode and hasattr(direction, 'source_enemy'):
            laser.source_enemy = direction.source_enemy
            
            # Track firing for accuracy metrics
            enemy_id = id(direction.source_enemy)
            if enemy_id in self.enemy_shot_tracker:
                self.enemy_shot_tracker[enemy_id][0] += 1  # Increment shots fired
                
        self.enemy_lasers.append(laser)
        play_sound(self.audio['enemy_laser'])
    
    def create_meteor(self):
        self.meteors.append(Meteor(self.assets['meteor']))
        
    def create_base_enemy(self):
        """Create a base enemy without adding it to the game yet"""
        enemy = Enemy(self.assets['enemy'], self.shoot_enemy_laser, self.assets['enemy_laser'])
        enemy.target_player = self.player
        return enemy
    
    def spawn_wave_enemy(self):
        """Spawn an enemy controlled by the wave manager"""
        if self.wave_manager.wave_active:
            enemy = self.wave_manager.spawn_enemy()
            
            # Apply genetic enhancements if AI mode is enabled
            if enemy and self.ai_mode:
                # Get enemy type
                enemy_type = getattr(enemy, 'debug_type', 'normal').lower()
                
                # Select a genome for this enemy
                genome_index = self.genetic_evolver.select_genome_for_enemy(enemy_type)
                
                # Apply genetic parameters
                enemy = self.genetic_evolver.apply_genome_to_enemy(enemy, genome_index)
                
                # Initialize tracking for this enemy
                enemy_id = id(enemy)
                self.enemy_damage_tracker[enemy_id] = 0
                self.enemy_lifetime_tracker[enemy_id] = 0
                self.enemy_shot_tracker[enemy_id] = [0, 0]  # [shots_fired, shots_hit]
            
            if enemy:
                self.enemies.append(enemy)
    
    def switch_wave_type(self, new_wave_type):
        """Switch between normal and neural wave managers with improved state preservation"""
        if not self.ai_mode:
            print("Cannot switch wave type in non-AI mode")
            return
            
        if new_wave_type == self.wave_type:
            print(f"Already using {new_wave_type} wave manager")
            return
            
        print(f"Switching from {self.wave_type} to {new_wave_type} wave manager")
        
        # Save current wave state
        current_wave = self.wave_manager.current_wave
        training_phase = getattr(self.wave_manager, 'training_phase', 4)
        wave_active = self.wave_manager.wave_active
        enemies_remaining = self.wave_manager.enemies_remaining
        
        # Initialize new wave manager
        if new_wave_type == "normal":
            self.wave_manager = WaveManager(self)
            self.enemy_factory = EnemyFactory(self)
        elif new_wave_type == "neural":
            self.wave_manager = NeuralWaveManager(self, device=self._get_device())
            self.enemy_factory = EnhancedEnemyFactory(self)
        
        # Restore wave state
        self.wave_manager.current_wave = current_wave
        if hasattr(self.wave_manager, 'training_phase'):
            self.wave_manager.training_phase = training_phase
        self.wave_manager.wave_active = wave_active
        self.wave_manager.enemies_remaining = enemies_remaining
        self.wave_type = new_wave_type
        
        print(f"Wave state preserved: Wave {current_wave}, Active: {wave_active}, Enemies: {enemies_remaining}")
    def discard_sprites(self):
        self.lasers = [laser for laser in self.lasers if not laser.discard]
        self.meteors = [meteor for meteor in self.meteors if not meteor.discard]
        self.explosions = [explosion for explosion in self.explosions if not explosion.discard]
        # Add cleanup for enemy ships and lasers
        self.enemies = [enemy for enemy in self.enemies if not enemy.discard]
        self.enemy_lasers = [laser for laser in self.enemy_lasers if not laser.discard]

    def draw_health(self):
        # Draw health indicators in the top-left corner
        for i in range(self.player.health):
            draw_texture_v(
                self.assets['health'],
                Vector2(20 + i * 40, 20),  # Position health icons with spacing
                WHITE
            )
    
    def draw_weapon_info(self):
        """Draw current weapon name"""
        if not self.player.weapons:
            return
            
        weapon = self.player.weapons[self.player.current_weapon_index]
        text = f"WEAPON: {weapon.name}"
        
        # Use a smaller font size
        small_font_size = FONT_SIZE // 4
        
        draw_text_ex(
            self.assets['font'],
            text,
            Vector2(20, WINDOW_HEIGHT - 40),
            small_font_size,
            0,
            WHITE
        )

    def _draw_ai_status(self):
        """Draw AI status indicators"""
        # Draw DRL agent status
        ai_text = f"AI: {'ON' if self.ai_mode else 'OFF'}"
        agent_text = f"Epsilon: {self.drl_agent.epsilon:.2f}"
        
        # Use a smaller font size
        small_font_size = FONT_SIZE // 4
        
        # Draw at top-right corner with padding
        draw_text_ex(
            self.assets['font'],
            ai_text,
            Vector2(WINDOW_WIDTH - 150, 60),
            small_font_size,
            0,
            GREEN if self.ai_mode else RED
        )
        
        draw_text_ex(
            self.assets['font'],
            agent_text,
            Vector2(WINDOW_WIDTH - 150, 60 + small_font_size + 5),
            small_font_size,
            0,
            WHITE
        )
        
        # Draw wave manager adaptation strategy if available
        if hasattr(self.wave_manager, 'adaptation_strategy'):
            strategy_text = f"Strategy: {self.wave_manager.adaptation_strategy}"
            difficulty_text = f"Difficulty: {self.wave_manager.difficulty_multiplier:.2f}x"
            
            draw_text_ex(
                self.assets['font'],
                strategy_text,
                Vector2(WINDOW_WIDTH - 150, 60 + (small_font_size + 5) * 2),
                small_font_size,
                0,
                YELLOW
            )
            
            draw_text_ex(
                self.assets['font'],
                difficulty_text,
                Vector2(WINDOW_WIDTH - 150, 60 + (small_font_size + 5) * 3),
                small_font_size,
                0,
                ORANGE
            )
        
        # Draw genetic algorithm status
        if hasattr(self, 'genetic_evolver'):
            gen_text = f"Gen: {self.genetic_evolver.generations}"
            draw_text_ex(
                self.assets['font'],
                gen_text,
                Vector2(WINDOW_WIDTH - 150, 60 + (small_font_size + 5) * 4),
                small_font_size,
                0,
                BLUE
            )

    def check_beam_collision(self, beam):
        """Check if a beam hits the player and deal damage"""
        if self.player and not self.player.invulnerable:
            hit = False
            player_center = self.player.get_center()
            
            # Line-circle intersection for beam collision
            # We'll check if player is near any point along the beam
            start_to_player = Vector2(
                player_center.x - beam.start_pos.x,
                player_center.y - beam.start_pos.y
            )
            
            beam_direction = Vector2(
                beam.end_pos.x - beam.start_pos.x,
                beam.end_pos.y - beam.start_pos.y
            )
            
            beam_length = sqrt(beam_direction.x**2 + beam_direction.y**2)
            
            if beam_length > 0:
                # Normalize beam direction
                beam_direction.x /= beam_length
                beam_direction.y /= beam_length
                
                # Project player position onto beam
                projection = start_to_player.x * beam_direction.x + start_to_player.y * beam_direction.y
                
                # Clamp projection to beam length
                projection = max(0, min(projection, beam_length))
                
                # Calculate closest point on beam to player
                closest_point = Vector2(
                    beam.start_pos.x + beam_direction.x * projection,
                    beam.start_pos.y + beam_direction.y * projection
                )
                
                # Check distance from player to closest point
                distance = sqrt(
                    (player_center.x - closest_point.x)**2 + 
                    (player_center.y - closest_point.y)**2
                )
                
                # Check if player is hit
                if distance <= self.player.collision_radius:
                    hit = True
            
            if hit and self.player.take_damage():
                # Create explosion at impact point with player
                impact_pos = player_center
                self.explosions.append(ExplosionAnimation(impact_pos, self.assets['explosion']))
                play_sound(self.audio['explosion'])
                
                # For AI mode, track the hit
                if self.ai_mode:
                    self.player_took_damage_this_frame = True
                
                # Check for game over
                if self.player.health <= 0:
                    self.game_over = True

    def check_collisions(self):
        # lasers and enemies
        for laser in self.lasers:
            for enemy in self.enemies:
                if check_collision_circle_rec(enemy.get_center(), enemy.collision_radius, laser.get_rect()):
                    laser.discard = True
                    
                    # Check if enemy is destroyed
                    if enemy.take_damage():
                        enemy.discard = True
                        
                        # Create explosion
                        pos = Vector2(enemy.pos.x + enemy.size.x / 2, enemy.pos.y + enemy.size.y / 2)
                        self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                        play_sound(self.audio['explosion'])
                        
                        # Add to score - remaining health is a multiplier
                        # For tougher enemies that had more starting health
                        if hasattr(enemy, 'health'):
                            self.score += 100  # Base score for any enemy
                        else:
                            self.score += 50   # Fallback if health attribute is missing
                        
                        # For AI mode, update genetic algorithm with enemy performance
                        if self.ai_mode:
                            enemy_id = id(enemy)
                            if enemy_id in self.enemy_lifetime_tracker:
                                lifetime = self.enemy_lifetime_tracker[enemy_id]
                                damage_dealt = self.enemy_damage_tracker.get(enemy_id, 0)
                                shots_fired, shots_hit = self.enemy_shot_tracker.get(enemy_id, [0, 0])
                                
                                # Update fitness in genetic algorithm
                                self.genetic_evolver.update_fitness(
                                    enemy, lifetime, damage_dealt, shots_fired, shots_hit
                                )
                                
                                # Clean up tracking
                                if enemy_id in self.enemy_lifetime_tracker:
                                    del self.enemy_lifetime_tracker[enemy_id]
                                if enemy_id in self.enemy_damage_tracker:
                                    del self.enemy_damage_tracker[enemy_id]
                                if enemy_id in self.enemy_shot_tracker:
                                    del self.enemy_shot_tracker[enemy_id]
                            
                            # Track enemy destruction for reward calculation
                            self.enemies_destroyed_this_frame += 1
                        
                        # Notify wave manager
                        self.wave_manager.enemy_destroyed()
        
        # lasers and meteors
        for laser in self.lasers:
            for meteor in self.meteors:
                if check_collision_circle_rec(meteor.get_center(), meteor.collision_radius, laser.get_rect()):
                    laser.discard, meteor.discard = True, True
                    pos = Vector2(laser.pos.x - laser.size.x / 2, laser.pos.y)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])

        # Player and enemy lasers
        for enemy_laser in self.enemy_lasers:
            if check_collision_circle_rec(self.player.get_center(), self.player.collision_radius, 
                                        enemy_laser.get_rect()):
                if self.player.take_damage():  # Only deal damage if not invulnerable
                    # Create explosion effect at the hit location
                    pos = Vector2(enemy_laser.pos.x, enemy_laser.pos.y)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])
                    
                    # For AI mode, track the hit
                    if self.ai_mode:
                        self.player_took_damage_this_frame = True
                        
                        # Update damage tracking for genetic algorithm
                        if hasattr(enemy_laser, 'source_enemy'):
                            enemy_id = id(enemy_laser.source_enemy)
                            if enemy_id in self.enemy_damage_tracker:
                                self.enemy_damage_tracker[enemy_id] += 1
                                # Also track hit
                                if enemy_id in self.enemy_shot_tracker:
                                    self.enemy_shot_tracker[enemy_id][1] += 1
                    
                    # Remove enemy laser
                    enemy_laser.discard = True
                    
                    # Check for game over
                    if self.player.health <= 0:
                        self.game_over = True

        # Player and meteors
        for meteor in self.meteors:
            if check_collision_circles(self.player.get_center(), self.player.collision_radius, 
                                    meteor.get_center(), meteor.collision_radius):
                if self.player.take_damage():  # Only deal damage if not invulnerable
                    # Create explosion effect
                    pos = Vector2(meteor.pos.x + meteor.size.x / 2, meteor.pos.y + meteor.size.y / 2)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])
                    
                    # For AI mode, track the hit
                    if self.ai_mode:
                        self.player_took_damage_this_frame = True
                    
                    # Remove meteor
                    meteor.discard = True
                    
                    # Check for game over
                    if self.player.health <= 0:
                        self.game_over = True
                
        # Player and enemies
        for enemy in self.enemies:
            if check_collision_circles(self.player.get_center(), self.player.collision_radius,
                                    enemy.get_center(), enemy.collision_radius):
                if self.player.take_damage():  # Only deal damage if not invulnerable
                    # Create explosion effect
                    pos = Vector2(enemy.pos.x + enemy.size.x / 2, enemy.pos.y + enemy.size.y / 2)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])
                    
                    # For AI mode, track the hit and update enemy's damage dealt
                    if self.ai_mode:
                        self.player_took_damage_this_frame = True
                        
                        enemy_id = id(enemy)
                        if enemy_id in self.enemy_damage_tracker:
                            self.enemy_damage_tracker[enemy_id] += 1
                    
                    # Remove enemy
                    enemy.discard = True
                    
                    # # Check for game over
                    if self.player.health <= 0:
                        self.game_over = True

    # Fixed Reward Function with improvements to prevent corner camping

    def _calculate_reward(self):
        """Simplified phase-specific reward function."""
        
        BASE_SCALE = 0.05
        CLOSE_DISTANCE = 150
        DAMAGE_PENALTY = -5.0  # Reduced from -15.0
        GAME_OVER_PENALTY = -7.5
        BOUNDARY_PENALTY_SCALE = -0.5  # Reduced from -1.0
        CENTER_REWARD_SCALE = 0.15
        MOVEMENT_REWARD_SCALE = 0.5  # Increased from 0.2
        SURVIVAL_REWARD = 0.01  # Small reward for each frame survived
        
        # Get current training phase
        training_phase = 0
        if hasattr(self.wave_manager, 'training_phase'):
            training_phase = self.wave_manager.training_phase
        
        # Initialize reward
        reward = 0
        
        # Get game state
        player = self.player
        player_max_health = getattr(player, "max_health", 15)
        enemy_count = len(self.enemies)
        
        # Common rewards across all phases
        
        # Damage penalty (highest priority penalty)
        if self.player_took_damage_this_frame:
            reward += DAMAGE_PENALTY
        
        # Boundary penalty to keep player away from edges - IMPROVED
        edge_distance = min(
            player.center_pos.x, 
            WINDOW_WIDTH - player.center_pos.x,
            player.center_pos.y, 
            WINDOW_HEIGHT - player.center_pos.y
        )
        
        # Start penalty earlier and make it exponential as player gets closer to edges
        if edge_distance < 300:  # Increased from 200 to start boundary penalty earlier
            edge_factor = 1.0 - edge_distance/300.0
            reward += BOUNDARY_PENALTY_SCALE * (edge_factor ** 2) * 3  # Exponential penalty
        
        # Game over penalty
        if self.game_over:
            reward += GAME_OVER_PENALTY
        
       # SIMPLIFIED MOVEMENT REWARDS - replacing all the complex movement code
        if not hasattr(self, "prev_player_pos") or self.prev_player_pos is None:
            self.prev_player_pos = Vector2(player.center_pos.x, player.center_pos.y)
            self.prev_rotation = player.rotation
            return 0.0  # Skip reward calculation for first frame

        current_pos = Vector2(player.center_pos.x, player.center_pos.y)
        # Calculate distance moved
        dx = current_pos.x - self.prev_player_pos.x
        dy = current_pos.y - self.prev_player_pos.y
        distance_moved = math.sqrt(dx*dx + dy*dy)

        # Calculate rotation change (absolute value)
        rotation_change = abs(player.rotation - self.prev_rotation)
        if rotation_change > 180:  # Handle wrap-around
            rotation_change = 360 - rotation_change

        # Simple rewards for movement and rotation
        if distance_moved > 5.0:  # Only reward significant movement
            movement_reward = 0.1 * min(distance_moved, 30.0)  # Cap at 30 pixels per frame
            reward += movement_reward

        if rotation_change > 5.0:  # Only reward significant rotation
            rotation_reward = 0.05 * min(rotation_change, 90.0)  # Cap at 90 degrees per frame
            reward += rotation_reward

        # Store current position and rotation for next frame
        self.prev_player_pos = current_pos
        self.prev_rotation = player.rotation
        
        # Phase-specific rewards
        if training_phase == 0:  # Movement Only
            # Movement rewards now handled in the common section
            pass
        
        elif training_phase == 1:  # Meteor Dodging
            # Reward for surviving with meteors
            # Check for meteors that came close but didn't hit (near misses)
             # Add survival reward near the beginning
            reward += SURVIVAL_REWARD  # Reward for surviving each frame
            near_miss_reward = 0
            if hasattr(self, 'meteors') and len(getattr(self, 'meteors', [])) > 0:
                for meteor in self.meteors:
                    # Calculate distance to meteor - FIX: use pos instead of center_pos
                    dx = meteor.pos.x - player.center_pos.x
                    dy = meteor.pos.y - player.center_pos.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Reward for close passes (near misses)
                    if distance < CLOSE_DISTANCE and distance > meteor.collision_radius + player.collision_radius:
                        # Calculate how close the miss was (closer = higher reward)
                        proximity_factor = 1.0 - max(0.2, (distance - player.collision_radius - meteor.collision_radius) / CLOSE_DISTANCE)
                        near_miss_reward += BASE_SCALE * 20.0 * proximity_factor**2
            
            # Add this reward
            reward += near_miss_reward
        elif training_phase == 2:  # Laser Dodging
            # Reward for surviving with lasers
            # Check if player is avoiding active lasers
            reward += SURVIVAL_REWARD  # Reward for surviving each frame
            if hasattr(self, 'lasers') and len(getattr(self, 'lasers', [])) > 0:
                for laser in self.lasers:
                    if getattr(laser, 'active', False):
                        # If player moved away from laser's path
                        if hasattr(laser, 'is_targeting_player') and laser.is_targeting_player:
                            if hasattr(self, 'player_dodged_laser_this_frame') and self.player_dodged_laser_this_frame:
                                reward += BASE_SCALE * 20.0  # Significant reward for actively dodging
                
                # Reward for successfully avoiding a laser that was targeting the player
                if hasattr(self, 'lasers_avoided_this_frame'):
                    reward += BASE_SCALE * 25.0 * self.lasers_avoided_this_frame
        
        elif training_phase == 3:  # Aim Training
            # Movement rewards now handled in common section
            is_shooting = getattr(player, "is_shooting", False) or getattr(player, "just_fired", False)
            # Reward destroying enemies
            if self.enemies_destroyed_this_frame > 0:
                reward += BASE_SCALE * 50.0 * self.enemies_destroyed_this_frame  # Increased destruction reward
            
            # Reward for shooting at enemies
            if enemy_count > 0:
                # Check if aiming at any enemy
                player_angle_rad = math.radians(player.rotation)
                facing_vector = (math.sin(player_angle_rad), -math.cos(player_angle_rad))
                
                best_aim_accuracy = 0
                is_aiming_at_enemy = False
                
                for enemy in self.enemies:
                    dx = enemy.center_pos.x - player.center_pos.x
                    dy = enemy.center_pos.y - player.center_pos.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # Vector to enemy
                        to_enemy = (dx/dist, dy/dist)
                        
                        # Calculate aim accuracy using dot product
                        dot_product = facing_vector[0]*to_enemy[0] + facing_vector[1]*to_enemy[1]
                        
                        # If aiming reasonably well at any enemy
                        if dot_product > 0.7:  # Within ~45 degrees
                            is_aiming_at_enemy = True
                            if dot_product > best_aim_accuracy:
                                best_aim_accuracy = dot_product
                
                # Reward aiming at enemy
                if is_aiming_at_enemy:
                    aim_accuracy = (best_aim_accuracy - 0.7) / 0.3
                    reward += BASE_SCALE * 8.0 * (aim_accuracy ** 2)
                    
                    # Extra reward for shooting while aiming
                    if is_shooting:
                        reward += BASE_SCALE * 40.0 * (aim_accuracy ** 2)  # Increased shooting reward

            # New: Add penalty for not aiming at enemies when they exist
            if enemy_count > 0 and not is_aiming_at_enemy:
                # Base penalty scaled by how many enemies are present
                reward += BASE_SCALE * -4.0 * (enemy_count / 5.0)  # -4 when 5 enemies
                
                # Additional penalty if player is rotating away from enemies
                if rotation_change > 5.0 and not is_aiming_at_enemy:
                    reward += BASE_SCALE * -2.0
        
        elif training_phase == 4:  # Full Combat
            # Movement rewards now handled in common section
            
            # Wave completion reward
            if hasattr(self, "prev_wave_active") and not self.wave_manager.wave_active and self.prev_wave_active:
                wave_num = self.wave_manager.current_wave
                reward += BASE_SCALE * 60.0 * (1.0 + 0.1*wave_num)  # Increased wave completion reward
            
            # Enemy destruction reward
            if self.enemies_destroyed_this_frame > 0:
                reward += BASE_SCALE * 50.0 * self.enemies_destroyed_this_frame  # Increased destruction reward

        # Store current position for next frame
        self.prev_player_pos = current_pos
        
        # Store wave state for next frame
        if not hasattr(self, "prev_wave_active"):
            self.prev_wave_active = getattr(self.wave_manager, "wave_active", False)
        else:
            self.prev_wave_active = getattr(self.wave_manager, "wave_active", False)
        
        self.current_reward = reward
        self.reward_history.append(reward)
        self.total_reward += reward
        
        return reward
    
    def _save_ai_models(self):
        """Simplified method to save all AI models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wave = self.wave_manager.current_wave
        
        # Create directory
        save_dir = os.path.join(SAVE_MODEL_DIR, f"wave_{wave}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save relevant models only
        drl_path = os.path.join(save_dir, f"drl_agent_{timestamp}.pt")
        self.drl_agent.save_model(drl_path)
        
        # Save metadata
        meta_path = os.path.join(save_dir, f"metadata_{timestamp}.txt")
        with open(meta_path, "w") as f:
            f.write(f"Wave: {wave}\n")
            f.write(f"Wave Type: {self.wave_type}\n")
            if hasattr(self.wave_manager, 'training_phase'):
                f.write(f"Training Phase: {self.wave_manager.training_phase}\n")
        
        print(f"Saved AI models at wave {wave}")

    def toggle_ai_mode(self):
        """Switch between AI and manual control"""
        self.ai_mode = not self.ai_mode
        print(f"AI mode {'enabled' if self.ai_mode else 'disabled'}")

    def draw_score(self):
        # Use self.score instead of get_time()
        text_size = measure_text_ex(self.assets['font'], str(self.score), FONT_SIZE, 0)
        draw_text_ex(self.assets['font'], str(self.score), Vector2(WINDOW_WIDTH / 2 - text_size.x / 2, 0), FONT_SIZE, 0, WHITE)

    def draw_rewards(self):
        if not self.show_rewards:
            return
            
        # Draw current reward
        reward_text = f"Reward: {self.current_reward:.3f}"
        draw_text_ex(
            self.assets['font'],
            reward_text,
            Vector2(20, 100),
            FONT_SIZE // 3,
            0,
            GREEN if self.current_reward >= 0 else RED
        )
        
        # Draw total reward
        total_text = f"Total: {self.total_reward:.1f}"
        draw_text_ex(
            self.assets['font'],
            total_text,
            Vector2(20, 130),
            FONT_SIZE // 3,
            0,
            WHITE
        )
        
        # Draw phase
        if hasattr(self.wave_manager, 'training_phase'):
            phase_names = ["MOVEMENT", "METEOR", "LASER", "AIM", "COMBAT"]
            phase = self.wave_manager.training_phase
            phase_name = phase_names[phase] if phase < len(phase_names) else f"PHASE_{phase}"
            phase_text = f"PHASE: {phase_name}"
            draw_text_ex(
                self.assets['font'],
                phase_text,
                Vector2(20, 160),
                FONT_SIZE // 3,
                0,
                YELLOW
            )

    def update(self):
        dt = get_frame_time()
        
        if not self.game_over:
            # AI-specific initial updates
            if self.ai_mode:
                # Increment frame counter and reset tracking variables
                self.frame_count += 1
                self.enemies_destroyed_this_frame = 0
                self.player_took_damage_this_frame = False
                self.prev_wave_active = self.wave_manager.wave_active
                
                # Update enemy lifetime tracking
                for enemy_id in list(self.enemy_lifetime_tracker.keys()):
                    if enemy_id in self.enemy_lifetime_tracker:
                        self.enemy_lifetime_tracker[enemy_id] += dt
                
                # AI agent action selection and execution
                state = self.drl_agent.get_state()
                action = self.drl_agent.act(state)
                self.drl_agent.execute_action(action)
            
                # Get current training phase once
                is_phase_0 = False
                if self.ai_mode and hasattr(self.wave_manager, 'training_mode') and self.wave_manager.training_mode:
                    current_phase = self.wave_manager.training_phase
                    is_phase_0 = current_phase == 0
                    
                    # Handle phase-specific updates
                    if is_phase_0:
                        # Phase 0 (movement only) - no meteors or enemies
                        self.meteors.clear()
                        self.meteor_timer.deactivate()
                    elif current_phase in [2, 3]:
                        # Phases 2 & 3 - NO METEORS
                        self.meteors.clear()
                        self.meteor_timer.deactivate()
                    else:
                        self.meteor_timer.update()
                else:
                    self.meteor_timer.update()

                # Core game updates
                self.enemy_timer.update() 
                self.game_timer.update()
                self.wave_manager.update(dt)
                self.player.update(dt)

                # Update all game entities
                for sprite in self.lasers + self.meteors + self.enemies + self.enemy_lasers + self.explosions:
                    sprite.update(dt)
                for beam in self.enemy_beams:
                    beam.update(dt)
                self.enemy_beams = [beam for beam in self.enemy_beams if not beam.discard]

                # Game logic
                self.check_collisions()
                self.discard_sprites()
            
            # Log updated player movement
            if hasattr(self, 'logger'):
                self.logger.log_player_movement(self.player)
            
            # AI learning updates after game state is updated
            if self.ai_mode:
                # Get next state and calculate reward
                next_state = self.drl_agent.get_state()
                reward = self._calculate_reward()
                
                # Store experience
                self.drl_agent.remember(state, action, reward, next_state, self.game_over)
                
                # Train periodically
                if self.frame_count % 4 == 0:
                    self.drl_agent.replay()
                
                # Check if wave was just completed
                if self.prev_wave_active and not self.wave_manager.wave_active:
                    self.wave_completed = True
                    
                    # Tell wave manager that the wave was completed
                    if hasattr(self.wave_manager, 'wave_completed'):
                        self.wave_manager.wave_completed()
                    
                    # Evolve enemies periodically
                    if self.wave_manager.current_wave % 3 == 0:
                        self.genetic_evolver.evolve()
                    
                    # Save models periodically
                    if self.wave_manager.current_wave % 10 == 0:
                        self._save_ai_models()
            
            # Debug command to show all enemies
            if is_key_pressed(KEY_F1):
                print("=== ACTIVE ENEMIES ===")
                for i, enemy in enumerate(self.enemies):
                    print(f"Enemy {i}: Type: {getattr(enemy, 'debug_type', 'unknown')}, "
                        f"Position: ({enemy.center_pos.x:.1f}, {enemy.center_pos.y:.1f}), "
                        f"Health: {enemy.health}, Discard: {enemy.discard}")
                print("=====================")
        else:
            # Just update explosions if there are any during game over
            for explosion in self.explosions:
                explosion.update(dt)
            self.explosions = [e for e in self.explosions if not e.discard]
        
        update_music_stream(self.audio['music'])

    def draw(self):
        begin_drawing()
        clear_background(BG_COLOR)
        
        if not self.game_over:
            self.draw_stars()
            self.draw_score()
            self.draw_health()  # Draw health indicators
            self.draw_timer()   
            self.player.draw()
            
            # Draw AI status if enabled
            if self.ai_mode:
                self._draw_ai_status()
                # Add model status to the UI display
                if hasattr(self, 'model_loaded') and self.model_loaded:
                    model_text = "AI MODEL: LOADED"
                    draw_text_ex(self.assets['font'], model_text, 
                                Vector2(20, WINDOW_HEIGHT - 80),
                                FONT_SIZE // 4, 0, GREEN)
                else:
                    model_text = "AI MODEL: DEFAULT"
                    draw_text_ex(self.assets['font'], model_text,
                                Vector2(20, WINDOW_HEIGHT - 80),
                                FONT_SIZE // 4, 0, YELLOW)
            
            # Draw all game objects
            for sprite in self.lasers + self.meteors + self.enemies + self.enemy_lasers + self.explosions:
                sprite.draw()
            
            # Draw beams
            for beam in self.enemy_beams:
                beam.draw()

            # Draw wave information
            self.wave_manager.draw()

            for enemy in self.enemies:
                if hasattr(enemy, 'health') and enemy.health > 1:
                    center_x = enemy.pos.x + enemy.size.x / 2
                    for i in range(enemy.health):
                        draw_rectangle(
                            int(center_x - enemy.health * 5 + i * 10), 
                            int(enemy.pos.y - 10),
                            8, 4, RED
                        )
            
            # Draw current weapon info
            self.draw_weapon_info()
        else:
            # Draw game over screen
            self.draw_game_over()
        self.draw_rewards()
        end_drawing()

    def draw_game_over(self):
        self.draw_stars()  # Keep drawing background
        
        game_over_text = "GAME OVER"
        score_text = f"Final Score: {self.score}"
        restart_text = "Press ENTER to restart"
        
        # Measure text sizes
        game_over_size = measure_text_ex(self.assets['font'], game_over_text, FONT_SIZE * 2, 0)
        score_size = measure_text_ex(self.assets['font'], score_text, FONT_SIZE, 0)
        restart_size = measure_text_ex(self.assets['font'], restart_text, FONT_SIZE, 0)
        
        # Draw text centered
        draw_text_ex(self.assets['font'], game_over_text, 
                    Vector2((WINDOW_WIDTH / 2) - (game_over_size.x / 2), (WINDOW_HEIGHT / 2) -(game_over_size.y / 2) ), 
                    FONT_SIZE * 2, 0, RED)
        
        draw_text_ex(self.assets['font'], score_text, 
                    Vector2((WINDOW_WIDTH / 2) - (score_size.x / 2), (WINDOW_HEIGHT / 2) +(score_size.y / 2) ), 
                    FONT_SIZE, 0, WHITE)
        
        draw_text_ex(self.assets['font'], restart_text, 
                    Vector2((WINDOW_WIDTH / 2) - (restart_size.x / 2), (WINDOW_HEIGHT / 2) + 200), 
                    FONT_SIZE, 0, WHITE)
    
    def run(self):
        while not window_should_close():
            # Handle AI toggle
            if is_key_pressed(KEY_F2):
                self.toggle_ai_mode()
            
            # Check for game restart
            if self.game_over and is_key_pressed(KEY_ENTER):
                # Reset the game state
                self.reset_game()
            
            # Save models on demand
            if is_key_pressed(KEY_F5) and self.ai_mode:
                self._save_ai_models()
            
            self.update()
            self.draw()
        
        unload_music_stream(self.audio['music'])
        close_audio_device()
        close_window()

    # Modify the reset_game method in main.py
    def reset_game(self):
        # Clear all game objects
        self.lasers.clear()
        self.meteors.clear()
        self.enemies.clear()
        self.enemy_lasers.clear()
        self.explosions.clear()
        self.enemy_beams.clear()
        
        # Reset player
        self.player = Player(self.assets['player'], 
                            Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2), 
                            self.shoot_laser)
        
        # Reinitialize weapons
        self.player.weapons = [
            Weapon(self),          # Standard laser
            TripleShot(self),      # Triple shot 
            RapidFire(self)        # Rapid fire
        ]
        
        # IMPORTANT: Only reset timer, not phase
        if hasattr(self.wave_manager, 'training_mode') and self.wave_manager.training_mode:
            self.wave_manager.phase_timer = 0
        
        # DON'T DEACTIVATE THE TIMERS HERE - THIS IS THE MAIN ISSUE
        # self.meteor_timer.deactivate() - REMOVE THIS
        # self.enemy_timer.deactivate() - REMOVE THIS
        
        # Keep weapons unlocked based on current wave
        if hasattr(self, 'wave_manager'):
            if self.wave_manager.current_wave >= 2:
                self.player.weapons[1].unlocked = True
            if self.wave_manager.current_wave >= 3:
                self.player.weapons[2].unlocked = True
        
        # Reset score and game state
        self.score = 0
        self.game_over = False
        self.wave_completed = False
        self.frame_count = 0
        
        # Reset tracking variables for AI
        if self.ai_mode:
            self.enemies_destroyed_this_frame = 0
            self.player_took_damage_this_frame = False
            self.prev_wave_active = False
            self.enemy_damage_tracker = {}
            self.enemy_lifetime_tracker = {}
            self.enemy_shot_tracker = {}
            self.prev_player_pos = None
            self.prev_rotation = None

        
        # Reset wave manager
        self.wave_manager.reset()
        
        # CRITICAL FIX: For phases 2 & 3, force settings
        if hasattr(self.wave_manager, 'training_mode') and self.wave_manager.training_mode:
            if self.wave_manager.training_phase in [2, 3]:
                print("FORCING ENEMY TIMER AND WAVE ACTIVE FOR PHASE 2/3")
                self.wave_manager.wave_active = True
                self.enemy_timer.duration = 0.5  # Faster spawning
                self.enemy_timer.activate()
                
                # Force immediate spawn of enemies
                for _ in range(self.wave_manager.max_training_enemies):
                    enemy = self.wave_manager.spawn_enemy()
                    if enemy:
                        self.enemies.append(enemy)
                        print(f"Force-spawned enemy: {getattr(enemy, 'debug_type', 'unknown')}")
        # Reset game timer
        self.game_timer = Timer(float('inf'), True, True)
        print("Game reset, starting new game")


class EnhancedEnemyFactory(EnemyFactory):
    """Extended enemy factory that works with the genetic algorithm"""
    
    def create_enemy(self, enemy_type):
        """Create an enemy with genetic enhancements"""
        # First create the base enemy
        enemy = super().create_enemy(enemy_type)
        
        # If we have a genetic evolver, it will be enhanced later
        # in the spawn_wave_enemy method
        
        return enemy


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Space Shooter Game")
    parser.add_argument("--ai", action="store_true", help="Enable AI mode")
    args = parser.parse_args()
    
    # Create and run the game
    game = Game(ai_mode=args.ai)
    game.run()