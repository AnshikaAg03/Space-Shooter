from sprites import *

class WaveManager:
    def __init__(self, game, phased_training=False, training_phase_override=None):
        """Initialize the wave manager with enhanced training capabilities"""
        self.game = game
        self.current_wave = 0
        self.enemies_remaining = 0      
        self.total_enemies_to_spawn = 0 
        self.enemies_spawned = 0        
        self.wave_active = False
        self.wave_cooldown = 5.0
        self.cooldown_timer = 0
        self.base_spawn_interval = ENEMY_SPAWN_INTERVAL
        self.current_spawn_interval = self.base_spawn_interval

        # Training phase system
        self.training_mode = phased_training
        # Phase definitions:
        # 0: Movement only - no enemies or meteors
        # 1: Meteor dodging - meteors only
        # 2: Laser dodging - stationary enemies that shoot
        # 3: Aim training - teleporting stationary enemies
        # 4: Full combat - normal game
        self.training_phase = 1 if training_phase_override is None else training_phase_override
        self.phase_timer = 0
        self.movement_phase_duration = 30.0
        self.meteor_phase_duration = 30.0
        self.laser_phase_duration = 30.0
        self.aim_phase_duration = 30.0
        
        # Enemy teleport timer (for aim training)
        self.teleport_timer = 0
        self.teleport_interval = 3.0  # Teleport enemies every 3 seconds
        
        # Wave announcement properties
        self.show_wave_message = False
        self.wave_message_timer = 0
        self.wave_message_duration = 3.0

        # Add a wave timeout to prevent stuck waves
        self.wave_timeout = 60.0  # 60 seconds maximum per wave
        self.wave_timer = 0
        
        # Max enemies for training phases
        self.max_training_enemies = 3

    def set_training_phase(self, phase):
        """Sets the training phase with better state management"""
        if phase < 0 or phase > 4:
            print(f"Invalid training phase: {phase}. Must be 0-4")
            return

        # Store previous phase for transition handling
        prev_phase = self.training_phase
        self.training_phase = phase
        self.phase_timer = 0
        
        # Clear objects based on new phase requirements
        if phase == 0:  # Movement only
            # Clear all enemies and meteors
            if hasattr(self.game, 'meteors'):
                self.game.meteors.clear()
            if hasattr(self.game, 'enemies'):
                self.game.enemies.clear()
            if hasattr(self.game, 'enemy_lasers'):
                self.game.enemy_lasers.clear()
            
            # Deactivate all spawning
            self.game.meteor_timer.deactivate()
            self.game.enemy_timer.deactivate()
            self.wave_active = False
            
        elif phase == 1:  # Meteor dodging
            # Clear enemies but keep meteors
            if hasattr(self.game, 'enemies'):
                self.game.enemies.clear()
            if hasattr(self.game, 'enemy_lasers'):
                self.game.enemy_lasers.clear()
            # Disable player shooting
            if hasattr(self.game.player, 'original_shoot_laser'):
                self.game.player.shoot_laser = lambda *args, **kwargs: None
            else:
                self.game.player.original_shoot_laser = self.game.player.shoot_laser
                self.game.player.shoot_laser = lambda *args, **kwargs: None
                        
            # Enable meteors, disable enemies
            self.game.meteor_timer.activate()
            self.game.enemy_timer.deactivate()
            self.wave_active = False
            
        elif phase == 2 or phase == 3:  # Laser dodging or Aim training
            # Clear meteors but allow enemies
            if hasattr(self.game, 'meteors'):
                self.game.meteors.clear()
                    # Disable player shooting
            if hasattr(self.game.player, 'original_shoot_laser'):
                self.game.player.shoot_laser = lambda *args, **kwargs: None
            else:
                self.game.player.original_shoot_laser = self.game.player.shoot_laser
                self.game.player.shoot_laser = lambda *args, **kwargs: None
                    
            # Disable meteors, enable enemies
            self.game.meteor_timer.deactivate()
            self.game.enemy_timer.activate()
            self.game.enemy_timer.duration = 1.0  # Spawn enemies quickly
            self.wave_active = True  # IMPORTANT: Make sure wave is active
            print(f"DEBUG: Phase {phase} - wave_active set to TRUE, meteors DISABLED")
            
        elif phase == 4:  # Full combat
            # Enable everything
            self.game.meteor_timer.activate()
            self.game.enemy_timer.activate()
            self.game.enemy_timer.duration = self.current_spawn_interval
            self.wave_active = True
            
            # Unlock weapons for full combat
            self.game.check_weapon_unlocks()
            
        if phase ==3 or phase ==4:
            # Enable player shooting
            if hasattr(self.game.player, 'original_shoot_laser'):
                self.game.player.shoot_laser = self.game.player.original_shoot_laser
            else:
                print("WARNING: Player shooting not restored - original function not found")

    def start_next_wave(self):
        """Start the next wave with increased difficulty"""
        print(f"DEBUG: Starting next wave - current wave: {self.current_wave}")
        self.current_wave += 1

        if self.training_mode:
            # In modular training, we don't change phases within a wave
            # Phase transitions are handled externally by the training script
            
            # Set enemy count based on phase
            if self.training_phase in [2, 3, 4]:
                # Phases 2, 3, 4 - use higher limit for continuous spawning
                self.total_enemies_to_spawn = 999  # Set high to ensure continuous respawning
                self.enemies_remaining = self.max_training_enemies
                self.enemies_spawned = 0
                print(f"DEBUG: Set training phase {self.training_phase} with {self.max_training_enemies} enemies")
            else:
                # Base enemy count calculation for other phases
                base_enemies = 5
                bonus_enemies = int(self.current_wave * 1.5)
                self.total_enemies_to_spawn = base_enemies + bonus_enemies
                self.enemies_remaining = self.total_enemies_to_spawn
                self.enemies_spawned = 0
            
            # Show wave message
            self.show_wave_message = True
            self.wave_message_timer = 0
            
            # Different setup based on current training phase
            if self.training_phase == 0:  # Movement only
                self.wave_active = False
                self.game.meteor_timer.deactivate()
                self.game.enemy_timer.deactivate()
                print(f"Wave {self.current_wave} - Movement training phase")
                
            elif self.training_phase == 1:  # Meteor dodging
                self.wave_active = False
                self.game.meteor_timer.activate()
                self.game.enemy_timer.deactivate()
                print(f"Wave {self.current_wave} - Meteor dodging phase")
                
            elif self.training_phase == 2 or self.training_phase == 3:  # Phases 2 & 3
                self.wave_active = True  # IMPORTANT: Make sure wave is active
                self.game.meteor_timer.deactivate()  # MAKE SURE METEORS ARE DISABLED
                
                # Force clear any existing meteors
                if hasattr(self.game, 'meteors'):
                    self.game.meteors.clear()
                    
                self.game.enemy_timer.activate()
                self.game.enemy_timer.duration = 1.0  # Spawn enemies more quickly
                self.teleport_timer = 0
                print(f"Wave {self.current_wave} - Phase {self.training_phase} (METEORS DISABLED)")
                print(f"DEBUG: Wave active: {self.wave_active}, enemy timer active: {self.game.enemy_timer.active}")
                
            elif self.training_phase == 4:  # Full combat
                self.wave_active = True
                self.game.meteor_timer.activate()
                
                # Adjust spawn interval for full combat
                self.current_spawn_interval = max(0.5, self.base_spawn_interval - (self.current_wave * 0.1))
                self.game.enemy_timer.duration = self.current_spawn_interval
                self.game.enemy_timer.activate()
                
                # Unlock weapons for full combat
                self.game.check_weapon_unlocks()
                print(f"Wave {self.current_wave} - Full combat phase! Enemies: {self.total_enemies_to_spawn}")

    def spawn_enemy(self):
        """Spawn an enemy if we haven't reached the total to spawn"""
        # Skip if wave is not active
        if not self.wave_active:
            print("DEBUG: Wave not active, not spawning")
            if self.training_mode and self.training_phase in [2, 3, 4]:  # ADDED PHASE 4
                # FORCE ACTIVE FOR PHASES 2/3/4
                print(f"FORCING WAVE ACTIVE FOR PHASE {self.training_phase}")
                self.wave_active = True
            else:
                return None
                
        # For training phases 2, 3 and 4, check if we already have max enemies
        if self.training_mode and self.training_phase in [2, 3, 4]:  # ADDED PHASE 4
            current_enemies = len(self.game.enemies)
            print(f"DEBUG: Training phase {self.training_phase}, current enemies: {current_enemies}, max: {self.max_training_enemies}")
            
            if current_enemies >= self.max_training_enemies:
                return None
        # For other phases, check if we've already spawned all enemies
        elif self.enemies_spawned >= self.total_enemies_to_spawn:
            return None
        
        # Create the appropriate enemy
        enemy = None
        
        if self.training_mode:
            if self.training_phase == 2:  # Laser dodging phase
                enemy = self.create_laser_dodging_enemy()
                print(f"DEBUG: Created laser dodging enemy: {enemy}")
            elif self.training_phase == 3:  # Aim training phase
                enemy = self.create_aim_training_enemy()
                print(f"DEBUG: Created aim training enemy: {enemy}")
            elif self.training_phase == 4:  # Full combat
                enemy = self.create_wave_enemy()
            else:
                # No enemies for phases 0-1
                return None
        else:
            # Standard wave enemy for normal game mode
            enemy = self.create_wave_enemy()
        
        # Increment spawn counter if we created an enemy
        if enemy:
            self.enemies_spawned += 1
            print(f"DEBUG: Enemy spawned, type: {getattr(enemy, 'debug_type', 'unknown')}")
        else:
            print("DEBUG: Failed to create enemy")
        
        return enemy
    
    def create_laser_dodging_enemy(self):
        """Create a stationary enemy for laser dodging phase that shoots and teleports"""
        # Randomly select from various enemy types
        enemy_types = ["normal", "sniper", "tank", "bomber"]
        enemy_type = random.choice(enemy_types)
        
        # Create the enemy using the enemy factory
        enemy = self.game.enemy_factory.create_enemy(enemy_type)
        
        # Make it stationary
        enemy.speed = 0
        
        # Add teleport ability
        enemy.can_teleport = True
        enemy.shoot_timer = 0
        enemy.shoot_duration = random.uniform(1.0, 3.0)  # Shoot for 1-3 seconds
        enemy.shooting_phase = True  # Start in shooting phase
        
        # Set health
        enemy.health = max(1, (self.current_wave - 1) // 2)
        
        # Set target player
        enemy.target_player = self.game.player
        
        # Set debug type
        enemy.debug_type = "LASER_DODGE"
        
        return enemy
    
    def create_aim_training_enemy(self):
        """Create a non-shooting enemy for aim training that DOESN'T teleport"""
        # Randomly select from various enemy types for visual variety
        enemy_types = ["normal", "swarm", "sniper", "tank", "bomber"]
        enemy_type = random.choice(enemy_types)
        
        # Create the enemy using the enemy factory
        enemy = self.game.enemy_factory.create_enemy(enemy_type)
        
        # Make it stationary and prevent shooting
        enemy.speed = 0
        enemy.shoot_laser = lambda *args, **kwargs: None  # Disable shooting
        
        # REMOVE teleport ability (changed this)
        enemy.can_teleport = False
        
        # Set health
        enemy.health = max(1, (self.current_wave - 1) // 2)
        
        # Set target player
        enemy.target_player = self.game.player
        
        # Set debug type
        enemy.debug_type = "AIM_TRAIN_STATIONARY"
        
        return enemy
    

    def create_phase_appropriate_enemy(self):
        """Create an enemy appropriate for the current training phase"""
        enemy = None
        
        if self.training_mode:
            if self.training_phase == 2:  # Laser dodging phase
                enemy = self.create_laser_dodging_enemy()
            elif self.training_phase == 3:  # Aim training phase
                enemy = self.create_aim_training_enemy()
            elif self.training_phase == 4:  # Full combat
                enemy = self.create_wave_enemy()
            else:  # Fallback
                enemy = self.create_wave_enemy()
        else:
            # Standard wave enemy for normal game mode
            enemy = self.create_wave_enemy()
        
        return enemy
    
    def create_stationary_enemy(self, can_teleport=False):
        """Create a stationary enemy that shoots but doesn't move"""
        # Use sniper type for stationary enemies
        enemy_type = "sniper"
        enemy = self.game.enemy_factory.create_enemy(enemy_type)
        
        # Make it stationary
        enemy.speed = 0
        
        # For teleporting enemies, add a flag
        if can_teleport:
            enemy.can_teleport = True
        
        # Set base health values
        enemy.health = max(1, (self.current_wave - 1) // 2)
        
        # Set debug type
        enemy.debug_type = "STATIONARY"
        
        # Set target player
        enemy.target_player = self.game.player
        
        return enemy
    
    def teleport_enemies(self):
        """Teleport all enemies with the can_teleport flag to random positions"""
        for enemy in self.game.enemies:
            if hasattr(enemy, 'can_teleport') and enemy.can_teleport:
                # For laser dodging enemies, only teleport if they're finished shooting
                if self.training_phase == 2 and hasattr(enemy, 'shooting_phase') and enemy.shooting_phase:
                    enemy.shoot_timer += self.teleport_interval
                    if enemy.shoot_timer >= enemy.shoot_duration:
                        enemy.shoot_timer = 0
                        enemy.shooting_phase = False
                        self._teleport_single_enemy(enemy)
                else:
                    # Always teleport aim training enemies or when not in shooting phase
                    self._teleport_single_enemy(enemy)
                    if self.training_phase == 2 and hasattr(enemy, 'shooting_phase'):
                        enemy.shooting_phase = True  # Start shooting again after teleport
    
    def _teleport_single_enemy(self, enemy):
        """Helper method to teleport a single enemy to a random position"""
        player = self.game.player
                
        # Keep trying until we find a position with minimum distance
        valid_position = False
        attempts = 0
        min_distance = 300  # Minimum distance from player
        max_distance = 700  # Maximum distance from player
        
        while not valid_position and attempts < 10:
            # Random position within screen bounds with padding
            new_x = random.randint(100, WINDOW_WIDTH - 100)
            new_y = random.randint(100, WINDOW_HEIGHT - 100)
            
            # Calculate distance to player
            dx = new_x - player.center_pos.x
            dy = new_y - player.center_pos.y
            distance = sqrt(dx*dx + dy*dy)
            
            # Check if distance is acceptable
            if min_distance <= distance <= max_distance:
                valid_position = True
                
                # Update enemy position
                enemy.pos.x = new_x - enemy.size.x / 2
                enemy.pos.y = new_y - enemy.size.y / 2
                
                # Update center position
                enemy.center_pos.x = new_x
                enemy.center_pos.y = new_y
                
                # Create a small "teleport" effect
                if hasattr(self.game, 'explosions'):
                    self.game.explosions.append(ExplosionAnimation(
                        Vector2(new_x, new_y), 
                        self.game.assets['explosion']
                    ))
            
            attempts += 1
    
    def create_wave_enemy(self):
        """Create an enemy with wave-appropriate attributes"""
        # Determine enemy type based on wave
        enemy_type = "normal"  # Default

        # Use random chance to select enemy type based on wave number
        r = random.random()

        if self.current_wave >= 5:
            # Higher waves - full variety
            if r < 0.3:
                enemy_type = "swarm"
            elif r < 0.5:
                enemy_type = "tank"
            elif r < 0.7:
                enemy_type = "sniper"
            elif r < 0.9:
                enemy_type = "bomber"
        elif self.current_wave >= 3:
            # Wave 3-4 - introduce tanks and snipers
            if r < 0.4:
                enemy_type = "swarm"
            elif r < 0.7:
                enemy_type = "tank"
            elif r < 0.9:
                enemy_type = "sniper"
        elif self.current_wave >= 2:
            # Wave 2 - introduce swarms
            if r < 0.7:
                enemy_type = "swarm"

        # Create the enemy using the factory
        enemy = self.game.enemy_factory.create_enemy(enemy_type)
        
        # Set base health values - first wave always has 1 health
        if self.current_wave == 1:
            enemy.health = 1
        else:
            # Scale health with wave number
            if enemy_type == "normal":
                # Normal enemies: Wave 1: 1, Wave 2: 1, Wave 3: 2, etc.
                enemy.health = max(1, (self.current_wave - 1) // 1)
            elif enemy_type == "tank":
                # Tanks are always tougher (wave number + 2)
                enemy.health = min(3 + (self.current_wave - 1) // 1, 7)
            elif enemy_type == "swarm":
                # Swarms always have low health
                enemy.health = 1
            elif enemy_type == "sniper":
                # Snipers scale with wave
                enemy.health = min(1 + (self.current_wave - 1) // 1, 4)
            elif enemy_type == "bomber":
                # Bombers scale with wave
                enemy.health = min(2 + (self.current_wave - 1) // 1, 5)
        
        # Set debug type for visualization
        enemy.debug_type = enemy_type.upper()

        # Apply wave-based speed scaling (gentle increase)
        speed_multiplier = 1.0 + min(0.2, (self.current_wave - 1) * 0.05)
        enemy.speed *= speed_multiplier

        return enemy
    
    def enemy_destroyed(self):
        """Called when an enemy is destroyed"""
        self.enemies_remaining -= 1
        print(f"Enemy destroyed! Remaining to kill: {self.enemies_remaining}")
        
        # For training phases 2, 3 & 4, ensure we maintain 3 enemies
        if self.training_mode and self.training_phase in [2, 3, 4]:  # ADDED PHASE 4 HERE
            current_enemy_count = len(self.game.enemies)
            print(f"Training phase {self.training_phase}: {current_enemy_count}/{self.max_training_enemies} enemies")
            
            # FORCE ACTIVE AND IMMEDIATE SPAWN
            self.wave_active = True
            self.game.enemy_timer.duration = 0.1  # Very short duration
            self.game.enemy_timer.activate()
            
            # DIRECT SPAWN - don't wait for timer
            if current_enemy_count < self.max_training_enemies:
                enemy = self.spawn_enemy()
                if enemy:
                    self.game.enemies.append(enemy)
                    print(f"IMMEDIATELY spawned new enemy after destruction: {getattr(enemy, 'debug_type', 'unknown')}")
        
        # Check if wave is complete for non-training phases
        elif not self.training_mode and self.enemies_remaining <= 0:
            self.wave_active = False
            self.cooldown_timer = 0
            
            # Give player a bonus
            self.game.score += 100 * self.current_wave
            
            print(f"Wave {self.current_wave} complete! Starting cooldown for next wave.")
            
    def update(self, dt):
        """Update the wave manager state"""
        # Handle wave message display
        if self.show_wave_message:
            self.wave_message_timer += dt
            if self.wave_message_timer >= self.wave_message_duration:
                self.show_wave_message = False
        
        # For training mode phases
        if self.training_mode:
            # Update phase timer
            self.phase_timer += dt
            
            # Handle phases 2, 3, and now 4 with enemy limits
            if self.training_phase in [2, 3, 4]:  # ADDED PHASE 4 HERE
                # FORCE WAVE ACTIVE AND TIMER
                if not self.wave_active:
                    print(f"FIXING: Setting wave_active to True for phase {self.training_phase}")
                    self.wave_active = True
                if not self.game.enemy_timer.active:
                    print(f"FIXING: Activating enemy timer for phase {self.training_phase}")
                    self.game.enemy_timer.activate()
                    
                # Only teleport enemies in phase 2 (laser dodging)
                if self.training_phase == 2:
                    self.teleport_timer += dt
                    if self.teleport_timer >= self.teleport_interval:
                        self.teleport_timer = 0
                        self.teleport_enemies()
                    
                # Make sure we always have enough enemies in these phases
                current_enemy_count = len(self.game.enemies)
                if current_enemy_count < self.max_training_enemies:
                    # First activate the timer
                    self.game.enemy_timer.activate()
                    
                    # Force direct spawn if necessary
                    if self.game.frame_count % 60 == 0:  # Check every ~1 second
                        missing_enemies = self.max_training_enemies - current_enemy_count
                        print(f"FORCE SPAWNING {missing_enemies} enemies for phase {self.training_phase}")
                        
                        for _ in range(missing_enemies):
                            enemy = self.spawn_enemy()
                            if enemy:
                                self.game.enemies.append(enemy)
        
        # If wave is not active and we're in appropriate phase
        if not self.wave_active and (not self.training_mode or self.training_phase >= 2):
            self.cooldown_timer += dt
            # Start next wave after cooldown
            if self.cooldown_timer >= self.wave_cooldown:
                self.start_next_wave()
    
    def draw(self):
        """Draw wave information"""
        if self.show_wave_message:
            # Draw wave announcement
            wave_text = f"WAVE {self.current_wave}"
            text_size = measure_text_ex(self.game.assets['font'], wave_text, FONT_SIZE, 0)
            
            # Position in center of screen
            draw_text_ex(
                self.game.assets['font'], 
                wave_text, 
                Vector2(
                    (WINDOW_WIDTH / 2) - (text_size.x / 2),
                    (WINDOW_HEIGHT / 2) - (text_size.y / 2)
                ),
                FONT_SIZE, 
                0, 
                RED
            )
        
        # Add phase indicator during training
        if self.training_mode:
            phase_names = ["MOVEMENT TRAINING", "METEOR TRAINING", "LASER DODGING", "AIM TRAINING", "FULL COMBAT"]
            phase_text = f"PHASE: {phase_names[self.training_phase]}"
            timer_text = f"PHASE TIME: {self.phase_timer:.1f}s"
            
            small_font_size = FONT_SIZE // 3
            
            # Draw phase name
            draw_text_ex(
                self.game.assets['font'],
                phase_text,
                Vector2(50, WINDOW_HEIGHT - 80),
                small_font_size,
                0,
                YELLOW
            )
            
            # Draw phase timer
            draw_text_ex(
                self.game.assets['font'],
                timer_text,
                Vector2(50, WINDOW_HEIGHT - 50),
                small_font_size,
                0,
                YELLOW
            )
        
        # Always show current wave in top right
        current_wave_text = f"WAVE: {self.current_wave}"
        enemies_text = f"ENEMIES: {self.enemies_remaining}"
        
        # Use a smaller font size for the persistent display
        small_font_size = FONT_SIZE // 3
        
        # Draw wave number
        draw_text_ex(
            self.game.assets['font'],
            current_wave_text,
            Vector2(WINDOW_WIDTH - 250, 20),
            small_font_size,
            0,
            WHITE
        )
        
        # Draw enemies remaining
        draw_text_ex(
            self.game.assets['font'],
            enemies_text,
            Vector2(WINDOW_WIDTH - 250, 20 + small_font_size + 10),
            small_font_size,
            0,
            WHITE
        )
    
    def reset(self):
        """Reset the wave manager to initial state with better state handling"""
        self.current_wave = 0
        self.enemies_remaining = 0
        self.total_enemies_to_spawn = 0
        self.enemies_spawned = 0
        self.wave_active = False
        self.cooldown_timer = 0
        self.current_spawn_interval = self.base_spawn_interval
        self.show_wave_message = False
        self.wave_message_timer = 0
        
        # Keep training phase but reset timers
        if self.training_mode:
            # Don't reset training_phase here - maintain current phase
            self.phase_timer = 0
            self.teleport_timer = 0
            
            # Configure timers based on current phase
            self.set_training_phase(self.training_phase)
        else:
            # Standard game mode
            if hasattr(self.game, 'meteor_timer'):
                self.game.meteor_timer.activate()
            if hasattr(self.game, 'enemy_timer'):
                self.game.enemy_timer.activate()
        
        print("Wave manager reset completed")
        
        # Start first wave 
        self.start_next_wave()



class EnemyFactory:
    """Creates different types of enemies based on game progression"""
    def __init__(self, game):
        self.game = game
        
    def create_enemy(self, enemy_type):
        """Create an enemy of the specified type with its specialized weapon"""
        enemy = None
        if enemy_type == "tank":
            enemy = TankEnemy(
                self.game.assets['enemy_tank'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        elif enemy_type == "swarm":
            enemy = SwarmEnemy(
                self.game.assets['enemy_swarm'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        elif enemy_type == "sniper":
            enemy = SniperEnemy(
                self.game.assets['enemy_sniper'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        elif enemy_type == "bomber":
            enemy = BomberEnemy(
                self.game.assets['enemy_bomber'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        else:
            # Default to normal enemy
            enemy = Enemy(
                self.game.assets['enemy'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser']
            )
        
        if enemy:
            # Always set target player
            enemy.target_player = self.game.player
            
            # Add debug text to display enemy type (temporary solution)
            enemy.debug_type = enemy_type.upper()
        
        return enemy