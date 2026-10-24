from settings import * 
from math import sin, sqrt, atan2, degrees,cos,radians,inf,sin, cos, radians, degrees, atan2, sqrt
import random 

class Sprite:
    def __init__(self, texture, pos, speed, direction):
        self.texture = texture
        self.pos = pos
        self.speed = speed
        self.direction = direction
        self.size = Vector2(texture.width, texture.height)
        self.discard = False
        self.collision_radius = self.size.y / 2
    
    def move(self, dt):
        self.pos.x += self.direction.x * self.speed * dt
        self.pos.y += self.direction.y * self.speed * dt

    def check_discard(self):
        # Check if this is an Enemy type
        if isinstance(self, Enemy):
            # For Enemy class, we check against extreme boundaries
            extreme_bounds = not -1000 < self.pos.y < WINDOW_HEIGHT + 1000 or \
                    not -1000 < self.pos.x < WINDOW_WIDTH + 1000
            if extreme_bounds:
                print(f"Warning: Enemy far outside bounds at ({self.center_pos.x}, {self.center_pos.y})")
                self.respawn()  # Respawn instead of discarding
        else:
            # For non-enemy sprites (lasers, meteors, etc.)
            # Use the original bounds checking logic
            self.discard = not -200 < self.pos.y < WINDOW_HEIGHT + 200 or \
                        not -200 < self.pos.x < WINDOW_WIDTH + 200

    def get_center(self):
        return Vector2(
            self.pos.x + self.size.x / 2,
            self.pos.y + self.size.y / 2,
        )

    def update(self, dt):
        self.move(dt)
        self.check_discard()

    def draw(self):
        draw_texture_v(self.texture, self.pos, WHITE)
class Player(Sprite):
    def __init__(self, texture, pos, shoot_laser, game=None):
        super().__init__(texture, pos, PLAYER_SPEED, Vector2())
        self.game = game
        self.shoot_laser = shoot_laser
        self.health = 20
        self.max_health = 20
        self.rotation = 0
        self.rotation_speed = 150
        self.rect = Rectangle(0, 0, self.size.x, self.size.y)
        self.rotate_direction = 0
        self.draw_offset = Vector2(50, 40)
        self.center_pos = Vector2(
            self.pos.x + texture.width/2,
            self.pos.y + texture.height/2
        )
        self.weapons = []
        self.current_weapon_index = 0
        self.collision_radius = min(texture.width, texture.height) / 3
        
        # Keep these for code compatibility but no actual invulnerability
        self.invulnerable = False
        self.invulnerable_timer = 0
        self.flash_timer = 0
        self.visible = True

    def take_damage(self):
        # Decrease health immediately
        self.health -= 1
        
        # Check for death
        if self.health <= 0:
            if self.game:
                self.game.game_over = True
        
        # Return true to indicate hit happened
        return True
        
    def constraint(self):
        # Keep the player within the screen boundaries
        min_x = 0 - self.draw_offset.x
        min_y = 0 - self.draw_offset.y
        max_x = WINDOW_WIDTH - self.size.x - self.draw_offset.x
        max_y = WINDOW_HEIGHT - self.size.y - self.draw_offset.y
        self.pos.x = max(min_x, min(self.pos.x, max_x))
        self.pos.y = max(min_y, min(self.pos.y, max_y))
        # Update center position
        self.center_pos.x = self.pos.x + self.size.x/2
        self.center_pos.y = self.pos.y + self.size.y/2

    def move(self, dt):
        # Log before movement
        initial_pos = Vector2(self.center_pos.x, self.center_pos.y)
        
        # Original movement code
        self.pos.x += self.direction.x * self.speed * dt
        self.pos.y += self.direction.y * self.speed * dt
        
        # Update center position
        self.center_pos.x = self.pos.x + self.size.x/2
        self.center_pos.y = self.pos.y + self.size.y/2
        
        # Log after movement
        final_pos = Vector2(self.center_pos.x, self.center_pos.y)
        delta_x = final_pos.x - initial_pos.x
        delta_y = final_pos.y - initial_pos.y
        
        # Check if we've moved as expected
        expected_dx = self.direction.x * self.speed * dt
        expected_dy = self.direction.y * self.speed * dt
        
        if hasattr(self.game, 'logger'):
            self.game.logger.log_debug(f"Move: dt={dt:.4f}, dir=({self.direction.x:.2f},{self.direction.y:.2f}), " +
                                    f"delta=({delta_x:.2f},{delta_y:.2f}), " +
                                    f"expected=({expected_dx:.2f},{expected_dy:.2f})")
    
    def input(self):
        # Store original direction
        original_dir = Vector2(self.direction.x, self.direction.y)
        self.direction.x = int(is_key_down(KEY_RIGHT)) - int(is_key_down(KEY_LEFT))
        self.direction.y = int(is_key_down(KEY_DOWN)) - int(is_key_down(KEY_UP))
        self.direction = Vector2Normalize(self.direction)

         # Handle rotation with A and S keys
        self.rotate_direction = int(is_key_down(KEY_D)) - int(is_key_down(KEY_A))

        if is_key_pressed(KEY_ONE):
            self.current_weapon_index = 0
        elif is_key_pressed(KEY_TWO) and len(self.weapons) > 1 and self.weapons[1].unlocked:
            self.current_weapon_index = 1
        elif is_key_pressed(KEY_THREE) and len(self.weapons) > 2 and self.weapons[2].unlocked:
            self.current_weapon_index = 2
        
        # Log if direction changed
        if original_dir.x != self.direction.x or original_dir.y != self.direction.y:
            print(f"INPUT: direction changed from ({original_dir.x:.2f},{original_dir.y:.2f}) "
                f"to ({self.direction.x:.2f},{self.direction.y:.2f})")

        if is_key_pressed(KEY_SPACE):
            self.fire_weapon()
            # # Calculate laser position and direction based on player rotation
            # angle_rad = radians(self.rotation )  # -90 because sprite faces up by default
            
            # # Calculate offset from center in the direction of rotation
            # offset_x = sin(angle_rad) * (self.size.y / 2)
            # offset_y = -cos(angle_rad) * (self.size.y / 2)
            
            # # Calculate center position of player
            # center = self.get_center()
            
            # # Calculate laser position (from the front of the ship)
            # laser_pos = Vector2(
            #     center.x + offset_x,
            #     center.y + offset_y
            # )
            
            # # Calculate direction vector based on rotation
            # laser_dir = Vector2(sin(angle_rad), -cos(angle_rad))
            
            # # Call shoot_laser with position and direction
            # self.shoot_laser(laser_pos, laser_dir)
    
    def fire_weapon(self):
        """Fire the current weapon"""
        if not self.weapons:
            return
            
        # Calculate firing position and direction
        angle_rad = radians(self.rotation )  # +90 because default sprite faces up
        
        # Calculate offset from the center (front of the ship)
        offset_x = sin(angle_rad) * (self.size.y / 2)
        offset_y = -cos(angle_rad) * (self.size.y / 2)
        
        # Calculate firing position
        laser_pos = Vector2(
            self.center_pos.x + offset_x,
            self.center_pos.y + offset_y
        )
        
        # Calculate direction vector - this is the key fix
        laser_dir = Vector2(sin(angle_rad), -cos(angle_rad))
        
        # Fire the current weapon
        self.weapons[self.current_weapon_index].fire(laser_pos, laser_dir)

    def update(self, dt):
        # Add debug print
        
        # Move player based on current direction and speed
        self.move(dt)
        self.constraint()
        
        # Apply rotation based on input and time delta
        self.rotation += self.rotate_direction * self.rotation_speed * dt
        
        # Normalize rotation to 0-360 range
        while self.rotation >= 360:
            self.rotation -= 360
        while self.rotation < 0:
            self.rotation += 360
        
        # Update weapons
        for weapon in self.weapons:
            weapon.update(dt)
        
        # Handle invulnerability period
        if hasattr(self, 'invulnerable') and self.invulnerable:
            self.invulnerable_timer += dt
            self.flash_timer += dt
            
            # Visual feedback - flash the player ship
            if self.flash_timer >= 0.1:  # Flash every 0.1 seconds
                self.visible = not self.visible
                self.flash_timer = 0
                
            # End invulnerability after 2 seconds
            if self.invulnerable_timer >= 2.0:
                self.invulnerable = False
                self.visible = True  # Ensure player is visible
        
    
    def get_center(self):
        # Return the actual center position (for collision detection)
        return self.center_pos
    
    def draw(self):
        # Only draw if visible (controls flashing effect)
        if self.visible:
            # Draw the texture with offset and rotation
            draw_texture_pro(
                self.texture,
                self.rect,
                Rectangle(
                    self.pos.x + self.draw_offset.x, 
                    self.pos.y + self.draw_offset.y,
                    self.size.x, 
                    self.size.y
                ),
                Vector2(self.size.x/2, self.size.y/2),
                self.rotation,
                WHITE
            )
            
            # Draw hitbox as a red circle at the actual center position
            draw_circle_lines(
                int(self.center_pos.x), 
                int(self.center_pos.y), 
                self.collision_radius, 
                RED
            )
class Laser(Sprite):
    def __init__(self, texture, pos, direction=Vector2(0,-1)):
        # Allow custom direction parameter with default to original up direction
        super().__init__(texture, pos, LASER_SPEED, direction)
        # Add rotation to match direction
        self.rotation = degrees(atan2(direction.y, direction.x)) + 90
        self.rect = Rectangle(0, 0, self.size.x, self.size.y)
    
    def get_rect(self):
        return Rectangle(self.pos.x, self.pos.y, self.size.x, self.size.y)
        
    def draw(self):
        # Draw laser with rotation
        target_rect = Rectangle(self.pos.x, self.pos.y, self.size.x, self.size.y)
        draw_texture_pro(
            self.texture,
            self.rect,
            target_rect,
            Vector2(self.size.x / 2, self.size.y / 2),
            self.rotation,
            WHITE
        )

class Meteor(Sprite):
    def __init__(self, texture):
        pos = Vector2(randint(0, WINDOW_WIDTH), randint(-150,-50))
        speed = randint(*METEOR_SPEED_RANGE)
        direction = Vector2(random.uniform(-0.5, 0.5),1)
        super().__init__(texture, pos, speed, direction)
        self.rotation = 0
        self.rect = Rectangle(0, 0, self.size.x, self.size.y)
    
    def update(self, dt):
        super().update(dt)
        self.rotation += 50 * dt
    
    def get_center(self):
        return self.pos

    def draw(self):
        target_rect = Rectangle(self.pos.x, self.pos.y, self.size.x, self.size.y)
        draw_texture_pro(self.texture,self.rect,target_rect,Vector2(self.size.x / 2, self.size.y / 2),self.rotation, WHITE)

class ExplosionAnimation:
    def __init__(self, pos, textures):
        self.textures = textures
        self.size = Vector2(textures[0].width, textures[0].height)
        self.pos = Vector2(pos.x - self.size.x / 2, pos.y - self.size.y / 2)
        self.index = 0
        self.discard = False

    def update(self, dt):
        self.index += 20 * dt
        if self.index >= len(self.textures) - 1:
            self.index = len(self.textures) - 1  # Clamp to last valid index
            self.discard = True
        

    def draw(self):
        draw_texture_v(self.textures[int(self.index)], self.pos, WHITE)

class Enemy(Sprite):
    def __init__(self, texture, shoot_laser, enemy_laser_texture):
        # Define the active play area (larger than screen but not too large)
        active_area_margin = 200  # How far beyond screen edges the active area extends
        
        # Choose a spawn position outside visible screen but inside active area
        spawn_options = [
            # Top area (above screen)
            (randint(0, WINDOW_WIDTH), randint(-active_area_margin, -10)),
            # Left area (left of screen)
            (randint(-active_area_margin, -10), randint(0, WINDOW_HEIGHT)),
            # Right area (right of screen)
            (randint(WINDOW_WIDTH + 10, WINDOW_WIDTH + active_area_margin), randint(0, WINDOW_HEIGHT))
        ]
        
        # Choose one of the spawn areas, with higher probability for top area
        r = random.random()
        if r < 0.7:  # 70% chance for top spawn
            spawn_index = 0
        elif r < 0.85:  # 15% chance for left spawn
            spawn_index = 1
        else:  # 15% chance for right spawn
            spawn_index = 2
            
        start_pos = Vector2(spawn_options[spawn_index][0], spawn_options[spawn_index][1])
        
        # Initialize direction to point toward center of screen
        center_x, center_y = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
        dx = center_x - start_pos.x
        dy = center_y - start_pos.y
        length = sqrt(dx*dx + dy*dy)
        
        # Base direction aims toward center with some randomness
        if length > 0:
            direction = Vector2(
                dx/length + random.uniform(-0.3, 0.3),
                dy/length + random.uniform(-0.3, 0.3)
            )
        else:
            direction = Vector2(random.uniform(-0.3, 0.3), 1)
        
        # Call the parent constructor with these calculated values
        super().__init__(
            texture=texture,
            pos=start_pos,
            speed=ENEMY_SPEED * 1.4,
            direction=Vector2Normalize(direction)  # Ensure normalized direction
        )
         # Add visual offset (adjust these values as needed)
        self.draw_offset = Vector2(50, 40)  # Start with (0,0)
        # Hitbox properties
        self.center_pos = Vector2(
            self.pos.x + texture.width/2,
            self.pos.y + texture.height/2
        )
        self.collision_radius = max(self.size.x, self.size.y) / 2

        # Add a flag to track when enemy has entered the screen
        self.entered_screen = False
         # NEW: Offscreen timer to detect lost enemies
        self.offscreen_timer = 0
        
        # Combat properties
        self.shoot_laser = shoot_laser
        self.enemy_laser_texture = enemy_laser_texture
        self.shoot_timer = 0
        self.shoot_interval = random.uniform(0.8, 1.5)
        
        # Movement/rotation properties
        self.movement_pattern = randint(0, 3)
        self.pattern_timer = 0
        self.health = randint(1, 3)
        self.target_player = None
        self.rotation = 0
        self.rotation_speed = 50
        self.rect = Rectangle(0, 0, self.size.x, self.size.y)

    def move(self, dt):
        # Update CENTER position first
        self.center_pos = Vector2Add(
            self.center_pos, 
            Vector2Scale(self.direction, self.speed * dt)
        )
        
        # Sync POSITION with CENTER (critical fix)
        self.pos.x = self.center_pos.x - self.texture.width/2
        self.pos.y = self.center_pos.y - self.texture.height/2

        # Check if enemy has entered the screen (passed a certain Y threshold)
        if self.center_pos.y > 50:  # Allow them to come onscreen a bit
            self.entered_screen = True
        
        # Apply constraints only after enemy has entered the screen
        if self.entered_screen:
            self.constrain()
        
        # Update rotation
        self.rotation += self.rotation_speed * dt
    
    def constrain(self):
        """Keep the enemy within the active play area or respawn them"""
        # Define active play area boundaries (larger than screen)
        active_area_margin = 200
        min_x = -active_area_margin
        max_x = WINDOW_WIDTH + active_area_margin
        min_y = -active_area_margin
        max_y = WINDOW_HEIGHT + active_area_margin
        
        # Check if enemy is WAY out of bounds (this would be unusual)
        far_out_of_bounds = (self.center_pos.x < min_x - 100 or 
                            self.center_pos.x > max_x + 100 or
                            self.center_pos.y < min_y - 100 or 
                            self.center_pos.y > max_y + 100)
        
        if far_out_of_bounds:
            # Something went wrong - enemy is WAY outside play area
            # Respawn them instead of trying to constrain
            self.respawn()
            return
        
        # Normal boundary constraints for enemies just at the edges
        center_x, center_y = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
        near_edge = False
        
        # Constrain X position
        if self.center_pos.x < min_x:
            self.center_pos.x = min_x
            self.direction.x = abs(self.direction.x) + 0.1
            near_edge = True
        elif self.center_pos.x > max_x:
            self.center_pos.x = max_x
            self.direction.x = -abs(self.direction.x) - 0.1
            near_edge = True
            
        # Constrain Y position
        if self.center_pos.y < min_y:
            self.center_pos.y = min_y
            self.direction.y = abs(self.direction.y) + 0.1
            near_edge = True
        elif self.center_pos.y > max_y:
            self.center_pos.y = max_y
            self.direction.y = -abs(self.direction.y) - 0.1
            near_edge = True
        
        # If near an edge, add a bias toward the center
        if near_edge:
            # Calculate direction toward center
            dx = center_x - self.center_pos.x
            dy = center_y - self.center_pos.y
            length = sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Blend current direction with center direction
                center_weight = 0.3  # How strongly to pull toward center
                self.direction.x = self.direction.x * (1 - center_weight) + (dx/length) * center_weight
                self.direction.y = self.direction.y * (1 - center_weight) + (dy/length) * center_weight
                
                # Normalize direction
                dir_length = sqrt(self.direction.x**2 + self.direction.y**2)
                if dir_length > 0:
                    self.direction.x /= dir_length
                    self.direction.y /= dir_length
        
        # Update position based on center
        self.pos.x = self.center_pos.x - self.texture.width/2
        self.pos.y = self.center_pos.y - self.texture.height/2

    def respawn(self):
        """Respawn the enemy at a new position at the top of the screen"""
        # Create a new random starting position
        self.center_pos = Vector2(
            randint(100, WINDOW_WIDTH - 100),  # Not too close to edges
            randint(-80, -30)                 # Just above the visible screen
        )
        
        # Reset position based on center
        self.pos.x = self.center_pos.x - self.texture.width/2
        self.pos.y = self.center_pos.y - self.texture.height/2
        
        # Reset direction to head downward with slight randomness
        self.direction = Vector2(random.uniform(-0.3, 0.3), 1)
        
        # Reset entered_screen flag
        self.entered_screen = False
        
        # Print debug info
        print(f"Enemy {getattr(self, 'debug_type', 'UNKNOWN')} respawned at ({self.center_pos.x:.1f}, {self.center_pos.y:.1f})")

    def update(self, dt):
        self.move(dt)
        self.check_discard()
        self.pattern_timer += dt

        # Movement patterns
        if self.movement_pattern == 1:  # Zigzag
            if self.pattern_timer > 0.8:
                self.direction.x = random.uniform(-0.7, 0.7)
                self.pattern_timer = 0
        elif self.movement_pattern == 2:  # Swooping
            self.direction.x = sin(self.pattern_timer * 3) * 0.8
        elif self.movement_pattern == 3 and self.target_player:  # Aggressive
            if self.pattern_timer > 1.5:
                player_pos = self.target_player.get_center()
                dx = player_pos.x - self.center_pos.x
                dy = player_pos.y - self.center_pos.y
                dist = sqrt(dx*dx + dy*dy)
                
                if dist > 0:
                    self.direction.x = dx / dist * 1.5
                    self.direction.y = dy / dist * 1.5
                self.pattern_timer = 0

        # NEW: Anti-stalling behavior - if enemy hasn't moved much in a while
        # Every 5 seconds, check if enemy has been moving and adjust if needed
        if hasattr(self, 'last_pos_check'):
            self.time_since_check += dt
            if self.time_since_check > 5.0:
                # Calculate distance moved since last check
                dx = self.center_pos.x - self.last_pos.x
                dy = self.center_pos.y - self.last_pos.y
                distance_moved = sqrt(dx*dx + dy*dy)
                
                # If enemy hasn't moved much
                if distance_moved < 50:
                    # Force movement toward center of screen
                    self.direction = Vector2Normalize(
                        Vector2(
                            (WINDOW_WIDTH/2) - self.center_pos.x,
                            (WINDOW_HEIGHT/2) - self.center_pos.y
                        )
                    )
                
                # Reset check
                self.last_pos = Vector2(self.center_pos.x, self.center_pos.y)
                self.time_since_check = 0
        else:
            # Initialize position tracking on first update
            self.last_pos = Vector2(self.center_pos.x, self.center_pos.y)
            self.time_since_check = 0
        
        # NEW: Offscreen check/update
        if (self.center_pos.x < 0 or self.center_pos.x > WINDOW_WIDTH or
            self.center_pos.y < 0 or self.center_pos.y > WINDOW_HEIGHT):
            self.offscreen_timer += dt
        else:
            self.offscreen_timer = 0

        if self.offscreen_timer > 5.0:
            self.respawn()
            self.offscreen_timer = 0

        # Rotation toward player
        if self.target_player:
            self.update_rotation()

        # Shooting
        self.shoot_timer += dt
        if self.shoot_timer >= self.shoot_interval:
            self.fire_laser()
            self.shoot_timer = 0
            self.shoot_interval = random.uniform(0.8, 1.5)

    def update_rotation(self):
        player_pos = self.target_player.get_center()
        dx = player_pos.x - self.center_pos.x
        dy = player_pos.y - self.center_pos.y
        angle = degrees(atan2(dy, dx))
        self.rotation = (angle + 90) + 180

    def fire_laser(self):
        angle_rad = radians(self.rotation - 180)
        offset_x = sin(angle_rad) * (self.size.y / 2)
        offset_y = -cos(angle_rad) * (self.size.y / 2)
        
        laser_pos = Vector2(
            self.center_pos.x + offset_x,
            self.center_pos.y + offset_y
        )
        
        if self.target_player:
            player_pos = self.target_player.get_center()
            dx = player_pos.x - laser_pos.x
            dy = player_pos.y - laser_pos.y
            dist = sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                direction = Vector2(dx/dist, dy/dist)
                self.shoot_laser(self.enemy_laser_texture, laser_pos, direction)
        else:
            self.shoot_laser(self.enemy_laser_texture, laser_pos, Vector2(0, 1))

    def take_damage(self):
        self.health -= 1
        return self.health <= 0

    def get_center(self):
        return self.center_pos

    def draw(self):
        # Apply offset to drawing position ONLY
        draw_texture_pro(
            self.texture,
            Rectangle(0, 0, self.texture.width, self.texture.height),
            Rectangle(
                self.pos.x + self.draw_offset.x,
                self.pos.y + self.draw_offset.y,
                self.texture.width,
                self.texture.height
            ),
            Vector2(self.texture.width/2, self.texture.height/2),
            self.rotation,
            WHITE
        )
        
        # Hitbox remains at original center_pos (no changes)
        draw_circle_lines(int(self.center_pos.x), int(self.center_pos.y), self.collision_radius, RED)

        # Draw enemy type and health for debugging
        if hasattr(self, 'debug_type'):
            debug_text = f"{self.debug_type} ({self.health}HP)"
            text_width = measure_text(debug_text, 12)
            draw_text(
                debug_text,
                int(self.center_pos.x - text_width/2),
                int(self.center_pos.y - self.collision_radius - 20),
                12,
                YELLOW
            )
        
        # If enemy is near screen edge, draw an arrow pointing to it
        margin = 50
        if (self.center_pos.x < margin or 
            self.center_pos.x > WINDOW_WIDTH - margin or
            self.center_pos.y < margin or 
            self.center_pos.y > WINDOW_HEIGHT - margin):
            
            # Calculate arrow position pointing toward the enemy
            arrow_x = max(margin, min(self.center_pos.x, WINDOW_WIDTH - margin))
            arrow_y = max(margin, min(self.center_pos.y, WINDOW_HEIGHT - margin))
            
            # Draw arrow
            draw_circle(int(arrow_x),int(arrow_y), 5, RED)

class TankEnemy(Enemy):
    """Slow but high-health enemy that can absorb more damage"""
    def __init__(self, texture, shoot_laser, enemy_laser_texture, game=None):
        super().__init__(texture, shoot_laser, enemy_laser_texture)
        self.health = 5
        self.game=game
        self.speed = ENEMY_SPEED * 0.6  # 60% of normal speed
        self.collision_radius *= 1.2  # Larger hitbox
        self.worth = 200  # Points when destroyed
        self.movement_pattern = 0  # Mostly straight path
        
        # Use tank's specific weapon
        self.weapon = TankCannon(game)
        
    def update(self, dt):
        super().update(dt)
        # Update weapon cooldown
        self.weapon.update(dt)
        
    def fire_laser(self):
        # Calculate EXACT firing position and direction like the base class
        angle_rad = radians(self.rotation - 180)
        offset_x = sin(angle_rad) * (self.size.y / 2)
        offset_y = -cos(angle_rad) * (self.size.y / 2)
        
        laser_pos = Vector2(
            self.center_pos.x + offset_x,
            self.center_pos.y + offset_y
        )
        
        if self.target_player:
            player_pos = self.target_player.get_center()
            dx = player_pos.x - laser_pos.x
            dy = player_pos.y - laser_pos.y
            dist = sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                direction = Vector2(dx/dist, dy/dist)
                # NOW pass to weapon system
                self.weapon.fire(laser_pos, direction)
        else:
            self.weapon.fire(laser_pos, Vector2(0, 1))


class SwarmEnemy(Enemy):
    """Fast but weak enemy that moves in erratic patterns"""
    def __init__(self, texture, shoot_laser, enemy_laser_texture, game=None):
        super().__init__(texture, shoot_laser, enemy_laser_texture)
        self.health = 1
        self.game = game
        self.speed = ENEMY_SPEED * 1.8  # Much faster
        self.collision_radius *= 0.8  # Smaller hitbox
        self.worth = 50  # Fewer points
        self.movement_pattern = 1  # Always zigzag pattern
        self.pattern_timer = 0
        self.direction_change_interval = 0.4  # Change direction frequently
        
        # Use swarm's specific weapon
        self.weapon = SwarmBlaster(game)
        
    def update(self, dt):
        """Override update to implement more erratic movement"""
        self.move(dt)
        self.check_discard()
        self.pattern_timer += dt
        
        # More erratic zigzag with shorter intervals
        if self.pattern_timer > self.direction_change_interval:
            self.direction.x = random.uniform(-0.9, 0.9)  # More extreme horizontal movement
            self.pattern_timer = 0
            
        # Random vertical speed changes
        if random.random() < 0.02:  # 2% chance per frame to change vertical direction
            self.direction.y = random.uniform(-0.5, 1.0)  # Occasionally move upward
            
        # Update weapon cooldown
        self.weapon.update(dt)

            # NEW: Anti-stalling behavior - if enemy hasn't moved much in a while
        # Every 5 seconds, check if enemy has been moving and adjust if needed
        if hasattr(self, 'last_pos_check'):
            self.time_since_check += dt
            if self.time_since_check > 5.0:
                # Calculate distance moved since last check
                dx = self.center_pos.x - self.last_pos.x
                dy = self.center_pos.y - self.last_pos.y
                distance_moved = sqrt(dx*dx + dy*dy)
                
                # If enemy hasn't moved much
                if distance_moved < 50:
                    # Force movement toward center of screen
                    self.direction = Vector2Normalize(
                        Vector2(
                            (WINDOW_WIDTH/2) - self.center_pos.x,
                            (WINDOW_HEIGHT/2) - self.center_pos.y
                        )
                    )
                
                # Reset check
                self.last_pos = Vector2(self.center_pos.x, self.center_pos.y)
                self.time_since_check = 0
        else:
            # Initialize position tracking on first update
            self.last_pos = Vector2(self.center_pos.x, self.center_pos.y)
            self.time_since_check = 0
        
        # Shooting logic - more frequent
        self.shoot_timer += dt
        if self.shoot_timer >= 0.7:  # Shorter interval
            self.fire_laser()
            self.shoot_timer = 0
            
        # Update rotation toward player if we have a target
        if self.target_player:
            self.update_rotation()
            
    def fire_laser(self):
        # Calculate EXACT firing position and direction like the base class
        angle_rad = radians(self.rotation - 180)
        offset_x = sin(angle_rad) * (self.size.y / 2)
        offset_y = -cos(angle_rad) * (self.size.y / 2)
        
        laser_pos = Vector2(
            self.center_pos.x + offset_x,
            self.center_pos.y + offset_y
        )
        
        if self.target_player:
            player_pos = self.target_player.get_center()
            dx = player_pos.x - laser_pos.x
            dy = player_pos.y - laser_pos.y
            dist = sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                direction = Vector2(dx/dist, dy/dist)
                # NOW pass to weapon system
                self.weapon.fire(laser_pos, direction)
        else:
            self.weapon.fire(laser_pos, Vector2(0, 1))


class SniperEnemy(Enemy):
    """Long-range enemy that fires accurately from a distance"""
    def __init__(self, texture, shoot_laser, enemy_laser_texture, game=None):
        super().__init__(texture, shoot_laser, enemy_laser_texture)
        self.health = 2
        self.game = game
        self.speed = ENEMY_SPEED * 0.8  # Slower than normal
        self.worth = 150
        self.target_distance = 400  # Tries to maintain this distance from player
        
        # Use sniper's specific weapon
        self.weapon = SniperRifle(game)
        
    def update(self, dt):
        """Override to implement distance-keeping behavior"""
        super().update(dt)
        
        # Update weapon cooldown
        self.weapon.update(dt)
        
        # Try to maintain optimal sniping distance from player
        if self.target_player and self.entered_screen:
            player_pos = self.target_player.get_center()
            dx = player_pos.x - self.center_pos.x
            dy = player_pos.y - self.center_pos.y
            dist = sqrt(dx*dx + dy*dy)
            
            if dist < self.target_distance - 50:
                # Too close, back away
                self.direction.x = -dx / dist if dist > 0 else 0
                self.direction.y = -dy / dist if dist > 0 else -1
            elif dist > self.target_distance + 50:
                # Too far, move closer
                self.direction.x = dx / dist if dist > 0 else 0
                self.direction.y = dy / dist if dist > 0 else 1
        
            # NEW: Anti-stalling behavior - if enemy hasn't moved much in a while
        # Every 5 seconds, check if enemy has been moving and adjust if needed
        if hasattr(self, 'last_pos_check'):
            self.time_since_check += dt
            if self.time_since_check > 5.0:
                # Calculate distance moved since last check
                dx = self.center_pos.x - self.last_pos.x
                dy = self.center_pos.y - self.last_pos.y
                distance_moved = sqrt(dx*dx + dy*dy)
                
                # If enemy hasn't moved much
                if distance_moved < 50:
                    # Force movement toward center of screen
                    self.direction = Vector2Normalize(
                        Vector2(
                            (WINDOW_WIDTH/2) - self.center_pos.x,
                            (WINDOW_HEIGHT/2) - self.center_pos.y
                        )
                    )
                
                # Reset check
                self.last_pos = Vector2(self.center_pos.x, self.center_pos.y)
                self.time_since_check = 0
        else:
            # Initialize position tracking on first update
            self.last_pos = Vector2(self.center_pos.x, self.center_pos.y)
            self.time_since_check = 0
        
        # Calculate firing position for the weapon's charging effect
        if hasattr(self.weapon, 'charging') and self.weapon.charging:
            angle_rad = radians(self.rotation - 180)
            offset_x = sin(angle_rad) * (self.size.y / 2)
            offset_y = -cos(angle_rad) * (self.size.y / 2)
            
            laser_pos = Vector2(
                self.center_pos.x + offset_x,
                self.center_pos.y + offset_y
            )
            
            # Update the position where the charging effect should be drawn
            self.weapon.set_charge_position(laser_pos)
                
    def fire_laser(self):
        """Use the sniper's specialized weapon"""
        angle_rad = radians(self.rotation - 180)
        offset_x = sin(angle_rad) * (self.size.y / 2)
        offset_y = -cos(angle_rad) * (self.size.y / 2)
        
        laser_pos = Vector2(
            self.center_pos.x + offset_x,
            self.center_pos.y + offset_y
        )
        
        if self.target_player:
            player_pos = self.target_player.get_center()
            dx = player_pos.x - laser_pos.x
            dy = player_pos.y - laser_pos.y
            dist = sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                direction = Vector2(dx/dist, dy/dist)
                self.weapon.fire(laser_pos, direction)
        else:
            self.weapon.fire(laser_pos, Vector2(0, 1))
    
    def draw(self):
        # First draw the enemy
        super().draw()
        
        # Then draw the charging effect if needed
        if hasattr(self.weapon, 'draw_charging_effect'):
            self.weapon.draw_charging_effect()


class BomberEnemy(Enemy):
    """Area-effect enemy that drops spread shots"""
    def __init__(self, texture, shoot_laser, enemy_laser_texture, game=None):
        super().__init__(texture, shoot_laser, enemy_laser_texture)
        self.health = 3
        self.speed = ENEMY_SPEED * 0.9
        self.worth = 175
        self.movement_pattern = 2  # Swooping pattern
        self.game=game
        
        # Use bomber's specific weapon
        self.weapon = BomberLauncher(game)
        
    def update(self, dt):
        super().update(dt)
        # Update weapon cooldown
        self.weapon.update(dt)
        
    def fire_laser(self):
        # Calculate EXACT firing position and direction like the base class
        angle_rad = radians(self.rotation - 180)
        offset_x = sin(angle_rad) * (self.size.y / 2)
        offset_y = -cos(angle_rad) * (self.size.y / 2)
        
        laser_pos = Vector2(
            self.center_pos.x + offset_x,
            self.center_pos.y + offset_y
        )
        
        if self.target_player:
            player_pos = self.target_player.get_center()
            dx = player_pos.x - laser_pos.x
            dy = player_pos.y - laser_pos.y
            dist = sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                direction = Vector2(dx/dist, dy/dist)
                # NOW pass to weapon system
                self.weapon.fire(laser_pos, direction)
        else:
            self.weapon.fire(laser_pos, Vector2(0, 1))


class EnemyLaser(Laser):
    def __init__(self, texture, pos, direction, speed_multiplier=1.0, scale=1.0, damage=1):
        super().__init__(texture, pos)
        self.direction = direction
        self.speed *= speed_multiplier
        self.scale = scale  # Visual size scaling
        self.damage = damage  # Damage amount
        self.rotation = degrees(atan2(direction.y, direction.x)) + 90
        self.rect = Rectangle(0, 0, self.size.x, self.size.y)
        
        # Add center offset for consistency
        self.center_offset = Vector2(self.size.x / 2, self.size.y / 2)
        
    def draw(self):
        # Draw laser with proper rotation to match its direction
        target_rect = Rectangle(
            self.pos.x, 
            self.pos.y, 
            self.size.x * self.scale, 
            self.size.y * self.scale
        )
        
        draw_texture_pro(
            self.texture,
            self.rect,
            target_rect,
            Vector2(self.size.x * self.scale / 2, self.size.y * self.scale / 2),
            self.rotation,
            WHITE
        )

class Weapon:
    """Base weapon class for the game"""
    def __init__(self, game):
        self.game = game
        self.name = "Standard Laser"
        self.description = "Basic single shot laser"
        self.cooldown = 0.35       # Time between shots
        self.current_cooldown = 0  # Current cooldown timer
        self.unlocked = True
        
    def update(self, dt):
        """Update weapon cooldown"""
        if self.current_cooldown > 0:
            self.current_cooldown -= dt
    
    def can_fire(self):
        """Check if weapon can fire"""
        return self.current_cooldown <= 0 and self.unlocked
    
    def fire(self, pos, direction):
        """Fire the weapon"""
        if not self.can_fire():
            return False
            
        # Play sound
        play_sound(self.game.audio['laser'])
        
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Create a single projectile
        self.game.lasers.append(self.game.create_laser(pos, direction))
        
        return True


class TripleShot(Weapon):
    """Fires three projectiles in a spread pattern"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Triple Shot"
        self.description = "Fires 3 projectiles in a spread pattern"
        self.cooldown = 0.6
        self.spread_angle = 25  # Good spread angle
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Play sound
        play_sound(self.game.audio['laser'])
        
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Get the player's rotation angle directly
        player_rotation = self.game.player.rotation
        base_angle = player_rotation
        
        # Calculate spread positions
        center_rad = radians(base_angle)
        left_rad = radians(base_angle - self.spread_angle)
        right_rad = radians(base_angle + self.spread_angle)
        
        # Add a slight offset to starting positions
        offset = 10  # Pixels to offset each projectile
        
        # Create center projectile
        center_dir = Vector2(sin(center_rad), -cos(center_rad))
        self.game.lasers.append(Laser(self.game.assets['laser'], pos, center_dir))
        
        # Create left projectile with offset
        left_dir = Vector2(sin(left_rad), -cos(left_rad))
        left_offset = Vector2(sin(left_rad) * offset, -cos(left_rad) * offset)
        left_pos = Vector2(pos.x + left_offset.x, pos.y + left_offset.y)
        self.game.lasers.append(Laser(self.game.assets['laser'], left_pos, left_dir))
        
        # Create right projectile with offset
        right_dir = Vector2(sin(right_rad), -cos(right_rad))
        right_offset = Vector2(sin(right_rad) * offset, -cos(right_rad) * offset)
        right_pos = Vector2(pos.x + right_offset.x, pos.y + right_offset.y)
        self.game.lasers.append(Laser(self.game.assets['laser'], right_pos, right_dir))
        
        return True



class RapidFire(Weapon):
    """Fires quickly but with slightly lower damage"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Rapid Fire"
        self.description = "High rate of fire laser"
        self.cooldown = 0.2  # Much faster firing rate
        self.unlocked = False  # Starts locked


class EnemyWeapon:
    """Base weapon class for enemies"""
    def __init__(self, game):
        self.game = game
        self.name = "Standard Enemy Laser"
        self.cooldown = 1.0  # Default cooldown
        self.current_cooldown = 0
        self.damage = 1  # Standard damage
        
    def update(self, dt):
        """Update weapon cooldown"""
        if self.current_cooldown > 0:
            self.current_cooldown -= dt
            
    def can_fire(self):
        """Check if weapon can fire"""
        return self.current_cooldown <= 0
        
    def fire(self, pos, direction):
        """Fire the weapon"""
        if not self.can_fire():
            return False
        
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Create a single projectile
        self.game.shoot_enemy_laser(
            self.game.assets['enemy_laser'], 
            pos, 
            direction
        )
        return True


class TankCannon(EnemyWeapon):
    """Heavy cannon for Tank enemies - slow but powerful"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Tank Cannon"
        self.cooldown = 2.0  # Very slow firing rate
        self.damage = 1.5  # Double damage
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Tank shots are larger if possible (you could add a scale parameter to your laser)
        # For now we'll just use the regular asset
        self.game.shoot_enemy_laser(
            self.game.assets['cannon_laser'], 
            pos,
            direction
        )
        
        # Play a stronger sound if available
        play_sound(self.game.audio['enemy_laser'])
        
        return True


class SwarmBlaster(EnemyWeapon):
    """Fast, light weapon for Swarm enemies"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Swarm Blaster"
        self.cooldown = 0.5  # Very fast firing
        self.damage = 0.5  # Lower damage
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Add some randomness to direction (inaccurate)
        spread = 0.1  # Amount of random spread
        random_dir = Vector2(
            direction.x + random.uniform(-spread, spread),
            direction.y + random.uniform(-spread, spread)
        )
        
        # Normalize the direction vector
        length = sqrt(random_dir.x**2 + random_dir.y**2)
        if length > 0:
            random_dir.x /= length
            random_dir.y /= length
        
        # Fire with slight randomness to direction
        self.game.shoot_enemy_laser(
            self.game.assets['enemy_laser'], 
            pos,
            random_dir
        )
        
        return True

class SniperRifle(EnemyWeapon):
    """Precision long-range weapon for Sniper enemies that fires a powerful beam"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Sniper Rifle"
        self.cooldown = 3.0  # Very slow firing rate
        self.damage = 2.5  # High damage
        
        # Charge-up animation properties
        self.charging = False
        self.charge_time = 1.5  # Time to fully charge (seconds)
        self.current_charge = 0
        self.charge_color = Color(255, 0, 0, 100)  # Start with red
        self.full_charge_color = Color(255, 50, 50, 220)  # Bright red when fully charged
        self.charge_radius_min = 5
        self.charge_radius_max = 15
        self.charge_positions = []  # Will store the position for drawing
        
    def update(self, dt):
        """Update weapon cooldown and charging state"""
        if self.current_cooldown > 0:
            self.current_cooldown -= dt
            
            # Begin charging when cooldown is low enough
            if self.current_cooldown <= self.charge_time:
                self.charging = True
                self.current_charge = self.charge_time - self.current_cooldown
        else:
            # Ready to fire - fully charged
            self.charging = self.current_charge > 0
            
    def begin_charging(self, enemy_pos):
        """Start the charging sequence"""
        self.charging = True
        self.current_charge = 0
        self.charge_positions = [enemy_pos]  # Store position for drawing
        
    def set_charge_position(self, pos):
        """Update the position where the charge effect should be drawn"""
        self.charge_positions = [pos]
        
    def fire(self, pos, direction):
        if not self.can_fire():
            # If we can't fire but we're not charging yet, start charging
            if not self.charging and self.current_cooldown <= self.charge_time:
                self.begin_charging(pos)
                self.set_charge_position(pos)
            return False
            
        # Reset cooldown and charging
        self.current_cooldown = self.cooldown
        fully_charged = self.charging and self.current_charge >= self.charge_time
        self.charging = False
        self.current_charge = 0
        
        # Play charge-complete sound
        if fully_charged:
            play_sound(self.game.audio['enemy_laser'])
            
            # Calculate end position by extending from the start position along the direction
            # Use raycasting to find where the beam hits something
            target_pos = None
            
            if self.game.player and not self.game.player.invulnerable:
                # Aim directly at player 
                player_pos = self.game.player.get_center()
                target_pos = Vector2(player_pos.x, player_pos.y)
                
                # Create a laser beam that connects source to target
                beam = LaserBeam(
                    self.game,
                    self.game.assets['enemy_laser'],  # Texture (not really used for beam)
                    pos,                             # Start position
                    target_pos,                      # End position
                    0.3,                            # Lifetime in seconds
                    15                              # Width at source
                )
                
                # Add beam to a list of active beams in the game
                self.game.enemy_beams.append(beam)
                
                # Deal damage immediately 
                # This would need to be implemented in your game logic
                self.game.check_beam_collision(beam)
        else:
            # Not fully charged, fire regular laser instead
            self.game.shoot_enemy_laser(
                self.game.assets['enemy_laser'],
                pos,
                direction
            )
            
        return True
    
    def draw_charging_effect(self):
        """Draw the charging animation"""
        if not self.charging or not self.charge_positions:
            return
            
        # Calculate charge percentage and current radius
        charge_percent = min(1.0, self.current_charge / self.charge_time)
        current_radius = self.charge_radius_min + (self.charge_radius_max - self.charge_radius_min) * charge_percent
        
        # Interpolate between starting and full charge colors
        r = int(self.charge_color.r + (self.full_charge_color.r - self.charge_color.r) * charge_percent)
        g = int(self.charge_color.g + (self.full_charge_color.g - self.charge_color.g) * charge_percent)
        b = int(self.charge_color.b + (self.full_charge_color.b - self.charge_color.b) * charge_percent)
        a = int(self.charge_color.a + (self.full_charge_color.a - self.charge_color.a) * charge_percent)
        current_color = Color(r, g, b, a)
        
        # Draw the charging effect at the enemy's position
        for pos in self.charge_positions:
            # Draw an outer ring that grows with charge
            draw_circle_lines(int(pos.x), int(pos.y), current_radius, current_color)
            
            # Draw inner circle that pulses
            pulse_size = current_radius * 0.6 * (0.8 + 0.2 * sin(get_time() * 10))
            draw_circle(int(pos.x), int(pos.y), pulse_size, current_color)
            
            # Add targeting laser sight when nearly charged
            if charge_percent > 0.7:
                # Find the direction to the player
                if self.game and self.game.player:
                    player_pos = self.game.player.get_center()
                    dx = player_pos.x - pos.x
                    dy = player_pos.y - pos.y
                    dist = sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # Draw a thin line indicating targeting
                        fade_color = Color(current_color.r, current_color.g, current_color.b, 50)
                        draw_line(
                            int(pos.x), 
                            int(pos.y),
                            int(pos.x + dx * 1.5), 
                            int(pos.y + dy * 1.5),
                            fade_color
                        )

class BomberLauncher(EnemyWeapon):
    """Area-effect weapon for Bomber enemies"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Bomber Launcher"
        self.cooldown = 2.5  # Moderate firing rate
        self.damage = 1  # Standard damage
        self.spread_count = 5  # Number of projectiles in spread
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Fire multiple projectiles in a spread pattern
        spread_angle = 50  # Total spread in degrees
        angle_step = spread_angle / (self.spread_count - 1)
        
        # Calculate base angle from direction vector
        base_angle = degrees(atan2(direction.y, direction.x))
        
        # Fire spread pattern
        for i in range(self.spread_count):
            shot_angle = base_angle - (spread_angle/2) + (angle_step * i)
            shot_rad = radians(shot_angle)
            spread_dir = Vector2(cos(shot_rad), sin(shot_rad))
            
            self.game.shoot_enemy_laser(
                self.game.assets['enemy_laser'], 
                pos,
                spread_dir
            )
        
        # Play special sound if available
        play_sound(self.game.audio['enemy_laser'])
        
        return True

class LaserBeam(Sprite):
    """A beam weapon that looks like a thick energy rod with uniform thinning over time"""
    def __init__(self,game, texture, start_pos, end_pos, lifetime=0.5, max_width=20):
        # Initialize with zero speed since beams don't move traditionally
        super().__init__(texture, start_pos, 0, Vector2(0, 0))
        self.game=game
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.lifetime = lifetime  # Increased lifetime for more visible effect
        self.time_alive = 0
        self.max_width = max_width  # Increased max width for more impact
        
        # Calculate beam length and angle
        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y
        self.length = sqrt(dx*dx + dy*dy)
        self.angle = degrees(atan2(dy, dx))
        
        # Beams don't use the traditional collision rectangle
        self.collision_points = []
        self.generate_collision_points()
        
        # Visual effects - more intense colors
        self.core_color = Color(255, 30, 30, 255)  # Bright red core
        self.glow_color = Color(255, 120, 120, 180)  # Softer red glow
        self.impact_color = Color(255, 220, 220, 255)  # Bright impact flash
        
        # Effect stages
        self.flash_stage = True  # Initial bright flash
        self.flash_duration = 0.1  # Duration of initial flash
        
        # Sound has been played flag (to avoid repeated plays)
        self.sound_played = False
    
    def generate_collision_points(self):
        """Generate points along the beam for collision detection"""
        # Create collision points every 10 pixels along the beam
        points_count = int(self.length / 10)
        
        if points_count > 0:
            for i in range(points_count):
                t = i / (points_count - 1) if points_count > 1 else 0
                x = self.start_pos.x + (self.end_pos.x - self.start_pos.x) * t
                y = self.start_pos.y + (self.end_pos.y - self.start_pos.y) * t
                self.collision_points.append(Vector2(x, y))
    
    def update(self, dt):
        """Update beam lifetime and effects"""
        self.time_alive += dt
        
        # Play beam sound if not already played
        if not self.sound_played:
            # Ideally, you'd have a specific beam sound in your audio assets
            # If not, we'll use the existing laser sound with a slight pitch shift
            play_sound(self.game.audio['enemy_laser'])
            self.sound_played = True
        
        if self.time_alive >= self.lifetime:
            self.discard = True
            
        # Transition from flash stage to normal beam
        if self.flash_stage and self.time_alive > self.flash_duration:
            self.flash_stage = False
    
    def draw(self):
        """Draw the beam with uniform thickness that thins over time"""
        # Calculate current width based on remaining lifetime
        # Start with max width and thin to 0 over lifetime
        life_ratio = 1.0 - (self.time_alive / self.lifetime)
        
        # Apply easing function to make thinning more visually pleasing
        # Start fast, then slow down (cubic easing out)
        ease_ratio = life_ratio * life_ratio * life_ratio
        current_width = self.max_width * ease_ratio
        
        # During flash stage, make beam thicker and brighter
        if self.flash_stage:
            current_width = self.max_width * 1.5
            core_color = Color(255, 255, 255, 255)  # Pure white for flash
            glow_color = Color(255, 200, 200, 220)  # Bright glow
        else:
            # Normal colors with alpha based on remaining life
            alpha_core = int(255 * min(1.0, life_ratio * 1.2))  # Keep core visible longer
            alpha_glow = int(180 * life_ratio)
            
            core_color = Color(self.core_color.r, self.core_color.g, self.core_color.b, alpha_core)
            glow_color = Color(self.glow_color.r, self.glow_color.g, self.glow_color.b, alpha_glow)
        
        # Calculate perpendicular vectors for uniform beam width
        angle_rad = radians(self.angle)
        perpendicular_angle = angle_rad + radians(90)
        perp_x = cos(perpendicular_angle)
        perp_y = sin(perpendicular_angle)
        
        # Calculate half width for vertex positions
        half_width = current_width / 2
        
        # Create uniform width beam (same width at both ends)
        vertices = []
        vertices.append(Vector2(
            self.start_pos.x + perp_x * half_width,
            self.start_pos.y + perp_y * half_width
        ))
        vertices.append(Vector2(
            self.start_pos.x - perp_x * half_width, 
            self.start_pos.y - perp_y * half_width
        ))
        vertices.append(Vector2(
            self.end_pos.x - perp_x * half_width,
            self.end_pos.y - perp_y * half_width
        ))
        vertices.append(Vector2(
            self.end_pos.x + perp_x * half_width,
            self.end_pos.y + perp_y * half_width
        ))
        
        # Draw inner beam (bright core)
        DrawTriangleFan(vertices, 4, core_color)
        
        # Draw outer glow (slightly larger)
        glow_half_width = half_width * 1.5
        glow_vertices = []
        glow_vertices.append(Vector2(
            self.start_pos.x + perp_x * glow_half_width,
            self.start_pos.y + perp_y * glow_half_width
        ))
        glow_vertices.append(Vector2(
            self.start_pos.x - perp_x * glow_half_width, 
            self.start_pos.y - perp_y * glow_half_width
        ))
        glow_vertices.append(Vector2(
            self.end_pos.x - perp_x * glow_half_width,
            self.end_pos.y - perp_y * glow_half_width
        ))
        glow_vertices.append(Vector2(
            self.end_pos.x + perp_x * glow_half_width,
            self.end_pos.y + perp_y * glow_half_width
        ))
        
        # Draw glow effect (drawn first so it appears behind the main beam)
        DrawTriangleFan(glow_vertices, 4, glow_color)
        
        # Draw impact effects at both ends
        # Impact at target end
        impact_size = current_width * (0.8 + 0.4 * sin(get_time() * 15))  # Pulsing effect
        DrawCircle(int(self.end_pos.x), int(self.end_pos.y), 
                  impact_size, 
                  core_color)
        
        # Smaller impact at source end
        DrawCircle(int(self.start_pos.x), int(self.start_pos.y), 
                  impact_size * 0.7, 
                  core_color)