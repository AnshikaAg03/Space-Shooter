from ai.utils import *
from sprites import *
class GeneticEnemyEvolver:
    """Evolves enemy parameters through a genetic algorithm to counter player strategies"""
    
    def __init__(self, game):
        self.game = game
        self.population_size = 20
        self.generations = 0
        self.elite_count = 2  # Number of top genomes to preserve unchanged
        
        # Genome parameters and their ranges
        self.genome_params = {
            'speed_factor': (0.7, 1.8),      # Speed multiplier
            'health_factor': (0.8, 2.0),     # Health multiplier
            'aggressiveness': (0.2, 1.0),    # Tendency to target player
            'firing_frequency': (0.5, 1.5),  # Multiplier for how often to fire
            'movement_pattern_weights': [(0.1, 1.0)] * 4,  # Weights for 4 movement patterns
            'target_distance': (100, 500),   # Preferred distance from player
            'evasion': (0.0, 1.0),          # Tendency to evade player shots
            'formation_cohesion': (0.0, 1.0) # Tendency to maintain formation with other enemies
        }
        
        # Initialize population - list of enemy parameter sets
        self.genomes = []
        self.initialize_population()
        
        # Fitness tracking
        self.fitness_scores = np.zeros(self.population_size)
        self.genome_usage_count = np.zeros(self.population_size)
        
        # Map to track which enemies are using which genomes
        self.enemy_genome_map = {}  # enemy_id -> genome_index
        
        # Evolution parameters
        self.mutation_rate = 0.2  # Probability of mutating a gene
        self.mutation_amount = 0.3  # How much to mutate by
        self.crossover_probability = 0.7  # Probability of crossover vs. copying
    
    def initialize_population(self):
        """Create initial random population of genomes"""
        for i in range(self.population_size):
            genome = {}
            
            # Generate random values for each parameter within its range
            for param, value_range in self.genome_params.items():
                if param == 'movement_pattern_weights':
                    # Handle special case of list parameter
                    genome[param] = [random.uniform(r[0], r[1]) for r in value_range]
                else:
                    # Handle scalar parameters
                    genome[param] = random.uniform(value_range[0], value_range[1])
            
            self.genomes.append(genome)
    
    def select_genome_for_enemy(self, enemy_type):
        """Select a genome to apply to a new enemy of the given type"""
        # Prioritize genomes with better fitness scores but that haven't been used much
        
        # Calculate selection weights based on fitness and usage
        if np.sum(self.fitness_scores) > 0:
            # Normalize fitness to 0-1 range
            normalized_fitness = self.fitness_scores / np.max(self.fitness_scores)
            
            # Adjust for usage count (prefer less used genomes)
            usage_penalty = self.genome_usage_count / (np.max(self.genome_usage_count) + 1)
            selection_weights = normalized_fitness - (0.3 * usage_penalty)
            
            # Ensure all weights are positive
            selection_weights = np.maximum(0.1, selection_weights)
        else:
            # If no fitness data yet, use uniform weights
            selection_weights = np.ones(self.population_size)
        
        # Select genome
        genome_index = random.choices(range(self.population_size), 
                                      weights=selection_weights, k=1)[0]
        
        # Update usage count
        self.genome_usage_count[genome_index] += 1
        
        return genome_index
    
    def apply_genome_to_enemy(self, enemy, genome_index):
        """Apply genetic parameters to a new enemy"""
        # Ensure valid genome index
        if genome_index < 0 or genome_index >= len(self.genomes):
            print(f"Warning: Invalid genome index {genome_index}")
            return enemy
            
        genome = self.genomes[genome_index]
        
        # Store reference to which genome this enemy uses
        self.enemy_genome_map[id(enemy)] = genome_index
        
        # Apply genome parameters to enemy
        
        # 1. Speed factor
        enemy.speed *= genome['speed_factor']
        
        # 2. Health factor (with type-specific adjustments)
        base_health = enemy.health
        enemy.health = max(1, int(base_health * genome['health_factor']))
        
        # 3. Movement pattern
        # Normalize weights
        weights = genome['movement_pattern_weights']
        weight_sum = sum(weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in weights]
            # Choose movement pattern based on weights
            enemy.movement_pattern = random.choices(range(len(weights)), 
                                                  weights=normalized_weights, k=1)[0]
        
        # 4. Type-specific adjustments
        if isinstance(enemy, SniperEnemy):
            # Adjust preferred target distance
            enemy.target_distance = genome['target_distance']
            
            # Adjust shooting interval
            if hasattr(enemy, 'weapon') and hasattr(enemy.weapon, 'cooldown'):
                enemy.weapon.cooldown /= genome['firing_frequency']
        
        elif isinstance(enemy, TankEnemy):
            # Make tanks more or less aggressive
            if hasattr(enemy, 'movement_pattern'):
                # Higher aggressiveness means more likely to use pattern 3 (aggressive)
                if random.random() < genome['aggressiveness']:
                    enemy.movement_pattern = 3
        
        elif isinstance(enemy, SwarmEnemy):
            # Adjust firing frequency for swarms
            if hasattr(enemy, 'weapon') and hasattr(enemy.weapon, 'cooldown'):
                enemy.weapon.cooldown /= (genome['firing_frequency'] * 1.2)  # Swarms fire more frequently
        
        elif isinstance(enemy, BomberEnemy):
            # Adjust firing frequency for bombers
            if hasattr(enemy, 'weapon') and hasattr(enemy.weapon, 'cooldown'):
                enemy.weapon.cooldown /= genome['firing_frequency']
        
        return enemy
    
    def update_fitness(self, enemy, lifetime, damage_dealt, shots_fired, shots_hit):
        """Update fitness score for genome used by this enemy"""
        enemy_id = id(enemy)
        if enemy_id in self.enemy_genome_map:
            genome_index = self.enemy_genome_map[enemy_id]
            
            # Calculate fitness based on several factors:
            # 1. Lifetime (longer is better)
            lifetime_score = min(lifetime / 20.0, 1.0) * 5.0  # Cap at 20 seconds
            
            # 2. Damage dealt to player (more is better)
            damage_score = damage_dealt * 2.0
            
            # 3. Accuracy (if shots were fired)
            accuracy_score = 0
            if shots_fired > 0:
                accuracy = shots_hit / shots_fired
                accuracy_score = accuracy * 3.0
            
            # 4. Efficiency (damage per second alive)
            efficiency_score = (damage_dealt / max(lifetime, 1.0)) * 2.0
            
            # Combine scores
            total_score = lifetime_score + damage_score + accuracy_score + efficiency_score
            
            # Update fitness with running average
            if self.genome_usage_count[genome_index] > 1:
                # Exponential moving average
                alpha = 0.3  # Weight for new observation
                self.fitness_scores[genome_index] = (1 - alpha) * self.fitness_scores[genome_index] + alpha * total_score
            else:
                # First observation
                self.fitness_scores[genome_index] = total_score
            
            # Clean up map entry
            del self.enemy_genome_map[enemy_id]
    
    def evolve(self):
        """Create a new generation based on fitness scores"""
        self.generations += 1
        print(f"Evolving generation {self.generations}")
        
        # 1. Select elite genomes
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_count:]
        elite_genomes = [self.genomes[i].copy() for i in elite_indices]
        
        # 2. Create new population through selection, crossover, and mutation
        new_population = []
        
        # First add elite genomes unchanged
        new_population.extend(elite_genomes)
        
        # Fill the rest through tournament selection and crossover
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_probability:
                child = self._crossover(parent1, parent2)
            else:
                # No crossover, clone parent1
                child = parent1.copy()
            
            # Mutation
            child = self._mutate(child)
            
            # Add to new population
            new_population.append(child)
        
        # 3. Replace old population with new one
        self.genomes = new_population
        
        # 4. Reset usage counts but keep fitness scores
        self.genome_usage_count = np.zeros(self.population_size)
        
        # 5. Print some information about the evolution
        avg_fitness = np.mean(self.fitness_scores)
        max_fitness = np.max(self.fitness_scores)
        print(f"Generation {self.generations}: Avg fitness = {avg_fitness:.2f}, Max fitness = {max_fitness:.2f}")
    
    def _tournament_selection(self, tournament_size=3):
        """Select a genome using tournament selection"""
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(self.genomes)), tournament_size)
        
        # Select the one with the highest fitness
        winner_index = max(tournament_indices, key=lambda i: self.fitness_scores[i])
        
        # Return a copy of the winner
        return self.genomes[winner_index].copy()
    
    def _crossover(self, parent1, parent2):
        """Create a child by combining genes from two parents"""
        child = {}
        
        for param in parent1.keys():
            if param == 'movement_pattern_weights':
                # For list parameters, perform uniform crossover on each element
                child[param] = []
                for i in range(len(parent1[param])):
                    # 50% chance to inherit from each parent
                    child[param].append(parent1[param][i] if random.random() < 0.5 else parent2[param][i])
            else:
                # For scalar parameters, use blend crossover (BLX-alpha)
                alpha = 0.3
                min_val = min(parent1[param], parent2[param])
                max_val = max(parent1[param], parent2[param])
                range_val = max_val - min_val
                
                # Extended range for blend crossover
                min_bound = max(self.genome_params[param][0], min_val - alpha * range_val)
                max_bound = min(self.genome_params[param][1], max_val + alpha * range_val)
                
                # Generate random value in the extended range
                child[param] = random.uniform(min_bound, max_bound)
        
        return child
    
    def _mutate(self, genome):
        """Apply random mutations to genome"""
        mutated = genome.copy()
        
        for param in mutated:
            # Decide whether to mutate this parameter
            if random.random() < self.mutation_rate:
                if param == 'movement_pattern_weights':
                    # Mutate each weight with some probability
                    for i in range(len(mutated[param])):
                        if random.random() < self.mutation_rate:
                            # Add Gaussian noise
                            noise = random.normalvariate(0, self.mutation_amount)
                            mutated[param][i] += noise
                            # Ensure it stays in valid range
                            min_val, max_val = self.genome_params[param][i]
                            mutated[param][i] = max(min_val, min(max_val, mutated[param][i]))
                else:
                    # Get parameter range
                    min_val, max_val = self.genome_params[param]
                    
                    # Three mutation strategies:
                    mutation_type = random.random()
                    
                    if mutation_type < 0.6:  # 60% chance: add Gaussian noise
                        # Calculate mutation amount (larger for larger ranges)
                        range_size = max_val - min_val
                        noise = random.normalvariate(0, range_size * self.mutation_amount)
                        mutated[param] += noise
                        
                    elif mutation_type < 0.8:  # 20% chance: reset to random value
                        mutated[param] = random.uniform(min_val, max_val)
                        
                    else:  # 20% chance: push toward extremes
                        if random.random() < 0.5:
                            # Push toward minimum
                            mutated[param] = min_val + (mutated[param] - min_val) * 0.5
                        else:
                            # Push toward maximum
                            mutated[param] = max_val - (max_val - mutated[param]) * 0.5
                    
                    # Ensure it stays in valid range
                    mutated[param] = max(min_val, min(max_val, mutated[param]))
        
        return mutated
    
    def save_state(self, filename):
        """Save genetic algorithm state to a file"""
        state = {
            'genomes': self.genomes,
            'fitness_scores': self.fitness_scores.tolist(),
            'genome_usage_count': self.genome_usage_count.tolist(),
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'mutation_amount': self.mutation_amount
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved genetic algorithm state to {filename}")
    
    def load_state(self, filename):
        """Load genetic algorithm state from a file"""
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            self.genomes = state['genomes']
            self.fitness_scores = np.array(state['fitness_scores'])
            self.genome_usage_count = np.array(state['genome_usage_count'])
            self.generations = state['generations']
            self.mutation_rate = state['mutation_rate']
            self.mutation_amount = state['mutation_amount']
            
            print(f"Loaded genetic algorithm state from {filename}")
            print(f"Generations: {self.generations}, Genomes: {len(self.genomes)}")
        else:
            print(f"No genetic algorithm state found at {filename}")
