"""
Q-Learning Agent for Tic-Tac-Toe
Implements Q-learning algorithm with state-action value table
"""
import numpy as np
import random
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tic_tac_toe_env import TicTacToeEnv, Player

class QLearningAgent:
    """Q-Learning agent for Tic-Tac-Toe"""

    def __init__(self,
                 player: Player,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent

        Args:
            player: Player type (X or O)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Training statistics
        self.training_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate_history': [],
            'epsilon_history': []
        }

    def get_q_value(self, state: str, action: int) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table[state][action]

    def get_best_action(self, state: str, available_actions: List[int]) -> int:
        """Get best action based on Q-values"""
        if not available_actions:
            return random.randint(0, 8)

        # Get Q-values for all available actions
        q_values = [self.get_q_value(state, action) for action in available_actions]

        # Find actions with maximum Q-value
        max_q = max(q_values)
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]

        # Random tie-breaking
        chosen_action = random.choice(best_actions)

        return chosen_action

    def choose_action(self, state: str, available_actions: List[int], training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if not available_actions:
            return random.randint(0, 8)

        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: best action
            return self.get_best_action(state, available_actions)

    def update_q_value(self, state: str, action: int, reward: float,
                      next_state: str, next_available_actions: List[int], done: bool):
        """Update Q-value using Q-learning update rule"""
        current_q = self.get_q_value(state, action)

        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state: Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
            next_q_values = [self.get_q_value(next_state, a) for a in next_available_actions] if next_available_actions else [0]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.discount_factor * max_next_q

        # Update Q-value
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)

    def update_epsilon(self):
        """Decay epsilon after each game"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_for_new_game(self):
        """Reset agent state for a new game"""
        pass  # Q-learning agents don't need to reset between games

    def update_training_stats(self, result: str):
        """Update training statistics"""
        self.training_stats['games_played'] += 1

        if result == 'win':
            self.training_stats['wins'] += 1
        elif result == 'loss':
            self.training_stats['losses'] += 1
        else:
            self.training_stats['draws'] += 1

        # Calculate win rate
        games = self.training_stats['games_played']
        wins = self.training_stats['wins']
        win_rate = wins / games if games > 0 else 0

        # Store history every 100 games
        if games % 100 == 0:
            self.training_stats['win_rate_history'].append(win_rate)
            self.training_stats['epsilon_history'].append(self.epsilon)

    def get_win_rate(self) -> float:
        """Get current win rate"""
        games = self.training_stats['games_played']
        wins = self.training_stats['wins']
        return wins / games if games > 0 else 0

    def print_stats(self):
        """Print training statistics"""
        stats = self.training_stats
        total_games = stats['games_played']

        if total_games == 0:
            print("No games played yet.")
            return

        win_rate = stats['wins'] / total_games
        loss_rate = stats['losses'] / total_games
        draw_rate = stats['draws'] / total_games

        print(f"Player {self.player.name} Statistics:")
        print(f"Total games: {total_games}")
        print(f"Wins: {stats['wins']} ({win_rate:.3f})")
        print(f"Losses: {stats['losses']} ({loss_rate:.3f})")
        print(f"Draws: {stats['draws']} ({draw_rate:.3f})")
        print(f"Current epsilon: {self.epsilon:.4f}")
        print(f"Q-table size: {len(self.q_table)}")

    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'player': self.player,
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'training_stats': self.training_stats,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # Restore Q-table as defaultdict
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in model_data['q_table'].items():
                for action, value in actions.items():
                    self.q_table[state][action] = value

            # Restore other attributes
            self.training_stats = model_data['training_stats']
            self.epsilon = model_data['epsilon']
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.epsilon_decay = model_data['epsilon_decay']
            self.epsilon_min = model_data['epsilon_min']

            print(f"Model loaded from {filepath}")
            print(f"Loaded Q-table with {len(self.q_table)} states")

        except FileNotFoundError:
            print(f"Model file {filepath} not found. Starting with empty Q-table.")
        except Exception as e:
            print(f"Error loading model: {e}")

class RandomAgent:
    """Random agent for baseline comparison"""

    def __init__(self, player: Player):
        self.player = player
        self.training_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        }

    def choose_action(self, state: str, available_actions: List[int], training: bool = True) -> int:
        """Choose random action from available actions"""
        return random.choice(available_actions) if available_actions else 0

    def update_q_value(self, *args, **kwargs):
        """Random agent doesn't learn"""
        pass

    def update_epsilon(self):
        """Random agent doesn't use epsilon"""
        pass

    def reset_for_new_game(self):
        """Reset for new game"""
        pass

    def update_training_stats(self, result: str):
        """Update training statistics"""
        self.training_stats['games_played'] += 1
        if result == 'win':
            self.training_stats['wins'] += 1
        elif result == 'loss':
            self.training_stats['losses'] += 1
        else:
            self.training_stats['draws'] += 1

    def get_win_rate(self) -> float:
        """Get current win rate"""
        games = self.training_stats['games_played']
        wins = self.training_stats['wins']
        return wins / games if games > 0 else 0

    def print_stats(self):
        """Print statistics"""
        stats = self.training_stats
        total_games = stats['games_played']

        if total_games == 0:
            print("No games played yet.")
            return

        win_rate = stats['wins'] / total_games
        print(f"Random Player {self.player.name} Statistics:")
        print(f"Total games: {total_games}")
        print(f"Win rate: {win_rate:.3f}")

def test_q_learning_agent():
    """Test the Q-learning agent"""
    print("Testing Q-Learning Agent")
    print("=" * 40)

    # Create agents
    agent_x = QLearningAgent(Player.X, epsilon=0.3)
    agent_o = QLearningAgent(Player.O, epsilon=0.3)

    # Test basic functionality
    env = TicTacToeEnv()
    state = env.get_state_key()
    available_actions = env.get_available_actions()

    print(f"Initial state: {state}")
    print(f"Available actions: {available_actions}")

    action = agent_x.choose_action(state, available_actions)
    print(f"Agent X chose action: {action}")

    # Test Q-value update
    new_state, reward, done, info = env.make_move(action)
    new_state_key = env.get_state_key()
    new_available_actions = env.get_available_actions()

    agent_x.update_q_value(state, action, reward, new_state_key, new_available_actions, done)
    print(f"Updated Q-value for state {state}, action {action}: {agent_x.get_q_value(state, action)}")

    print("Q-Learning agent test completed!")

if __name__ == "__main__":
    test_q_learning_agent()