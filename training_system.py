"""
Training System for Tic-Tac-Toe Q-Learning
Self-play training with progress tracking and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional
from tic_tac_toe_env import TicTacToeEnv, Player, GameState
from q_learning_agent import QLearningAgent, RandomAgent

class TicTacToeTrainer:
    """Training system for Tic-Tac-Toe Q-Learning agents"""

    def __init__(self):
        """Initialize the trainer"""
        self.training_history = {
            'episode': [],
            'x_win_rate': [],
            'o_win_rate': [],
            'draw_rate': [],
            'x_epsilon': [],
            'o_epsilon': [],
            'games_per_checkpoint': 1000
        }

    def play_single_game(self, agent_x, agent_o, env: TicTacToeEnv,
                        training: bool = True, verbose: bool = False) -> dict:
        """
        Play a single game between two agents

        Returns:
            dict: Game result information
        """
        env.reset()
        agent_x.reset_for_new_game()
        agent_o.reset_for_new_game()

        # Track game history for learning
        game_history = []
        previous_state = None
        previous_action = None
        previous_agent = None

        if verbose:
            print("Starting new game:")
            env.print_board()

        while env.game_state == GameState.ONGOING:
            current_agent = agent_x if env.current_player == Player.X else agent_o

            # Get current state and available actions
            state = env.get_state_key()
            available_actions = env.get_available_actions()

            # Update Q-value for previous move (if exists and training)
            if training and previous_state is not None:
                # Intermediate reward (0 for ongoing game)
                intermediate_reward = 0.0
                next_available_actions = available_actions
                previous_agent.update_q_value(previous_state, previous_action,
                                            intermediate_reward, state,
                                            next_available_actions, done=False)

            # Agent chooses action
            action = current_agent.choose_action(state, available_actions, training=training)

            # Store current move info
            previous_state = state
            previous_action = action
            previous_agent = current_agent

            # Store state-action pair for learning
            if training:
                game_history.append({
                    'player': env.current_player,
                    'state': state,
                    'action': action,
                    'agent': current_agent
                })

            # Make move
            new_state, reward, done, info = env.make_move(action)

            if verbose:
                print(f"Player {env.current_player.name if not done else 'previous'} played action {action}")
                env.print_board()
                if done:
                    print(f"Game ended: {info['game_state'].name}")

        # Update Q-values for final move if training
        if training and previous_state is not None:
            final_reward = self._get_final_reward(previous_agent.player, info)
            previous_agent.update_q_value(previous_state, previous_action,
                                        final_reward, env.get_state_key(),
                                        [], done=True)

            # Also update the opponent's last move if there was one
            if len(game_history) > 1:
                second_last_move = game_history[-2]
                opponent_agent = second_last_move['agent']
                opponent_reward = self._get_final_reward(opponent_agent.player, info)
                opponent_agent.update_q_value(second_last_move['state'],
                                            second_last_move['action'],
                                            opponent_reward, env.get_state_key(),
                                            [], done=True)

        # Update training statistics
        if training:
            self._update_agent_stats(agent_x, agent_o, info)

        return {
            'winner': info.get('winner'),
            'game_state': info['game_state'],
            'moves': info['move_count'],
            'history': game_history if training else []
        }

    def _get_final_reward(self, player: Player, game_info: dict) -> float:
        """Get final reward for a player based on game outcome"""
        winner = game_info.get('winner')

        if winner is None:  # Draw
            return 0.0
        elif winner == player:  # Win
            return 1.0
        else:  # Loss
            return -1.0

    def _update_agent_stats(self, agent_x, agent_o, game_info: dict):
        """Update training statistics for both agents"""
        winner = game_info.get('winner')

        if winner == Player.X:
            agent_x.update_training_stats('win')
            agent_o.update_training_stats('loss')
        elif winner == Player.O:
            agent_x.update_training_stats('loss')
            agent_o.update_training_stats('win')
        else:  # Draw
            agent_x.update_training_stats('draw')
            agent_o.update_training_stats('draw')

    def train_self_play(self, num_episodes: int = 10000,
                       checkpoint_interval: int = 1000,
                       save_models: bool = True,
                       verbose: bool = False) -> Tuple[QLearningAgent, QLearningAgent]:
        """
        Train two Q-Learning agents through self-play

        Args:
            num_episodes: Number of training episodes
            checkpoint_interval: How often to print progress
            save_models: Whether to save models during training
            verbose: Whether to print detailed game info

        Returns:
            Tuple of trained agents (X, O)
        """
        print(f"Starting self-play training for {num_episodes} episodes")
        print("=" * 60)

        # Create agents
        agent_x = QLearningAgent(
            player=Player.X,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.9995,
            epsilon_min=0.01
        )

        agent_o = QLearningAgent(
            player=Player.O,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.9995,
            epsilon_min=0.01
        )

        env = TicTacToeEnv()

        # Training loop
        start_time = time.time()

        for episode in range(num_episodes):
            # Play one game
            game_result = self.play_single_game(agent_x, agent_o, env, training=True,
                                              verbose=verbose and episode < 5)

            # Update epsilon
            agent_x.update_epsilon()
            agent_o.update_epsilon()

            # Checkpoint progress
            if (episode + 1) % checkpoint_interval == 0:
                self._print_training_progress(episode + 1, agent_x, agent_o, start_time)
                self._record_training_history(episode + 1, agent_x, agent_o)

                # Save models at checkpoints
                if save_models:
                    agent_x.save_model(f"model_x_episode_{episode + 1}.pkl")
                    agent_o.save_model(f"model_o_episode_{episode + 1}.pkl")

        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")

        # Final statistics
        print("\nFinal Training Statistics:")
        print("-" * 40)
        agent_x.print_stats()
        print()
        agent_o.print_stats()

        return agent_x, agent_o

    def train_vs_random(self, num_episodes: int = 5000) -> QLearningAgent:
        """
        Train a Q-Learning agent against random opponent

        Args:
            num_episodes: Number of training episodes

        Returns:
            Trained Q-Learning agent
        """
        print(f"Training Q-Learning agent vs Random opponent for {num_episodes} episodes")
        print("=" * 70)

        agent_x = QLearningAgent(
            player=Player.X,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.01
        )

        random_agent = RandomAgent(Player.O)
        env = TicTacToeEnv()

        start_time = time.time()

        for episode in range(num_episodes):
            game_result = self.play_single_game(agent_x, random_agent, env, training=True)
            agent_x.update_epsilon()

            if (episode + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                win_rate = agent_x.get_win_rate()
                print(f"Episode {episode + 1:5d} | Win Rate: {win_rate:.3f} | "
                      f"Epsilon: {agent_x.epsilon:.4f} | Time: {elapsed:.1f}s")

        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
        agent_x.print_stats()

        return agent_x

    def _print_training_progress(self, episode: int, agent_x: QLearningAgent,
                               agent_o: QLearningAgent, start_time: float):
        """Print training progress"""
        elapsed = time.time() - start_time
        x_win_rate = agent_x.get_win_rate()
        o_win_rate = agent_o.get_win_rate()

        print(f"Episode {episode:5d} | "
              f"X Win Rate: {x_win_rate:.3f} | "
              f"O Win Rate: {o_win_rate:.3f} | "
              f"X ε: {agent_x.epsilon:.4f} | "
              f"O ε: {agent_o.epsilon:.4f} | "
              f"Time: {elapsed:.1f}s")

    def _record_training_history(self, episode: int, agent_x: QLearningAgent,
                                agent_o: QLearningAgent):
        """Record training history for plotting"""
        self.training_history['episode'].append(episode)
        self.training_history['x_win_rate'].append(agent_x.get_win_rate())
        self.training_history['o_win_rate'].append(agent_o.get_win_rate())

        # Calculate draw rate
        x_stats = agent_x.training_stats
        total_games = x_stats['games_played']
        draws = x_stats['draws']
        draw_rate = draws / total_games if total_games > 0 else 0
        self.training_history['draw_rate'].append(draw_rate)

        self.training_history['x_epsilon'].append(agent_x.epsilon)
        self.training_history['o_epsilon'].append(agent_o.epsilon)

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        if not self.training_history['episode']:
            print("No training history to plot.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Win rates
        ax1.plot(self.training_history['episode'], self.training_history['x_win_rate'],
                label='Player X', color='blue')
        ax1.plot(self.training_history['episode'], self.training_history['o_win_rate'],
                label='Player O', color='red')
        ax1.plot(self.training_history['episode'], self.training_history['draw_rate'],
                label='Draw Rate', color='green')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Rate')
        ax1.set_title('Win/Draw Rates During Training')
        ax1.legend()
        ax1.grid(True)

        # Epsilon decay
        ax2.plot(self.training_history['episode'], self.training_history['x_epsilon'],
                label='Player X ε', color='blue')
        ax2.plot(self.training_history['episode'], self.training_history['o_epsilon'],
                label='Player O ε', color='red')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Epsilon Decay During Training')
        ax2.legend()
        ax2.grid(True)

        # Moving average of win rates (last 10 checkpoints)
        if len(self.training_history['episode']) >= 10:
            window = min(10, len(self.training_history['x_win_rate']))
            x_ma = np.convolve(self.training_history['x_win_rate'],
                              np.ones(window)/window, mode='valid')
            o_ma = np.convolve(self.training_history['o_win_rate'],
                              np.ones(window)/window, mode='valid')
            episodes_ma = self.training_history['episode'][window-1:]

            ax3.plot(episodes_ma, x_ma, label='Player X (MA)', color='blue', linewidth=2)
            ax3.plot(episodes_ma, o_ma, label='Player O (MA)', color='red', linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Win Rate (Moving Average)')
            ax3.set_title('Smoothed Win Rates')
            ax3.legend()
            ax3.grid(True)

        # Game statistics
        if len(self.training_history['episode']) > 0:
            episodes = self.training_history['episode']
            x_rates = self.training_history['x_win_rate']
            o_rates = self.training_history['o_win_rate']
            draw_rates = self.training_history['draw_rate']

            ax4.stackplot(episodes, x_rates, o_rates, draw_rates,
                         labels=['X Wins', 'O Wins', 'Draws'],
                         colors=['blue', 'red', 'green'], alpha=0.7)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Cumulative Rate')
            ax4.set_title('Cumulative Game Outcomes')
            ax4.legend(loc='upper left')
            ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")

        plt.show()

    def evaluate_agents(self, agent_x, agent_o, num_games: int = 1000) -> dict:
        """Evaluate trained agents without learning"""
        print(f"Evaluating agents over {num_games} games...")

        env = TicTacToeEnv()
        results = {'x_wins': 0, 'o_wins': 0, 'draws': 0}

        for _ in range(num_games):
            game_result = self.play_single_game(agent_x, agent_o, env,
                                              training=False, verbose=False)

            winner = game_result['winner']
            if winner == Player.X:
                results['x_wins'] += 1
            elif winner == Player.O:
                results['o_wins'] += 1
            else:
                results['draws'] += 1

        # Print evaluation results
        print(f"\nEvaluation Results ({num_games} games):")
        print(f"Player X wins: {results['x_wins']} ({results['x_wins']/num_games:.3f})")
        print(f"Player O wins: {results['o_wins']} ({results['o_wins']/num_games:.3f})")
        print(f"Draws: {results['draws']} ({results['draws']/num_games:.3f})")

        return results

def main():
    """Main training function"""
    trainer = TicTacToeTrainer()

    print("Tic-Tac-Toe Q-Learning Training System")
    print("=" * 50)

    # Option 1: Self-play training
    print("\n1. Self-play training (recommended)")
    agent_x, agent_o = trainer.train_self_play(
        num_episodes=10000,
        checkpoint_interval=1000,
        save_models=True
    )

    # Evaluate the trained agents
    trainer.evaluate_agents(agent_x, agent_o, num_games=1000)

    # Plot training progress
    trainer.plot_training_progress("training_progress.png")

    # Save final models
    agent_x.save_model("final_model_x.pkl")
    agent_o.save_model("final_model_o.pkl")

    print("\nTraining completed! Models saved.")

if __name__ == "__main__":
    main()