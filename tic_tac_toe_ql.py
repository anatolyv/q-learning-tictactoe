"""
Tic-Tac-Toe Q-Learning Main Application
Complete reinforcement learning system with training and gameplay
"""
import os
import sys
import argparse
from typing import Optional

from tic_tac_toe_env import TicTacToeEnv, Player
from q_learning_agent import QLearningAgent, RandomAgent
from training_system import TicTacToeTrainer
from human_vs_ai import HumanVsAI

class TicTacToeQLearning:
    """Main application class for Tic-Tac-Toe Q-Learning"""

    def __init__(self):
        """Initialize the application"""
        self.trainer = TicTacToeTrainer()
        self.game_interface = HumanVsAI()

    def train_agents(self, num_episodes: int = 10000,
                    checkpoint_interval: int = 1000,
                    training_type: str = 'self_play') -> tuple:
        """
        Train Q-Learning agents

        Args:
            num_episodes: Number of training episodes
            checkpoint_interval: Progress reporting interval
            training_type: 'self_play' or 'vs_random'

        Returns:
            Tuple of trained agents or single agent
        """
        print(f"🚀 Starting {training_type} training...")
        print("=" * 60)

        if training_type == 'self_play':
            agent_x, agent_o = self.trainer.train_self_play(
                num_episodes=num_episodes,
                checkpoint_interval=checkpoint_interval,
                save_models=True,
                verbose=False
            )

            # Evaluate the trained agents
            print("\n📊 Evaluating trained agents...")
            self.trainer.evaluate_agents(agent_x, agent_o, num_games=1000)

            # Save final models
            agent_x.save_model("final_model_x.pkl")
            agent_o.save_model("final_model_o.pkl")

            print("\n✅ Self-play training completed!")
            return agent_x, agent_o

        elif training_type == 'vs_random':
            agent = self.trainer.train_vs_random(num_episodes=num_episodes)
            agent.save_model("model_vs_random.pkl")

            print("\n✅ Training vs random completed!")
            return agent,

        else:
            raise ValueError("training_type must be 'self_play' or 'vs_random'")

    def play_human_vs_ai(self, model_path: Optional[str] = None):
        """
        Start human vs AI gameplay

        Args:
            model_path: Path to trained model file
        """
        print("🎮 Starting Human vs AI mode...")
        self.game_interface.load_and_play(model_path)

    def demo_training_progress(self):
        """Demonstrate training with visualization"""
        print("📈 Running training demonstration with progress visualization...")

        # Quick training run
        agent_x, agent_o = self.trainer.train_self_play(
            num_episodes=5000,
            checkpoint_interval=500,
            save_models=False
        )

        # Show training progress
        try:
            self.trainer.plot_training_progress("demo_training_progress.png")
        except Exception as e:
            print(f"Note: Could not create plot (matplotlib not available): {e}")

        return agent_x, agent_o

    def benchmark_agents(self):
        """Benchmark different agents against each other"""
        print("⚔️ Running agent benchmarks...")

        # Create different types of agents
        print("Creating agents...")
        q_agent_strong = QLearningAgent(Player.X, epsilon=0.0)  # No exploration
        q_agent_weak = QLearningAgent(Player.O, epsilon=0.2)    # Some exploration
        random_agent = RandomAgent(Player.O)

        # Try to load trained models
        if os.path.exists("final_model_x.pkl"):
            print("Loading trained Q-Learning agent...")
            q_agent_strong.load_model("final_model_x.pkl")
        else:
            print("No trained model found. Training a quick agent...")
            q_agent_strong = self.trainer.train_vs_random(2000)

        # Benchmark games
        benchmarks = [
            ("Trained Q-Learning vs Random", q_agent_strong, random_agent),
            ("Trained Q-Learning vs Weak Q-Learning", q_agent_strong, q_agent_weak)
        ]

        results = {}
        for name, agent1, agent2 in benchmarks:
            print(f"\n🥊 {name}")
            print("-" * 50)

            # Reset agents
            agent1.player = Player.X
            agent2.player = Player.O

            wins = {'X': 0, 'O': 0, 'draws': 0}
            num_games = 500

            env = TicTacToeEnv()
            for _ in range(num_games):
                game_result = self.trainer.play_single_game(
                    agent1, agent2, env, training=False, verbose=False
                )

                winner = game_result['winner']
                if winner == Player.X:
                    wins['X'] += 1
                elif winner == Player.O:
                    wins['O'] += 1
                else:
                    wins['draws'] += 1

            # Print results
            print(f"Results over {num_games} games:")
            print(f"Player X wins: {wins['X']} ({wins['X']/num_games:.2%})")
            print(f"Player O wins: {wins['O']} ({wins['O']/num_games:.2%})")
            print(f"Draws: {wins['draws']} ({wins['draws']/num_games:.2%})")

            results[name] = wins

        return results

    def show_model_info(self, model_path: str):
        """Show information about a saved model"""
        if not os.path.exists(model_path):
            print(f"❌ Model file {model_path} not found.")
            return

        print(f"📋 Model Information: {model_path}")
        print("=" * 50)

        # Load and display model info
        agent = QLearningAgent(Player.X)
        agent.load_model(model_path)

        print(f"Q-table size: {len(agent.q_table)} states")
        print(f"Current epsilon: {agent.epsilon:.6f}")
        print(f"Learning parameters:")
        print(f"  - Learning rate: {agent.learning_rate}")
        print(f"  - Discount factor: {agent.discount_factor}")
        print(f"  - Epsilon decay: {agent.epsilon_decay}")
        print(f"  - Epsilon min: {agent.epsilon_min}")

        if agent.training_stats['games_played'] > 0:
            stats = agent.training_stats
            print(f"\nTraining statistics:")
            print(f"  - Games played: {stats['games_played']}")
            print(f"  - Wins: {stats['wins']} ({stats['wins']/stats['games_played']:.3f})")
            print(f"  - Losses: {stats['losses']} ({stats['losses']/stats['games_played']:.3f})")
            print(f"  - Draws: {stats['draws']} ({stats['draws']/stats['games_played']:.3f})")

        # Show some example Q-values
        print(f"\nSample Q-values (first 10 states):")
        for i, (state, actions) in enumerate(agent.q_table.items()):
            if i >= 10:
                break
            if actions:
                best_action, best_value = max(actions.items(), key=lambda x: x[1])
                print(f"  State {state}: Best action {best_action} (Q={best_value:.3f})")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Tic-Tac-Toe Q-Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --train --episodes 10000              # Train with self-play
  %(prog)s --train --type vs_random --episodes 5000  # Train vs random
  %(prog)s --play                                 # Play against AI
  %(prog)s --play --model final_model_x.pkl      # Play against specific model
  %(prog)s --demo                                 # Show training demo
  %(prog)s --benchmark                            # Benchmark different agents
  %(prog)s --info final_model_x.pkl              # Show model information
        """
    )

    parser.add_argument('--train', action='store_true',
                       help='Train Q-Learning agents')
    parser.add_argument('--play', action='store_true',
                       help='Play against trained AI')
    parser.add_argument('--demo', action='store_true',
                       help='Show training demonstration with visualization')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark different agents')
    parser.add_argument('--info', metavar='MODEL_PATH',
                       help='Show information about a saved model')

    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes (default: 10000)')
    parser.add_argument('--checkpoint', type=int, default=1000,
                       help='Progress reporting interval (default: 1000)')
    parser.add_argument('--type', choices=['self_play', 'vs_random'],
                       default='self_play',
                       help='Training type (default: self_play)')
    parser.add_argument('--model', metavar='MODEL_PATH',
                       help='Path to model file for playing')

    args = parser.parse_args()

    # Create main application
    app = TicTacToeQLearning()

    # Handle different modes
    if args.train:
        app.train_agents(
            num_episodes=args.episodes,
            checkpoint_interval=args.checkpoint,
            training_type=args.type
        )

        # Plot training progress if possible
        try:
            app.trainer.plot_training_progress("training_progress.png")
        except Exception as e:
            print(f"Note: Could not create training plot: {e}")

    elif args.play:
        app.play_human_vs_ai(args.model)

    elif args.demo:
        app.demo_training_progress()

    elif args.benchmark:
        app.benchmark_agents()

    elif args.info:
        app.show_model_info(args.info)

    else:
        # Interactive menu if no arguments provided
        print("🎯 Tic-Tac-Toe Q-Learning System")
        print("=" * 50)

        while True:
            print("\nChoose an option:")
            print("1. Train new agents (self-play)")
            print("2. Quick train vs random")
            print("3. Play against AI")
            print("4. Show training demo")
            print("5. Benchmark agents")
            print("6. Show model info")
            print("7. Test environment")
            print("8. Exit")

            choice = input("\nEnter your choice (1-8): ").strip()

            if choice == '1':
                episodes = input("Enter number of episodes (default 10000): ").strip()
                episodes = int(episodes) if episodes.isdigit() else 10000
                app.train_agents(num_episodes=episodes, training_type='self_play')

            elif choice == '2':
                app.train_agents(num_episodes=5000, training_type='vs_random')

            elif choice == '3':
                model_path = input("Enter model path (or press Enter for auto-detect): ").strip()
                app.play_human_vs_ai(model_path if model_path else None)

            elif choice == '4':
                app.demo_training_progress()

            elif choice == '5':
                app.benchmark_agents()

            elif choice == '6':
                model_path = input("Enter model path: ").strip()
                if model_path:
                    app.show_model_info(model_path)

            elif choice == '7':
                print("Testing environment...")
                from tic_tac_toe_env import test_environment
                test_environment()

            elif choice == '8':
                print("Thanks for using Tic-Tac-Toe Q-Learning! 👋")
                break

            else:
                print("Invalid choice. Please enter 1-8.")

if __name__ == "__main__":
    main()