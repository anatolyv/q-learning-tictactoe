"""
Human vs AI Gameplay Interface
Interactive game where humans can play against trained Q-Learning agents
"""
import os
from typing import Optional
from tic_tac_toe_env import TicTacToeEnv, Player, GameState
from q_learning_agent import QLearningAgent, RandomAgent

class HumanVsAI:
    """Interface for human vs AI gameplay"""

    def __init__(self):
        """Initialize the game interface"""
        self.env = TicTacToeEnv()
        self.human_player = None
        self.ai_player = None
        self.human_symbol = None
        self.ai_symbol = None

    def display_board_with_positions(self):
        """Display board with position numbers for easy reference"""
        print("\nPosition reference:")
        print(" 0 | 1 | 2 ")
        print("-----------")
        print(" 3 | 4 | 5 ")
        print("-----------")
        print(" 6 | 7 | 8 ")
        print("\nCurrent board:")
        self.env.print_board()

    def get_human_move(self) -> int:
        """Get move input from human player"""
        while True:
            try:
                move = input(f"\nYour turn (Player {self.human_symbol})! Enter position (0-8) or 'q' to quit: ").strip()

                if move.lower() == 'q':
                    return -1

                move = int(move)

                if not (0 <= move <= 8):
                    print("Please enter a number between 0 and 8.")
                    continue

                if not self.env.is_valid_action(move):
                    print("That position is already taken! Choose another.")
                    continue

                return move

            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return -1

    def play_single_game(self, ai_agent: QLearningAgent, human_starts: bool = True) -> dict:
        """
        Play a single game between human and AI

        Args:
            ai_agent: Trained AI agent
            human_starts: Whether human plays first

        Returns:
            dict: Game result
        """
        self.env.reset()

        # Set up players
        if human_starts:
            self.human_player = Player.X
            self.ai_player = Player.O
            self.human_symbol = 'X'
            self.ai_symbol = 'O'
        else:
            self.human_player = Player.O
            self.ai_player = Player.X
            self.human_symbol = 'O'
            self.ai_symbol = 'X'

        print(f"\nStarting new game!")
        print(f"You are playing as {self.human_symbol}")
        print(f"AI is playing as {self.ai_symbol}")

        if human_starts:
            print("You go first!")
        else:
            print("AI goes first!")

        move_count = 0

        while self.env.game_state == GameState.ONGOING:
            self.display_board_with_positions()

            if self.env.current_player == self.human_player:
                # Human's turn
                move = self.get_human_move()
                if move == -1:  # Quit
                    return {'winner': None, 'quit': True}

                state, reward, done, info = self.env.make_move(move)
                move_count += 1

            else:
                # AI's turn
                print(f"\nAI ({self.ai_symbol}) is thinking...")
                state_key = self.env.get_state_key()
                available_actions = self.env.get_available_actions()

                ai_move = ai_agent.choose_action(state_key, available_actions, training=False)
                print(f"AI chooses position {ai_move}")

                state, reward, done, info = self.env.make_move(ai_move)
                move_count += 1

            if done:
                break

        # Game ended, show final board
        print("\nGame Over!")
        self.display_board_with_positions()

        # Determine and announce result
        winner = info.get('winner')
        if winner == self.human_player:
            print("🎉 Congratulations! You won!")
            result = 'human_win'
        elif winner == self.ai_player:
            print("🤖 AI wins! Better luck next time.")
            result = 'ai_win'
        else:
            print("🤝 It's a draw! Good game.")
            result = 'draw'

        return {
            'winner': winner,
            'result': result,
            'moves': move_count,
            'quit': False
        }

    def play_tournament(self, ai_agent: QLearningAgent, num_games: int = 5) -> dict:
        """
        Play a tournament of multiple games

        Args:
            ai_agent: Trained AI agent
            num_games: Number of games to play

        Returns:
            dict: Tournament results
        """
        print(f"\n{'='*60}")
        print(f"TOURNAMENT MODE: Playing {num_games} games")
        print(f"{'='*60}")

        results = {
            'human_wins': 0,
            'ai_wins': 0,
            'draws': 0,
            'games_played': 0
        }

        for game_num in range(num_games):
            print(f"\n--- Game {game_num + 1} of {num_games} ---")

            # Alternate who starts
            human_starts = game_num % 2 == 0

            game_result = self.play_single_game(ai_agent, human_starts)

            if game_result['quit']:
                break

            # Update results
            results['games_played'] += 1
            if game_result['result'] == 'human_win':
                results['human_wins'] += 1
            elif game_result['result'] == 'ai_win':
                results['ai_wins'] += 1
            else:
                results['draws'] += 1

            # Show running score
            print(f"\nRunning score after {results['games_played']} games:")
            print(f"You: {results['human_wins']}")
            print(f"AI: {results['ai_wins']}")
            print(f"Draws: {results['draws']}")

            if game_num < num_games - 1:
                continue_game = input("\nPress Enter for next game, or 'q' to quit tournament: ").strip()
                if continue_game.lower() == 'q':
                    break

        # Tournament summary
        print(f"\n{'='*60}")
        print("TOURNAMENT RESULTS")
        print(f"{'='*60}")
        games = results['games_played']
        if games > 0:
            human_rate = results['human_wins'] / games
            ai_rate = results['ai_wins'] / games
            draw_rate = results['draws'] / games

            print(f"Games played: {games}")
            print(f"Human wins: {results['human_wins']} ({human_rate:.2%})")
            print(f"AI wins: {results['ai_wins']} ({ai_rate:.2%})")
            print(f"Draws: {results['draws']} ({draw_rate:.2%})")

            if results['human_wins'] > results['ai_wins']:
                print("🏆 Congratulations! You won the tournament!")
            elif results['ai_wins'] > results['human_wins']:
                print("🤖 AI wins the tournament! Keep practicing!")
            else:
                print("🤝 Tournament tied! Well played!")
        else:
            print("No games completed.")

        return results

    def load_and_play(self, model_path: Optional[str] = None):
        """
        Load a trained model and start playing

        Args:
            model_path: Path to saved model file
        """
        # Try to load a trained model
        ai_agent = QLearningAgent(Player.X)  # Placeholder

        if model_path and os.path.exists(model_path):
            print(f"Loading AI model from {model_path}...")
            ai_agent.load_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            # Look for default models
            default_models = [
                "final_model_x.pkl",
                "final_model_o.pkl",
                "model_x_episode_10000.pkl",
                "model_o_episode_10000.pkl"
            ]

            loaded = False
            for model_file in default_models:
                if os.path.exists(model_file):
                    print(f"Found trained model: {model_file}")
                    ai_agent.load_model(model_file)
                    print("✅ Model loaded successfully!")
                    loaded = True
                    break

            if not loaded:
                print("⚠️ No trained model found. Training a quick model against random opponent...")
                from training_system import TicTacToeTrainer
                trainer = TicTacToeTrainer()
                ai_agent = trainer.train_vs_random(num_episodes=2000)
                print("✅ Quick training completed!")

        # Start the game interface
        self.game_menu(ai_agent)

    def game_menu(self, ai_agent: QLearningAgent):
        """
        Main game menu

        Args:
            ai_agent: Trained AI agent
        """
        print(f"\n{'='*60}")
        print("🎮 TIC-TAC-TOE vs AI 🤖")
        print(f"{'='*60}")

        while True:
            print("\nChoose an option:")
            print("1. Play single game")
            print("2. Play tournament (5 games)")
            print("3. Show AI statistics")
            print("4. Quit")

            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == '1':
                self.play_single_game(ai_agent)
            elif choice == '2':
                self.play_tournament(ai_agent)
            elif choice == '3':
                self.show_ai_stats(ai_agent)
            elif choice == '4':
                print("Thanks for playing! 👋")
                break
            else:
                print("Invalid choice. Please enter 1-4.")

    def show_ai_stats(self, ai_agent: QLearningAgent):
        """Show AI agent statistics"""
        print(f"\n{'='*60}")
        print("AI AGENT STATISTICS")
        print(f"{'='*60}")

        print(f"Q-table size: {len(ai_agent.q_table)} states")
        print(f"Current epsilon: {ai_agent.epsilon:.6f}")
        print(f"Learning rate: {ai_agent.learning_rate}")
        print(f"Discount factor: {ai_agent.discount_factor}")

        if hasattr(ai_agent, 'training_stats') and ai_agent.training_stats:
            stats = ai_agent.training_stats
            if stats['games_played'] > 0:
                print(f"\nTraining history:")
                print(f"Total training games: {stats['games_played']}")
                print(f"Training wins: {stats['wins']}")
                print(f"Training win rate: {stats['wins']/stats['games_played']:.3f}")
        else:
            print("\nNo training statistics available.")

        # Show a few example Q-values
        print(f"\nSample Q-values (first 5 states):")
        for i, (state, actions) in enumerate(ai_agent.q_table.items()):
            if i >= 5:
                break
            best_action = max(actions.items(), key=lambda x: x[1]) if actions else ("None", 0)
            print(f"State {state}: Best action {best_action[0]} (Q={best_action[1]:.3f})")

def main():
    """Main function to start human vs AI interface"""
    game_interface = HumanVsAI()

    print("🎮 Tic-Tac-Toe Q-Learning Game 🤖")
    print("=" * 50)

    # Allow user to specify model path
    model_path = input("Enter path to trained model (or press Enter for auto-detect): ").strip()
    if not model_path:
        model_path = None

    game_interface.load_and_play(model_path)

if __name__ == "__main__":
    main()