"""
Tic-Tac-Toe Game Environment
Basic game logic, board state management, and move validation
"""
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum

class Player(Enum):
    """Player enumeration"""
    EMPTY = 0
    X = 1
    O = 2

class GameState(Enum):
    """Game state enumeration"""
    ONGOING = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3

class TicTacToeEnv:
    """Tic-Tac-Toe game environment"""

    def __init__(self):
        """Initialize the game environment"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_state = GameState.ONGOING
        self.move_count = 0

    def reset(self) -> np.ndarray:
        """Reset the game to initial state"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_state = GameState.ONGOING
        self.move_count = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get current board state as a flattened array"""
        return self.board.flatten()

    def get_state_key(self) -> str:
        """Get string representation of state for Q-table indexing"""
        return ''.join(map(str, self.board.flatten()))

    def get_available_actions(self) -> List[int]:
        """Get list of available actions (empty positions)"""
        return [i for i in range(9) if self.board.flatten()[i] == Player.EMPTY.value]

    def is_valid_action(self, action: int) -> bool:
        """Check if an action is valid"""
        if not 0 <= action <= 8:
            return False
        row, col = divmod(action, 3)
        return self.board[row, col] == Player.EMPTY.value

    def make_move(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Make a move on the board
        Returns: (new_state, reward, done, info)
        """
        if not self.is_valid_action(action):
            # Invalid move penalty
            return self.get_state(), -10, True, {"invalid_move": True}

        # Make the move
        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player.value
        self.move_count += 1

        # Check for game end conditions
        reward = self._check_game_end()
        done = self.game_state != GameState.ONGOING

        # Switch players if game continues
        if not done:
            self.current_player = Player.O if self.current_player == Player.X else Player.X

        info = {
            "game_state": self.game_state,
            "winner": self._get_winner(),
            "move_count": self.move_count
        }

        return self.get_state(), reward, done, info

    def _check_game_end(self) -> float:
        """Check if game has ended and return reward"""
        # Check rows
        for row in self.board:
            if len(set(row)) == 1 and row[0] != Player.EMPTY.value:
                self.game_state = GameState.X_WINS if row[0] == Player.X.value else GameState.O_WINS
                return 1.0 if self.current_player.value == row[0] else -1.0

        # Check columns
        for col in self.board.T:
            if len(set(col)) == 1 and col[0] != Player.EMPTY.value:
                self.game_state = GameState.X_WINS if col[0] == Player.X.value else GameState.O_WINS
                return 1.0 if self.current_player.value == col[0] else -1.0

        # Check diagonals
        diag1 = [self.board[i, i] for i in range(3)]
        diag2 = [self.board[i, 2-i] for i in range(3)]

        for diag in [diag1, diag2]:
            if len(set(diag)) == 1 and diag[0] != Player.EMPTY.value:
                self.game_state = GameState.X_WINS if diag[0] == Player.X.value else GameState.O_WINS
                return 1.0 if self.current_player.value == diag[0] else -1.0

        # Check for draw
        if self.move_count == 9:
            self.game_state = GameState.DRAW
            return 0.0

        # Game continues
        return 0.0

    def _get_winner(self) -> Optional[Player]:
        """Get the winner of the game"""
        if self.game_state == GameState.X_WINS:
            return Player.X
        elif self.game_state == GameState.O_WINS:
            return Player.O
        return None

    def render(self) -> str:
        """Render the current board state"""
        symbols = {Player.EMPTY.value: ' ', Player.X.value: 'X', Player.O.value: 'O'}
        board_str = ""

        for i, row in enumerate(self.board):
            row_str = " | ".join([symbols[cell] for cell in row])
            board_str += row_str
            if i < 2:
                board_str += "\n---------\n"

        return board_str

    def print_board(self):
        """Print the current board state"""
        print("\n" + self.render())
        print(f"Current player: {self.current_player.name}")
        print(f"Game state: {self.game_state.name}")

    def get_symmetries(self) -> List[Tuple[np.ndarray, int]]:
        """
        Get all symmetric states of current board and corresponding action mappings
        Used for data augmentation in training
        """
        board = self.board.copy()
        symmetries = []

        # Original
        symmetries.append((board.flatten(), list(range(9))))

        # 90-degree rotations
        for k in range(1, 4):
            rotated = np.rot90(board, k)
            # Map actions accordingly
            action_map = []
            for i in range(9):
                row, col = divmod(i, 3)
                for _ in range(k):
                    row, col = col, 2 - row
                action_map.append(row * 3 + col)
            symmetries.append((rotated.flatten(), action_map))

        # Horizontal flip
        flipped = np.fliplr(board)
        action_map = []
        for i in range(9):
            row, col = divmod(i, 3)
            col = 2 - col
            action_map.append(row * 3 + col)
        symmetries.append((flipped.flatten(), action_map))

        return symmetries

def test_environment():
    """Test the tic-tac-toe environment"""
    print("Testing Tic-Tac-Toe Environment")
    print("=" * 40)

    env = TicTacToeEnv()

    # Test basic functionality
    print("Initial state:")
    env.print_board()

    # Test some moves
    moves = [0, 4, 1, 3, 2]  # X wins
    for move in moves:
        print(f"\nPlayer {env.current_player.name} makes move {move}")
        state, reward, done, info = env.make_move(move)
        env.print_board()
        print(f"Reward: {reward}, Done: {done}")

        if done:
            print(f"Game ended: {info['game_state'].name}")
            if info['winner']:
                print(f"Winner: {info['winner'].name}")
            break

    # Test invalid move
    print("\nTesting invalid move:")
    env2 = TicTacToeEnv()
    state, reward, done, info = env2.make_move(0)  # Valid move
    state, reward, done, info = env2.make_move(0)  # Invalid move (position occupied)
    print(f"Invalid move reward: {reward}, Done: {done}")

if __name__ == "__main__":
    test_environment()