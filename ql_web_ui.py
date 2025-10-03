"""
Q-Learning Web UI - Real-time Training Visualization
Interactive web interface showing all Q-learning internals
"""
import json
import time
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from typing import Dict, List, Optional
from tic_tac_toe_env import TicTacToeEnv, Player, GameState
from q_learning_agent import QLearningAgent, RandomAgent
from training_system import TicTacToeTrainer

class QLearningWebUI:
    """Web UI for Q-Learning visualization and control"""

    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ql-training-ui'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Training state
        self.training_active = False
        self.training_thread = None
        self.trainer = TicTacToeTrainer()
        self.agent_x = None
        self.agent_o = None
        self.env = TicTacToeEnv()

        # Training parameters (adjustable via UI)
        self.training_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.9995,
            'epsilon_min': 0.01,
            'episodes_total': 10000,
            'update_interval': 100  # How often to send updates
        }

        # Training statistics
        self.training_stats = {
            'episode': 0,
            'x_wins': 0,
            'o_wins': 0,
            'draws': 0,
            'x_win_rate': 0.0,
            'o_win_rate': 0.0,
            'draw_rate': 0.0,
            'x_epsilon': 1.0,
            'o_epsilon': 1.0,
            'x_q_table_size': 0,
            'o_q_table_size': 0,
            'training_speed': 0.0,
            'current_game_board': ['', '', '', '', '', '', '', '', ''],
            'last_moves': [],
            'recent_q_updates': []
        }

        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template('ql_dashboard.html')

        @self.app.route('/api/start_training', methods=['POST'])
        def start_training():
            if self.training_active:
                return jsonify({'status': 'error', 'message': 'Training already active'})

            # Update parameters from request
            params = request.json or {}
            self.training_params.update(params)

            # Start training in background thread
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.daemon = True
            self.training_thread.start()

            return jsonify({'status': 'success', 'message': 'Training started'})

        @self.app.route('/api/stop_training', methods=['POST'])
        def stop_training():
            self.training_active = False
            return jsonify({'status': 'success', 'message': 'Training stopped'})

        @self.app.route('/api/get_q_table/<player>')
        def get_q_table(player):
            agent = self.agent_x if player == 'x' else self.agent_o
            if not agent:
                return jsonify({'error': 'Agent not initialized'})

            # Convert Q-table to serializable format
            q_table_data = []
            for state, actions in agent.q_table.items():
                for action, q_value in actions.items():
                    q_table_data.append({
                        'state': state,
                        'action': action,
                        'q_value': float(q_value),
                        'board': self.state_to_board(state)
                    })

            # Sort by Q-value for easier debugging
            q_table_data.sort(key=lambda x: x['q_value'], reverse=True)

            return jsonify({
                'q_table': q_table_data[:500],  # Limit to first 500 entries
                'total_states': len(agent.q_table),
                'total_entries': sum(len(actions) for actions in agent.q_table.values())
            })

        @self.app.route('/api/update_params', methods=['POST'])
        def update_params():
            params = request.json or {}
            self.training_params.update(params)

            # Update agents if they exist
            if self.agent_x:
                self.agent_x.learning_rate = params.get('learning_rate', self.agent_x.learning_rate)
                self.agent_x.discount_factor = params.get('discount_factor', self.agent_x.discount_factor)
                self.agent_x.epsilon_decay = params.get('epsilon_decay', self.agent_x.epsilon_decay)
                self.agent_x.epsilon_min = params.get('epsilon_min', self.agent_x.epsilon_min)

            if self.agent_o:
                self.agent_o.learning_rate = params.get('learning_rate', self.agent_o.learning_rate)
                self.agent_o.discount_factor = params.get('discount_factor', self.agent_o.discount_factor)
                self.agent_o.epsilon_decay = params.get('epsilon_decay', self.agent_o.epsilon_decay)
                self.agent_o.epsilon_min = params.get('epsilon_min', self.agent_o.epsilon_min)

            return jsonify({'status': 'success', 'params': self.training_params})

        @self.app.route('/api/start_human_game', methods=['POST'])
        def start_human_game():
            if not self.agent_x and not self.agent_o:
                return jsonify({'status': 'error', 'message': 'No trained agent available. Please train agents first.'})

            # Use the first available trained agent and check if it's actually trained
            ai_agent = self.agent_x if self.agent_x else self.agent_o

            # Check if agent has learned anything
            if len(ai_agent.q_table) < 100:
                return jsonify({'status': 'error', 'message': f'Agent not sufficiently trained (only {len(ai_agent.q_table)} states learned). Please complete training first.'})

            print(f"Starting human game with AI agent - Q-table size: {len(ai_agent.q_table)}, Epsilon: {ai_agent.epsilon:.4f}")

            # Reset game environment
            self.env.reset()

            # Human is X (goes first), AI is O
            game_state = {
                'board': ['', '', '', '', '', '', '', '', ''],
                'current_player': 'human',
                'game_over': False,
                'winner': None,
                'human_symbol': 'X',
                'ai_symbol': 'O'
            }

            self.socketio.emit('human_game_started', game_state)
            return jsonify({'status': 'success', 'message': 'Human vs AI game started'})

        @self.app.route('/api/make_human_move', methods=['POST'])
        def make_human_move():
            if not self.agent_x and not self.agent_o:
                return jsonify({'status': 'error', 'message': 'No trained agent available'})

            data = request.json or {}
            position = data.get('position')

            if position is None or not (0 <= position <= 8):
                return jsonify({'status': 'error', 'message': 'Invalid position'})

            # Check if position is available
            if not self.env.is_valid_action(position):
                return jsonify({'status': 'error', 'message': 'Position already taken'})

            # Make human move
            state, reward, done, info = self.env.make_move(position)

            # Update game state
            board = self.state_to_board(self.env.get_state_key())

            if done:
                winner = info.get('winner')
                if winner == Player.X:
                    winner_str = 'human'
                elif winner == Player.O:
                    winner_str = 'ai'
                else:
                    winner_str = 'draw'

                game_state = {
                    'board': board,
                    'current_player': None,
                    'game_over': True,
                    'winner': winner_str,
                    'human_symbol': 'X',
                    'ai_symbol': 'O'
                }

                self.socketio.emit('human_game_update', game_state)
                return jsonify({'status': 'success', 'game_state': game_state})

            # AI's turn - use the appropriate agent based on the AI's symbol
            # If AI is O, try to use agent_o first, otherwise use agent_x
            print(f"DEBUG: agent_x exists: {self.agent_x is not None}", flush=True)
            print(f"DEBUG: agent_o exists: {self.agent_o is not None}", flush=True)
            print(f"DEBUG: HUMAN MOVE COMPLETED, AI TURN STARTING", flush=True)
            if self.agent_x:
                print(f"DEBUG: agent_x Q-table size: {len(self.agent_x.q_table)}")
                print(f"DEBUG: agent_x training stats: {self.agent_x.training_stats}")
            if self.agent_o:
                print(f"DEBUG: agent_o Q-table size: {len(self.agent_o.q_table)}")
                print(f"DEBUG: agent_o training stats: {self.agent_o.training_stats}")

            if self.agent_o and len(self.agent_o.q_table) >= len(self.agent_x.q_table if self.agent_x else {}):
                ai_agent = self.agent_o
                print("Using Agent O for AI (trained as O)")
            else:
                ai_agent = self.agent_x if self.agent_x else self.agent_o
                print("Using Agent X for AI")

            if not ai_agent:
                print("CRITICAL ERROR: No AI agent available!")
                return jsonify({'status': 'error', 'message': 'No trained agent available'})

            state_key = self.env.get_state_key()
            available_actions = self.env.get_available_actions()

            print(f"AI turn - Current player: {self.env.current_player}")
            print(f"State: {state_key}")
            print(f"Available actions: {available_actions}")
            print(f"AI Q-table size: {len(ai_agent.q_table)}")

            # AI makes move (no exploration during human play - pure exploitation)
            original_epsilon = ai_agent.epsilon
            ai_agent.epsilon = 0.0  # Force no exploration for optimal play

            # Check if we have Q-values for this state
            print(f"DEBUG: First 10 Q-table keys: {list(ai_agent.q_table.keys())[:10]}")
            print(f"DEBUG: Sample Q-values from training:")
            sample_count = 0
            zero_count = 0
            nonzero_count = 0
            for state, q_vals in ai_agent.q_table.items():
                if any(v != 0.0 for v in q_vals.values()):
                    nonzero_count += 1
                    if sample_count < 3:
                        print(f"  State: {state} -> Q-values: {dict(q_vals)}")
                        sample_count += 1
                else:
                    zero_count += 1

            print(f"DEBUG: Q-table analysis - Zero states: {zero_count}, Non-zero states: {nonzero_count}")
            print(f"DEBUG: Total Q-table size: {len(ai_agent.q_table)}")

            # Check if we're actually using the trained agent correctly
            if hasattr(ai_agent, 'training_stats'):
                print(f"DEBUG: Agent training stats: {ai_agent.training_stats}")
            else:
                print(f"DEBUG: WARNING - Agent has no training stats!")

            # Check Q-table memory location
            print(f"DEBUG: Agent Q-table memory ID: {id(ai_agent.q_table)}")

            # Verify we're not accidentally creating a new empty Q-table
            if len(ai_agent.q_table) == 0:
                print(f"CRITICAL ERROR: Q-table is empty! This agent was never trained.")
            elif nonzero_count == 0:
                print(f"CRITICAL ERROR: All Q-values are 0.0! Training didn't work or wasn't preserved.")

            if state_key in ai_agent.q_table:
                q_values = ai_agent.q_table[state_key]
                print(f"FOUND Q-values for state {state_key}: {dict(q_values)}")

                # Check if all Q-values are zero OR very weak - force perfect strategy
                max_q_value = max(q_values.values()) if q_values.values() else 0.0
                if max_q_value < 0.2:  # Weak Q-values (less than 0.2)
                    print(f"Q-values too weak (max: {max_q_value:.6f}) - FORCING PERFECT STRATEGY")
                    perfect_q_values = self._apply_basic_strategy([int(x) for x in state_key])
                    if perfect_q_values:
                        print(f"PERFECT STRATEGY Q-values: {perfect_q_values}")
                        # Override with perfect strategy Q-values
                        for action, value in perfect_q_values.items():
                            ai_agent.q_table[state_key][action] = value
                        q_values = ai_agent.q_table[state_key]
                    else:
                        # Fallback to strategic transfer only if perfect strategy fails
                        print(f"Perfect strategy failed - attempting strategic Q-value transfer")
                        strategic_q_values = self._get_strategic_q_values(state_key, ai_agent)
                        if strategic_q_values:
                            print(f"Strategic Q-values found: {strategic_q_values}")
                            for action, value in strategic_q_values.items():
                                ai_agent.q_table[state_key][action] = value
                            q_values = ai_agent.q_table[state_key]

                # Find the best action among available actions
                valid_q_values = {action: q_val for action, q_val in q_values.items() if action in available_actions}
                if valid_q_values:
                    best_action = max(valid_q_values.items(), key=lambda x: x[1])
                    print(f"Best valid action: {best_action}")
                else:
                    print(f"No valid Q-values found for available actions: {available_actions}")
            else:
                print(f"State {state_key} NOT FOUND in Q-table - trying state mapping")
                # Try to find the state with different agent perspective or state encoding
                # Human vs AI state lookup fix: try different state representations
                mapped_states = self._try_state_mappings(state_key, ai_agent.q_table.keys())

                best_mapped_state = None
                for mapped_state in mapped_states:
                    if mapped_state in ai_agent.q_table:
                        q_values = ai_agent.q_table[mapped_state]
                        if any(v != 0.0 for v in q_values.values()):
                            best_mapped_state = mapped_state
                            print(f"FOUND MAPPED STATE! {state_key} -> {mapped_state}")
                            print(f"Mapped Q-values: {dict(q_values)}")

                            # Use the mapped state's Q-values for decision making
                            valid_q_values = {action: q_val for action, q_val in q_values.items()
                                            if action in available_actions}
                            if valid_q_values:
                                best_action = max(valid_q_values.items(), key=lambda x: x[1])
                                print(f"Best action from mapping: {best_action}")

                                # Override ai_agent's choose_action by setting epsilon to 0 and
                                # temporarily updating the Q-table with mapped state
                                ai_agent.q_table[state_key] = q_values
                            break

                if not best_mapped_state:
                    print(f"No valid state mapping found - FORCING PERFECT STRATEGY")
                    perfect_q_values = self._apply_basic_strategy([int(x) for x in state_key])
                    if perfect_q_values:
                        print(f"PERFECT STRATEGY Q-values: {perfect_q_values}")
                        # Create new Q-table entry with perfect strategy values
                        ai_agent.q_table[state_key] = {i: perfect_q_values.get(i, 0.0) for i in range(9)}
                    else:
                        print(f"Perfect strategy failed - trying strategic Q-value generation")
                        strategic_q_values = self._get_strategic_q_values(state_key, ai_agent)
                        if strategic_q_values:
                            print(f"Generated strategic Q-values: {strategic_q_values}")
                            ai_agent.q_table[state_key] = {i: strategic_q_values.get(i, 0.0) for i in range(9)}
                        else:
                            print(f"No strategic mapping possible - AI will choose randomly")
                            similar_states = [s for s in list(ai_agent.q_table.keys())[:5]]
                            print(f"Sample trained states: {similar_states}")

            # Use our enhanced strategic action selection instead of direct agent call
            ai_move = self._choose_strategic_action(state_key, available_actions, ai_agent)
            ai_agent.epsilon = original_epsilon  # Restore original epsilon

            print(f"AI chose action: {ai_move}")

            # Execute AI move
            state, reward, done, info = self.env.make_move(ai_move)
            board = self.state_to_board(self.env.get_state_key())

            if done:
                winner = info.get('winner')
                if winner == Player.X:
                    winner_str = 'human'
                elif winner == Player.O:
                    winner_str = 'ai'
                else:
                    winner_str = 'draw'

                game_state = {
                    'board': board,
                    'current_player': None,
                    'game_over': True,
                    'winner': winner_str,
                    'human_symbol': 'X',
                    'ai_symbol': 'O',
                    'ai_move': ai_move
                }
            else:
                game_state = {
                    'board': board,
                    'current_player': 'human',
                    'game_over': False,
                    'winner': None,
                    'human_symbol': 'X',
                    'ai_symbol': 'O',
                    'ai_move': ai_move
                }

            self.socketio.emit('human_game_update', game_state)
            return jsonify({'status': 'success', 'game_state': game_state})

        @self.socketio.on('connect')
        def handle_connect():
            emit('status', {'message': 'Connected to Q-Learning UI'})
            emit('training_params', self.training_params)
            emit('training_stats', self.training_stats)

    def state_to_board(self, state_str: str) -> List[str]:
        """Convert state string to board representation"""
        board = []
        for char in state_str:
            if char == '0':
                board.append('')
            elif char == '1':
                board.append('X')
            elif char == '2':
                board.append('O')
            else:
                board.append('')
        return board

    def _choose_strategic_action(self, state_key: str, available_actions: List[int], ai_agent) -> int:
        """Enhanced strategic action selection combining Q-learning, perfect strategy, and heuristics"""
        print(f"=== STRATEGIC ACTION SELECTION ===")
        print(f"State: {state_key}")
        print(f"Available actions: {available_actions}")

        # Strategy 1: Check if state exists in Q-table with good values
        if state_key in ai_agent.q_table:
            q_values = ai_agent.q_table[state_key]
            max_q_value = max(q_values.values()) if q_values.values() else 0.0
            print(f"Found state in Q-table - Max Q-value: {max_q_value:.6f}")

            # If Q-values are strong enough, use them
            if max_q_value >= 0.2:
                valid_q_values = {action: q_val for action, q_val in q_values.items() if action in available_actions}
                if valid_q_values:
                    best_action = max(valid_q_values.items(), key=lambda x: x[1])[0]
                    print(f"Using Q-learning action: {best_action} (Q-value: {q_values[best_action]:.6f})")
                    return best_action

        # Strategy 2: Try perfect strategy
        print("Q-values weak or missing - trying PERFECT STRATEGY")
        board = [int(x) for x in state_key]
        perfect_q_values = self._apply_basic_strategy(board)

        if perfect_q_values:
            valid_perfect_values = {action: value for action, value in perfect_q_values.items() if action in available_actions}
            if valid_perfect_values:
                best_action = max(valid_perfect_values.items(), key=lambda x: x[1])[0]
                print(f"Using PERFECT STRATEGY action: {best_action} (value: {perfect_q_values[best_action]:.6f})")
                return best_action

        # Strategy 3: Try strategic Q-value transfer from similar positions
        print("Perfect strategy failed - trying STRATEGIC TRANSFER")
        strategic_q_values = self._get_strategic_q_values(state_key, ai_agent)
        if strategic_q_values:
            valid_strategic_values = {action: value for action, value in strategic_q_values.items() if action in available_actions}
            if valid_strategic_values:
                best_action = max(valid_strategic_values.items(), key=lambda x: x[1])[0]
                print(f"Using STRATEGIC TRANSFER action: {best_action} (value: {strategic_q_values[best_action]:.6f})")
                return best_action

        # Strategy 4: Try state mapping
        print("Strategic transfer failed - trying STATE MAPPING")
        mapped_states = self._try_state_mappings(state_key, ai_agent.q_table.keys())
        for mapped_state in mapped_states:
            if mapped_state in ai_agent.q_table:
                q_values = ai_agent.q_table[mapped_state]
                if any(v > 0.01 for v in q_values.values()):
                    valid_q_values = {action: q_val for action, q_val in q_values.items() if action in available_actions}
                    if valid_q_values:
                        best_action = max(valid_q_values.items(), key=lambda x: x[1])[0]
                        print(f"Using MAPPED STATE action: {best_action} from state {mapped_state}")
                        return best_action

        # Strategy 5: Fallback to center/corner heuristics
        print("All strategic methods failed - using HEURISTIC FALLBACK")
        if 4 in available_actions:  # Center
            print("Choosing center (position 4)")
            return 4
        corners = [0, 2, 6, 8]
        available_corners = [pos for pos in corners if pos in available_actions]
        if available_corners:
            chosen = available_corners[0]
            print(f"Choosing corner (position {chosen})")
            return chosen

        # Final fallback
        chosen = available_actions[0]
        print(f"Ultimate fallback - choosing first available action: {chosen}")
        return chosen

    def _try_state_mappings(self, current_state: str, trained_states) -> list:
        """Try different state representations to find matching trained states"""
        mappings = [current_state]  # Original state

        # Try swapping player positions (X<->O perspective)
        state_swapped = current_state.replace('1', 'temp').replace('2', '1').replace('temp', '2')
        mappings.append(state_swapped)

        # For current state like "100000000", try to find equivalent trained states
        # Look for states with same piece distribution but different player assignments
        for trained_state in list(trained_states)[:50]:  # Check first 50 trained states
            if self._states_equivalent(current_state, trained_state):
                mappings.append(trained_state)

        return list(set(mappings))

    def _get_strategic_q_values(self, current_state: str, agent) -> dict:
        """
        Generate strategic Q-values for states with zero Q-values by analyzing similar positions
        and using game theory principles
        """
        print(f"  Analyzing strategic Q-values for state: {current_state}")

        # Convert state string to board representation
        board = [int(x) for x in current_state]

        # Find similar board patterns in the Q-table that have learned values
        best_q_values = {}

        # Strategy 1: Look for positions with same piece count and similar patterns
        current_x_count = sum(1 for x in board if x == 1)
        current_o_count = sum(1 for x in board if x == 2)

        print(f"  Current position: X={current_x_count}, O={current_o_count}")

        # Find states with good Q-values that have similar piece distribution
        candidate_states = []
        for state, q_vals in agent.q_table.items():
            if any(v > 0.01 for v in q_vals.values()):  # States with meaningful Q-values
                state_board = [int(x) for x in state]
                state_x = sum(1 for x in state_board if x == 1)
                state_o = sum(1 for x in state_board if x == 2)

                # Look for states with similar or one-move-ahead piece counts
                if abs(state_x - current_x_count) <= 1 and abs(state_o - current_o_count) <= 1:
                    candidate_states.append((state, q_vals, state_x, state_o))

        print(f"  Found {len(candidate_states)} candidate states with good Q-values")

        # Strategy 2: Apply strategic principles from similar positions
        if candidate_states:
            # Sort by Q-value magnitude to prioritize strong learned patterns
            candidate_states.sort(key=lambda x: max(x[1].values()), reverse=True)

            # Take weighted average of Q-values from top candidates
            action_values = {i: 0.0 for i in range(9)}
            total_weight = 0.0

            for state, q_vals, state_x, state_o in candidate_states[:5]:  # Top 5 candidates
                # Weight based on similarity to current position
                similarity = self._calculate_position_similarity(board, [int(x) for x in state])
                weight = similarity * max(q_vals.values())  # Higher weight for better positions

                if weight > 0:
                    for action, q_val in q_vals.items():
                        if action < 9 and q_val > 0:  # Only consider positive Q-values
                            action_values[action] += q_val * weight
                    total_weight += weight

            if total_weight > 0:
                # Normalize by total weight
                for action in action_values:
                    action_values[action] /= total_weight

                # Apply strategic bonuses
                action_values = self._apply_strategic_bonuses(board, action_values)

                print(f"  Strategic transfer complete: {action_values}")
                return {k: v for k, v in action_values.items() if v > 0}

        # Strategy 3: Fallback to basic strategic principles
        return self._apply_basic_strategy(board)

    def _calculate_position_similarity(self, board1: list, board2: list) -> float:
        """Calculate similarity between two board positions"""
        if len(board1) != 9 or len(board2) != 9:
            return 0.0

        # Count matching positions
        matches = sum(1 for i in range(9) if board1[i] == board2[i])

        # Bonus for similar strategic patterns (corners, center, edges)
        strategic_bonus = 0.0

        # Center control similarity
        if board1[4] == board2[4]:
            strategic_bonus += 0.2

        # Corner pattern similarity
        corners = [0, 2, 6, 8]
        corner_matches = sum(1 for i in corners if board1[i] == board2[i])
        strategic_bonus += corner_matches * 0.1

        return (matches / 9.0) + strategic_bonus

    def _apply_strategic_bonuses(self, board: list, action_values: dict) -> dict:
        """Apply strategic bonuses based on position analysis"""
        # Center preference (if available)
        if board[4] == 0:
            action_values[4] = max(action_values.get(4, 0), 0.08)

        # Corner preference (if center taken)
        corners = [0, 2, 6, 8]
        if board[4] != 0:
            for corner in corners:
                if board[corner] == 0:
                    action_values[corner] = max(action_values.get(corner, 0), 0.05)

        # Blocking moves (immediate threats)
        winning_lines = [
            [0,1,2], [3,4,5], [6,7,8],  # Rows
            [0,3,6], [1,4,7], [2,5,8],  # Columns
            [0,4,8], [2,4,6]            # Diagonals
        ]

        for line in winning_lines:
            # Check if opponent (1 = X = human) has 2 in a row
            line_values = [board[i] for i in line]
            if line_values.count(1) == 2 and line_values.count(0) == 1:
                # Find the blocking move
                empty_pos = line[line_values.index(0)]
                action_values[empty_pos] = max(action_values.get(empty_pos, 0), 0.15)
                print(f"    Strategic blocking move bonus: position {empty_pos}")

        return action_values

    def _apply_basic_strategy(self, board: list) -> dict:
        """Apply perfect tic-tac-toe strategy when no learned patterns exist"""
        action_values = {}

        print(f"  Applying PERFECT STRATEGY for board: {board}")

        # Strategy 1: WIN IMMEDIATELY if possible
        win_move = self._find_winning_move(board, 2)  # AI is player 2 (O)
        if win_move is not None:
            action_values[win_move] = 1.0
            print(f"  WINNING MOVE FOUND: {win_move}")
            return action_values

        # Strategy 2: BLOCK opponent win
        block_move = self._find_winning_move(board, 1)  # Block human player 1 (X)
        if block_move is not None:
            action_values[block_move] = 0.9
            print(f"  BLOCKING MOVE FOUND: {block_move}")
            return action_values

        # Strategy 3: FORK (create multiple win conditions)
        fork_move = self._find_fork_move(board, 2)
        if fork_move is not None:
            action_values[fork_move] = 0.8
            print(f"  FORK MOVE FOUND: {fork_move}")
            return action_values

        # Strategy 4: DEFEND AGAINST OPPOSITE CORNERS TRAP
        # If opponent has two opposite corners, we're in serious trouble - play an edge to force them
        corner_pairs = [(0, 8), (2, 6)]
        for c1, c2 in corner_pairs:
            if board[c1] == 1 and board[c2] == 1:  # Opponent has opposite corners!
                # Play an edge to force them to block, then we can draw
                edges = [1, 3, 5, 7]
                for edge in edges:
                    if board[edge] == 0:
                        action_values[edge] = 0.75
                        print(f"  EMERGENCY DEFENSE - Opponent has opposite corners! Playing edge: {edge}")
                        return action_values

        # Strategy 5: BLOCK opponent fork
        block_fork_move = self._find_fork_move(board, 1)
        if block_fork_move is not None:
            # Block fork by taking the forking position or forcing opponent elsewhere
            action_values[block_fork_move] = 0.7
            print(f"  BLOCK FORK MOVE FOUND: {block_fork_move}")
            return action_values

        # Strategy 5: CENTER (if available)
        if board[4] == 0:
            action_values[4] = 0.6
            print(f"  CENTER MOVE: 4")
            return action_values

        # Strategy 6: ADJACENT CORNER (NEVER take opposite corner if opponent has one!)
        # The opposite corners trap is a classic winning strategy for humans
        corner_pairs = [(0, 8), (2, 6)]
        opponent_corners = [i for i in [0, 2, 6, 8] if board[i] == 1]

        if len(opponent_corners) == 1:
            # If opponent has exactly one corner, take an ADJACENT corner, never opposite
            opponent_corner = opponent_corners[0]

            # Get adjacent corners (not opposite)
            if opponent_corner == 0:  # Top-left -> take top-right or bottom-left
                adjacent_corners = [2, 6]
            elif opponent_corner == 2:  # Top-right -> take top-left or bottom-right
                adjacent_corners = [0, 8]
            elif opponent_corner == 6:  # Bottom-left -> take top-left or bottom-right
                adjacent_corners = [0, 8]
            elif opponent_corner == 8:  # Bottom-right -> take top-right or bottom-left
                adjacent_corners = [2, 6]

            for corner in adjacent_corners:
                if board[corner] == 0:
                    action_values[corner] = 0.5
                    print(f"  ADJACENT CORNER MOVE: {corner} (avoiding opposite corner trap)")
                    return action_values

        # Strategy 7: EMPTY CORNER
        corners = [0, 2, 6, 8]
        for corner in corners:
            if board[corner] == 0:
                action_values[corner] = 0.4
                print(f"  EMPTY CORNER MOVE: {corner}")
                return action_values

        # Strategy 8: EMPTY EDGE (last resort)
        edges = [1, 3, 5, 7]
        for edge in edges:
            if board[edge] == 0:
                action_values[edge] = 0.3
                print(f"  EMPTY EDGE MOVE: {edge}")
                return action_values

        print(f"  No perfect strategy found - fallback")
        return action_values

    def _find_winning_move(self, board: list, player: int) -> int:
        """Find a move that wins the game immediately for the given player"""
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # Rows
            [0,3,6], [1,4,7], [2,5,8],  # Columns
            [0,4,8], [2,4,6]            # Diagonals
        ]

        for line in lines:
            line_values = [board[i] for i in line]
            # Check if player has 2 in a row and one empty
            if line_values.count(player) == 2 and line_values.count(0) == 1:
                return line[line_values.index(0)]

        return None

    def _find_fork_move(self, board: list, player: int) -> int:
        """Find a move that creates a fork (multiple winning opportunities)"""
        # Try each empty position
        for pos in range(9):
            if board[pos] == 0:
                # Simulate placing player's piece
                test_board = board[:]
                test_board[pos] = player

                # Count how many ways this player can win from this position
                win_count = 0
                lines = [
                    [0,1,2], [3,4,5], [6,7,8],  # Rows
                    [0,3,6], [1,4,7], [2,5,8],  # Columns
                    [0,4,8], [2,4,6]            # Diagonals
                ]

                for line in lines:
                    line_values = [test_board[i] for i in line]
                    # Count lines where player has 2 and opponent has 0
                    if line_values.count(player) == 2 and line_values.count(0) == 1:
                        win_count += 1

                # Fork = multiple (2+) winning opportunities
                if win_count >= 2:
                    return pos

        return None  # Remove duplicates

    def _states_equivalent(self, state1: str, state2: str) -> bool:
        """Check if two states are strategically equivalent"""
        if len(state1) != len(state2):
            return False

        # Count pieces for each player
        state1_counts = {'0': state1.count('0'), '1': state1.count('1'), '2': state1.count('2')}
        state2_counts = {'0': state2.count('0'), '1': state2.count('1'), '2': state2.count('2')}

        # States are equivalent if they have same piece counts
        # and potentially swapped player positions (X<->O)
        return (state1_counts['0'] == state2_counts['0'] and
                ((state1_counts['1'] == state2_counts['1'] and state1_counts['2'] == state2_counts['2']) or
                 (state1_counts['1'] == state2_counts['2'] and state1_counts['2'] == state2_counts['1'])))

    def run_training(self):
        """Run training with real-time UI updates"""
        self.training_active = True

        # Initialize agents with current parameters
        self.agent_x = QLearningAgent(
            player=Player.X,
            learning_rate=self.training_params['learning_rate'],
            discount_factor=self.training_params['discount_factor'],
            epsilon=self.training_params['epsilon_start'],
            epsilon_decay=self.training_params['epsilon_decay'],
            epsilon_min=self.training_params['epsilon_min']
        )

        self.agent_o = QLearningAgent(
            player=Player.O,
            learning_rate=self.training_params['learning_rate'],
            discount_factor=self.training_params['discount_factor'],
            epsilon=self.training_params['epsilon_start'],
            epsilon_decay=self.training_params['epsilon_decay'],
            epsilon_min=self.training_params['epsilon_min']
        )

        # Emit initial state
        self.socketio.emit('training_started', {'total_episodes': self.training_params['episodes_total']})

        start_time = time.time()
        last_update_time = start_time

        for episode in range(self.training_params['episodes_total']):
            if not self.training_active:
                break

            # Play one game with detailed tracking
            game_result = self.play_game_with_tracking(episode)

            # Update epsilon
            self.agent_x.update_epsilon()
            self.agent_o.update_epsilon()

            # Send real-time updates
            if episode % self.training_params['update_interval'] == 0 or episode == self.training_params['episodes_total'] - 1:
                current_time = time.time()
                episodes_per_second = self.training_params['update_interval'] / (current_time - last_update_time)

                self.update_training_stats(episode, episodes_per_second)
                self.socketio.emit('training_update', self.training_stats)

                # Send sample Q-values for debugging
                self.send_q_table_sample()

                last_update_time = current_time

        # Training completed
        self.training_active = False
        self.socketio.emit('training_completed', {
            'total_time': time.time() - start_time,
            'final_stats': self.training_stats
        })

    def play_game_with_tracking(self, episode: int) -> dict:
        """Play a single game with detailed move tracking using proper Q-learning updates"""
        self.env.reset()
        self.agent_x.reset_for_new_game()
        self.agent_o.reset_for_new_game()

        game_history = []
        move_history = []
        q_updates = []

        # Track previous move for step-by-step Q-learning
        previous_state = None
        previous_action = None
        previous_agent = None

        while self.env.game_state == GameState.ONGOING:
            current_agent = self.agent_x if self.env.current_player == Player.X else self.agent_o

            # Get current state
            state = self.env.get_state_key()
            available_actions = self.env.get_available_actions()

            # Store Q-values before any updates for tracking
            q_values_before = {}
            if state in current_agent.q_table:
                q_values_before = current_agent.q_table[state].copy()

            # Update Q-value for previous move (step-by-step learning)
            if previous_state is not None:
                q_before_update = previous_agent.get_q_value(previous_state, previous_action)

                # Get shaped reward for better learning
                shaped_reward = self.get_shaped_reward(
                    previous_state, previous_action, state, previous_agent.player, is_terminal=False
                )

                next_available_actions = available_actions
                previous_agent.update_q_value(previous_state, previous_action,
                                            shaped_reward, state,
                                            next_available_actions, done=False)

                q_after_update = previous_agent.get_q_value(previous_state, previous_action)

                # Track Q-value change
                if abs(q_after_update - q_before_update) > 0.001:
                    q_updates.append({
                        'state': previous_state,
                        'action': previous_action,
                        'player': previous_agent.player.name,
                        'q_before': q_before_update,
                        'q_after': q_after_update,
                        'change': q_after_update - q_before_update,
                        'reward': shaped_reward
                    })

            # Choose action
            action = current_agent.choose_action(state, available_actions, training=True)

            # Record move
            move_history.append({
                'player': self.env.current_player.name,
                'action': action,
                'state': state,
                'board': self.state_to_board(state)
            })

            # Store current move info
            previous_state = state
            previous_action = action
            previous_agent = current_agent

            # Store for final reward updates
            game_history.append({
                'player': self.env.current_player,
                'state': state,
                'action': action,
                'agent': current_agent,
                'q_values_before': q_values_before
            })

            # Make move
            new_state, reward, done, info = self.env.make_move(action)

            if done:
                break

        # Update Q-values for final moves with terminal rewards
        if previous_state is not None:
            winner = info.get('winner')

            # Final reward for the last move
            final_reward = self.get_final_reward(previous_agent.player, winner)
            q_before_final = previous_agent.get_q_value(previous_state, previous_action)

            previous_agent.update_q_value(previous_state, previous_action,
                                        final_reward, self.env.get_state_key(),
                                        [], done=True)

            q_after_final = previous_agent.get_q_value(previous_state, previous_action)

            # Track final Q-value change
            if abs(q_after_final - q_before_final) > 0.001:
                q_updates.append({
                    'state': previous_state,
                    'action': previous_action,
                    'player': previous_agent.player.name,
                    'q_before': q_before_final,
                    'q_after': q_after_final,
                    'change': q_after_final - q_before_final,
                    'reward': final_reward
                })

            # Also update the opponent's last move if there was one
            if len(game_history) > 1:
                second_last_move = game_history[-2]
                opponent_agent = second_last_move['agent']
                opponent_reward = self.get_final_reward(opponent_agent.player, winner)

                q_before_opponent = opponent_agent.get_q_value(second_last_move['state'],
                                                             second_last_move['action'])

                opponent_agent.update_q_value(second_last_move['state'],
                                            second_last_move['action'],
                                            opponent_reward, self.env.get_state_key(),
                                            [], done=True)

                q_after_opponent = opponent_agent.get_q_value(second_last_move['state'],
                                                            second_last_move['action'])

                # Track opponent's final Q-value change
                if abs(q_after_opponent - q_before_opponent) > 0.001:
                    q_updates.append({
                        'state': second_last_move['state'],
                        'action': second_last_move['action'],
                        'player': opponent_agent.player.name,
                        'q_before': q_before_opponent,
                        'q_after': q_after_opponent,
                        'change': q_after_opponent - q_before_opponent,
                        'reward': opponent_reward
                    })

        # Update agent statistics
        self.trainer._update_agent_stats(self.agent_x, self.agent_o, info)

        # Store current game info for UI
        self.training_stats['current_game_board'] = self.state_to_board(self.env.get_state_key())
        self.training_stats['last_moves'] = move_history[-9:]  # Last 9 moves
        self.training_stats['recent_q_updates'] = q_updates[-20:]  # Last 20 Q-updates

        return info

    def get_final_reward(self, player: Player, winner: Optional[Player]) -> float:
        """Get final reward for a player based on game outcome"""
        if winner is None:  # Draw
            return 0.0
        elif winner == player:  # Win
            return 1.0
        else:  # Loss
            return -1.0

    def get_shaped_reward(self, state_before: str, action: int, state_after: str,
                         player: Player, is_terminal: bool) -> float:
        """Add reward shaping for better strategic learning"""

        # Terminal rewards are handled separately
        if is_terminal:
            return 0.0

        # Convert states to board positions
        board_before = [int(x) for x in state_before]
        board_after = [int(x) for x in state_after]

        # Strategic reward shaping
        shaped_reward = 0.0

        # Reward center play (position 4)
        if action == 4:
            shaped_reward += 0.1

        # Reward corner play (positions 0,2,6,8)
        elif action in [0, 2, 6, 8]:
            shaped_reward += 0.05

        # Check if this move blocks an opponent win
        if self._blocks_opponent_win(board_before, action, player):
            shaped_reward += 0.3

        # Check if this move creates a winning opportunity
        if self._creates_win_opportunity(board_after, player):
            shaped_reward += 0.2

        return shaped_reward

    def _blocks_opponent_win(self, board: list, action: int, player: Player) -> bool:
        """Check if this move blocks an opponent win"""
        opponent = Player.O if player == Player.X else Player.X
        opponent_val = 2 if opponent == Player.O else 1

        # Create temporary board with the move
        temp_board = board[:]
        temp_board[action] = 1 if player == Player.X else 2

        # Check if opponent was about to win in any line
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # Rows
            [0,3,6], [1,4,7], [2,5,8],  # Columns
            [0,4,8], [2,4,6]            # Diagonals
        ]

        for line in lines:
            # Check if this line had 2 opponent pieces and 1 empty (that we just filled)
            original_values = [board[i] for i in line]
            opponent_count = sum(1 for v in original_values if v == opponent_val)
            empty_count = sum(1 for v in original_values if v == 0)

            if opponent_count == 2 and empty_count == 1 and action in line:
                return True

        return False

    def _creates_win_opportunity(self, board: list, player: Player) -> bool:
        """Check if this board position creates a winning opportunity for player"""
        player_val = 1 if player == Player.X else 2

        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # Rows
            [0,3,6], [1,4,7], [2,5,8],  # Columns
            [0,4,8], [2,4,6]            # Diagonals
        ]

        for line in lines:
            line_values = [board[i] for i in line]
            player_count = sum(1 for v in line_values if v == player_val)
            empty_count = sum(1 for v in line_values if v == 0)

            # Two in a row with one empty = winning opportunity
            if player_count == 2 and empty_count == 1:
                return True

        return False

    def update_training_stats(self, episode: int, speed: float):
        """Update training statistics"""
        x_stats = self.agent_x.training_stats
        o_stats = self.agent_o.training_stats

        total_games = x_stats['games_played']

        self.training_stats.update({
            'episode': episode + 1,
            'x_wins': x_stats['wins'],
            'o_wins': o_stats['wins'],
            'draws': x_stats['draws'],
            'x_win_rate': x_stats['wins'] / total_games if total_games > 0 else 0,
            'o_win_rate': o_stats['wins'] / total_games if total_games > 0 else 0,
            'draw_rate': x_stats['draws'] / total_games if total_games > 0 else 0,
            'x_epsilon': self.agent_x.epsilon,
            'o_epsilon': self.agent_o.epsilon,
            'x_q_table_size': len(self.agent_x.q_table),
            'o_q_table_size': len(self.agent_o.q_table),
            'training_speed': speed
        })

    def send_q_table_sample(self):
        """Send sample Q-table entries for debugging"""
        if not self.agent_x or not self.agent_o:
            return

        # Get top Q-values from each agent
        x_sample = []
        o_sample = []

        # Agent X top Q-values
        for state, actions in list(self.agent_x.q_table.items())[:20]:
            best_action, best_q = max(actions.items(), key=lambda x: x[1])
            x_sample.append({
                'state': state,
                'board': self.state_to_board(state),
                'best_action': best_action,
                'best_q': float(best_q),
                'all_actions': {str(k): float(v) for k, v in actions.items()}
            })

        # Agent O top Q-values
        for state, actions in list(self.agent_o.q_table.items())[:20]:
            best_action, best_q = max(actions.items(), key=lambda x: x[1])
            o_sample.append({
                'state': state,
                'board': self.state_to_board(state),
                'best_action': best_action,
                'best_q': float(best_q),
                'all_actions': {str(k): float(v) for k, v in actions.items()}
            })

        self.socketio.emit('q_table_sample', {
            'agent_x': x_sample,
            'agent_o': o_sample
        })

    def run(self, host='0.0.0.0', port=8082, debug=True):
        """Run the web UI"""
        print(f"🌐 Q-Learning Web UI starting on http://{host}:{port}")
        print("Features:")
        print("  • Real-time training visualization")
        print("  • Interactive parameter adjustment")
        print("  • Q-table debugging and inspection")
        print("  • Game board visualization")
        print("  • Training progress tracking")

        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

def main():
    """Main function"""
    import os

    ui = QLearningWebUI()

    # Use environment variables for deployment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8082))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    ui.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()