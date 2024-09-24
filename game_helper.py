"""Game Helper"""

from copy import deepcopy
from random import randrange

BOARD_DIMENSION = 12
DIRECTIONS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)]
DEPTH = 4
EMPTY_CELL = "."

STATIC_WEIGHTED_BOARD = [
    [50, -10, 10, -15, 10, 8, 8, 10, -15, 10, -10, 50],
    [-10, -20, -5, -5, -5, -8, -8, -5, -5, -5, -20, -10],
    [10, -5, 4, 2, 3, 4, 4, 3, 2, 4, -5, 10],
    [-15, -5, 2, 3, 4, 2, 2, 4, 3, 2, -5, -15],
    [10, -5, 3, 4, 5, 4, 4, 5, 4, 3, -5, 10],
    [8, -8, 4, 2, 4, 2, 2, 4, 2, 4, -8, 8],
    [8, -8, 4, 2, 4, 2, 2, 4, 2, 4, -8, 8],
    [10, -5, 3, 4, 5, 4, 4, 5, 4, 3, -5, 10],
    [-15, -5, 2, 3, 4, 2, 2, 4, 3, 2, -5, -15],
    [10, -5, 4, 2, 3, 4, 4, 3, 2, 4, -5, 10],
    [-10, -20, -5, -5, -5, -8, -8, -5, -5, -5, -20, -10],
    [50, -10, 10, -15, 10, 8, 8, 10, -15, 10, -10, 50],
]


def get_opponent_disc(player_disc):
    """Given the player disc gets the opponents disc"""
    return "X" if player_disc == "O" else "O"


def get_valid_moves(board, player_disc):
    """Computes valid moves for the current state of the board"""
    valid_moves = []
    for row in range(BOARD_DIMENSION):
        for col in range(BOARD_DIMENSION):
            if board[row][col] == EMPTY_CELL:
                for delta in DIRECTIONS:
                    if is_valid_cell(board, player_disc, row, col, delta):
                        valid_moves.append((row, col))
                        break

    return valid_moves


def is_valid_cell(board, player_disc, row, col, delta):
    """validates the grid location"""
    opponent_symbol = get_opponent_disc(player_disc)
    x, y = delta

    # Check if there is an opponent's disc in the specified direction
    if (
        not is_inside_board(row + x, col + y)
        or board[row + x][col + y] != opponent_symbol
    ):
        return False

    # Continue checking in the specified direction
    while is_inside_board(row + x, col + y):
        if board[row + x][col + y] == player_disc:
            return True
        if board[row + x][col + y] == EMPTY_CELL:
            break
        row += x
        col += y

    return False


def is_inside_board(row, col):
    """validates is the given coordinates are in the cell"""
    return 0 <= row < BOARD_DIMENSION and 0 <= col < BOARD_DIMENSION


def count_discs(board, player_disc, opponent_disc):
    """counts the number of player's and opponent's discs"""
    player_disc_count = 0
    opponent_disc_count = 0
    for row in range(BOARD_DIMENSION):
        for col in range(BOARD_DIMENSION):
            if board[row][col] == player_disc:
                player_disc_count += 1
            elif board[row][col] == opponent_disc:
                opponent_disc_count += 1
    return player_disc_count, opponent_disc_count


def compute_corner_count(board, player_disc, opponent_disc):
    """counts the number of player's and opponent's corner discs"""
    player_corners_count = 0
    opponent_corners_count = 0
    corner_positions = [(0, 0), (0, 11), (11, 0), (11, 11)]
    for row, col in corner_positions:
        if board[row][col] == player_disc:
            player_corners_count += 1
        elif board[row][col] == opponent_disc:
            opponent_corners_count += 1

    return player_corners_count, opponent_corners_count


def compute_edge_count(board, player_disc, opponent_disc):
    """counts the number of player's and opponent's edges"""
    player_edges = 0
    opponent_edges = 0
    # count top and bottom
    for row in range(1, BOARD_DIMENSION - 1):
        # top edge
        if board[0][row] == player_disc:
            player_edges += 1
        elif board[0][row] == opponent_disc:
            opponent_edges += 1
        # bottom edge
        if board[BOARD_DIMENSION - 1][row] == player_disc:
            player_edges += 1
        elif board[BOARD_DIMENSION - 1][row] == opponent_disc:
            opponent_edges += 1

    # count left and right edges
    for col in range(1, BOARD_DIMENSION - 1):
        # left edge
        if board[col][0] == player_disc:
            player_edges += 1
        elif board[col][0] == opponent_disc:
            opponent_edges += 1
        # right edge
        if board[col][BOARD_DIMENSION - 1] == player_disc:
            player_edges += 1
        elif board[col][BOARD_DIMENSION - 1] == opponent_disc:
            opponent_edges += 1

    return player_edges, opponent_edges


def is_stable(board, row, col, player_disc):
    """validates stability of a player's disc"""
    for direction in DIRECTIONS:
        dr, dc = direction
        r, c = row + dr, col + dc

        while is_inside_board(r, c):
            if board[r][c] == ".":
                break  # Empty cell, not stable in this direction
            if board[r][c] == player_disc:
                return True  # Found a stable disc in this direction
            r += dr
            c += dc

    return False  # No stable disc found in any direction


def compute_stability(board, player_disc):
    """counts the number stable discs for a player"""
    stability_count = 0

    for row in range(BOARD_DIMENSION):
        for col in range(BOARD_DIMENSION):
            if board[row][col] == player_disc:
                if is_stable(board, row, col, player_disc):
                    stability_count += 1

    return stability_count


def compute_static_score(board, player_disc):
    """returns the static score for the given board"""
    score = sum(
        STATIC_WEIGHTED_BOARD[row][col]
        for row in range(BOARD_DIMENSION)
        for col in range(BOARD_DIMENSION)
        if board[row][col] == player_disc
    )
    return score


def evaluate(board, player_disc, thresh):
    """Core evaluation funtion. Evaluates based on number of discs primarily
    If the number if discs are less than thresh, static score is computed
    otherwise dynamic score based on mobility, edge count, corner count, disc
    count and stability"""

    opponent_disc = get_opponent_disc(player_disc)

    # count discs
    player_disc_count, opponent_disc_count = count_discs(
        board, player_disc, opponent_disc
    )

    # return static evaluation if the total number of discs are less than a thresh
    if player_disc_count + opponent_disc_count < thresh:
        return compute_static_score(board, player_disc)

    # compute mobility
    mobility = len(get_valid_moves(board, player_disc))
    opponent_mobility = len(get_valid_moves(board, opponent_disc))

    # compute stability
    stability = compute_stability(board, player_disc)
    opponent_stability = compute_stability(board, opponent_disc)

    # compute corner counts
    player_corners, opponent_corners = compute_corner_count(
        board, player_disc, opponent_disc
    )

    # compute edge counts
    player_edges, opponent_edges = compute_edge_count(board, player_disc, opponent_disc)

    disc_count_h = evaluate_heuristic(player_disc_count, opponent_disc_count)
    mobility_h = evaluate_heuristic(mobility, opponent_mobility)
    corners_h = evaluate_heuristic(player_corners, opponent_corners)
    edges_h = evaluate_heuristic(player_edges, opponent_edges)
    stability_h = evaluate_heuristic(stability, opponent_stability)

    combined_eval = round(
        disc_count_h + mobility_h + corners_h + edges_h + stability_h, 3
    )
    return combined_eval


def evaluate_heuristic(player_value, opponent_value):
    """Heuristic calculator"""
    if (player_value + opponent_value) == 0:
        return 0
    return 100 * ((player_value - opponent_value) / (player_value + opponent_value))


def make_move(board, move, player_disc):
    """Makes the move on the board and flips discs for that move"""
    row, col = move
    new_board = deepcopy(board)
    # new_board = [row[:] for row in board]  # Create a deep copy of the board

    # Place the player's disc at the specified position
    new_board[row][col] = player_disc

    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        discs_to_flip = []

        while (
            0 <= r < BOARD_DIMENSION
            and 0 <= c < BOARD_DIMENSION
            and new_board[r][c] != player_disc
            and new_board[r][c] != EMPTY_CELL
        ):
            discs_to_flip.append((r, c))
            r += dr
            c += dc

        if (
            0 <= r < BOARD_DIMENSION
            and 0 <= c < BOARD_DIMENSION
            and new_board[r][c] == player_disc
        ):
            # Flip the discs in this direction
            for flip_row, flip_col in discs_to_flip:
                new_board[flip_row][flip_col] = player_disc

    return new_board


def game_over(board):
    """checks if the given board state terminates the game"""
    return not any(get_valid_moves(board, player_color) for player_color in ["X", "O"])


def minimax(board, depth, maximizing_player, alpha, beta, player_disc, thresh):
    """implementation of minimax with alpha-beta pruning"""

    if depth == 0 or game_over(board):
        return evaluate(board, player_disc, thresh)

    # maximizing player
    if maximizing_player:
        max_eval = float("-inf")
        valid_moves = get_valid_moves(board, player_disc)
        for move in valid_moves:
            new_board = make_move(board, move, player_disc)
            eval_value = minimax(
                new_board, depth - 1, False, alpha, beta, player_disc, thresh
            )
            max_eval = max(max_eval, eval_value)
            alpha = max(alpha, eval_value)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval

    # minimizing player
    min_eval = float("inf")
    opponent_disc = get_opponent_disc(player_disc)
    valid_moves = get_valid_moves(board, opponent_disc)
    for move in valid_moves:
        new_board = make_move(board, move, opponent_disc)
        eval_value = minimax(
            new_board, depth - 1, True, alpha, beta, player_disc, thresh
        )
        min_eval = min(min_eval, eval_value)
        beta = min(beta, eval_value)
        if beta <= alpha:
            break  # Alpha cut-off
    return min_eval


def compute_best_move(board, player_disc, remaining_time):
    """Computes next best move based on evaluations"""
    # print("computing best move")

    # get valid moves for player
    valid_moves_list = get_valid_moves(board, player_disc)
    print(len(valid_moves_list))

    # based on previous runs
    thresh = 50 if player_disc == "X" else 66

    best_move = None
    max_eval = float("-inf")

    # if remaining time is less than 5s return random move
    if remaining_time < 5:
        best_move = valid_moves_list[randrange(0, len(valid_moves_list))]
    else:
        for move in valid_moves_list:
            new_board = make_move(board, move, player_disc)
            eval_value = minimax(
                new_board,
                DEPTH,
                False,
                float("-inf"),
                float("inf"),
                player_disc,
                thresh,
            )
            if eval_value > max_eval:
                max_eval = eval_value
                best_move = move

    # if move is not computed then depth = 2 failed, so complete evaluation with
    # depth = 0
    if best_move is None:
        print("Doomed!!")
        for move in valid_moves_list:
            new_board = make_move(board, move, player_disc)
            eval_value = minimax(
                new_board, 0, False, float("-inf"), float("inf"), player_disc, thresh
            )
            if eval_value > max_eval:
                max_eval = eval_value
                best_move = move

    # print("Best Move: ", best_move)
    return best_move
