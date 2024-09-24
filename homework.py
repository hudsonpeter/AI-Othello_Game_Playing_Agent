"""Game Playing"""

from game_helper import compute_best_move, BOARD_DIMENSION


def parse_input(file_path):
    """parse_input: parses input file"""
    with open(file_path, "r", encoding="utf-8") as file:
        # first line - player symbol
        player = file.readline().strip()

        # second line - remaining player time and opponent time
        times = file.readline().strip().split()
        remaining_time, opponent_time = float(times[0]), float(times[1])

        # next 12 lines - game board
        game_board = [
            [line[col] for col in range(BOARD_DIMENSION)]
            for _, line in zip(range(BOARD_DIMENSION), file)
        ]

    # print("File operations completed")
    return player, remaining_time, opponent_time, game_board


def move_mapper(move):
    """Maps coordinates to baord notation"""
    letters = "abcdefghijkl"
    letter = letters[move[1]]
    column_number = move[0] + 1
    return f"{letter}{column_number}"


# parse input
player_disc, player_rt, opponent_rt, board = parse_input("input.txt")

# game play move evaluation
best_move = compute_best_move(board, player_disc, player_rt)

# print(best_move)
final_move = move_mapper(best_move)
# print(final_move)

with open("output.txt", "w", encoding="utf-8") as output_file:
    if final_move is not None:
        output_file.write(final_move)
    output_file.close()
