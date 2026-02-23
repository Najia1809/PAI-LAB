def is_safe(board, row, col):
    for i in range(row):
        if board[i] == col or abs(board[i] - col) == abs(i - row):
            return False
    return True
def solve(board, row):
    if row == 4:
        print("Solution:")
        print_board(board)
        return
    for col in range(4):
        if is_safe(board, row, col):
            board[row] = col
            solve(board, row + 1)
def print_board(board):
    for i in range(4):
        for j in range(4):
            if board[i] == j:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()
    print()
board = [-1] * 4
solve(board, 0)