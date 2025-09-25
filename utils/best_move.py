from utils.piece_positions_to_fen import piece_positions_to_fen

fen = piece_positions_to_fen(piece_positions)

print(fen)


from stockfish import Stockfish

stockfish = Stockfish(path="C:/Users/wojte\Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish.set_fen_position(fen)
best_move = stockfish.get_best_move()
print(best_move)