def piece_positions_to_fen(piece_positions):
    fen_rows = []
    piece_map = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
    }
    N = 8
    for row in range(N):
        fen_row = ''
        empty_count = 0
        for col in range(N):
            idx = row * 8 + col
            pieces = piece_positions.get(idx, [])
            if pieces:
                piece = max(pieces, key=lambda x: x[3])  # największe confidence
                class_id = piece[2]
                fen_char = piece_map.get(class_id, None)
                if fen_char is None:
                    fen_char = ' '  # albo raise Exception
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += fen_char
            else:
                empty_count += 1
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen_rows.reverse()  # FEN idzie od 8 do 1
    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
    return fen
