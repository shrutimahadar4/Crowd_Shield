# counter.py

entered_ids = set()
exited_ids = set()

def reset_counter():
    entered_ids.clear()
    exited_ids.clear()

def update_count(track_id, prev_y, curr_y, entry_line):
    if prev_y < entry_line and curr_y >= entry_line:
        entered_ids.add(track_id)

    elif prev_y > entry_line and curr_y <= entry_line:
        exited_ids.add(track_id)

    return max(len(entered_ids) - len(exited_ids), 0)
