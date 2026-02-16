litre_A= 4
litre_B= 3
visited=set()

def dfs(state,path):
    x,y =state
    if x == 2:
        print("\nSolution Path:")
        for p in path:
            print(p)
        return True

    if state in visited:
        return False

    visited.add(state)

    moves = [
        ((litre_A, y), "1. fill x till full"),
        ((x, litre_B), "2. fill y till full"),
        ((0, y), "3. empty x fully"),
        ((x, 0), "4. empty y fully"),
        ((x - min(x, litre_B-y), y + min(x, litre_B-y)),
         "5. pour x into y until x is empty"),
        ((x - min(x, litre_B-y), y + min(x, litre_B-y)),
         "6. pour x into y until y is full"),
        ((x + min(y, litre_A-x), y - min(y, litre_A-x)),
         "7. pour y into x until y is empty"),
        ((x + min(y, litre_A-x), y - min(y, litre_A-x)),
         "8. pour y into x until x is full")
    ]

    for new_state, rule in moves:
        if new_state not in visited:
            if dfs(new_state, path + [(new_state, rule)]):
                return True

    return False
start = (0,0)
dfs(start, [(start, "Start")])
