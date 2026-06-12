import random

def simulate_match(home_win, draw, away_win):

    r = random.random()

    if r < away_win:
        return "away"

    elif r < away_win + draw:
        return "draw"

    return "home"