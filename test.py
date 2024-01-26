import pandas as pd
from utils import *



raw_dataset = pd.read_csv('data/sudoku-3m.csv')

# columns: id, puzzle, solution, clues, difficulty
# lignes: 1,
#         1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312,
#         198543726643278591527619843914735268876192435235486179462351987381927654759864312,
#         27,
#         2.2

print(raw_dataset.columns)
print(raw)
print(print_grid(raw_dataset.loc[['puzzle']]))