import pandas as pd
import numpy as np
import chart_studio.plotly as plt
import plotly.graph_objects as go
import plotly.express as px
import cufflinks as cf

inp = open('./debug.txt', 'r')

#input: moves & stuff
disks = int(inp.readline().strip())
fps = float(inp.readline().strip())
movesn = int(inp.readline().strip())

moves = [] #[disk, from, to, pausePrev, len]
for i in range(movesn):
    moves.append(inp.readline().strip().split(' '))
    for j in range(5):
        moves[-1][j] = int(moves[-1][j])

endPause = int(inp.readline().strip())

#section: data extraction

#move lengths
moveLengths = []
for move in moves:
    moveLengths.append(round(move[4] / fps, 3))

#pauses
pauses = []
for move in moves:
    pauses.append(round(move[3] / fps, 3))
pauses.append(endPause)

#disk amounts
diskAmounts = []
for i in range(disks):
    diskAmounts.append([i+1, 0, 'Nein'])
for move in moves:
    diskAmounts[move[0]][1] += 1
optMoves = np.zeros(disks)
for i in range(disks):
    optMoves *= 2
    optMoves[i] += 1
for i in range(disks):
    diskAmounts.append([i+1, optMoves[i], 'Ja'])
dadf = pd.DataFrame(data=diskAmounts, columns=['Scheibe', 'Anzahl', 'Optimal'])

#disk trails

diskTrails = []
positions = np.zeros(disks)
for i in range(movesn+1):
    for j in range(disks):
        diskTrails.append([i, j+1, positions[j]+1])
    if (i != movesn):
        positions[moves[i][0]] = moves[i][2]

dtdf = pd.DataFrame(data=diskTrails, columns=['Zug', 'Scheibe', 'Position'])

#output: disk amounts
diskAmBar = px.bar(
    dadf,
    x='Scheibe',
    y='Anzahl',
    title='Anzahl Bewegungen pro Scheibe',
    color='Optimal',
    barmode='group'
)
diskAmBar.update_layout(
    yaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    )
)
diskAmBar.show()

#output: disk trails
diskTrLine = px.line(
    dtdf,
    x='Zug',
    y='Position',
    title='Wege der Scheiben',
    color='Scheibe',
    hover_data={'Zug': False}
)

diskTrLine.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 1
    ),
    yaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1
    ),
    hovermode="x"
)
diskTrLine.show()