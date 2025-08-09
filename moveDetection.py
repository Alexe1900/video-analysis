from ultralytics import YOLO
import cv2
from time import perf_counter
import openpyxl as xl
import openpyxl.styles as xlst

JA = True
NEIN = False

txtOut = JA # Ausgabe in TXT
xlOut = JA # Ausgabe in Excel

debugOut = JA
zeit = JA

DISKS = 3   # ANZAHL VON SCHEIBEN
source = './TEST VIDEOS/3 disks/4.mp4'    # WEG ZUM VIDEO

#section: evaluation
start = perf_counter()

normalClasses = list(range(DISKS))

model = YOLO(f'./sds/{DISKS}d.pt')

cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

results = model(source, stream=True)

#section: analysis
analysis = perf_counter()

frames = []

state = []

for result in results:
    names = result.names

    xyxys = result.boxes.xyxy
    
    frame = result.orig_img

    classes = list(map(int, result.boxes.cls))
    sortedClasses = sorted(classes)

    if sortedClasses != normalClasses:
        state.append(['movement/mistake'])
        continue

    valid = True
    state.append([None] * DISKS)
    for i, xyxy in enumerate(xyxys):
        centerX = int((xyxy[0] + xyxy[2]) / 2)
        centerY = int((xyxy[1] + xyxy[3]) / 2)
        center = (centerX, centerY)

        classId = classes[i]
        diskName = names[classId]
        diskId = int(diskName[-1]) - 1

        centerX = centerX * 640 / width

        if (centerX >= 233) and (centerX <= 239):
            state[-1][diskId] = 0
        elif (centerX >= 315) and (centerX <= 321):
            state[-1][diskId] = 1
        elif (centerX >= 398) and (centerX <= 404):
            state[-1][diskId] = 2
        else:
            valid = False
            break

        cv2.circle(frame, center, radius=3, color=(255, 0, 0), thickness=-1)
        cv2.putText(frame, diskName, center, cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)

    if valid:
        frames.append(frame)
    else:
        state.pop()
        state.append(['movement/mistake'])

#section: filtering state
filtering = perf_counter()

filteredState = [[0] * DISKS]
filteredState[0].append(1)
for i in range(1, len(state)):
    if state[i] == ['movement/mistake']:
        if filteredState[-1][0] == 'movement/mistake':
            filteredState[-1][-1] += 1
        else:
            filteredState.append(state[i])
            filteredState[-1].append(1)
    elif state[i] != filteredState[-1][:DISKS]:
        filteredState.append(state[i])
        filteredState[-1].append(1)
    else:
        filteredState[-1][-1] += 1

state = filteredState

#section: extracting moves
moves = []
iter_state = 'normal'
x = -1
for i in range(1, len(state)):
    if state[i][0] == 'movement/mistake':
        iter_state = 'mistake check'
        x = i - 1
    else:
        if iter_state == 'mistake check':
            iter_state = 'normal'
            if state[x][:DISKS] != state[i][:DISKS]:
                for j in range(DISKS-1, -1, -1):
                    if state[x][j] != state[i][j]:
                        moves.append((j, state[x][j], state[i][j], state[x][-1], state[i-1][-1]))
                        break

#section: filtering double moves
filteredMoves = []
i = 1
while i < len(moves):
    if moves[i-1][0] == moves[i][0] and moves[i-1][2] == moves[i][1] and moves[i][3] <= 10:
        if moves[i-1][1] == 0 and moves[i][2] == 2:
            filteredMoves.append((moves[i][0], 0, 2, moves[i-1][3], moves[i-1][4]+moves[i][3]+moves[i][4]))
            i += 2
        elif moves[i-1][1] == 2 and moves[i][2] == 0:
            filteredMoves.append((moves[i][0], 2, 0, moves[i-1][3], moves[i-1][4]+moves[i][3]+moves[i][4]))
            i += 2
        else:
            filteredMoves.append(moves[i-1])
            i += 1

            if i == len(moves)-1:
                filteredMoves.append(moves[i])
    else:
        filteredMoves.append(moves[i-1])

        if i == len(moves)-1:
            filteredMoves.append(moves[i])
            
        i += 1

moves = filteredMoves

#section: TXT output
outputTXT = perf_counter()

if txtOut:
    output = open('./result.txt', 'w')
    output.write('')
    output.close()

    output = open('./result.txt', 'a')

    for move in moves:
        output.write(f'{round(move[3] / fps, 4)} Sekunden lange Pause\n')
        output.write(f'Scheibe {move[0] + 1} von {move[1] + 1} zu {move[2] + 1}, {round(move[4] / fps, 4)} Sekunden lang\n')

    output.write(f'{round(state[-1][-1] / fps, 4)} Sekunden lange Pause\n')

    output.close()

#section: XL output
outputXL = perf_counter()

if xlOut:
    book = xl.Workbook()

    outXL = book.create_sheet('result', 0)

    for letter in 'ACDEFGI':
        outXL[letter+'1'].font = xlst.Font(bold=True, size=12)

    outXL['A1'] = 'Typ'
    outXL['C1'] = 'Zeit (Sek.)'
    outXL['D1'] = 'Zeit (Frames)'
    outXL['E1'] = 'Scheibe'
    outXL['F1'] = 'Von'
    outXL['G1'] = 'Nach'

    outXL['I1'] = 'Bewegungen:'
    outXL['I2'] = len(moves)

    outXL.column_dimensions['B'].width = 3
    outXL.column_dimensions['C'].width = 10
    outXL.column_dimensions['D'].width = 12
    outXL.column_dimensions['E'].width = 8
    outXL.column_dimensions['F'].width = 5.5
    outXL.column_dimensions['G'].width = 5.5
    outXL.column_dimensions['I'].width = 12.5

    for letter in 'CDEFGI':
        for i in range(1, len(moves)*2+4):
            outXL[letter+str(i)].alignment = xlst.Alignment(horizontal='center', vertical='center')

    for i, move in enumerate(moves):
        outXL['A' + str(i*2+3)] = 'Pause'
        outXL['C' + str(i*2+3)] = round(move[3] / fps, 4)
        outXL['D' + str(i*2+3)] = move[3]

        outXL['A' + str(i*2+4)] = 'Bewegung'
        outXL['C' + str(i*2+4)] = round(move[4] / fps, 4)
        outXL['D' + str(i*2+4)] = move[4]
        outXL['E' + str(i*2+4)] = move[0] + 1
        outXL['F' + str(i*2+4)] = move[1] + 1
        outXL['G' + str(i*2+4)] = move[2] + 1

    outXL['A' + str(len(moves)*2+3)] = 'Pause'
    outXL['C' + str(len(moves)*2+3)] = round(state[-1][-1] / fps, 4)
    outXL['D' + str(len(moves)*2+3)] = state[-1][-1]

    book.save('result.xlsx')

#section: benchmarking
end = perf_counter()

if zeit:
    print(f'Time: {end - start}')
    print(f'Eval: {analysis - start}')
    print(f'Analysis: {filtering - analysis}')
    print(f'Filtering: {outputTXT - filtering}')
    print(f'Out: {outputXL - outputTXT}')
    print(f'XL: {end - outputXL}')

    print(f'Sec/fr: {(end-start) / frameCount}')
    print(f'Sec/sec: {(end-start) / (frameCount / fps)}')

#section: debug output
if debugOut:
    debug = open('./debug.txt', 'w')
    debug.write('')
    debug.close()

    debug = open('./debug.txt', 'a')
    debug.write(str(DISKS) + '\n')
    debug.write(str(fps) + '\n')
    debug.write(str(len(moves)) + '\n')
    for move in moves:
        debug.write(' '.join([str(x) for x in move]) + '\n')
    debug.write(str(state[-1][-1]) + '\n')

print('FERTIG')