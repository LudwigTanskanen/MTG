import pytesseract
import numpy as np
import re
import difflib
import json
import cv2
from pynput import keyboard
import threading
import time


############### CONFIG ###############
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # if you have second camera you can set first parameter as 1
if not cap.isOpened():
    print("Error: Could not open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 1600) # you should chose a value that the camera supports
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 1200) # you should chose a value that the camera supports
cap.set(cv2.CAP_PROP_FPS, 5)    
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)


lastkey = ""
stop_flag = threading.Event()

# Structure is Phase (key) : steps (value) [list], if value is None, then there are no individual steps inside of that phase.
PAS = {          
        'beginning phase':[
            'untap',
            'upkeep',
            'draw'
            ], 
        'pre-combat main phase': None, 
        'combat phase': [
            'beginning of combat', 
            'declare attacker', 
            'declare blockers', 
            'first strike damage', 
            'damage', 
            'end of combat'
            ], 
        'post-combat main phase': None, 
        'end phase': [
            'beginning of endstep', 
            'clean up'
            ]
        }

############### CLASSES ###############

class GameState:
    def __init__(self):
        self.cardsInPlay = []
        self.currentPhaseIndex = 0
        self.currentStepIndex = 0
        self.currentPhase = list(PAS.keys())[0]
        self.currentStep = PAS[self.currentPhase][0] if PAS[self.currentPhase] else None

state = GameState()

############### HELP FUNCTIONS ###############
def empty(a):
    pass


def advance_game_state(state: GameState):
    phases = list(PAS.keys())

    # Advance step if the phase has steps
    if PAS[state.currentPhase]:
        state.currentStepIndex += 1
        if state.currentStepIndex < len(PAS[state.currentPhase]):
            state.currentStep = PAS[state.currentPhase][state.currentStepIndex]
            return

    # Otherwise, move to next phase
    state.currentPhaseIndex += 1
    if state.currentPhaseIndex < len(phases):
        state.currentPhase = phases[state.currentPhaseIndex]
        state.currentStepIndex = 0
        state.currentStep = PAS[state.currentPhase][0] if PAS[state.currentPhase] else None
    else:
        # Loop back to start of turn
        state.currentPhaseIndex = 0
        state.currentPhase = phases[0]
        state.currentStepIndex = 0
        state.currentStep = PAS[state.currentPhase][0] if PAS[state.currentPhase] else None


def extract_triggers(text):
    text = text.lower()
    triggers = []
    trigger_keywords = {
        "untap": "untap",
        "upkeep": "upkeep",
        "draw": "draw",
        "beginning of combat": "beginning of combat",
        "declare attackers": "declare attacker",
        "declare blockers": "declare blockers",
        "end step": "beginning of endstep"
    }

    for keyword, step in trigger_keywords.items():
        if keyword in text:
            triggers.append(step)
    return triggers


def show_trigger_reminders(state: GameState):
    reminders = []
    for card in state.cardsInPlay:
        if isinstance(card, dict):  # Cards stored as dict with triggers
            if state.currentStep in card.get('triggers', []):
                reminders.append(card['name'])
    if reminders:
        print(f"[Reminder] Step '{state.currentStep}': Abilities trigger for {', '.join(reminders)}")

############### OPEN CV ###############

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getName(img, width, height, json_file, state, non_permanents):
    card_title = img[25:75, 15:400] #img[15:80, 10:450] 
    cv2.imshow("Card Name",card_title)

    gray = cv2.cvtColor(card_title, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    card_name = re.sub('[^a-zA-Z0-9,+ ]', '', pytesseract.image_to_string(thresh))  # was checking cardtitle before.
    card_name = card_name.strip().lower()

    names = [x['name'].lower() for x in json_file]
    matched_name = None

    closest_match = difflib.get_close_matches(card_name, names, cutoff=0.8)
    if closest_match:
        matched_name = closest_match[0]
        for card in json_file:
            if card['name'].lower() == matched_name and card['type_line'] not in non_permanents:
                # Check if card already exists in play
                for existing in state.cardsInPlay:
                    if existing['name'] == matched_name:
                        # Merge triggers, avoiding duplicates
                        new_triggers = extract_triggers(card.get("oracle_text", ""))
                        existing['triggers'] = list(set(existing['triggers']) | set(new_triggers))
                        return 

                # Add new card with its triggers
                triggers = extract_triggers(card.get("oracle_text", ""))
                state.cardsInPlay.append({
                    'name': matched_name,
                    'triggers': triggers,
                })
                return matched_name
    elif matched_name != None:
        return matched_name
                
    else:
        closest_match = "Not Found"
    
    print(f"[Debug] Current cards in play: {state.cardsInPlay}")
    return closest_match


def getContours(img, imgContour, originalImg, imgConts, json_file, state, non_permanents):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    width, height = 500, 700

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if area > areaMin and len(approx) == 4:
            cv2.drawContours(imgConts, cnt, -1, (255, 0, 255), 7)
            rectX, rectY, rectW, rectH = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (rectX, rectY), (rectX + rectW, rectY + rectH), (0, 255, 0), 5)
            points = []
            
            for point in approx:
                x, y = point[0]
                points.append([x, y])
            card = np.float32(points)
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            x4, y4 = points[3]

            # distance formula
            # sqrt( (x2-x1)^2 + (y2-y1)^2 )
            # This should make it so if it's cocked it still gets put into up, down regardless if it's cocked
            # left or cocked right
            if np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)) < np.sqrt(np.square(x1 - x4) + np.square(y1 - y4)):
                # top point goes to top right
                cardWarped = np.float32([[width, 0], [0, 0], [0, height], [width, height]])
            else:
                # top point goes to top left
                cardWarped = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
            matrix = cv2.getPerspectiveTransform(card, cardWarped)
            imgOutput = cv2.warpPerspective(originalImg, matrix, (width, height))
            card_name = getName(imgOutput, width, height, json_file, state, non_permanents)
            textColor = (255, 50, 0)

            if card_name:
                cv2.putText(imgContour, card_name + " ", (rectX, rectY - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, textColor, 2)
            else:
                cv2.putText(imgContour, "Unknown", (rectX, rectY - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, textColor, 2)

############### INPUT HANDLING ###############

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    print('{0} released'.format(key))
    global lastkey

    if key == keyboard.Key.esc:
        print("[keyboard] ESC detected, setting stop_flag")
        stop_flag.set()
        return False
    elif key == keyboard.Key.space:
        advance_game_state(state)
        show_trigger_reminders(state)
    else:
        lastkey = key
        print(type(lastkey))

############### THREADS ###############    

def thread_boot(state, json_file, non_permanents):
    listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            name="Listener")
    mainThread = threading.Thread(target=main, args=(state,))
    
    print("[boot] Starting threads")

    listener.start()
    mainThread.start()

    runCv(state, json_file, non_permanents)

    while not stop_flag.is_set():
        time.sleep(0.1)

    print("[boot] Stop flag detected, waiting for threads")
    
    listener.stop()
    mainThread.join(timeout=5)
    listener.join(timeout=5)

    print("[boot] All threads joined. Program exiting.")
    
############### OPEN CV MAIN LOOP ###############

def runCv(state, json_file, non_permanents):
    print("[runCv] Started")

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 640, 180)
    cv2.createTrackbar("Threshold1", "Parameters", 100, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 140, 255, empty)
    cv2.createTrackbar("Area", "Parameters", 35000, 100000, empty)
    
    try:
        while not stop_flag.is_set():
            success,img = cap.read()
            if not success:
                continue


            imgContour = img.copy()
            imgConts = img.copy()

            imgBlur = cv2.GaussianBlur(img, (5,5), 0)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)


            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
            imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
            kernel = np.ones((5, 5), np.uint8)
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

            getContours(imgDil, imgContour, img, imgConts, json_file, state, non_permanents)

            imgStack = stackImages(0.8, ([img, imgGray, imgCanny],
                                        [imgConts, imgContour, img]))
            
            cv2.imshow('Stack', imgStack)                 
            cv2.imshow('Image',imgContour)

            if cv2.waitKey(1) & 0xFF == 27:
                stop_flag.set()
                break
        
    finally:
        cap.release()
        for i in range(5):
            cv2.waitKey(1)  # let GUI events flush
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[runCv] Error destroying OpenCV windows: {e}")


def main(state):
    global lastkey
    while not stop_flag.is_set():
        if lastkey:
            advance_game_state(state)
            show_trigger_reminders(state)
            print(f"[main] Last key: {lastkey}")
            print(f"[main] Current cards in play: {state.cardsInPlay}")
            lastkey = None
        time.sleep(0.1)

if __name__ == '__main__':
    with open('mtg.json', encoding='utf-8') as f:
        d = json.load(f)
    state = GameState()
    permanents = ['Creature', 'Enchantment', 'Artifact']
    non_permanents = ['Instant', 'Sorcery', 'Legendary sorcery']   

    thread_boot(state, d, non_permanents)

    print(f"cards in play: {state.cardsInPlay}")