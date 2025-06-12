# MTG

This is a part of my thesis.

The software is based on [Ian McKee's MTG scanner](https://github.com/iancmckee/MTG-TCG-CV)

## Installation
Tesseract OCR must be installed locally, and you'll need to set the location inside MTG.py (at the beginning)<br>
``pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe``

[Tesseract OCR](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract)

The program also requires that you have a webcam connected to your computer. <br>
**NOTE!**<br>
If you have more than one (1) video device connected, such as a capture card, you MIGHT need to change the source in the code<br>
``cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # if you have second camera you can set first parameter as 1``<br>
Also you might need to change your resolution, if your camera supportes the one that is set currently<br>
1600 x 1200

I've provided a requirements.txt, pip install this then start the program.<br>
Run ``pip install --upgrade -r requirements.txt`` in the folder where you've downloaded the repo.

### Usage
Run ``python mtg.py`` (in the location of the repo)<br>
Your webcam should start, point it down towards a light colored surface, and then place your MTG card in the center.
The card should be scanned <br>
*There currently is a bug where the name will fluctuate between **Unknown** and **Not found*** <br>

After than you can press the SPACE-key on your keyboard to proceed to next phase or step, or ESC-key to close the program.<br>
If a card would trigger during any of those steps or phases, a reminder will be printed in the terminal.

