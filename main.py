import re

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QTextEdit, QHBoxLayout
)
import sys
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input, 5 hidden, 1 output
        self.fc1 = nn.Linear(11,12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, 12)
        self.fc5 = nn.Linear(12, 12)
        self.fc6 = nn.Linear(12, 12)
        self.fc7 = nn.Linear(12, 5)
    def forward(self, x):
        x = (F.relu(self.fc1(x))) #YOUR CODE. Apply layers created in __init__.
        x = (F.relu(self.fc2(x)))
        x = (F.relu(self.fc3(x)))
        x = (F.relu(self.fc4(x)))
        x = (F.relu(self.fc5(x)))
        x = (F.relu(self.fc6(x)))
        x = ((self.fc7(x)))
        return x


model = LeNet()
model.load_state_dict(torch.load("LeNet_weights.pth", map_location=DEVICE))
model = model.to(DEVICE)

sdd = [
   [ 0,3,4,2,0,0,1,0,0,0,4,5,2,6,0,2,0,4,4,3,0,6,0,0,3,5],
   [ 0,0,0,0,6,0,0,0,0,9,0,7,0,0,0,0,0,0,0,0,7,0,0,0,7,0],
    [3,0,0,0,2,0,0,6,0,0,8,0,0,0,6,0,5,0,0,0,3,0,0,0,0,0],
    [1,6,0,0,1,0,0,0,4,4,0,0,0,0,0,0,0,0,0,1,0,0,4,0,1,0],
    [0,0,4,5,0,0,0,0,0,3,0,0,3,2,0,3,6,5,4,0,0,4,3,8,0,0],
    [3,0,0,0,0,5,0,0,2,1,0,0,0,0,5,0,0,2,0,4,1,0,0,0,0,0],
    [2,0,0,0,1,0,0,6,1,0,0,0,0,0,2,0,0,1,0,0,2,0,0,0,0,0],
    [5,0,0,0,7,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,5,0,0,0,4,0,0,0,1,1,3,7,0,0,0,0,5,3,0,5,0,0,0,8],
    [0,0,0,0,6,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,9,0,0,0,0,0],
    [0,0,0,0,6,0,0,0,5,0,0,0,0,4,0,0,0,0,0,0,0,0,1,0,0,0],
    [2,0,0,4,2,0,0,0,3,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,7,0],
    [5,5,0,0,5,0,0,0,2,0,0,0,0,0,2,6,0,0,0,0,2,0,0,0,6,0],
    [0,0,4,7,0,0,8,0,0,2,2,0,0,0,0,0,3,0,0,4,0,0,0,0,0,0],
    [0,2,0,0,0,8,0,0,0,0,4,0,5,5,0,2,0,4,0,0,7,4,5,0,0,0],
    [3,0,0,0,3,0,0,0,0,0,0,5,0,0,5,7,0,6,0,0,3,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0],
    [1,0,0,0,4,0,0,0,2,0,4,0,0,0,2,0,0,0,0,0,0,0,0,0,5,0],
    [1,1,0,0,0,0,0,1,2,0,0,0,0,0,1,4,4,0,1,4,2,0,4,0,0,0],
    [0,0,0,0,0,0,0,8,3,0,0,0,0,0,3,0,0,0,0,0,0,0,2,0,0,0],
    [0,4,3,0,0,0,5,0,0,0,0,6,2,3,0,6,0,6,5,3,0,0,0,0,0,6],
    [0,0,0,0,8,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [6,0,0,0,2,0,0,6,6,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
    [3,0,7,0,1,0,0,0,2,0,0,0,0,0,0,9,0,0,0,5,0,0,0,6,0,0],
    [1,6,2,0,0,2,0,0,0,6,0,0,2,0,6,2,1,0,2,1,0,0,6,0,0,0],
    [2,0,0,0,8,0,0,0,0,6,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,9]
]
ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
eng_alp = {}
for i in range (len(ascii_lowercase)):
    eng_alp[ascii_lowercase[i]] = i
def SDD(strk): #normalize?
    strk = strk.lower()
    mass = strk.split()
    sumi = 0
    counter = 0
    for it in mass:
        for i in range (len(it) - 1):
            sumi += sdd[eng_alp[it[i]]][eng_alp[it[i + 1]]]
            counter += 1
    if counter == 0: return 0
    return sumi / counter

freq_dict = {
  'a' : 0.08167, 'b' : 0.01492, 'c' : 0.02782, 'd' : 0.04253, 'e' : 0.12702, 'f' : 0.02228,
  'g' : 0.02015, 'h' : 0.06094, 'i' : 0.06966, 'j' : 0.00153, 'k' : 0.00772, 'l' : 0.04025,
  'm' : 0.02406, 'n' : 0.06749, 'o' : 0.07507, 'p' : 0.01929, 'q' : 0.00095, 'r' : 0.05987,
  's' : 0.06327, 't' : 0.09056, 'u' : 0.07258, 'v' : 0.00978, 'w' : 0.02360, 'x' : 0.00150,
  'y' : 0.01974, 'z' : 0.00074
}

def CSS(strk):
    strk = strk.lower()
    sumi = 0
    for w in freq_dict:
        if w in freq_dict: sumi += (strk.count(w) / len(strk) - freq_dict[w])**2 / freq_dict[w]
    return sumi

def NUC(strk):
    strk = strk.lower()
    return len(set(strk))

def IC(strk):
    if len(strk) == 0 or len(strk) == 1:
        return 0
    strk = strk.lower()
    sumi = 0
    for w in freq_dict:
        a = strk.count(w)
        b = len(strk)
        if w in freq_dict: sumi += a * (a - 1) / (b*(b-1))
    return sumi

def MIC(strk):
    maxi = -1
    for period in range(1, 16):
        begin = period - 1
        sumi = 0
        for i in range(period):
            cur = ''
            counter = 0
            for j in range(begin, len(strk), period):
                cur += strk[j]
            begin += 1
            a = IC(cur)
            sumi += a
        if (sumi / period) > maxi: maxi = sumi / period
    return maxi

def MKA(strk):
    maxi = -1
    for period in range (1,16):
        st = strk[0:len(strk)-period]
        st = (period * '!') + st
        cur = 0
        for i in range (len(strk)):
            if (strk[i] == st[i]): cur += 1
        if maxi < cur/len(strk): maxi = cur/len(strk)
    return maxi

def DIC(strk):
    strk = strk.lower()
    sumi = 0
    t = set()
    for i in range(len(strk) - 1):
        if (strk[i] in freq_dict and strk[i+1] in freq_dict and (strk[i] + strk[i+1]) not in t):
            st = strk[i] + strk[i+1]
            t.add(st)
            a = 0
            for k in range (len(strk) - 1):
                if (strk[k] + strk[k+1] == st):
                    a += 1
            b = len(strk)
            sumi += a * (a - 1) / ((b-1)*(b-2))
    return sumi

def EDI(strk):
    strk = strk.lower()
    sumi = 0
    t = set()
    for i in range(0, len(strk) - 2, 2):
        if (strk[i] in freq_dict and strk[i+2] in freq_dict and (strk[i] + strk[i+2]) not in t):
            st = strk[i] + strk[i+1]
            t.add(st)
            a = 0
            for k in range (len(strk) - 1):
                if (strk[k] + strk[k+1] == st):
                    a += 1
            b = len(strk)
            sumi += a * (a - 1) / ((b-1)*(b-2))
    return sumi

from math import sqrt
def LR(strk):
    strk = strk.lower()
    st = set()
    sumi = 0
    for w in freq_dict:
        if w not in st:
            a = strk.count(w)
            st.add(w)
            if a == 3: sumi += 3
    return sqrt(sumi)/len(strk)


def ROD(strk):
    strk = strk.lower()
    st = set()
    sum_all = 0
    sum_odd = 0
    for i in range(len(strk) - 1):
        cur_all = 0
        cur_odd = 0
        if (strk[i] not in st):
            st.add(strk[i])
            for j in range(i, len(strk)):
                if (strk[i] == strk[j]):
                    cur_all += 1
                    if j % 2 == 1:
                        cur_odd += 1
        if cur_all > 1: sum_all += cur_all
        if cur_odd > 1: sum_odd += cur_odd
    if (sum_all == 0): return (0.5)
    return sum_odd / sum_all

logdi =[
    [4,7,8,7,4,6,7,5,7,3,6,8,7,9,3,7,3,9,8,9,6,7,6,5,7,4],
    [7,4,2,0,8,1,1,1,6,3,0,7,2,1,7,1,0,6,5,3,7,1,2,0,6,0],
    [8,2,5,2,7,3,2,8,7,2,7,6,2,1,8,2,2,6,4,7,6,1,3,0,4,0],
    [7,6,5,6,8,6,5,5,8,4,3,6,6,5,7,5,3,6,7,7,6,5,6,0,6,2],
    [9,7,8,8,8,7,6,6,7,4,5,8,7,9,7,7,5,9,9,8,5,7,7,6,7,3],
    [7,4,5,3,7,6,4,4,7,2,2,6,5,3,8,4,0,7,5,7,6,2,4,0,5,0],
    [7,5,5,4,7,5,5,7,7,3,2,6,5,5,7,5,2,7,6,6,6,3,5,0,5,1],
    [8,5,4,4,9,4,3,4,8,3,1,5,5,4,8,4,2,6,5,7,6,2,5,0,5,0],
    [7,5,8,7,7,7,7,4,4,2,5,8,7,9,7,6,4,7,8,8,4,7,3,5,0,5],
    [5,0,0,0,4,0,0,0,3,0,0,0,0,0,5,0,0,0,0,0,6,0,0,0,0,0],
    [5,4,3,2,7,4,2,4,6,2,2,4,3,6,5,3,1,3,6,5,3,0,4,0,5,0],
    [8,5,5,7,8,5,4,4,8,2,5,8,5,4,8,5,2,4,6,6,6,5,5,0,7,1],
    [8,6,4,3,8,4,2,4,7,1,0,4,6,4,7,6,1,3,6,5,6,1,4,0,6,0],
    [8,6,7,8,8,6,9,6,8,4,6,6,5,6,8,5,3,5,8,9,6,5,6,3,6,2],
    [6,6,7,7,6,8,6,6,6,3,6,7,8,9,7,7,3,9,7,8,9,6,8,4,5,3],
    [7,3,3,3,7,3,2,6,7,2,1,7,3,2,7,6,0,7,6,6,6,0,3,0,4,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0],
    [8,6,6,7,9,6,6,5,8,3,6,6,6,6,8,6,3,6,8,8,6,5,6,0,7,1],
    [8,6,7,6,8,6,5,7,8,4,6,6,6,6,8,7,4,5,8,9,7,4,7,0,6,2],
    [8,6,6,5,8,6,5,9,8,3,3,6,6,5,9,6,2,7,8,8,7,4,7,0,7,2],
    [6,6,7,6,6,4,6,4,6,2,3,7,7,8,5,6,0,8,8,8,3,3,4,3,4,3],
    [6,1,0,0,8,0,0,0,7,0,0,0,0,0,5,0,0,0,1,0,2,1,0,0,3,0],
    [7,3,3,4,7,3,2,8,7,2,2,4,4,6,7,3,0,5,5,5,2,1,4,0,3,1],
    [4,1,4,2,4,2,0,3,5,1,0,1,1,0,3,5,0,1,2,5,2,0,2,2,3,0],
    [6,6,6,6,6,6,5,5,6,3,3,5,6,5,8,6,3,5,7,6,4,3,6,2,4,2],
    [4,0,0,0,5,0,0,0,3,0,0,2,0,0,3,0,0,0,1,0,2,0,0,0,4,4]
]

ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
eng_alp = {}
for i in range (len(ascii_lowercase)):
    eng_alp[ascii_lowercase[i]] = i
def LDI(strk):
    strk = strk.lower()
    mass = strk.split()
    sumi = 0
    counter = 0
    for it in mass:
        for i in range (len(it) - 1):
            sumi += logdi[eng_alp[it[i]]][eng_alp[it[i + 1]]]
            counter += 1
    if counter == 0: return 0
    return sumi / counter

def predict_cipher_probabilities(text: str):
    txt = re.sub(r'[^a-zA-Z ]', '', text.lower())
    features = extract_features(txt).unsqueeze(0).to(DEVICE)
    #print("FEATURE SHAPE:", features.shape)
    model.eval()
    with torch.no_grad():
        output = model(features)
        probs = torch.nn.functional.softmax(output, dim=-1).cpu().numpy()[0]

    cipher_names = ["Bazeries", "Beaufort", "Bifid", 'Playfair', 'TwoSquare']
    return dict(zip(cipher_names, probs))


class CipherPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Определение шифра по тексту")
        self.setGeometry(100, 100, 500, 400)

        self.layout = QVBoxLayout()

        self.input_label = QLabel("Введите зашифрованный текст:")
        self.text_edit = QTextEdit()

        self.predict_button = QPushButton("Определить шифр")
        self.predict_button.clicked.connect(self.predict)

        self.result_labels = {
            "Bazeries": QLabel("Bazeries: "),
            "Beaufort": QLabel("Beaufort: "),
            "Playfair": QLabel("Playfair: "),
            "TwoSquare": QLabel("TwoSquare: "),
            "Bifid": QLabel("Bifid: "),
        }

        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.predict_button)
        self.layout.addSpacing(10)

        for label in self.result_labels.values():
            self.layout.addWidget(label)

        self.setLayout(self.layout)

    def predict(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            return

        probs = predict_cipher_probabilities(text)

        for cipher, prob in probs.items():
            self.result_labels[cipher].setText(f"{cipher}: {prob:.2%}")

def extract_features(strk):
    features = [
        NUC(strk),
        CSS(strk),
        IC(strk),
        MIC(strk),
        MKA(strk),
        DIC(strk),
        EDI(strk),
        LR(strk),
        ROD(strk),
        LDI(strk),
        SDD(strk)
    ]
    #print(features)
    return torch.tensor(features, dtype=torch.float32)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CipherPredictor()
    window.show()
    sys.exit(app.exec())
