import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import random
from cvzone.HandTrackingModule import HandDetector
from tkinter import filedialog
from tkinter import messagebox
from  tkinter import *

os.chdir("E:\GoogleDrive\pycv")
os.getcwd()
classes=['Volume Up','Volume Down','10second Backward','10second Forward','Stop']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['SEED']) # Seed 고정

def openfile():
    root = Tk()
    root.filename =  filedialog.askopenfilenames(initialdir = "E:\GoogleDrive\pycv\리모콘 제스쳐",title = "choose your file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
    # print (root.filename)
    root.withdraw()
    return root

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list

        #self.video_id = video_id
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(CFG['FPS']):
            _, img = cap.read()
            img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


def visual(path,result,prob):
    cap=cv2.VideoCapture(path)
    while True:
        ret, img_bgr = cap.read()
        img_bgr = cv2.resize(img_bgr, (480, 480))
        #img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hands, img_bgr = detector.findHands(img_bgr)  # With Draw

        cv2.putText(img_bgr, f'Class:{result}', (5, 440), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 2)
        cv2.putText(img_bgr, f'Prob:{prob:.6f}', (5, 470), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 2)

        # img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if ret == False:
            break
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        # 무한반복
        if (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        # vertical=np.hstack(img_bgr[0],img_bgr[1])
        cv2.imshow("Result", img_bgr)
        # cv2.imshow("Result", self.img_bgr[1])
        # if (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        #     break
    cap.release()
    # img_bgr[1].release()
    cv2.destroyAllWindows()
    return


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    prob=[]
    result=[]
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)
            logit = model(videos)
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
        prob.append(softmax(logit))
    return preds,prob
detector = HandDetector(detectionCon=0.8, maxHands=2)

softmax = torch.nn.Softmax()
model=torch.load(f'리모콘 제스쳐/resnet34_premodel_.pt')

root = openfile()
test_dataset = CustomDataset(root.filename, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

preds,pro = inference(model, test_loader, device)
prob=[pro[0][i].max().item() for i in range(len(pro[0]))]

result = [classes[i] for i in preds]

visual(root.filename[0],result[0],prob[0])


