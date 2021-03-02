import os
import sys 
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from utils import unet # full size version 173.6 MB
from utils import *
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='bg_removal')
parser.add_argument('--src_img', required=True, help='Path for foreground image')
parser.add_argument('--bg_img', required=True, help='Path for background image')
parser.add_argument('--out', required=False, default="./out/")
parser.add_argument('--model', required=False, default="./models/human.pth")
args = parser.parse_args()

def find_human(img):
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose

  # For static images:
  pose = mp_pose.Pose(
      static_image_mode=True, min_detection_confidence=0.5)
  image = cv2.imread(img)
  # Convert the BGR image to RGB before processing.
  try:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks == None:
      print("human not found out!!!")
      sys.exit(0)
  except Exception:
    print("err in image try another image...")
    sys.exit(0)


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,src_img,back_img):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    #imo.save(d_dir+imidx+'.png')
    

    # Read the images
    foreground = cv2.imread(src_img)
    background = cv2.imread(back_img)
    alpha = np.array(imo) 

    #resizing
    h,w=foreground.shape[:2]
    background = cv2.resize(background, (h , w),  
                  interpolation = cv2.INTER_NEAREST) 


    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    background = cv2.blur(background, (7,7))  

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    cv2.imwrite(d_dir+imidx+'.png',outImage)


def main():

    #find_human(args.src_img)

    # --------- 1. get image path and name ---------
    model_name='unet'

    image = glob.glob(args.src_img)
    prediction_dir = args.out
    model_dir = args.model

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = image,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='unet'):
        net = UNET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        #print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(image[i_test],pred,prediction_dir,args.src_img,args.bg_img)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
