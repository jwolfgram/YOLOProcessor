import cv2
from ultralytics import YOLO
import time
import types
import os
import sys
import torch

# Load the model
yolo = YOLO('./yolov8x.pt')

def runYOLO(toProcess):
    hasSiliconGPU = torch.backends.mps.is_available()
    hasNvidiaGPU = torch.cuda.device_count() or torch.cuda.is_available()
    # Nvidia > MPS > CPU
    device = 'cpu'
    if hasSiliconGPU:
        device = 'mps'
    if hasNvidiaGPU: 
        device = 'gpu'
    return yolo(toProcess, verbose=False, device=device)

def handleCLIArgs():
    argsDict = {}
    def printError(msg): 
        print('\x1b[31;20m' + msg + '\x1b[0m')
    def checkArg(name):
        if name not in argsDict:
            printError('Missing ' + name + ' arg.')
            return False
        return argsDict[name]
    for arg in sys.argv:
        if arg.startswith("--"):
            try:
                key, val = arg.split('=', 1)
                argsDict[key] = val
            except:
                argsDict[key] = None
    if checkArg("--processType") is False:
       raise
    processTypeVal = argsDict["--processType"]
    match processTypeVal:
        case "file":
            res = runSingleImg(checkArg("--filePath"), checkArg("--processedDir"))
            print(res)
        case "stream":
            readFromSrc(checkArg("--videoSrc"))
        case "directory":
            srcDir = checkArg("--srcDir")
            processedDir = checkArg("--processedDir")
            processImagesFromDir(srcDir, processedDir)
            print('DONE')
            # need to do the directory stuff... should be quick then print(done)

# inputDir = "~/Documents/img_stor/original/"
# outputDir = "~/Documents/img_stor/processed/"
def processImagesFromDir(inputDir, outputDir):
    # for dir process images, need to get intake and output folder
    # simplicity just 1:1 naming in new folder.
    ext = ('.png', '.jpg', '.jpeg')
    for file in os.listdir(inputDir):
        # if file exists in output skip
        if file.endswith(ext) and not os.path.isfile(outputDir + file):
            srcFile = inputDir + file
            # res = runSingleImg(srcFile)
            img = cv2.imread(srcFile)
            result = runYOLO(img)
            drawResults(result, img)
            # cv2.imshow('frame', img)
            # cv2.waitKey(0)
            cv2.imwrite(outputDir + file, img)
            

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def processResults(results):
    dectResults = {}

    for result in results:
        # get the classes names
        classes_names = result.names

        for dectCls in result.boxes.cls:
            dectInt = int(dectCls.item())
            if classes_names[dectInt] in dectResults:
                dectResults[classes_names[dectInt]] = dectResults[classes_names[dectInt]] + 1
            else:
                dectResults[classes_names[dectInt]] = 1
    return dectResults

def drawResults(results, frame):
    for result in results:
        classes_names = result.names
        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
def runSingleImg(path, outputPath):
    img = cv2.imread(path)
    result = runYOLO(img)
    if outputPath:
        print(path.rsplit('/'))
        cv2.imwrite(outputPath + path.rsplit('/'), img)
    processed = processResults(result)
    return processed

def readFromSrc(srcURL):
    # Load the video capture
    videoCap = cv2.VideoCapture(srcURL or 0)

    while True:
        ret, frame = videoCap.read()
        if not ret:
            continue
        results = yolo.track(frame, stream=True)

        # processResults(results)
        drawResults(results, frame)
                    
        cv2.imshow('frame', frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    videoCap.release()


# processImagesFromDir()
handleCLIArgs()
# release the video capture and destroy all windows
# cv2.destroyAllWindows()
