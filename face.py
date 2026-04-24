'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

     # (3, H, W) -> (H, W, 3)
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    imgNP = img.detach().cpu().numpy()


    if imgNP.ndim != 3:
        return []
    

    if str(imgNP.dtype) != "uint8":
        imgNP = (imgNP * 255).clip(0, 255).astype("uint8")
    faceLocations = face_recognition.face_locations(imgNP, model = "hog")

    # fig, ax = matplotlib.pyplot.subplots(1)
    # ax.imshow(img)
    for (top, right, bottom, left) in faceLocations:
        left = float(left)
        top = float(top)
        width = float(right - left)
        height = float(bottom - top)
        detection_results.append([left, top, width, height])
        
        #draw bounding boxes
    #     rect = matplotlib.patches.Rectangle(
    #         (left, top), width, height,
    #         linewidth = 3,
    #         edgecolor = 'g',
    #         facecolor = 'none')
    
    #     ax.add_patch(rect)

    # matplotlib.pyplot.show()

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####

    imgNames = []
    encodings = []
    for imgName, img in imgs.items():
        if not isinstance(img, torch.Tensor):
            continue
        # if img.ndim != 3 or img.shape[-1] != 3:
        #     continue
        if img.ndim != 3:
            continue
        if img.shape[-1] != 3 and img.shape[0] != 3:
            continue

        if img.shape[0] == 3: # convert torch img (C, H, W) to image process required (H, W, C)
            img = img.permute(1, 2, 0)

        imgNP = img.detach().cpu().numpy()

        # if imgNP.dtype != "uint8":
        #     imgNP = (imgNP * 255).clip(0, 255).astype("uint8")
        if imgNP.max() <= 1.0:
            imgNP = (imgNP * 255).clip(0, 255).astype("uint8")
        else:
            imgNP = imgNP.clip(0, 255).astype("uint8")

       

        # print(type(img))
        # print(img.shape)
        # print(img.dtype)
        boxes = face_recognition.face_locations(imgNP, model = "hog")
        # print(type(boxes))
        # print(type(boxes[0]))
        # print(boxes)
        h, w = img.shape[0], img.shape[1]
        if len(boxes) == 0: # no face detected, use full img size
            
            boxes = [(0, w, h, 0)]

        enc = face_recognition.face_encodings(imgNP, boxes)
        if len(enc) == 0: # skip fail encoding image
            continue
        encTensor = torch.tensor(enc[0], dtype=torch.float32)
        imgNames.append(imgName)
        encodings.append(encTensor)

    E = torch.stack(encodings, dim=0) # (N, 128) feature matrix
    N = len(encodings)
    iteration = 100
    cluster_results = kmeans(N, E, K, imgNames, iteration)
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def calculateDistance(x: torch.Tensor, c: torch.Tensor):
    dt = x.unsqueeze(1) - c.unsqueeze(0)
    dt2 = (dt ** 2).sum(dim=2)
    return dt2

def kmeans(N, E, K, imgnames, iteration):
    torch.manual_seed(42)
    indices = torch.randperm(N)[:K] # randomly select K unrepeat point as initial centroids
    centroids = E[indices].clone() 

    labels = torch.zeros(N, dtype=torch.long)
    for i in range(iteration):
        dist2 = calculateDistance(E, centroids)
        labelsTemp = dist2.argmin(dim=1)
        if torch.equal(labels, labelsTemp): # converge
            break
        labels = labelsTemp

        for k in range(K):
            idx = torch.where(labels == k)[0]
            if len(idx) > 0:
                centroids[k] = E[idx].mean(dim=0) # update centroids within each cluster
            else:
                idx = torch.randint(0, N, (1,)).item() # randomly select a point as centroid
                centroids[k] = E[idx].clone()

    
    results = [[] for _ in range(K)]
    for i, name in enumerate(imgnames):
        idx = labels[i].item()
        results[idx].append(name)
    return results
