# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import simplejson as json

def stitch(imgmark, N=5, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    # if imgmark == 't3':
        # for i in imgs:
        #     # i = cv2.resize(i, (min(i.shape[1],250),min(i.shape[0],200)),interpolation = cv2.INTER_AREA)
        #     print(i.shape)

    print("Total images: ",len(imgs))
        
    def prep_img(img_1,img_2):
            img_ = img_1
            img = img_2
            topoffset = img_.shape[0]
            bottomoffset = img_.shape[0]
            leftoffset = img_.shape[1]
            img = cv2.copyMakeBorder(img, topoffset, bottomoffset, leftoffset, 0, cv2.BORDER_CONSTANT, None, value = 0)

            img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            return img1, img2, topoffset, bottomoffset, leftoffset, img

    def check_overlap(img1,img2):
            sift = cv2.xfeatures2d.SIFT_create()
    
            key1, des1 = sift.detectAndCompute(img1,None)
            key2, des2 = sift.detectAndCompute(img2,None)

            # match = cv2.BFMatcher()
            # matches = match.match(des1,des2)
            # # matches = sorted(matches,key = lambda x:x.distance)
            # ssddist = list(filter(lambda x : x.distance < 50, matches))


            class SsdDist:
                pass
            ssddist = []
            for i in range(len(des1)):
                if len(ssddist)>25:
                    break
                for j in range(len(des2)):
                    obj = SsdDist()
                    # obj.dist = np.sqrt(np.sum(np.square(np.subtract(des1[i],des2[j]))))
                    obj.dist = np.linalg.norm(des1[i]-des2[j])
                    if obj.dist < 50:
                        obj.xidx = i
                        obj.yidx = j
                        ssddist.append(obj)

            return key1, key2, ssddist

    def stitch_two(key1, key2, good, img1, img2):

        h,w = img1.shape

        src_pts = np.float32([ key1[m.xidx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ key2[m.yidx].pt for m in good ]).reshape(-1,1,2)

        # src_pts = np.float32([ key1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        # dst_pts = np.float32([ key2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        dst = cv2.warpPerspective(img2,M,(img.shape[1] + img2.shape[1], img.shape[0]))

        for i in range(img.shape[0]-topoffset):
            for j in range(img.shape[1]):
                if np.all(img[topoffset+i][j] < 50):
                    pass
                else:
                    dst[topoffset+i,j] = img[topoffset+i,j]
            
        def crop(full_img):
            while(np.sum(full_img[0]) == 0 ):
                full_img=full_img[1:]
            while(np.sum(full_img[-1]) == 0 ):
                full_img=full_img[:-2]
            while(np.sum(full_img[:,0]) == 0 ):
                full_img=full_img[:,1:]
            while(np.sum(full_img[:,-1]) == 0 ):
                full_img=full_img[:,:-2]
            return full_img

        return crop(dst)
    
    print("Started..")
    
    MIN_MATCH_COUNT = 10
    cnt = 1 
    overlap_arr = np.eye(len(imgs), dtype=int)
    for p in range(len(imgs)):
        q=p+1
        while q < len(imgs):
            img1, img2, topoffset, bottomoffset, leftoffset, img =   prep_img(imgs[p],imgs[q])
            key1, key2, good_match = check_overlap(img1,img2)
            if len(good_match) > MIN_MATCH_COUNT:
                overlap_arr[p][q] = 1
                overlap_arr[q][p] = 1
            q+=1
    print(overlap_arr)
    arr_sum = np.sum(overlap_arr, axis=1)
    for i in range(len(arr_sum)):
        if arr_sum[i] == 1:
            imgs.pop(i)

    while(len(imgs)>1):
        print("stiches remaining: ",len(imgs)-1)
        img1, img2, topoffset, bottomoffset, leftoffset, img =   prep_img(imgs[0],imgs[cnt])
        key1, key2, good_match = check_overlap(img1,img2)

        if len(good_match) > MIN_MATCH_COUNT:
            stitched_img = stitch_two(key1, key2, good_match, img1, imgs[0])
            print("one stitch done")
            imgs.pop(0)
            imgs.insert(0,stitched_img)
            imgs.pop(cnt)
            
            cnt = 1
        else: 
            cnt = cnt+1
            if cnt == len(imgs):
                imgs.pop(0)
                cnt = 1
                
    cv2.imwrite(savepath, imgs[0])
    # cv2.waitKey(2000)
    print("End")
    
    return overlap_arr
if __name__ == "__main__":
    # # #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # # #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
