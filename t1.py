#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    imgs = []
    imgs.append(img1)
    imgs.append(img2)

    img_cpy = []
    img_cpy.append(img1)
    img_cpy.append(img2)


    def full_stitch(imgs):

        def check_overlap(img1,img2):
                sift = cv2.xfeatures2d.SIFT_create()
        
                key1, des1 = sift.detectAndCompute(img1,None)
                key2, des2 = sift.detectAndCompute(img2,None)

                # match = cv2.BFMatcher()
                # matches = match.match(des1,des2)
                # # matches = sorted(matches,key = lambda x:x.distance)
                # good = list(filter(lambda x : x.distance < 100, matches))


                class SsdDist:
                    pass
                ssddist = []
                for i in range(len(des1)):
                    for j in range(len(des2)):
                        obj = SsdDist()
                        # obj.dist = np.sqrt(np.sum(np.square(np.subtract(des1[i],des2[j]))))
                        obj.dist = np.linalg.norm(des1[i]-des2[j])
                        if obj.dist < 200:
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
   
            trans_img = np.copy(dst)
            dst[topoffset:topoffset+img_cpy[1].shape[0],leftoffset:leftoffset+img_cpy[1].shape[1]] = img_cpy[1]

            difference = cv2.subtract(dst, trans_img)
            Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(Conv_hsv_Gray, 10, 255,cv2.THRESH_BINARY_INV)
  
            trans_img[mask != 255] = [0, 0, 0]

            c_t =0
            for i in range(trans_img.shape[0]):
                for j in range(trans_img.shape[1]):
                    if np.all(trans_img[i][j] < 50):
                        pass
                    else:
                        dst[i,j] = trans_img[i,j] 
                        c_t+=1

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
        
        MIN_MATCH_COUNT = 10
        img1, img2, topoffset, bottomoffset, leftoffset, img =   prep_img(imgs[0],imgs[1])
        key1, key2, good_match = check_overlap(img1,img2)

        if len(good_match) > MIN_MATCH_COUNT:
            stitch_img = stitch_two(key1, key2, good_match, img1, imgs[0])
        else:
            print("Not enough matches found")
            return imgs[0]
        return stitch_img
    
    print("Started")
    stitched_img = full_stitch(imgs)
    cv2.imwrite(savepath, stitched_img)
    print("End")
        
    return

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

