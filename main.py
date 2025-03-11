import cv2
import numpy as np
import matplotlib.pyplot as plt



def detectAndDescribe(image, method=None):

    assert method is not None, "You need to define a feature detection method."

    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()


    kps, features = descriptor.detectAndCompute(image, None)
    return (kps, features)



def createMatcher(method, crossCheck):
    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)

    return bf

def getHomography(method, kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4):

    matcher = createMatcher(method, crossCheck=False)
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    good = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            good.append(m)
    if len(good) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (good, H, status)
    else:
        return None



def crop(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    crop = image[y:y + h, x:x + w]
    return crop

def stitchImages(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    method = 'sift'


    kpsA, featuresA = detectAndDescribe(gray1, method=method)
    kpsB, featuresB = detectAndDescribe(gray2, method=method)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(gray1, kpsA, None, color=(0, 0, 255)))
    ax1.set_xlabel("(a) Key Points", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(gray2, kpsB, None, color=(0, 0, 255)))
    ax2.set_xlabel("(b) Key Points", fontsize=14)
    plt.show()


    M = getHomography(method, kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4)
    if M is None:
        print("Error! - Homography could not be computed")
        return None
    (matches, H, status) = M


    img_matches = cv2.drawMatches(img1, kpsA, img2, kpsB, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title("Matches between Images")
    plt.show()


    width = img1.shape[1] + img2.shape[1]
    height = max(img1.shape[0], img2.shape[0])
    result = cv2.warpPerspective(img1, H, (width, height))


    print("Transformed img1 size:", result.shape)


    overlap_start = 0
    overlap_end = img2.shape[1]


    sigma = 1
    x = np.linspace(-3 * sigma, 3 * sigma, overlap_end - overlap_start)
    alpha = np.exp(-x ** 2 / (2 * sigma ** 2))


    for row in range(img2.shape[0]):
        for col in range(overlap_start, overlap_end):
            if np.any(result[row, col] != 0):
                beta = 1 - alpha[col - overlap_start]
                result[row, col] = alpha[col - overlap_start] * img2[row, col] + beta * result[row, col]

    result[0:img2.shape[0], 0:img2.shape[1]] = np.maximum(img2, result[0:img2.shape[0], 0:img2.shape[1]])


    result = crop(result)

    return result




if __name__ == "__main__":

    img1 = cv2.imread('data/1.png')
    img2 = cv2.imread('data/2.png')

    stitched_image = stitchImages(img2, img1)

    if stitched_image is not None:
        stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
        plt.imshow(stitched_image_rgb)
        plt.axis('on')
        plt.title('Stitched Image')
        plt.show()

    else:
        print("Stitching failed.")
        stitched_image_path = None
