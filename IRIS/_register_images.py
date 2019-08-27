#!/usr/bin/env python3
"""
This model is used to register the images which contained in pixel-matrix.

The process of registration can be split into 3 parts:
1) To detect the key points. Its results will be used in finding good matched key point pairs.
2) To find the matched key point pairs. Then, to filter the bad matched pairs and retain good one.
3) To compute the transform matrix, which can be used to transform the images of channel between different cycles.

Here, we used 2 algorithms for key points detecting, the one is BRISK (S Leutenegger. et al., IEEE, 2011) and the other
is ORB (E Rublee. et al., Citeseer, 2011). The bad matched key points would be marked, and be filtered subsequently, for
ensuring the accuracy of calculating for transform matrix.

The transform matrix will be used to register the images by rigid registration. It means there are only translation
and rotation between images, no zooming and retortion.
"""


from sys import stderr
from cv2 import (getStructuringElement, morphologyEx,
                 BRISK, ORB, BFMatcher, findHomography, estimateRigidTransform,
                 MORPH_CROSS, MORPH_GRADIENT, MORPH_CLOSE, NORM_HAMMING, RANSAC)
from numpy import (array, float32)


def register_cycles(f_reference_cycle, f_transform_cycle, f_detection_method=None):
    """
    For computing the transform matrix between reference image and transform image.

    Input reference image, transform image and one of the algorithms of detector.
    Returning transform matrix.

    :param f_reference_cycle: The image that will be used to register other images.
    :param f_transform_cycle: The image will be registered.
    :param f_detection_method: The detection algorithm of feature points.
    :return: A transformation matrix from transformed image to reference.
    """
    def _get_key_points_and_descriptors(f_gray_image, f_method=None):
        """
        For detecting the key points and their descriptions by BRISK or ORB.

        Here, we employed morphology transforming to pre-process image for exposing the key points, under a kernel of
        5x5. A BRISK or ORB detector used to scan the image for locating the key point, and computed their descriptions
        as well.

        Input a gray scale image and one of the algorithms of detector.
        Returning the key points and their descriptions.

        :param f_gray_image: The 8-bit image.
        :param f_method: The detection algorithm of feature points.
        :return: A tuple including a group of feature points and their descriptions.
        """
        ksize = (15, 15)
        kernel = getStructuringElement(MORPH_CROSS, ksize)

        f_gray_image = morphologyEx(f_gray_image, MORPH_GRADIENT, kernel, iterations=2)
        f_gray_image = morphologyEx(f_gray_image, MORPH_CLOSE,    kernel, iterations=2)

        det = ''
        ext = ''

        f_method = 'BRISK' if f_method is None else 'ORB'

        if f_method == 'BRISK':
            det = BRISK.create()
            ext = BRISK.create()

        elif f_method == 'ORB':
            det = ORB.create()
            ext = ORB.create()

        else:
            print('Only BRISK and ORB would be suggested.', file=stderr)

        f_key_points = det.detect(f_gray_image)

        _, f_descriptions = ext.compute(f_gray_image, f_key_points)

        return f_key_points, f_descriptions

    def _get_good_matched_pairs(f_description1, f_description2):
        """
        For finding the good matched pairs of key points.

        The matched pairs of key points would be filtered to generate a group of good matched pairs.
        These good matched pairs of key points would be used to compute the transform matrix.

        Input two groups of description.
        Returning the good matched pairs of key points.

        :param f_description1: The description of feature points group 1.
        :param f_description2: The description of feature points group 2.
        :return: The good matched pairs between those two groups of feature points.
        """
        matcher = BFMatcher.create(normType=NORM_HAMMING, crossCheck=True)

        matched_pairs = matcher.knnMatch(f_description1, f_description2, 1)

        f_good_matched_pairs = [best_match_pair[0] for best_match_pair in matched_pairs if len(best_match_pair) > 0]
        f_good_matched_pairs = sorted(f_good_matched_pairs, key=lambda x: x.distance)

        return f_good_matched_pairs

    f_transform_matrix = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float32)

    kp1, des1 = _get_key_points_and_descriptors(f_reference_cycle, f_detection_method)
    kp2, des2 = _get_key_points_and_descriptors(f_transform_cycle, f_detection_method)

    good_matches = _get_good_matched_pairs(des1, des2)

    if len(good_matches) >= 4:
        pts_a = float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
        pts_b = float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

        _, mask = findHomography(pts_a, pts_b, RANSAC)

        good_matches = [good_matches[_] for _ in range(0, mask.size) if mask[_][0] == 1]

        pts_a_filtered = float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
        pts_b_filtered = float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

        f_transform_matrix = estimateRigidTransform(pts_b_filtered, pts_a_filtered, fullAffine=False)

    else:
        print('REGISTERING ERROR.', file=stderr)

    return f_transform_matrix


if __name__ == '__main__':
    pass