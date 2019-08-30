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
from cv2 import (convertScaleAbs, GaussianBlur, getStructuringElement, morphologyEx,
                 BRISK, ORB, BFMatcher, estimateAffinePartial2D,
                 MORPH_CROSS, MORPH_GRADIENT, NORM_HAMMING, RANSAC)
from numpy import (array, mean, float32)


def register_cycles(reference_cycle, transform_cycle, detection_method=None):
    """
    For computing the transform matrix between reference image and transform image.

    Input reference image, transform image and one of the algorithms of detector.
    Returning transform matrix.

    :param reference_cycle: The image that will be used to register other images.
    :param transform_cycle: The image will be registered.
    :param detection_method: The detection algorithm of feature points.
    :return: A transformation matrix from transformed image to reference.
    """
    def _get_key_points_and_descriptors(gray_image, method=None):
        """
        For detecting the key points and their descriptions by BRISK or ORB.

        Here, we employed morphology transforming to pre-process image for exposing the key points, under a kernel of
        5x5. A BRISK or ORB detector used to scan the image for locating the key point, and computed their descriptions
        as well.

        Input a gray scale image and one of the algorithms of detector.
        Returning the key points and their descriptions.

        :param gray_image: The 8-bit image.
        :param method: The detection algorithm of feature points.
        :return: A tuple including a group of feature points and their descriptions.
        """
        gray_image = GaussianBlur(gray_image, (5, 5), 0)

        ksize = (15, 15)
        kernel = getStructuringElement(MORPH_CROSS, ksize)

        gray_image = morphologyEx(gray_image, MORPH_GRADIENT, kernel, iterations=3)

        det = ''
        ext = ''

        method = 'BRISK' if method is None else method

        if method == 'BRISK':
            det = BRISK.create()
            ext = BRISK.create()

        elif method == 'ORB':
            det = ORB.create()
            ext = ORB.create()

        else:
            print('Only BRISK and ORB would be suggested.', file=stderr)

        f_key_points = det.detect(gray_image)

        _, f_descriptions = ext.compute(gray_image, f_key_points)

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

    # Lightness Rectification #
    transform_cycle = convertScaleAbs(transform_cycle * (mean(reference_cycle) / mean(transform_cycle)))

    kp1, des1 = _get_key_points_and_descriptors(reference_cycle, detection_method)
    kp2, des2 = _get_key_points_and_descriptors(transform_cycle, detection_method)

    good_matches = _get_good_matched_pairs(des1, des2)

    n = 1
    while n:
        pts_a = float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
        pts_b = float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

        _, mask = estimateAffinePartial2D(pts_b, pts_a, RANSAC)

        good_matches = [good_matches[_] for _ in range(0, mask.size) if mask[_][0] == 1]

        n = sum([mask[_][0] for _ in range(0, mask.size)]) - mask.size

    if len(good_matches) >= 4:
        pts_a_filtered = float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
        pts_b_filtered = float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

        f_transform_matrix, _ = estimateAffinePartial2D(pts_b_filtered, pts_a_filtered, RANSAC)

        if f_transform_matrix is None:
            print('MATRIX GENERATION FAILED.', file=stderr)

    else:
        print('NO ENOUGH MATCHED FEATURES, REGISTRATION FAILED.', file=stderr)

    return f_transform_matrix


if __name__ == '__main__':
    pass
