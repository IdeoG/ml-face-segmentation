import logging
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
logging.basicConfig()


class HairSegmentationTestCase(unittest.TestCase):

    def test_inference(self):
        from face_segmentation.hair_segmentation import prepare_model, inference

        image_path = 'images/1.jpg'
        model_path = '../'

        prepare_model(model_path)

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = inference(image, mode='rgb')

        self.assertIsInstance(mask[0][0], np.bool_)
        self.assertEqual(mask.shape, image.shape[:-1])

        mask = np.array(mask, dtype=np.uint8) * 255
        sobel_edges = cv2.Canny(mask, 100, 200)
        image[sobel_edges != 0] = [255, 0, 0]

        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    unittest.main()
