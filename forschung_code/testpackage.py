import torch
import numpy as np
from skimage import io, data
import yaml
import cv2
import odl
import astra

def test_torch():
    print("Testing PyTorch...")
    x = torch.rand(5, 3)
    print("PyTorch tensor:\n", x)

def test_numpy():
    print("Testing NumPy...")
    x = np.array([[1, 2], [3, 4]])
    print("NumPy array:\n", x)

def test_skimage():
    print("Testing scikit-image...")
    camera = data.camera()
    print("Sample image shape:", camera.shape)

def test_yaml():
    print("Testing PyYAML...")
    sample_yaml = {'a': 1, 'b': 2, 'c': 3}
    yaml_str = yaml.dump(sample_yaml)
    print("Sample YAML:\n", yaml_str)

def test_opencv():
    print("Testing OpenCV...")
    img = np.zeros((100,100), dtype=np.uint8)
    cv2.line(img, (0, 0), (99, 99), (255), 1)
    print("OpenCV image with a line:\n", img)

def test_odl():
    print("Testing ODL...")
    space = odl.uniform_discr([0, 0], [1, 1], [100, 100])
    print("ODL space:\n", space)

def test_astra():
    print("Testing ASTRA toolbox...")
    vol_geom = astra.create_vol_geom(256, 256)
    print("ASTRA volume geometry:\n", vol_geom)

if __name__ == "__main__":
    test_torch()
    test_numpy()
    test_skimage()
    test_yaml()
    test_opencv()
    test_odl()
    test_astra()