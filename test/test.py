from src.potrace import Potrace
from skimage.io import imread

if __name__ == "__main__":
    Potrace(imread("imgs/smiley.png")).save("result/smiley.svg")
