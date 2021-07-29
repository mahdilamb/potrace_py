from src.potrace import Potrace
from skimage.io import imread

if __name__ == "__main__":
    Potrace(imread("imgs/yao.jpg")).to_svg("result/yao.svg")
