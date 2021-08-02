from potrace.potrace import potrace
from skimage.io import imread

if __name__ == "__main__":
    potrace(imread("imgs/yao.jpg"), output="result/yao.svg")
