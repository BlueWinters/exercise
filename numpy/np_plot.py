# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt



def get_10color_list():
    color = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
             [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
             [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
             [1.0, 1.0, 0.0], [1.0, 1.0, 0.5], # 2^3=8，[1,1,1]是白色，
             [0.5, 1.0, 1.0], [1.0, 0.5, 1.0]] # 所以最后三个是随机选择的
    return color

def get_8color_list():
    color = ['b','r','g','k','m','c','w','y']
    return color

def mix_10_2D_guassian(n_samples, mean=0.0, std=1.0):
    samples = np.zeros(shape=[n_samples, 2])
    for n in range(10):
        mean_list = np.zeros([n_samples, 2])
        std_list = np.ones([n_samples, 2])
        epsilon = np.random.normal(loc=mean, scale=std, size=[n_samples, 2])
        samples += mean_list + epsilon*std_list
    return samples

def plotPoints_10class():
    # np.random.seed(2017)
    point = mix_10_2D_guassian(100)
    color = get_10color_list()
    for n in range(9):
        x = point[10*n:10*(n+1),0]
        y = point[10*n:10*(n+1),1]
        plt.scatter(x, y, c=color[n], edgecolors='face')
    plt.show()



if __name__ == '__main__':
    plotPoints_10class()
