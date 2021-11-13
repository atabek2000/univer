import numpy as np # массивтермен жұмыс жасау үшін
import math as m # математикалық функциялар үшін
import matplotlib.pyplot as plt # визуализация үшін

# кластерлердің классы
class Cluster:
    def __init__(self, pixels, center): # класстың конструкторы, класты құрғанда пикселдер мен центін береміз
        self.pixels = pixels
        self.center = center
    # класты екіге бөлуге арналған функция, кірісті 1 клатер қабылдайды, шығыста 2 кластер қайтарады
    def split(cluster):
        gamma = 0.5
        c1 = np.array([0, 0])
        c2 = np.array([0, 0])
        std = cluster.std()
        c1[0] = cluster.center[0] + std[0]*gamma
        c1[1] = cluster.center[1] + std[1]*gamma
        
        c2[0] = cluster.center[0] - std[0]*gamma
        c2[1] = cluster.center[1] - std[1]*gamma
        pixels1 = list()
        pixels2 = list()
        for i in cluster.pixels:
            if Cluster.distance(i,c1) > Cluster.distance(i, c2):
                pixels1.append(i)
            else: 
                pixels2.append(i)
        cluster1 = Cluster(np.array(pixels1), c1)
        cluster2 = Cluster(np.array(pixels2), c2)
        return cluster1, cluster2
    # кластерлерді біріктіруге арналған функция, параметрлері біріктірілетін 2 клас, қайтаратын мәні біріккен 1 класс
    def merge(cluster1, cluster2):
        center1 = cluster1.center
        center2 = cluster2.center
        pixels = np.concatenate((cluster1.pixels, cluster2.pixels))
        x = (center1[0] + center2[0])/2
        y = (center1[1] + center2[1])/2
        new_center = np.array([x,y])
        new_cluster = Cluster(pixels, new_center)
        return new_cluster
    # кластерді өшіруге арналған функция, бұл кластердің пикселдерін басқа кластерлерге беріп, бұл кластерді жадыдан өшіреді
    def delete(clusters, number):
        for i in clusters[number].pixels:
            mn = float("inf")
            mn_j = -1
            for j in range(len(clusters)):
                if j != number:
                    if Cluster.distance(i, clusters[j].center) < mn:
                        mn = Cluster.distance(i, clusters[j].center)
                        mn_j = j
            temp = list(clusters[mn_j].pixels)
            temp.append(i)
            clusters[mn_j].pixels = np.array(temp)
        del clusters[number]
    # стандартты ауытқуды есептеуге арналған функция
    def std(self):
        x = 0
        y = 0
        ct = self.center
        for i in self.pixels:
            x += (i[0] - ct[0])**2
            y += (i[1] - ct[1])**2
        x = m.sqrt(x/len(pixels))
        y = m.sqrt(y/len(pixels))
        return np.array([x, y])
    # центр мен пикселдер ара қашықтығын есептеуге арналған функция
    def distance(pixel, center):
        return m.sqrt((pixel[0] - center[0])**2 + (pixel[1] - center[1])**2)
    # әлі кластерлерге бөлінбеген пикселдерді берілген центрлер бойынша кластерлерге бөлуге арналған функция
    def distribute(pixels, centers):
        pxs = list()
        for i in range(len(centers)):
            pxs.append(list())
        mn = float("inf")
        mn_j = 0
        for i in range(len(pixels)):
            mn = float("inf")
            for j in range(len(centers)):
                if Cluster.distance(pixels[i], centers[j]) < mn:
                    mn = Cluster.distance(pixels[i], centers[j])
                    mn_j = j
            pxs[mn_j].append(pixels[i])
            
        clusters = [0]*len(centers)
        for i in range(len(centers)):
            clusters[i] = Cluster(np.array(pxs[i]), centers[i])
        return clusters
    # кластерлерді экранға шыгаруға ареалған функция
    def print_clusters(clusters):
        for i in range(len(clusters)):
            print('Center ', i, ': ', clusters[i].center)
            print('Pixels ', i, ': ', clusters[i].pixels)

        
        
        
# k - необходимое число кластеров
# qn - параметр, с которым сравнивается количество выборочных образов, вошедших в кластер
# qs - параметр, характеризующий среднеквадратичное отклонение
# qc - параметр, характеризующий компактность
# l -  максимальное количество пар центров кластеров, которые можно объединить
# i - допустимое число циклов итерации


# ISODATA алгоритміне арналған функция
def isodata(pixels, centers, Nc=1, z1=[0,0],  k=2, qn=2, qs=0.2, qc=10, l=5, iter=10):
    clusters = Cluster.distribute(pixels, centers)
    for t in range(iter):
        stop = True
        # 7-шагтағы шарт орындалмай қалғанға дейін цикл қайталана береді
        while stop:
            j = 0
            for i in range(len(clusters)):
                if len(clusters[j].pixels) < qn:
                    Cluster.delete(clusters, j) # кластерде пикселдер саны qn нен аз болса онда кластерді өшіреміз
                else:
                    j += 1
            for i in clusters:
                sm_x = 0
                sm_y = 0
                for j in i.pixels:
                    sm_x += j[0]
                    sm_y += j[1]
                i.center[0] = sm_x/len(i.pixels)
                i.center[1] = sm_y/len(i.pixels)
            Dj = list()
            for i in clusters:
                sm =  0
                for j in i.pixels:
                    sm += m.sqrt((j[0] - i.center[0])**2 + (j[1] - i.center[1])**2)
                Dj.append(sm/len(i.pixels))
            D = 0
            sm = 0
            count_pxl = 0
            for i in range(len(clusters)):
                count_pxl += len(clusters[i].pixels)
                sm += len(clusters[i].pixels)*Dj[i]
            D = sm/count_pxl
            if t == (iter-1):
                qc = 0
                break
            else:
                split = False
                std_mx = -float("inf")
                mx_i = 0
                for i in range(len(clusters)):
                    std_x, std_y = clusters[i].std()
                    if std_x > std_mx:
                        std_mx = std_x
                        mx_i = i   
                    if std_y > std_mx:
                        std_mx = std_y
                        mx_i = i
                if std_mx > qs:
                    split = True
                    cltr1, cltr2 = Cluster.split(clusters[mx_i]) # стандартты ауытқу qs - тен үлкен болса класты бөлеміз
                    clusters.append(cltr1)
                    clusters.append(cltr2)
                    del clusters[mx_i]
                    Nc += 1
                if ~split: #егер кластер бөлінбесе 11-шагқа өтеміз
                    break
            break
        # 11 - шаг
        Dij = np.full((len(clusters), len(clusters)), 0.0)
        mn = float("inf")
        mn_i = float("inf")
        mn_j = float("inf")
        for i in range(len(Dij)):
            for j in range(len(Dij)):
                Dij[i][j] = m.sqrt((clusters[i].center[0] - clusters[j].center[0])**2 + (clusters[i].center[1] - clusters[j].center[1])**2)
                if Dij[i][j] < mn and i != j:
                    mn = Dij[i][j]
                    mn_i = i
                    mn_j = j
        if(mn < qc):
            clusters.append(Cluster.merge(clusters[mn_i], clusters[mn_j]))
            del clusters[mn_i]
            del clusters[mn_j-1]
        else:
            break
    return clusters
        

# бастапқы, ешқандай кластерге бөлінбеген пикселдер            
pixels = np.array([[2,3],[3,4],[4,5],[5,6],[6,3],[1,6],[8,12],[5,1],[7,9]])
# бастапқы центрлер 
centers = np.array([[2,5], [1,1], [7,9]])

# бастапқы пикселдер мен центрлерді негізгі функцияға береміз
clstrs = isodata(pixels = pixels, centers = centers)
# нәтижені консольге шыгарамыз
Cluster.print_clusters(clstrs)


# визуализация жасаймыз
colors = ['red','green','pink','black','orange']
plt.figure()
for i in range(len(pixels)):
    plt.scatter(pixels[i][0], pixels[i][1], c='blue')
for i in range(len(centers)):
    plt.scatter(centers[i][0], centers[i][1], marker="^", c=colors[i], linewidths=(12), alpha=(0.5))

plt.figure()
for i in range(len(clstrs)):
    plt.scatter(clstrs[i].center[0], clstrs[i].center[1], marker="^", c=colors[i], linewidths=(12), alpha=(0.5))
    for j in clstrs[i].pixels:
        plt.scatter(j[0], j[1], c=colors[i])
 
