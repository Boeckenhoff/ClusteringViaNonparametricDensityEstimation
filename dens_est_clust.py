from scipy.spatial import Delaunay
from sklearn.neighbors import KernelDensity
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from treelib import Node, Tree
from sklearn.metrics.pairwise import euclidean_distances


class pdfCluster:
    """The class which implements the described method"""

    def __init__(self, n_clusters = 2, kernel='gaussian', bandwidth=0.95, stepsize=1):
        """Initalizes the model. The number of cluster will be used"""
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.stepsize = stepsize

    def fit_predict(self, data_points):
        self.fit(data_points)
        return self.predict()

    def fit(self, data_points):
        self.clusters_at_different_scales = self.compute_M_of_p(data_points)
        tree = self.build_tree(self.clusters_at_different_scales)
        self.clustertree = self.add_remaining_points(tree, data_points)

    def predict(self):    
        return self.predict_n(self.n_clusters)

    def predict_n(self, n):    
        n = min(self.clustertree.depth(), n-1)
        allNID = [node.identifier for node in self.clustertree.all_nodes()]
        levelNID = [nid for nid in allNID if self.clustertree.level(nid)==n]
        results = [self.clustertree.get_node(nid).data[1] for nid in levelNID]
        return self.convert_labels(results)

    def compute_M_of_p(self, points):
        n = len(points)

        simplices = Delaunay(points).simplices
        edges = [edge for sublist in [self.createEdges(x) for x in simplices] for edge in sublist]

        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(points)
        density = np.exp(kde.score_samples(points))
        density = MinMaxScaler().fit_transform(density.reshape(-1, 1)) # My personal addition
        density = [item for sublist in density for item in sublist] # my personal addition
        #This way we avoid duplicates and have the minimal number of steps to get guaranted all m
        sorted_densities = sorted(list(set(density)))

        MofP = [(0, 0, 1, [set(range(n))])] 
        
        rest_edges = edges 
        for i, c in enumerate(sorted_densities):
            if i % self.stepsize == 0:
                #remove points of low density  from DT
                SofC = density >= c
                low_density_points,  = np.where( np.logical_not(SofC))
                low_density_points = set(low_density_points.tolist())

                #rest_edges = [edge for edge in edges if not low_density_points.intersection(edge)] #SLOW?!
                rest_edges = [edge for edge in rest_edges if not low_density_points.intersection(edge)]

                #determine connected sets of retained Points
                cc = self.connected_Components(rest_edges)
                #compute m(p)
                p = (n- SofC.sum()) /n #the bigger c the smaller is p
                m = len(cc)
                MofP.append( (c, p, m, cc) ) 

                #print( str(SofC.sum()) +"/"+str(n) + " Points in "+str(m) +" Clusters with density >" +str(c)+":")
        MofP.append( (1, 1, 0, set()) )
        return MofP

    def createEdges(self, a):
        return (a[0],a[1]), (a[0],a[2]), (a[1],a[2])
        
    def connected_Components(self, edges):
        g = nx.Graph(edges)
        cc = nx.connected_components(g)
        return list(cc)
    
    
    def build_tree(self, mop):
        ctree = Tree()
        counter = 0

        first_iteration = True
        lastm = 1
        id = 0
        #from left to right
        for c, p, m, clusters in mop:
            clusters = [(p, x) for x in clusters]   
            #The root contains all the datapoints
            if first_iteration:
                ctree.create_node(str(id), id, data=clusters[0])
                first_iteration = False 
            #Increment
            if lastm < m:
                allNID = [node.identifier for node in ctree.all_nodes()]
                levelNID = [nid for nid in allNID if ctree.level(nid)==counter]
                leaves = levelNID
                counter = counter +1
                for cluster in clusters:
                    for leafnid in leaves:
                        leaf = ctree.get_node(leafnid)
                        if cluster[1].issubset(leaf.data[1]):
                            id = id + 1
                            ctree.create_node(cluster[0], id, parent=leaf.identifier, data=cluster)
            lastm = m
        return ctree

    def add_remaining_points(self, tree, points):
        for level in range(tree.depth()+1):
            levelNID = [node.identifier for node in tree.all_nodes() if tree.level(node.identifier)==level]
            for id, point in enumerate(points):
                found = False
                for NID in levelNID:
                    found = id in tree.get_node(NID).data[1]
                    if found:
                        break
                if not found:
                    #find the best node
                    min_dis =  float("inf")
                    curNID = 0
                    for NID in levelNID:
                        for x in list(tree.get_node(NID).data[1]):
                            x = points[x]
                            if euclidean_distances(point.reshape(1, -1),x.reshape(1, -1) ) < min_dis:
                                min_dis = euclidean_distances(point.reshape(1, -1),x.reshape(1, -1) ) 
                                curNID = NID
                    #Add point to node
                    tree.get_node(curNID).data[1].add(id)
        return tree

    def convert_labels(self, results):
        sum = 0
        for x in results:
            sum = sum + len(x)
        labels = np.zeros(sum, dtype=int)
        for label, cluster in enumerate(results):
            for id in cluster:
                labels[id]= label
        return labels
        