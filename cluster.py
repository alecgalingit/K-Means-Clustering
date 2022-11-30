"""
Module representing a cluster for K-Means clustering.
"""
import math
import random
import numpy
import dataset


class Cluster(object):
    """
    A class representing a cluster, a subset of the points in a dataset.

    A cluster is represented as a list of integers that give the indices in the dataset
    of the points contained in the cluster.  For instance, a cluster consisting of the
    points with indices 0, 4, and 5 in the dataset's data array would be represented by
    the index list [0,4,5].

    A cluster instance also contains a centroid that is used as part of the k-means
    algorithm.  This centroid is an n-D point (where n is the dimension of the dataset),
    represented as a list of n numbers, not as an index into the dataset. (This is because
    the centroid is generally not a point in the dataset, but rather is usually in between
    the data points.)
    """
    def getIndices(self):
        """
        Returns the indices of points in this cluster

        This method returns the indices directly (not a copy). Any changes made to this
        list will modify the cluster.
        """
        return self._indices

    def getCentroid(self):
        """
        Returns the centroid of this cluster.

        This getter method is to protect access to the centroid, and prevent someone
        from changing it accidentally. Because the centroid is a tuple, it is not
        necessary to copy the centroid before returning it.
        """
        return self._centroid

    def __init__(self, dset, centroid):
        """
        Initializes a new empty cluster with the given centroid

        Parameter dset: the dataset
        Precondition: dset is an instance of Dataset

        Parameter centroid: the cluster centroid
        Precondition: centroid is a tuple of dset.getDimension() numbers
        """
        assert isinstance(dset,dataset.Dataset)
        assert type(centroid) == tuple and len(centroid) == dset.getDimension()

        self._dataset = dset
        self._centroid = centroid
        self._indices = []

    def addIndex(self, index):
        """
        Adds the given dataset index to this cluster.

        If the index is already in this cluster, this method leaves the
        cluster unchanged.

        Precondition: index is a valid index into this cluster's dataset.
        That is, index is an int >= 0, but less than the dataset size.
        """
        assert type(index) == int and index >= 0 and index < self._dataset.getSize()

        if index not in self._indices:
            self._indices.append(index)


    def clear(self):
        """
        Removes all points from this cluster, but leaves the centroid unchanged.
        """
        self._indices.clear()


    def getContents(self):
        """
        Returns a new list containing copies of the points in this cluster.

        The result is a list of points (tuples of int/float). It has to be computed
        from the list of indices.
        """
        temp = []
        for i in self.getIndices():
            temp.append(self._dataset.getPoint(i))
        return temp


    # Part B
    def distance(self, point):
        """
        Returns the euclidean distance from point to this cluster's centroid.

        Parameter point: The point to be measured
        Precondition: point is a tuple of numbers (int or float), with the same dimension
        as the centroid.
        """
        assert dataset.is_point(point)
        assert len(point) == self._dataset.getDimension()

        #Creates a nested listed with values of the point and centroid at each
        #index i
        all = []
        for i in range(len(point)):
            all.append([point[i]])
        for i in range(len(self.getCentroid())):
            all[i].append(self.getCentroid()[i])
        sum = 0
        for i in range(len(all)):
            sum += ((all[i][0] - all[i][1])**2)

        return sum**0.5


    def getRadius(self):
        """
        Returns the maximum distance from any point in this cluster, to the centroid.

        This method loops over the contents of this cluster to find the maximum distance
        from the centroid.
        """
        distances = []
        for point in self.getContents():
            distances.append(self.distance(point))

        return max(distances)

    def update(self):
        """
        Returns True if the centroid remains the same after recomputation; False otherwise.

        This method recomputes the centroid of this cluster. The new centroid is the
        average of the of the contents (To average a point, average each coordinate
        separately).

        Whether the centroid "remained the same" after recomputation is determined by
        numpy.allclose. The return value should be interpreted as an indication of
        whether the starting centroid was a "stable" position or not.

        If there are no points in the cluster, the centroid. does not change.
        """
        #0 case
        if len(self.getContents()) == 0:
            return True

        #Creates an empty nested list containing len(self.getContents()[0]) empty
        #lists
        recomputation = []
        for i in range(len(self.getContents()[0])):
            recomputation.append([])

        #Adds lists of index of points from self.getContents() to recomputation
        for point in self.getContents():
            for i in range(len(point)):
                recomputation[i].append(point[i])

        #Sets recomputation[i] to mean of recomputation[i], computing the final
        #recomputation
        for i in range(len(recomputation)):
            recomputation[i] = (sum(recomputation[i])/len(recomputation[i]))

        old_centroid = self._centroid
        self._centroid = recomputation

        return numpy.allclose(old_centroid,recomputation)


    def __str__(self):
        """
        Returns a String representation of the centroid of this cluster.
        """
        return str(self._centroid)+':'+str(self._indices)

    def __repr__(self):
        """
        Returns an unambiguous representation of this cluster.
        """
        return str(self.__class__) + str(self)
