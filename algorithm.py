"""
Algorithm for k-Means clustering
"""
import math
import random
import numpy
import dataset
import cluster

def valid_seeds(value, size):
    """
    Returns True if value is a valid list of seeds for clustering.

    A list of seeds is a k-element list OR tuple of integers between 0 and size-1.
    In addition, no seed element can appear twice.

    Parameter valud: a value to check
    Precondition: value can be anything

    Parameter size: The database size
    Precondition: size is an int > 0
    """
    assert type(size) == int and size > 0

    if type(value) != list:
        return False
    for seed in value:
        if not isinstance(seed,int):
            return False
        if seed < 0 or seed > (size - 1):
            return False
        if value.count(seed) > 1:
            return False

    return True

class Algorithm(object):
    """
    A class to manage and run the k-means algorithm.

    The method step() performs one step of the calculation.  The method run() will
    continue the calculation until it converges (or reaches a maximum number of steps).
    """

    def getClusters(self):
        """
        Returns the list of clusters in this object.

        This method returns the cluster list directly (it does not copy).  Any changes
        made to this list will modify the set of clusters.
        """
        return self._cluster

    def __init__(self, dset, k, seeds=None):
        """
        Initializes the algorithm for the dataset ds, using k clusters.

        If the optional argument seeds is supplied, those seeds will be a list OR
        tuple of indices into the dataset. They specify which points should be the
        initial cluster centroids. Otherwise, the clusters are initialized by randomly
        selecting k different points from the database to be the cluster centroids.

        Parameter dset: the dataset
        Precondition: dset is an instance of Dataset

        Parameter k: the number of clusters
        Precondition: k is an int, 0 < k <= dset.getSize()

        Paramter seeds: the initial cluster indices (OPTIONAL)
        Precondition: seeds is None, or a list/tuple of valid seeds.
        """
        assert isinstance(dset,dataset.Dataset)
        assert isinstance(k,int) and k > 0 and k <= dset.getSize()
        assert seeds == None or valid_seeds(seeds,dset.getSize())

        self._dataset = dset
        if seeds != None:
            #assert len(seeds) == k
            self._cluster = [cluster.Cluster(dset,dset.getPoint(seed)) for seed in seeds]
        else:
            random_seeds = [random.sample(range(dset.getSize()),k)][0]
            self._cluster = [cluster.Cluster(dset,dset.getPoint(seed)) for seed in random_seeds]


    def _nearest(self, point):
        """
        Returns the cluster nearest to point

        This method uses the distance method of each Cluster to compute the distance
        between point and the cluster centroid. It returns the Cluster that is closest.

        Ties are broken in favor of clusters occurring earlier in the list returned
        by getClusters().

        Parameter point: The point to compare.
        Precondition: point is a tuple of numbers (int or float). Its length is the
        same as the dataset dimension.
        """
        assert dataset.is_point(point) and len(point) == \
        self._dataset.getDimension()

        distances = {}
        for cluster in self._cluster:
            distances[cluster] = cluster.distance(point)

        minimum = min(distances.values())
        return [key for key in distances if distances[key]==minimum][0]


    def _partition(self):
        """
        Repartitions the dataset so each point is in exactly one Cluster.
        """
        # First, clear each cluster of its points.  Then, for each point in the
        # dataset, find the nearest cluster and add the point to that cluster.
        for cluster in self._cluster:
            cluster.clear()
        for index in range(len(self._dataset.getContents())):
            self._nearest(self._dataset.getPoint(index)).addIndex(index)


    def _update(self):
        """
        Returns True if all centroids are unchanged after an update; False otherwise.

        This method first updates the centroids of all clusters'.  When it is done, it
        checks whether any of them have changed. It returns False if just one has
        changed. Otherwise, it returns True.
        """
        temp = True
        for cluster in self._cluster:
            update = cluster.update()
            if update == False:
                temp = False

        return temp


    def step(self):
        """
        Returns True if the algorithm converges after one step; False otherwise.

        This method performs one cycle of the k-means algorithm. It then checks if
        the algorithm has converged and returns the appropriate result (True if
        converged, false otherwise).
        """
        self._partition()
        no_update = self._update()

        return no_update


    def run(self, maxstep):
        """
        Continues clustering until either it converges or performs maxstep steps.

        After the maxstep call to step, if this calculation did not converge, this
        method will stop.

        Parameter maxstep: The maximum number of steps to perform
        Precondition: maxstep is an int >= 0
        """
        # Call k_means_step repeatedly, up to maxstep times, until the algorithm
        # converges.  Stop once you reach maxstep iterations even if the algorithm
        # has not converged.
        # You do not need a while loop for this.  Just write a for-loop, and exit
        # the for-loop (with a return) if you finish early.
        assert isinstance(maxstep, int) and maxstep >= 0

        for i in range(maxstep):
            converge = self.step()
            if converge:
                return
