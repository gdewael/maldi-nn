"""`topf`: a package for topological peak filtering."""

# Copyright (c) 2018 Bastian Rieck
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__version__ = "0.1.0"

import numpy as np
import collections.abc


class UnionFind:
    """An implementation of a Union--Find class.

    Basic implementation of a Union--Find algorithm for a set of
    vertices with contiguous vertex indices. The class performs path
    compression by default and uses integers internally for storing
    a disjoint set.

    The class requires the vertices to form a contiguous sequence, so no
    gaps are allowed. The vertices also have to be zero-indexed.
    """

    def __init__(self, n_vertices):
        """Create new instance of the class.

        Creates a new Union--Find class with `n_vertices` vertices. The
        vertices are expected to be zero-indexed.

        Parameters
        ----------
        n_vertices : int
            Number of vertices
        """
        self.n_vertices = n_vertices
        self.parent = {x: x for x in range(n_vertices)}

    def find(self, u):
        """Find and return parent of `u`."""
        if self.parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self.parent[u] = self.find(self.parent[u])
            return self.parent[u]

    def merge(self, u, v):
        """Merge `u` into the component of `v`.

        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.

        Parameters
        ----------
        u : int
            Source vertex
        v : int
            Target vertex
        """
        if u != v:
            self.parent[self.find(u)] = self.find(v)


class PersistenceDiagram(collections.abc.Sequence):
    """Persistence diagram storage class.

    Simple class for storing the pairs of a persistence diagram. This is
    nothing but a light-weight wrapper for additional convenience.
    """

    def __init__(self, pairs):
        """Create new diagram from sequence of pairs."""
        self._pairs = pairs

    def __len__(self):
        """Return number of persistence pairs."""
        return len(self._pairs)

    def __getitem__(self, index):
        """Return persistence at index `index`."""
        return self._pairs[index]

    def __str__(self):
        """Return string representation of the persistence diagram."""
        return str(self._pairs)

    def total_persistence(self, p=1.0):
        """Calculate total persistence of the persistence diagram.

        Calculates the sum of all persistence values in the diagram,
        weighted by the specified power.

        Parameters
        ----------
        p : float
            Exponent for the weight calculation. Must be positive, else
            an exception will be raised.

        Returns
        -------
        Total persistence of the persistence diagram.
        """
        assert p > 0.0

        persistence_values = self._pairs[:, 0] - self._pairs[:, 1]
        persistence_values = np.power(persistence_values, p)

        return np.power(np.sum(persistence_values), 1.0 / p)


class PersistenceTransformer:
    """Persistence transformer class.

    Transforms a function that is represented as an input array of (x,y)
    pairs into (x, persistence(x)) pairs. Optionally, all tuples of the
    corresponding persistence diagram are calculated as well.
    """

    def __init__(
        self,
        calculate_persistence_diagram=False,
        n_peaks=None,
        enforce_n_peaks=True,
    ):
        """Create new instance of the persistence transformer class.

        Parameters
        ----------
        calculate_persistence_diagram : bool
            If set, calculates the persistence diagram along with
            transformed function. Use the `persistence_diagram`
            property of the class to access it.

        n_peaks : bool
            If set, keeps only the specified number of peaks. Peaks will
            be eliminated in top-down order starting from the one with
            the lowest persistence. Thus, if `n_peaks` is 1, only the
            highest peak will be kept.

        enforce_n_peaks : bool
            If set, ensures that `n_peaks` peaks will be returned, even in
            cases where two peaks have the same persistence. This behaviour
            is on by default because if users request a certain number of
            peaks, they should get them. This follows the principle of least
            surprise.
        """
        self._calculate_persistence_diagram = calculate_persistence_diagram
        self._n_peaks = n_peaks
        self._persistence_diagram = None
        self._enforce_n_peaks = enforce_n_peaks

    def fit_transform(self, a):
        """Transform a single function according to its peaks.

        Parameters
        ----------
        a : array_like of shape (n, 2)
            Function with `n` 2D samples to be transformed into
            persistence space.

        Returns
        -------
        `np.array` of shape (n, 2), following the original input order
        of the function, but with each value having been transformed to
        a persistence value.
        """
        a = np.asarray(a)

        if len(a.shape) != 2 or a.shape[1] != 2:
            raise RuntimeError("Unexpected array format")

        # This way of sorting ensures that points with the same
        # y value will be sorted according to their x value. It
        # ensures that left-most points are detected first.
        indices = np.argsort(-a[:, 1], kind="stable")

        # Optionally, the function can also return a proper persistence
        # diagram, i.e. a set of tuples that describe the merges.
        if self._calculate_persistence_diagram:
            b = np.zeros_like(a)
            b[:, 0] = a[:, 1]  # y
            b[:, 1] = a[:, 1]  # y (everything is paired with itself)
        else:
            b = None

        # Prepare Union--Find data structure; by default, every vertex
        # is initialized to be its own parent.
        n_vertices = len(a)
        uf = UnionFind(n_vertices)

        # By default, all points that are not explicitly handled will be
        # assigned a persistence value of zero.
        persistence = np.zeros(n_vertices)

        for index in indices:
            left_index = index - 1
            right_index = index + 1

            y = a[index, 1]

            # Inner point: both neighbours are defined; this is easy to
            # handle because we just have to check both of them.
            if left_index >= 0 and right_index <= n_vertices - 1:
                y_left = a[left_index, 1]
                y_right = a[right_index, 1]

                # The point is a local minimum, so we have to merge the
                # two neighbours.
                if y_left >= y and y <= y_right:
                    # The left neighbour is the younger neighbour, so it
                    # will be merged into the right one.
                    if a[uf.find(left_index), 1] < a[uf.find(right_index), 1]:
                        persistence[uf.find(left_index)] = a[uf.find(left_index), 1] - y

                        if self._calculate_persistence_diagram:
                            b[uf.find(left_index), 1] = y

                        uf.merge(left_index, index)
                        uf.merge(index, right_index)
                    else:
                        persistence[uf.find(right_index)] = (
                            a[uf.find(right_index), 1] - y
                        )

                        if self._calculate_persistence_diagram:
                            b[uf.find(right_index), 1] = y

                        uf.merge(right_index, index)
                        uf.merge(index, left_index)

                # The point is a regular point, i.e. one neighbour
                # has a higher function value, the other one has a
                # lower function value.
                elif not (y > y_left and y > y_right):
                    # Always merge the current point into the higher one
                    # of its neighbours. This merge does not result in a
                    # pair.
                    if a[uf.find(left_index), 1] < a[uf.find(right_index), 1]:
                        uf.merge(index, right_index)
                    else:
                        uf.merge(index, left_index)

        # Assign the persistence value to the global maximum of the
        # function to ensure that all tuples have been paired.
        if n_vertices > 0:
            global_maximum_index = indices[0]
            global_minimum_index = indices[-1]

            persistence[global_maximum_index] = (
                a[global_maximum_index, 1] - a[global_minimum_index, 1]
            )

            if self._calculate_persistence_diagram:
                b[global_maximum_index, 1] = a[global_minimum_index, 1]

        # Only create a persistence diagram if we have some persistence
        # tuples to store.
        if b is not None:
            self._persistence_diagram = PersistenceDiagram(b)

        # Perform peak filtering: reduce the number of peaks such that
        # only `n_peaks` remain. If this is not possible (because of a
        # strange value distribution, try to approximate the number).
        if self._n_peaks is not None:
            persistence_values = sorted(persistence)[::-1]
            n_peaks = self._n_peaks

            assert n_vertices == len(persistence_values)

            # Error condition: no filtering should be done because the
            # number of peaks coincides with the number of points. The
            # client will not be notified here, because this condition
            # is only provided for readability.
            if n_peaks == len(persistence_values):
                pass

            # Error condition: there are fewer values than there are
            # peaks requested. In this case, we raise an error.
            elif n_peaks > len(persistence_values):
                raise RuntimeError(
                    f"Specified {n_peaks} peaks, but only {n_vertices} "
                    f"peaks are available."
                )

            # Perform the filtering
            threshold = persistence_values[n_peaks - 1]
            persistence[persistence < threshold] = 0

            # Duplicate values exist. Depending on our parameters, we
            # either ignore this or cut off the peaks that come later
            # in the sense of their x position.
            if persistence_values[n_peaks - 1] == persistence_values[n_peaks]:
                if self._enforce_n_peaks:
                    persistence[n_peaks + 1 :] = 0

        return np.vstack((a[:, 0], persistence)).T

    @property
    def persistence_diagram(self):
        """Return persistence diagram.

        Returns
        -------
        The persistence diagram that was optionally calculated by
        calling `fit_transform()`. The diagram, if available, will
        be returned as a 2D `numpy.array`.
        """
        return self._persistence_diagram
