"""Vendored from https://github.com/deepthought-initiative/neural_name_dismabiguator/blob/main/dbscanc.py

DBSCAN with an added instance-level name-compatibility filter (two mentions can't be neighbors if
their names are incompatible, e.g. different first initials) and a cluster-level year-gap filter
(a candidate can't join a cluster if it would create a publication-year gap wider than year_eps).

One fix from the original: `import re` was missing (name_compatible calls re.sub but the module
never imported re) -- name_compatible would NameError the first time it ran. Added below. Otherwise
unchanged from upstream.
"""
import re

import numpy as np
from sklearn.neighbors import NearestNeighbors


def dbscanc(X, years, names, eps=0.5, min_samples=5, year_eps=3, metric='cosine'):
    """
    DBSCAN with cluster-level and instance-level constraints.

    Parameters
    ----------
    X : ndarray, (n,d) or (n,n) if precomputed
    years : ndarray of shape (n,) or list; can contain None
    names : list of str, shape (n,)
    eps : float, radius for spatial neighborhood
    min_samples : int, min points to be core
    year_eps : int, year window for temporal proximity
    metric : str, distance metric or 'precomputed'

    Returns
    -------
    core_samples : ndarray of int
    labels : ndarray of int, -1 means noise
    """
    X = np.asarray(X)
    years = np.array(years)
    n = X.shape[0]

    # Step 1: Build neighborhoods
    if metric == 'precomputed':
        if X.ndim != 2 or X.shape[1] != n:
            raise ValueError("With metric='precomputed', X must be square.")
        neighbors = [np.where(X[i] <= eps)[0].tolist() for i in range(n)]
    else:
        nbrs = NearestNeighbors(radius=eps, metric=metric).fit(X)
        neighbors = nbrs.radius_neighbors(X, return_distance=False)

    # Instance-level constraints
    def satisfies_instance_constraints(i, j):
        # Name compatibility (instance-level)
        if name_compatible(names[i], names[j]) is None:
            print(names[i], names[j])
        return name_compatible(names[i], names[j])

    # Filter neighborhoods with instance-level constraints
    filtered_neighbors = []
    for i, nb in enumerate(neighbors):
        filtered = [j for j in nb if satisfies_instance_constraints(i, j)]
        filtered_neighbors.append(filtered)

    # Step 2: Core points
    is_core = np.array([len(nb) >= min_samples for nb in filtered_neighbors], dtype=bool)

    # Step 3: Cluster expansion with cluster-level constraints
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i] or not is_core[i]:
            visited[i] = True
            continue

        # Start a new cluster
        labels[i] = cluster_id
        visited[i] = True
        seeds = set(filtered_neighbors[i]) - {i}
        cluster_members = {i}

        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                if is_core[j]:
                    seeds.update(filtered_neighbors[j])
            if labels[j] == -1:
                # Cluster-level constraints
                if cluster_level_compatible(j, cluster_members, names, years, year_eps):
                    labels[j] = cluster_id
                    cluster_members.add(j)

        cluster_id += 1

    core_samples = np.where(is_core)[0]
    return core_samples, labels


def cluster_level_compatible(j, cluster_members, names, years, year_eps):
    # Gather all years from current cluster + candidate
    member_years = [int(years[m]) for m in cluster_members if years[m] is not None]
    if years[j] is not None:
        member_years.append(int(years[j]))
    else:
        return False  # Cannot assess if candidate has no year

    if len(member_years) <= 1:
        return True  # No gaps possible with one or two papers

    # Sort and check for temporal gaps
    member_years.sort()
    for i in range(1, len(member_years)):
        if member_years[i] - member_years[i - 1] > year_eps:
            return False
    return True


def get_first_name(name):
    # Fix vs. upstream: the original unconditionally indexed [1], assuming every name has a given
    # name token (i.e. "Family, Given ..."). A single-token name (just a surname, or a malformed
    # string with no given name recorded) has fewer than 2 tokens and raised "list index out of
    # range" here. Empty string means "no given name" -- name_compatible() already treats two empty
    # first names as compatible (falls through to the middle-name check), same as it would for any
    # other exact match, so this doesn't change behavior for well-formed names.
    tokens = name.lower().split()
    return re.sub(r'\W+', '', tokens[1]) if len(tokens) > 1 else ''


def get_middle_name(name):
    if len(name.lower().split()) > 2:
        return re.sub(r'\W+', '', name.lower().split()[2][0])
    else:
        return None


def name_compatible(name1, name2):
    """name matching."""
    first_name1 = get_first_name(name1)
    first_name2 = get_first_name(name2)
    middle_name1 = get_middle_name(name1)
    middle_name2 = get_middle_name(name2)

    if len(first_name1) != 1 and len(first_name2) != 1 and first_name1 != first_name2:
        return False

    # Only compare middle names if both are present
    if middle_name1 is not None and middle_name2 is not None and middle_name1 != middle_name2:
        return False

    return True
