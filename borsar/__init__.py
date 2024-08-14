__version__ = '0.2.0'

from borsar import (channels, cluster, csd, freq, project, stats, utils, viz)

from borsar.cluster import (Clusters, find_clusters, read_cluster,
                            cluster_based_regression,
                            permutation_cluster_test_array)
from borsar.utils import find_index, find_range, write_info, read_info
