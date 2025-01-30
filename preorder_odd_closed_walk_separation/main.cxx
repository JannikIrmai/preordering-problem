#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <algorithm>

#include <stdexcept>


double EPSILON = 1e-6;

/**
 * Separation of odd closed walk inequalities according to Mueller (1996).
 * The algorithm works as follows:
 * Construct auxiliary graph with two nodes ij0 and ij1 for each arc ij in original graph and
 * edges (ij0, jk1) and (ij1, jk0) with weights 2*x_ik + 1 - x_ij - x_jk for all triplets ijk.
 * Every ij0-ij1-path in the auxiliary graph corresponds to an odd closed walk in the original 
 * graph that contains the arc ij and vice versa.
 * If the weighted length of such a path is less than 1, the corresponding odd closed walk inequality
 * is violated.
 * Therefore, a violated odd closed walk inequality can be found by searching for shortest paths in the
 * auxiliary graph.
 * INPUT:
 *  - n x n matrix x of fractional values
 *  - maximum walk length max_length
 * OUTPUT:
 *  - List of odd closed walks
 */
std::vector<std::vector<std::size_t>> separate(std::vector<std::vector<double>> x, size_t max_length)
{
    size_t n = x.size();  // number of elements

    // struct for representing nodes in auxiliary graph during shortest path computation
    struct Edge
    {
        size_t i;  // start
        size_t j;  // end
        size_t p;  // parity
        double d;  // distance
        size_t depth;

        bool operator<(const Edge& other) const
        {
            return d < other.d;
        }
    };

    // list of computed odd closed walks
    std::vector<std::vector<size_t>> walks;

    // iterate over all arcs (i0, j0) as start nodes for shortest path computations
    for (size_t i0 = 0; i0 < n; ++i0)
    for (size_t j0 = 0; j0 < n; ++j0)
    {
        if (i0 == j0)
            continue;
        if ((x[i0][j0] <= EPSILON) || (x[i0][j0] >= 1 - EPSILON))
            continue;  // inequality cannot be violated

        // search for shortest path in auxiliary graph with Dijkstra
        std::priority_queue<Edge> queue;
        std::vector<std::vector<std::vector<double>>> dist(2, std::vector<std::vector<double>>(n, std::vector<double>(n, std::numeric_limits<double>::infinity())));
        dist[0][i0][j0] = 0;
        std::vector<std::vector<std::vector<size_t>>> pred(2, std::vector<std::vector<size_t>>(n, std::vector<size_t>(n, 0)));
        queue.push({i0, j0, 0, 0, 0});

        while (!queue.empty())
        {
            auto e = queue.top();
            queue.pop();

            if (dist[e.p][e.i][e.j] < e.d)
                continue;  // outdated

            if (e.i == i0 && e.j == j0 && e.p == 1)
                break;  // found shortest path
            
            // iterate over all neighbors
            for (size_t k = i0; k < n; ++k)
            {    
                if (k == e.i || k == e.j)
                    continue; 
                if (x[e.i][k] >= 1 - EPSILON || x[e.j][k] >= 1 - EPSILON || x[e.j][k] <= EPSILON)
                    continue;  // cannot be violated
                // compute new distance to neighbor
                double alt_distance = e.d + std::max(0.0, 2 * x[e.i][k] + 1 - x[e.i][e.j] - x[e.j][k]);
                size_t p = 1-e.p;  // if ij was odd, jk will be even and vice versa
                if (dist[p][e.j][k] <= alt_distance + EPSILON)
                    continue;  // jk was already visited and has less distance
                if (alt_distance >= 1 - EPSILON)
                    continue; // path is to long
                if (e.depth >= max_length)
                    continue;  // path to long
                // update distance to jk, its predecessor and insert into queue
                dist[p][e.j][k] = alt_distance;
                pred[p][e.j][k] = e.i;
                queue.push({e.j, k, p, alt_distance, e.depth+1});
            } 
        }

        if (dist[1][i0][j0] >= 1 - EPSILON)
            continue;  // not violated

        // extract shortest cycle
        std::vector<size_t> walk;
        size_t y = j0;
        size_t x = i0;
        size_t p = 1;
        while (p != 0 || x != i0 || y != j0)
        {
            walk.push_back(y);
            if (walk.size() > max_length)
                throw std::runtime_error("Error: Walk to long!");
                
            size_t new_x = pred[p][x][y];
            p = 1 - p;
            y = x;
            x = new_x;
        }
        // assert that the walk has odd length
        assert (walk.size() % 2 == 1);
        // reduce max_length if shorter walk was found
        if (walk.size() < max_length)
            max_length = walk.size();

        walks.push_back(walk);
    }
    return walks;
}


// wrap as Python module
PYBIND11_MODULE(preorder_odd_closed_walk_separation, m)
{
    m.def("separate", &separate);
}