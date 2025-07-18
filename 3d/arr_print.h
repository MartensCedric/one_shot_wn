#ifndef _PRINT_ARR_H_
#define _PRINT_ARR_H_

//-----------------------------------------------------------------------------
// Print all neighboring vertices to a given arrangement vertex.
//
template<class Arrangement>
void print_neighboring_vertices(typename Arrangement::Vertex_const_handle v)
{
    if (v->is_isolated())
    {
        std::cout << "The vertex (" << v->point() << ") is isolated" << std::endl;
        return;
    }

    typename Arrangement::Halfedge_around_vertex_const_circulator  first, curr;
    typename Arrangement::Vertex_const_handle                      u;

    std::cout << "The neighbors of the vertex (" << v->point() << ") are:";
    first = curr = v->incident_halfedges();
    do
    {
        // Note that the current halfedge is (u -> v):
        u = curr->source();
        std::cout << " (" << u->point() << ")";

        ++curr;
    } while (curr != first);
    std::cout << std::endl;

    return;
}

//-----------------------------------------------------------------------------
// Print all vertices (points) and edges (curves) along a connected component
// boundary.
//
template<class Arrangement>
void print_ccb(typename Arrangement::Ccb_halfedge_const_circulator circ)
{
    typename Arrangement::Ccb_halfedge_const_circulator  curr = circ;
    typename Arrangement::Halfedge_const_handle          he;

    std::cout << "(" << curr->source()->point() << ")";
    do
    {
        he = curr;
        std::cout << "   [" << he->curve() << "]   "
            << "(" << he->target()->point() << ")";

        ++curr;
    } while (curr != circ);
    std::cout << std::endl;

    return;
}


//-----------------------------------------------------------------------------
// Print the given arrangement.
//
template<class Arrangement>
void print_arrangement(const Arrangement& arr)
{
    CGAL_precondition(arr.is_valid());

    // Print the arrangement vertices.
    typename Arrangement::Vertex_const_iterator  vit;

    std::cout << arr.number_of_vertices() << " vertices:" << std::endl;
    for (vit = arr.vertices_begin(); vit != arr.vertices_end(); ++vit)
    {
        std::cout << "(" << vit->point().dx() << ")";
        if (vit->is_isolated())
            std::cout << " - Isolated." << std::endl;
        else
            std::cout << " - degree " << vit->degree() << std::endl;
    }

    // Print the arrangement edges.
    typename Arrangement::Edge_const_iterator    eit;

    std::cout << arr.number_of_edges() << " edges:" << std::endl;
    for (eit = arr.edges_begin(); eit != arr.edges_end(); ++eit)
        std::cout << "[" << eit->curve() << "]" << std::endl;

    return;
}

#endif