
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import copy
import pickle


# In[ ]:


def find_closest_node_to_center(node_coords):
    # create array with IDs and coordinates
    num_nodes = node_coords.shape[0]
    node_ids = np.arange(num_nodes)
    geometry = np.column_stack((node_ids, node_coords))

    # find center of geometry
    center = np.mean(geometry[:, 1:], axis=0)
    
    # find closest node to center
    distances = np.linalg.norm(geometry[:, 1:] - center, axis=1)
    closest_node_idx = np.argmin(distances)
    closest_node_id = geometry[closest_node_idx, 0]

    return int(closest_node_id)

def poly_vert(myRVE): # Find vertices that make up the polyhedra
    n_poly = myRVE.n_poly
    poly_face_f_all = myRVE.poly_face_f_all
    face_vert_f_all = myRVE.face_vert_f_all
    verts_poly = []

    for i in range(n_poly): #Loop over polyhedrons
        #verts = [] # Vertices IDs
        verts = np.array([])
        current_poly_faces = poly_face_f_all[i]
        # let's give a max here and break if possible or do nothing 
        for j in range(len(current_poly_faces)): #Loop over faces
            current_face_vert = face_vert_f_all[int(current_poly_faces[j])-1]
            #current_face_edge = face_edge_f_all[int(current_poly_faces[j])-1]
            for m in range(len(current_face_vert)):
                verts = np.hstack((verts,current_face_vert[m]))
                
        verts_poly.append(np.unique(verts))     
           
    return verts_poly

def Vol_SA_Dist(node_pos,myRVE):
    #node_pos = jnp.array(node_pos)
    n_poly = myRVE.n_poly
    #poly_face_f_all = myRVE.poly_face_f_all
    #face_vert_f_all = myRVE.face_vert_f_all
    edge_pos = myRVE.edge_pos
    v_poly = myRVE.v_poly
    
    poly_vol = np.zeros((n_poly,1))
    poly_SA = np.zeros((n_poly,1))

    kappa = 100. # Strength of convexity constraint 
    convex = 0. # We want to keep this value equal to zero to be sure all the cells and the geometry is convex
    
    node_p = []
    
    for i in range(10): #I just want the first 5 cells of my RVE
        #print(v_poly)
        current_v_poly = v_poly[i]
        #print(current_v_poly)
        v_all = node_pos[current_v_poly.astype(int)-1]
        centroid = calculate_hull_centroid(v_all)
        hull = ConvexHull(v_all-centroid) #Remplazar esto por nuestra funcion que con el input de v_all, nos da el volumen y surface area
        poly_vol[i] = hull.volume
        poly_SA[i] = hull.area
        temp = (v_all-centroid)#.reshape(1,-1)
        node_p.append(temp)
        node_p.append(hull.volume)
        node_p.append(hull.area)
        if hull.vertices.size != len(v_all):
            convex += kappa  
        #print(v_all)
            
        
    v1_coords = node_pos[edge_pos[:,0].astype(int)-1] # Coordinates of the first points
    v2_coords = node_pos[edge_pos[:,1].astype(int)-1] # Coordinates of the second points
    dist = ((v2_coords[:,0]-v1_coords[:,0])**2 + (v2_coords[:,1]-v1_coords[:,1])**2 + (v2_coords[:,2]-v1_coords[:,2])**2)**0.5
    edge_distance = dist.astype(float).reshape(-1,1)
    
    node_p = np.array(node_p,dtype='object')

    return poly_vol, poly_SA, edge_distance, convex, node_p

def calculate_hull_centroid(v_all):
    centroid = v_all.mean(axis=0)
    return centroid


# In[ ]:


all_cells = []
for ID in range(5000):
    file = '/scratch/brown/jbarsima/RVE/RVE_adip_size/tess_files/RVE_' + str(ID+1) + '.tess'
    data = []
    with open(file, 'r') as f:
        for line in f:
            line_data = line.strip().split()
            # append the data from this line to the list
            data.append(line_data)

    ## VERTEX DATA
    # loop through each element of the 'data' list
    for i in range(len(data)):
        # check if the current element matches the search value
        if data[i] == ['**vertex']:
            #print(f"Found '**vertex' at index {i}.")
            vertex_in = i #Index for vertex
            break

    n_vertex = int(data[vertex_in + 1][0]) #Number of vertices

    node_ref = np.zeros((1,3))
    for i in range(n_vertex):
        ii = i + vertex_in + 2
        x = float(data[ii][1])
        y = float(data[ii][2])
        z = float(data[ii][3])
        temp = np.array([x,y,z])
        node_ref = np.vstack((node_ref,temp))
    node_ref = node_ref[1:,:] #Vertices

    ## EDGE DATA
    # loop through each element of the 'data' list
    for i in range(n_vertex,len(data)):
        # check if the current element matches the search value
        if data[i] == ['**edge']:
            #print(f"Found '**edge' at index {i}.")
            edge_in = i #Index for edge

    n_edge = int(data[edge_in + 1][0]) #Number of edges

    edge_pos = np.zeros((1,2))
    for i in range(n_edge):
        ii = i + edge_in + 2
        ini_ = int(data[ii][1])
        end_ = int(data[ii][2])
        temp = np.array([ini_,end_])
        edge_pos = np.vstack((edge_pos,temp))
    edge_pos = edge_pos[1:,:] #edges

    ## FACE DATA
    # loop through each element of the 'data' list
    for i in range(n_edge + edge_in,len(data)):
        # check if the current element matches the search value
        if data[i] == ['**face']:
            #print(f"Found '**face' at index {i}.")
            face_in = i #Index for face
            break

    n_face = int(data[face_in + 1][0]) #Number of faces

    off = 2
    face_vert_f_all = []
    face_edge_f_all = []
    for i in range(n_face):
        ii = 4*i + face_in + off
        face_vert = data[ii] #Vertices that form the face
        face_edge = data[ii+1] #Edges that form the face
        face_vert_f = [] #Float
        face_edge_f = [] #Float
        for j in range(len(face_vert)):
            face_vert_f.append(float(face_vert[j]))
        face_vert_f = np.abs(face_vert_f[2:])
        # Array that containts all the vertices for the faces. Useful to find volume without
        # having to index edges
        face_vert_f_all.append(face_vert_f) 
        for m in range(len(face_edge)):
            face_edge_f.append(float(face_edge[m]))
        face_edge_f = np.abs(face_edge_f[1:])
        # Array that containts all the edges for the faces
        # Note that this has "the same" data as face_vert_f_all but 
        # with edge information instead of vertex information directly.
        # This one can be used to create a mesh as it lists the node
        # connectivity by edge. They are ordered correctly so they can be indexed.
        # For example, face 4 contains edges [4. 25. 26. 15. 27. 28.] which have nodes
        # [8. 2. 4. 1. 14. 15.], in that order. These edges can be used to index
        # the array edge_pos. For example, edge_pos[3] will give the nodes that make up
        # edge 4 (8 2). Then, using this data, one can index node_ref to obtain the coordinates
        # of the vertex. THIS PARTICULAR NUMBERS WERE OBTAINED FROM RVE_2
        face_edge_f_all.append(face_edge_f) 
    #print(face_edge_f_all[3])
    #print(face_vert_f_all[3])
    #print(edge_pos[3])
    #print(node_ref[7])

    ## POLYHEDRON DATA
    # loop through each element of the 'data' list
    for i in range(n_face + face_in,len(data)):
        # check if the current element matches the search value
        if data[i] == ['**polyhedron']:
            #print(f"Found '**polyhedron' at index {i}.")
            poly_in = i #Index for face
            break

    n_poly = int(data[poly_in + 1][0]) #Number of polyhedrons

    poly_face_f_all = []
    for i in range(n_poly):
        ii = i + poly_in + 2
        poly_face = data[ii] #Faces that form the poly
        poly_face_f = [] #Float
        for j in range(len(poly_face)):
            poly_face_f.append(float(poly_face[j]))
        poly_face_f = np.abs(poly_face_f[2:]) 
        poly_face_f_all.append(poly_face_f) #Array that containts all the faces
    #print(poly_face_f_all)

    ##############################################################
    ## PERIODICITY INFO
    ## PERIODICITY SECTION INDEX
    # loop through each element of the 'data' list
    for i in range(n_poly + poly_in,len(data)):
        # check if the current element matches the search value
        if data[i] == ['**periodicity']:
            #print(f"Found '**periodicity' at index {i}.")
            periodicity_in = i #Index for periodicity section
            break

    ## VERTEX INFO
    # loop through each element of the 'data' list
    for i in range(periodicity_in,len(data)):
        # check if the current element matches the search value
        if data[i] == ['*vertex']:
            #print(f"Found '*vertex' at index {i}.")
            vertex_per_in = i #Index for periodicity section
            break
    n_vertex_per = int(data[vertex_per_in + 1][0]) #Number of periodic vertices

    vert_pair = np.zeros((n_vertex_per,5)) # <secondary_ver_id> <primary_ver_id> <per_shift_x> <per_shift_y> <per_shift_z>
    vert_pair_coord = np.zeros((1,3)) # Coordinates of the periodic vertices. Twice the length of vert_pair always
    # Ordered as <secondary_ver_coords> in the 2n row, <primary_ver_coords> in the 2n+1 row. n = 0,1,2,...
    for i in range(n_vertex_per):
        ii = i + vertex_per_in + 2
        vert_pair_temp = data[ii]  
        vert_pair[i,0] = vert_pair_temp[0] #Secondary vertex
        vert_pair[i,1] = vert_pair_temp[1] #Primary vertex
        vert_pair[i,2] = vert_pair_temp[2] #Periodic shift x
        vert_pair[i,3] = vert_pair_temp[3] #Periodic shift y
        vert_pair[i,4] = vert_pair_temp[4] #Periodic shift z
        #print(vert_pair_temp)
        vert_pair_coord = np.vstack((vert_pair_coord,node_ref[int(vert_pair[i,0])-1]))
        vert_pair_coord = np.vstack((vert_pair_coord,node_ref[int(vert_pair[i,1])-1]))
    vert_pair_coord = vert_pair_coord[1:,:]
    #print(len(vert_pair_coord))

    ## EDGE INFO
    # loop through each element of the 'data' list
    for i in range(n_vertex_per + vertex_per_in,len(data)):
        # check if the current element matches the search value
        if data[i] == ['*edge']:
            #print(f"Found '*edge' at index {i}.")
            edge_per_in = i #Index for periodicity section
            break
    n_edge_per = int(data[edge_per_in + 1][0]) #Number of periodict vertices

    edge_pair = np.zeros((n_edge_per,2)) # Pair of periodic edges
    for i in range(n_edge_per):
        ii = i + edge_per_in + 2
        edge_pair_temp = data[ii] 
        edge_pair[i,0] = edge_pair_temp[0]
        edge_pair[i,1] = edge_pair_temp[1]
        #print(edge_pair_temp)
    #print(edge_pair)   

    ## FACE INFO
    # loop through each element of the 'data' list
    for i in range(n_edge_per + edge_per_in,len(data)):
        # check if the current element matches the search value
        if data[i] == ['*face']:
            #print(f"Found '*face' at index {i}.")
            face_per_in = i #Index for periodicity section
            break
    n_face_per = int(data[face_per_in + 1][0]) #Number of periodic faces

    face_pair = np.zeros((n_face_per,2)) # Pair of periodic faces
    for i in range(n_face_per):
        ii = i + face_per_in + 2
        face_pair_temp = data[ii]
        face_pair[i,0] = face_pair_temp[0]
        face_pair[i,1] = face_pair_temp[1]

    #Find the closest node to the center of the geometry
    ID_center = find_closest_node_to_center(node_ref)

    # initial guess for deformed RVE is just the original RVE 
    node_def = copy.deepcopy((node_ref))

    ## trying to create a hashable object 
    class RVE:
        def __init__(self,n_poly,poly_face_f_all,face_vert_f_all,face_edge_f_all,edge_pos,node_ref,vert_pair,ID_center,n_edge):
            self.n_poly = n_poly
            self.poly_face_f_all = poly_face_f_all
            self.face_vert_f_all = face_vert_f_all
            self.face_edge_f_all = face_edge_f_all
            self.edge_pos = edge_pos
            self.node_ref = node_ref
            self.vert_pair = vert_pair
            self.ID_center = ID_center
            self.n_edge = n_edge
            self.v_poly = 0. #Dummy variable for until we find the actual value

    myRVE = RVE(n_poly,poly_face_f_all,face_vert_f_all,face_edge_f_all,edge_pos,node_ref,vert_pair,ID_center,n_edge)
    poly_v = poly_vert(myRVE)
    myRVE.v_poly = poly_v # Replace dummy variable for actual value

    vol_ref,SA_ref,length_ref,convex_ref,node_r = Vol_SA_Dist(node_ref,myRVE)
    all_cells.append(node_r)
all_cells_np = np.array(all_cells)


# In[ ]:


all_cells_np[0]


# In[ ]:


# pickle the variable
with open('all_cells_np.pkl', 'wb') as f:
    pickle.dump(all_cells_np, f)

