import numpy as np

from neuron import Neuron, Vertex, PCA

def get_angle(v1, v2):
    v1 = v1/max(np.linalg.norm(v1),1e-10)
    v2 = v2/max(np.linalg.norm(v2),1e-10)
    dp = np.dot(v1, v2)
    ang = np.arccos(max(min(dp,1),-1))
    return ang

def get_vector_plane_angle(v0, v1, v2):
    vn = np.cross(v1, v2)
    ang = np.pi/2 - get_angle(v0, vn)
    return np.abs(ang)

class Branch():
    def __init__(self, vertices, idx):
        self.idx = idx
        self.vertices = vertices # list of vertices from parent to the child
        self.parent = None
        self.child = []
        self.labels = {} # name:value pair of labels
        self.sequence_steps=8

    def populate_branch_features(self):
        if self.child:
            self.labels['is_tip'] = [0]
        else:
            self.labels['is_tip'] = [1]
        self.labels['center'] = self.get_center()
        self.labels['length'] = [self.get_length()]
        self.labels['contraction'] = [self.get_contraction()]

    def populate_branch_sequence(self):
        seq = self.get_sequence()
        for sid in range(self.sequence_steps):
            self.labels['seq-%d'%sid] = list(seq[sid, :])

    def get_sequence(self):
        length = self.get_length()
        if length == 0 or len(self.vertices) <= 1:
            return np.zeros((self.sequence_steps,3))
        step = length/self.sequence_steps
        coord = []
        for v in self.vertices:
            coord.append(v.coord)
        coord = np.array(coord)
        
        # set start to origin
        coord -= coord[0,:]
        # rotate coord
        dir_1 = coord[-1,:]
        dir_1 = dir_1/np.linalg.norm(dir_1)
        coord_1 = np.dot(coord, dir_1)
        coord_23 = coord - (dir_1[np.newaxis,:]*coord_1[:,np.newaxis])
        pca = PCA()
        pca.fit(coord_23)
        coord = pca.transform(coord)
        coord = coord[:,[2,0,1]]

        # interpolate
        coord_new = [coord[0,:]]
        accu,nid = 0,1
        coord_prev = coord[0,:]
        while nid < coord.shape[0]:
            cur_len = np.linalg.norm(coord[nid,:]-coord_prev)
            if accu+cur_len >= step:
                coord_new.append(coord_prev+(coord[nid,:]-coord_prev)/cur_len*(step-accu))
                accu = 0
                coord_prev = coord_new[-1]
            else:
                accu+=cur_len
                coord_prev = coord[nid,:]
                nid+=1
        if len(coord_new) == self.sequence_steps:
            coord_new.append(coord[-1,:])
        coord = np.array(coord_new)

        # generate vector
        vec = np.array([coord[i+1,:]-coord[i,:] for i in range(coord.shape[0]-1)])

        return vec

    def get_center(self):
        center = np.zeros(3)
        for v in self.vertices:
            center += v.coord/len(self.vertices)
        return center

    def get_average_radius(self):
        r = .0 
        for v in self.vertices:
            r += v.radius/len(self.vertices)
        return r

    def get_contraction(self):
        length = self.get_length()
        if length:
            dis = np.linalg.norm(self.vertices[0].coord-self.vertices[1].coord)
            contraction = dis/length
        else:
            contraction = 1 
        return contraction

    def get_length(self):
        length = 0
        for v in self.vertices[1:]:
            length += v.len
        return length

    def get_remote_angle_between_branch(self, branch):
        v1 = self.vertices[-1].coord - self.vertices[0].coord
        v2 = branch.vertices[-1].coord - branch.vertices[0].coord
        return get_angle(v1, v2)

    def get_local_angle_between_branch(self, branch):
        if len(self.vertices) <= 1 or len(branch.vertices) <= 1:
            return np.arccos(0)
        v1 = self.vertices[1].coord - self.vertices[0].coord
        v2 = branch.vertices[1].coord - branch.vertices[0].coord
        return get_angle(v1, v2)

    def get_remote_angle_between_plane(self, branch1, branch2):
        v0 = self.vertices[-1].coord - self.vertices[0].coord
        v1 = branch1.vertices[-1].coord - branch1.vertices[0].coord
        v2 = branch2.vertices[-1].coord - branch2.vertices[0].coord
        return get_vector_plane_angle(v0, v1, v2)

    def get_local_angle_between_plane(self, branch1, branch2):
        if len(self.vertices) <= 1 or \
                len(branch1.vertices) <= 1 or \
                len(branch2.vertices) <= 1:
            return np.arccos(0)
        v0 = self.vertices[1].coord - self.vertices[0].coord
        v1 = branch1.vertices[1].coord - branch1.vertices[0].coord
        v2 = branch2.vertices[1].coord - branch2.vertices[0].coord
        return get_vector_plane_angle(v0, v1, v2)
        
class NeuronSimplifier(): 
    def __init__(self, neuron):
        self._load_neuron(neuron)
        self.branch_list = []

    def _load_neuron(self, neuron):
        if len(neuron.roots) > 1:
            neuron.warning('has %s roots.'%len(neuron.roots)+
                    ' Only largest tree will be summarized.')
            print([neuron.vertices[i].vid for i in neuron.roots])
            neuron.remove_fragements()
        self.neuron = neuron

    def _neuron_to_branch(self, vidx, flag_root = False):
        vertices = []
        while len(self.neuron.vertices[vidx].child) == 1 and not flag_root:
            vertices.append(self.neuron.vertices[vidx])
            vidx = self.neuron.vertices[vidx].child[0]
        vertices.append(self.neuron.vertices[vidx])
        branch = Branch(vertices, len(self.branch_list))
        self.branch_list.append(branch)
        for cidx in self.neuron.vertices[vidx].child:
            tmp = self._neuron_to_branch(cidx)
            tmp.vertices = [self.neuron.vertices[vidx],] + tmp.vertices
            tmp.parent = branch
            branch.child.append(tmp)
        return branch

    def _compute_branch_feature(self, cur, parent, sibling):
        cur.populate_branch_features()
        if parent:
            cur.labels['angle_remote_parent'] = [cur.get_remote_angle_between_branch(parent)]
            cur.labels['angle_local_parent'] = [cur.get_local_angle_between_branch(parent)]
        else:
            cur.labels['angle_remote_parent'] = [0]
            cur.labels['angle_local_parent'] = [0] 
        if sibling:
            cur.labels['angle_remote_sibling'] = [cur.get_remote_angle_between_branch(sibling)]
            cur.labels['angle_local_sibling'] = [cur.get_local_angle_between_branch(sibling)]
        else:
            cur.labels['angle_remote_sibling'] = [0]
            cur.labels['angle_local_sibling'] = [0] 
        if parent and sibling:
            cur.labels['angle_remote_plane'] = [cur.get_remote_angle_between_plane(parent, sibling)]
            cur.labels['angle_local_plane'] = [cur.get_local_angle_between_plane(parent, sibling)]
        else:
            cur.labels['angle_remote_plane'] = [0] 
            cur.labels['angle_local_plane'] = [0] 
        cur.populate_branch_sequence()

    def summarize_by_branch(self, fname_output=None):
        #sepereate neuron branches
        btree = self._neuron_to_branch(self.neuron.roots[0], True)

        #level traversal compute branch features
        cur_lvl = [(btree, None, None)]
        while cur_lvl:
            next_lvl = []
            for node, parent, sibling in cur_lvl:
                node.labels['sibling_num'] = [len(cur_lvl)]
                self._compute_branch_feature(node, parent, sibling)

                if node.child:
                    for c0 in range(len(node.child)):
                        if len(node.child) == 1:
                            next_lvl += [(node.child[c0], node, None)]
                        for c1 in range(len(node.child)):
                            if c0 == c1: continue
                            next_lvl += [(node.child[c0], node, node.child[c1])]
                            break
            cur_lvl = next_lvl

        #generate summarized neuron
        bneuron = Neuron()
        for node in self.branch_list:
            pidx = node.parent.idx if node.parent else -2
            vtx = Vertex(node.idx+1,
                node.vertices[-1].coord,
                node.vertices[-1].type,
                pidx+1,
                node.get_average_radius())
            vtx.labels = node.labels
            bneuron.add_vertex(vtx)

        bneuron.compute_subtree_size()

        if fname_output is not None:
            bneuron.save_eswc(fname_output)
            bneuron.fname = fname_output

        return bneuron

def script_test():
    N = Neuron()
    N.load_eswc('/home/hanbo/workspace/external/morphvae/data/M1_exc_data/neurons/20190410_sample_10.swc')
    N.normalize_neuron(False)
    NS = NeuronSimplifier(N)
    NS.summarize_by_branch('/home/hanbo/datasets/tmp/test.eswc')

if __name__ == '__main__':
    script_test()
