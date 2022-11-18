import numpy as np
import json
import math
import vtk
#from sklearn.decomposition import PCA
from scipy import linalg as LA
from scipy.spatial.transform import Rotation as R
import colorsys
import copy
from collections import defaultdict
import random

class PCA():
    def __init__(self):
        self.evecs = None

    def fit(self, x):
        cov = np.cov(x, rowvar = False) 
        evals , evecs = LA.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        self.evecs = evecs

    def transform(self, x):
        if self.evecs is not None:
            return np.dot(x, self.evecs) 
        else:
            return x

class Vertex():
    def __init__(self,
            vid,
            coordinate,
            ntype,
            parent=-1,
            radius=1):
        self.vid = vid # vertex id
        self.coord = coordinate #
        self.type = ntype # 2: axon, 3/4: dendrite
        self.radius = radius
        self.parent = parent # vertex index of parent
        self.child = [] # vertex index of child
        self.labels = {} # name:value pair of labels
        self.len = 0

    def add_child(self, child_id):
        self.child.append(child_id)

    def add_label(self, name, value):
        if isinstance(value, (int, float)):
            self.labels[name] = [value] 
        elif isinstance(value, (tuple, list)):
            self.labels[name] = value 
        else:
            print("error: cannot recognize type of vertex lable %s:"%name, value)
            exit(0)

class Neuron():
    def __init__(self):
        self.reset()

    def reset(self):
        self.vertices = [] # vertices 
        self.dict_vid_to_index = {} #key is vertex id, value is vertex(row) index
        self.roots = [] # index of root vertex
        self.labels = {} # labels of neuron
        self.fname = ''
        self.roots_dir = []

    def warning(self, msg):
        print('WARNING: file %s, %s'%(self.fname, msg))

    def load_json(self, fname_json):
        # clean existing file
        self.reset()
        self.fname = fname_json

        # load data
        with open(fname_json) as fp:
            data = json.load(fp)

        # soma
        vid = len(self.vertices)+1
        ntype = 1
        coord = np.array([data['neuron']['soma']['x'],
            data['neuron']['soma']['y'],
            data['neuron']['soma']['z']])
        radius = 1
        parent_vid = -1

        self.dict_vid_to_index[vid] = len(self.vertices)
        self.vertices.append(Vertex(vid, coord, ntype, radius=radius))
        self.roots.append(0)
        self.vertices[0].len = 0

        # dendrite or axon
        for typename in ['axon','dendrite']:
            dict_samplenumber_to_index = {}
            for node in data['neuron'][typename]:
                if node['structureIdentifier'] == 1: # soma node, skip
                    dict_samplenumber_to_index[node['sampleNumber']] = 0
                    continue
                # create vertex
                dict_samplenumber_to_index[node['sampleNumber']] = len(self.vertices)
                vid = len(self.vertices)+1
                ntype = node['structureIdentifier']
                coord = np.array([node['x'],node['y'],node['z']])
                radius = node['radius']
                self.dict_vid_to_index[vid] = len(self.vertices)
                self.vertices.append(Vertex(vid, coord, ntype, radius=radius))

                # connect with parent
                parent_vid = self.vertices[dict_samplenumber_to_index[node['parentNumber']]].vid
                pidx = parent_vid-1
                cidx = vid-1
                self.vertices[pidx].add_child(cidx)
                self.vertices[cidx].parent = pidx
                self.vertices[cidx].len = np.linalg.norm(self.vertices[pidx].coord-self.vertices[cidx].coord)

        if len(self.vertices) == 0:
            self.warning('empty reconstruction')

    def load_eswc(self, fname_eswc):
        # clean existing file
        self.reset()
        self.fname = fname_eswc

        # load data
        with open(fname_eswc) as fp:
            rows = fp.read().splitlines()

        # parse lines
        list_pvid_cvid_pair = []
        col_names = []
        for row in rows:
            if row.startswith('#'):
                col_names = row.split(',')
                continue
            items = row.strip().split(' ')
            try:
                vid = int(items[0])
                ntype = int(items[1])
                coord = np.array([float(x) for x in items[2:5]])
                radius = float(items[5])
                parent_vid = int(items[6])
            except:
                self.warning('skip row when parsing: %s'%row)
                continue

            if vid in self.dict_vid_to_index:
                self.warning('ignore duplicate vertices: %s'%row)
                continue

            self.dict_vid_to_index[vid] = len(self.vertices)
            self.vertices.append(Vertex(vid,coord,ntype,radius=radius))
            if parent_vid != vid:
                list_pvid_cvid_pair.append((parent_vid, vid))
            else:
                self.warning('ignore self connection node: %s'%row)

            for lid in range(len(items[7:])):
                label = items[lid+7]
                label_name = col_names[lid+7] if lid+7 < len(col_names) else 'label_%d'%lid
                if label[0] == '#': #comment
                    self.vertices[-1].add_label('comment',' '.join(items[(lid+7):]))
                    break
                try:
                    self.vertices[-1].add_label(label_name,float(label))
                except:
                    self.warning('cannot recognize colume %d: %s, use default value -1.'%(lid+7,row))
                    self.vertices[-1].add_label(label_name,-1)

        # update parent to child
        for pvid, cvid in list_pvid_cvid_pair:
            if pvid not in self.dict_vid_to_index:
                cidx = self.dict_vid_to_index[cvid]
                self.roots.append(cidx)
                self.vertices[cidx].len = 0
                continue
            if cvid not in self.dict_vid_to_index:
                # this should not happen
                self.warning('missing vertex with id %d, check code.'%cvid)
                exit()
            pidx = self.dict_vid_to_index[pvid]
            cidx = self.dict_vid_to_index[cvid]
            self.vertices[pidx].add_child(cidx)
            self.vertices[cidx].parent = pidx
            self.vertices[cidx].len = np.linalg.norm(self.vertices[pidx].coord-self.vertices[cidx].coord)

        if len(self.vertices) == 0:
            self.warning('empty reconstruction')

    def check_and_flip(self, axis=2, plane=228, ntype=[3,4]):
        mean_coord = 0
        ncount = 0
        for vtx in self.vertices:
            if vtx.type in ntype:
                mean_coord += vtx.coord[axis]
                ncount += 1
        for vid in self.roots:
            mean_coord += self.vertices[vid].coord[axis]
            ncount += 1
        if ncount == 0:
            self.warning('has not type %s'%','.join([str(x) for x in ntype]))
            return
        mean_coord = mean_coord/ncount
        if mean_coord > plane:
            for vtx in self.vertices:
                vtx.coord[axis] = plane*2-vtx.coord[axis]

    def has_global_axon(self, dis_thr=40, length_thr=200):
        len_sum = 0
        for vtx in self.vertices:
            if vtx.type != 2:
                continue
            if vtx.labels['root_radius'][0] > dis_thr:
                len_sum += vtx.len
        return len_sum > length_thr

    def add_vertex(self, vtx):
        # check if vid has been used
        if vtx.vid in self.dict_vid_to_index:
            self.warning('vertex %d already in neuron tree, cannot add it. skip'%vid)
            return False
        # check if vertex label is valid
        if len(self.vertices) > 0:
            for label_name in self.vertices[0].labels:
                if label_name not in vtx.labels:
                    self.warning('new vertex does not have label value %s. skip'%label_name)
                    return False
        # connect with parent and child
        for vid in range(len(self.vertices)):
            if self.vertices[vid].vid == vtx.parent:
                vtx.parent = vid
                self.vertices[vid].add_child(len(self.vertices))
                break
        else:
            # update root if necessary
            self.roots.append(len(self.vertices))
        # add vertex
        self.dict_vid_to_index[vtx.vid] = len(self.vertices)
        self.vertices.append(vtx)


    def compute_distance_from_root(self):
        dis_id_pairs = [(0,rid,rid) for rid in self.roots]
        while len(dis_id_pairs) > 0:
            dis, pid, rid = dis_id_pairs.pop()
            self.vertices[pid].labels['root_distance'] = [dis]
            tmp_dis = self.vertices[rid].coord - self.vertices[pid].coord
            tmp_dis = np.linalg.norm(tmp_dis)
            self.vertices[pid].labels['root_radius'] = [tmp_dis]
            for cid in self.vertices[pid].child:
                tmp_dis = self.vertices[cid].coord - self.vertices[pid].coord
                tmp_dis = np.linalg.norm(tmp_dis)
                tmp_dis += dis
                dis_id_pairs.append((tmp_dis, cid, rid))
        return

    def compute_subtree_size(self):
        def DFS_size(vid):
            size = 1
            if 'length' in self.vertices[vid].labels:
                length = self.vertices[vid].labels['length'][0]
            else:
                length = self.vertices[vid].len
            for cid in self.vertices[vid].child:
                tmp_s, tmp_l = DFS_size(cid)
                length += tmp_l
                size += tmp_s
            self.vertices[vid].labels['subtree_node_count'] = [size]
            self.vertices[vid].labels['subtree_length_sum'] = [length]
            return size, length

        for rid in self.roots:
            DFS_size(rid)
        return

    def get_vtk_unstructured_data_grid(self, type_filter=None):
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        radius = vtk.vtkFloatArray()
        radius.SetName('radius')
        radius.SetNumberOfComponents(1)
        ntype = vtk.vtkFloatArray()
        ntype.SetName('node_type')
        ntype.SetNumberOfComponents(1)
        if len(self.vertices) > 0 and len(self.vertices[0].labels) > 0:
            flag_has_label = True
            labels = {}
            for label_name in self.vertices[0].labels:
                tmp_label = vtk.vtkFloatArray()
                tmp_label.SetName(label_name)
                tmp_label.SetNumberOfComponents(len(self.vertices[0].labels[label_name]))
                labels[label_name] = tmp_label
        else:
            flag_has_label = False
        for vtx in self.vertices:
            points.InsertNextPoint(vtx.coord[0],vtx.coord[1],vtx.coord[2])
            radius.InsertNextTypedTuple([vtx.radius])
            ntype.InsertNextTypedTuple([vtx.type])
            if flag_has_label:
                for label_name in labels:
                    labels[label_name].InsertNextTypedTuple(vtx.labels[label_name])
            if vtx.parent >= 0:
                norm = np.abs(vtx.coord - self.vertices[vtx.parent].coord)
                norm /= (np.linalg.norm(norm)+1e-10)
                norm = (norm*255).astype('uint8')
                colors.InsertNextTypedTuple(list(norm))
            else:
                colors.InsertNextTypedTuple([0,0,0])
        ug = vtk.vtkUnstructuredGrid()
        ug.SetPoints(points)
        ug.GetPointData().SetScalars(colors)
        ug.GetPointData().AddArray(radius)
        ug.GetPointData().AddArray(ntype)
        if flag_has_label:
            for label_name in labels:
                ug.GetPointData().AddArray(labels[label_name])

        line_head = []
        for pidx in self.roots:
            for cidx in self.vertices[pidx].child:
                line_head.append((pidx, cidx))

        if len(self.labels) > 0:
            flag_has_label = True
            labels = {}
            for label_name in self.labels:
                tmp_label = vtk.vtkFloatArray()
                tmp_label.SetName(label_name)
                tmp_label.SetNumberOfComponents(len(self.labels[label_name]))
                labels[label_name] = tmp_label
        else:
            flag_has_label = False
        while len(line_head) > 0:
            pidx, idx = line_head.pop()

            polyline = vtk.vtkPolyLine()
            if type_filter is None or self.vertices[idx].type in type_filter:
                polyline.GetPointIds().InsertId(0, pidx)
                line_len = 1
            else:
                line_len = 0
            while 1:
                if type_filter is None or self.vertices[idx].type in type_filter:
                    polyline.GetPointIds().InsertId(line_len, idx)
                    line_len += 1
                if len(self.vertices[idx].child) == 1:
                    idx = self.vertices[idx].child[0]
                elif len(self.vertices[idx].child) == 0:
                    break
                else:
                    for cidx in self.vertices[idx].child:
                        line_head.append((idx, cidx))
                    break
            if line_len > 0:
                ug.InsertNextCell(polyline.GetCellType(), polyline.GetPointIds())
                if flag_has_label:
                    for label_name in labels:
                        labels[label_name].InsertNextTypedTuple(self.labels[label_name])
        if flag_has_label:
            for label_name in labels:
                ug.GetCellData().AddArray(labels[label_name])

        return ug

    def get_root_vtk_unstructured_data_grid(self):
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        radius = vtk.vtkFloatArray()
        radius.SetName('radius')
        radius.SetNumberOfComponents(1)
        if len(self.vertices) > 0 and len(self.vertices[0].labels) > 0:
            flag_point_has_label = True
            labels_point = {}
            for label_name in self.vertices[0].labels:
                tmp_label = vtk.vtkFloatArray()
                tmp_label.SetName(label_name)
                tmp_label.SetNumberOfComponents(len(self.vertices[0].labels[label_name]))
                labels_point[label_name] = tmp_label
        else:
            flag_point_has_label = False
        if len(self.labels) > 0:
            flag_vtx_has_label = True
            labels_vtx = {}
            for label_name in self.labels:
                tmp_label = vtk.vtkFloatArray()
                tmp_label.SetName(label_name)
                tmp_label.SetNumberOfComponents(len(self.labels[label_name]))
                labels_vtx[label_name] = tmp_label
        else:
            flag_vtx_has_label = False
        if len(self.roots_dir) > 0:
            flag_root_has_dir = True
            label_dir = vtk.vtkFloatArray()
            label_dir.SetName('axon_dir')
            label_dir.SetNumberOfComponents(3)
        else:
            flag_root_has_dir = False
        for ridx, rid in enumerate(self.roots):
            vtx = self.vertices[rid]
            pid = points.InsertNextPoint(vtx.coord[0],vtx.coord[1],vtx.coord[2])
            radius.InsertNextTypedTuple([vtx.radius])
            if flag_point_has_label:
                for label_name in labels_point:
                    labels_point[label_name].InsertNextTypedTuple(vtx.labels[label_name])
            if flag_vtx_has_label:
                for label_name in labels_vtx:
                    labels_vtx[label_name].InsertNextTypedTuple(self.labels[label_name])
            if flag_root_has_dir:
                label_dir.InsertNextTypedTuple(self.roots_dir[ridx])
        ug = vtk.vtkUnstructuredGrid()
        ug.SetPoints(points)
        ug.GetPointData().AddArray(radius)
        if flag_point_has_label:
            for label_name in labels_point:
                ug.GetPointData().AddArray(labels_point[label_name])
        for pid in range(len(self.roots)):
            vtx = vtk.vtkPolyVertex()
            vtx.GetPointIds().InsertId(0, pid)
            ug.InsertNextCell(vtx.GetCellType(), vtx.GetPointIds())
        if flag_vtx_has_label:
            for label_name in labels_vtx:
                ug.GetPointData().AddArray(labels_vtx[label_name])
        if flag_root_has_dir:
            ug.GetPointData().AddArray(label_dir)

        return ug

    def save_vtk(self, fname_vtk, type_filter=None):
        ug = self.get_vtk_unstructured_data_grid(type_filter=type_filter)
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fname_vtk)
        writer.SetInputData(ug)
        writer.Write()

    def save_root_vtk(self, fname_vtk):
        ug = self.get_root_vtk_unstructured_data_grid()
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fname_vtk)
        writer.SetInputData(ug)
        writer.Write()

    def save_swc(self, fname_swc):
        with open(fname_swc, 'w') as fp:
            fp.write('##n,type,x,y,z,radius,parent\n')
            for vtx in self.vertices:
                fp.write('%d %d %.3f %.3f %.3f %.3f '%(
                    vtx.vid, vtx.type, vtx.coord[0], vtx.coord[1], vtx.coord[2],
                    vtx.radius))
                if vtx.parent >= 0:
                    fp.write('%d'%self.vertices[vtx.parent].vid)
                else:
                    fp.write('-1')
                fp.write('\n')

    def save_eswc(self, fname_eswc):
        with open(fname_eswc, 'w') as fp:
            fp.write('##n,type,x,y,z,radius,parent')
            if self.vertices:
                for lname in self.vertices[0].labels:
                    if len(self.vertices[0].labels[lname]) == 1:
                        fp.write(',%s'%lname)
                    else:
                        for k in range(len(self.vertices[0].labels[lname])):
                            fp.write(',%s-%d'%(lname,k))
            fp.write('\n')
            for vtx in self.vertices:
                fp.write('%d %d %.8f %.8f %.8f %.8f '%(
                    vtx.vid, vtx.type, vtx.coord[0], vtx.coord[1], vtx.coord[2],
                    vtx.radius))
                if vtx.parent >= 0:
                    fp.write('%d'%self.vertices[vtx.parent].vid)
                else:
                    fp.write('-1')
                for lname in self.vertices[0].labels:
                    for k in range(len(self.vertices[0].labels[lname])):
                        fp.write(' %.3f'%(vtx.labels[lname][k]))
                fp.write('\n')

    def save_svg(self, fname_svg, max_node=None):
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=[1,1], dpi=100)
        if max_node and max_node < len(self.vertices):
            vtx_list = random.sample(self.vertices, max_node)
        else:
            vtx_list = self.vertices
        for vtx in vtx_list:
            vtx_p = self.vertices[vtx.parent]
            plt.plot([vtx.coord[1],vtx_p.coord[1]],[vtx.coord[0],vtx_p.coord[0]],'w',linewidth=0.5)
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(fname_svg, transparent=True)
        plt.close('all')

    def compute_root_axon_dir(self, axon_len=10):
        axon_types = [2]
        for rid in self.roots:
            #march from roots to find axon
            queue = [(rid, 0, 0)]
            coord_root = self.vertices[rid].coord
            dir_sum = np.zeros(3)
            axon_count = 0
            while len(queue) > 0:
                cur_pid, cur_len, sum_len = queue.pop()
                # add direction if it is axon
                if self.vertices[cur_pid].type in axon_types:
                    tmp_dir = self.vertices[cur_pid].coord-coord_root
                    tmp_dir /= (np.linalg.norm(tmp_dir)+1e-10)
                    dir_sum += tmp_dir*cur_len
                    axon_count += cur_len
                # march child
                for child_pid in self.vertices[cur_pid].child:
                    tmp_dir = self.vertices[child_pid].coord-self.vertices[cur_pid].coord
                    tmp_len = np.linalg.norm(tmp_dir)
                    if tmp_len + sum_len < axon_len:
                        queue.append((child_pid, tmp_len, tmp_len+sum_len))
            if axon_count > 0:
                self.roots_dir.append(dir_sum/np.linalg.norm(dir_sum))
            else:
                self.warning('no axon connected to root %d'%self.vertices[rid].vid)
                self.roots_dir.append((0,1,0))

    def scale_coordiante(self, scale_x, scale_y, scale_z):
        scale = np.array([scale_x, scale_y, scale_z])
        for vtx in self.vertices:
            vtx.coord *= scale
        for vtx in self.vertices:
            if vtx.parent:
                vtx.len = np.linalg.norm(vtx.coord-self.vertices[vtx.parent].coord)
    
    def normalize_neuron(self, flag_rotate=True, rotate_method='dendrite_pca_meanshift', ntype=None,
            dir_order='xyz'):
        if ntype is None:
            ntype=[3,4]
        if len(self.roots) > 1:
            self.warning('has multiple roots, use the first 1 as center')
        center = np.array(self.vertices[self.roots[0]].coord)
        for vidx, vtx in enumerate(self.vertices):
            tmp_coord = self.vertices[vidx].coord
            tmp_coord -= center
            self.vertices[vidx].coord = np.array(tmp_coord)
        if flag_rotate:
            if rotate_method == 'dendrite_pca_meanshift':
                self.rotate_neuron_by_pca_meanshift(ntype=ntype, dir_order=dir_order)
            elif rotate_method == 'dendrite_meanshift':
                self.rotate_neuron_by_meanshift(ntype=ntype, dir_order=dir_order)
            elif rotate_method == 'dendrite_tracemap':
                self.rotate_neuron_by_tracemap(ntype=ntype, dir_order=dir_order)
            elif rotate_method == 'dendrite_max_dir':
                self.rotate_neuron_by_max_dir(ntype=ntype, dir_order=dir_order)
            elif rotate_method == 'dendrite_pca':
                self.rotate_neuron_by_pca(ntype=ntype, dir_order=dir_order)
            elif rotate_method == 'neuron_pca':
                self.rotate_neuron_by_pca(dir_order=dir_order)
            else:
                self.rotate_neuron_by_axon(dir_order=dir_order)

    def _override_type_DFS(self, vid, type_weight):
        stack_all = [vid]
        stack_DFS = [vid]
        while stack_DFS:
            pid = stack_DFS.pop()
            for cid in self.vertices[pid].child:
                stack_DFS.append(cid)
                stack_all.append(cid)
        while stack_all:
            pid = stack_all.pop()
            maxt, maxv = self.vertices[pid].type, type_weight[self.vertices[pid].type]
            for cid in self.vertices[pid].child:
                tmpt, tmpv = self.vertices[cid].type, type_weight[self.vertices[cid].type]
                if tmpv > maxv:
                    maxv = tmpv
                    maxt = tmpt
            self.vertices[pid].type = maxt

    def override_type(self):
        # correct the root type
        type_weight = defaultdict(int)
        type_weight[0] = 100
        type_weight[1] = 100
        type_weight[4] = 2
        type_weight[3] = 3
        type_weight[2] = 1
        for vid in self.roots:
            self._override_type_DFS(vid, type_weight)

    def _compute_max_mean_dir(self, ntype):
        coord = np.array([vtx.coord for vtx in self.vertices if vtx.type in ntype])
        weight = np.array([vtx.len for vtx in self.vertices if vtx.type in ntype])
        # compute first direction
        dir_1 = np.mean(coord*weight[:,np.newaxis], axis=0)
        dir_1 = dir_1/np.linalg.norm(dir_1)
        return dir_1

    def _compute_max_tmap_dir(self, ntype):
        tmap = N.get_branch_tracemap(ntype = ntype)
        dir_1 = tmap.get_max_dir()
        return dir_1

    def _rotate_neuron_by_transform(self, transform_coord):
        for vidx, vtx in enumerate(self.vertices):
            self.vertices[vidx].coord = transform_coord(vtx.coord)
        for idx in range(len(self.roots_dir)):
            self.roots_dir[idx] = transform_coord(self.roots_dir[idx])

    def _flip_neuron_negative_dir(self, ntype):
        # make positive direction more dense
        for order in range(3):
            count = np.sum([vtx.coord[order]*vtx.len for vtx in self.vertices if vtx.type in ntype]) 
            if count < 0:
                for vidx, vtx in enumerate(self.vertices):
                    self.vertices[vidx].coord[order] *= -1

    def _vector_cos(self,a_norm,b):
        return np.dot(a_norm,b)/(np.linalg.norm(b)+1e-16)

    def _vector_to_vectors_cos(self,a_norm,b):
        return np.dot(a_norm.T,b.T)/(np.linalg.norm(b, axis=1)+1e-16)

    def _vector_decouple(self,vec_tar,vec_ref):
        vec_ref = np.array(vec_ref)
        vec_ref /= np.linalg.norm(vec_ref)
        vec_tar -= vec_ref*np.dot(vec_tar, vec_ref)
        vec_tar /= np.linalg.norm(vec_tar)
        return vec_tar

    def _angle_range_sample_searcher(self,init_dir,search_angle,ntype=None,constrain_dir=None):
        thr_angle = math.cos(math.pi/180.0*search_angle) 
        init_dir = np.array(init_dir)
        init_dir /= np.linalg.norm(init_dir)
        sample_dir = np.array([vtx.coord for vtx in self.vertices if ntype is None or vtx.type in ntype])
        sample_weight = np.array([vtx.len*np.linalg.norm(vtx.coord) for vtx in self.vertices if ntype is None or vtx.type in ntype])
        if constrain_dir is not None:
            constrain_dir = np.array(constrain_dir)
            constrain_dir /= np.linalg.norm(constrain_dir)
            project_dir = constrain_dir[:,np.newaxis]*np.dot(constrain_dir.T, sample_dir.T)
            sample_dir -= project_dir.T
        angle = self._vector_to_vectors_cos(init_dir, sample_dir)
        return np.sum(sample_weight[angle>thr_angle])

    def _mean_shift_direction_searcher(self,init_dir,search_angle,ntype=None,constrain_dir=None):
        thr_angle = math.cos(math.pi/180.0*search_angle) 
        init_dir = np.array(init_dir)
        init_dir /= np.linalg.norm(init_dir)
        sample_dir = np.array([vtx.coord*vtx.len for vtx in self.vertices if ntype is None or vtx.type in ntype])
        if constrain_dir is not None:
            constrain_dir = np.array(constrain_dir)
            constrain_dir /= np.linalg.norm(constrain_dir)
            project_dir = constrain_dir[:,np.newaxis]*np.dot(constrain_dir.T, sample_dir.T)
            sample_dir -= project_dir.T
        angle = self._vector_to_vectors_cos(init_dir, sample_dir)
        if np.sum(angle>thr_angle) == 0:
            return init_dir
        new_dir = np.sum(sample_dir[angle>thr_angle,:], axis=0)
        new_dir /= np.linalg.norm(new_dir)
        return new_dir

    def _mean_shift_direction_searcher_v0(self,init_dir,search_angle,ntype=None,constrain_dir=None):
        '''
        deprecate less efficient version
        '''
        thr_angle = math.cos(math.pi/180.0*search_angle) 
        init_dir = np.array(init_dir)
        init_dir /= np.linalg.norm(init_dir)
        new_dir = np.zeros(3)
        for vidx, vtx in enumerate(self.vertices):
            if ntype is not None and vtx.type not in ntype:
                continue
            if self._vector_cos(init_dir, vtx.coord) > thr_angle:
                new_dir += vtx.coord *vtx.len
        if np.linalg.norm(new_dir)  < 1e-16:
            return init_dir
        new_dir /= np.linalg.norm(new_dir)
        return new_dir

    def _mean_shift_direction(self,init_dir,constrain_dir=None,search_angle=30,converge_thr=1,max_iter=16,
            ntype=None):
        thr_angle = math.cos(math.pi/180.0*converge_thr) 
        prev_dir = np.array(init_dir)
        for i in range(max_iter):
            new_dir = self._mean_shift_direction_searcher(prev_dir, search_angle, ntype=ntype,
                    constrain_dir=constrain_dir)
            if self._vector_cos(prev_dir, new_dir) > thr_angle:
                break
            prev_dir = new_dir
        return new_dir

    def rotate_neuron_by_tracemap(self,ntype=None,dir_order='zyx',attempt=0):
        tmap = self.get_branch_tracemap(ntype = ntype)
        dir_1, dir_2, dir_3 = tmap.get_transform_dir()
        dir_1 = self._mean_shift_direction(dir_1,ntype=ntype)
        dir_2 = self._mean_shift_direction(dir_2, constrain_dir=dir_1, ntype=ntype)
        dir_3 = np.cross(dir_1, dir_2)
        def transform_coord(x):
            x = np.array(x)
            c1 = np.dot(x, dir_1)
            c2 = np.dot(x, dir_2)
            c3 = np.dot(x, dir_3)
            if dir_order == 'xyz':
                c_new = [c1, c2, c3]
            else:
                c_new = [c3, c2, c1]
            return np.array(c_new)

        self._rotate_neuron_by_transform(transform_coord)
        self._flip_neuron_negative_dir(ntype)

        if attempt > 0:
            self.rotate_neuron_by_tracemap(ntype,dir_order,attempt-1)

    def rotate_neuron_by_pca_meanshift(self,ntype=None,dir_order='zyx',attempt=0,
            search_angle_1=60):
        # use pca decide z direction
        pca = PCA()
        coord = np.array([vtx.coord for vtx in self.vertices if ntype is None or vtx.type in ntype])
        coord -= np.mean(coord, axis=0)
        if len(coord) < 3:
            return
        pca.fit(coord)
        dir_3 = pca.evecs[:,2] 
        coord = np.array([vtx.coord for vtx in self.vertices if ntype is None or vtx.type in ntype])
        tmp = np.dot(dir_3, coord.T)
        if np.sum(tmp) < 0:
            dir_3 *= -1

        # search primary direction with mean shift
        tmp_dir = [-1,0,0]
        if 1-np.abs(np.dot(tmp_dir, dir_3)) <1e-3:
            tmp_dir = [0,1,0]
        tmp_dir = self._vector_decouple(tmp_dir, dir_3)
        tmp_dir /= np.linalg.norm(tmp_dir)
        num_circle_samples = 6
        r = R.from_rotvec(np.pi/num_circle_samples*2*np.array(dir_3))
        max_weight=0
        for i in range(num_circle_samples):
            tmp_weight = self._angle_range_sample_searcher(tmp_dir, search_angle_1, ntype, dir_3)
            if tmp_weight > max_weight:
                max_weight = tmp_weight
                dir_1 = tmp_dir
            tmp_dir = r.apply(tmp_dir)
        dir_1 = self._mean_shift_direction(dir_1, ntype=ntype, constrain_dir=dir_3,
                search_angle=search_angle_1)

        # use primary and last direction to decide secondary direction
        dir_2 = np.cross(dir_3, dir_1)
        coord = np.array([vtx.coord for vtx in self.vertices if ntype is None or vtx.type in ntype])
        tmp = np.dot(dir_2, coord.T)
        if np.sum(tmp) < 0:
            dir_2 *= -1

        def transform_coord(x):
            x = np.array(x)
            c1 = np.dot(x, dir_1)
            c2 = np.dot(x, dir_2)
            c3 = np.dot(x, dir_3)
            if dir_order == 'xyz':
                c_new = [c1, c2, c3]
            else:
                c_new = [c3, c2, c1]
            return np.array(c_new)

        self._rotate_neuron_by_transform(transform_coord)

        if attempt > 0:
            self.rotate_neuron_by_pca_meanshift(ntype,dir_order,attempt-1)


    def rotate_neuron_by_meanshift(self,ntype=None,dir_order='zyx',attempt=0,
            search_angle_1=90, search_angle_2=45):
        # search primary direction
        max_weight=0
        for vtx in self.vertices:
            if len(vtx.child)!=0:
                continue
            if ntype is not None and vtx.type not in ntype:
                continue
            tmp_weight = self._angle_range_sample_searcher(vtx.coord, search_angle_1, ntype)
            if tmp_weight > max_weight:
                max_weight = tmp_weight
                dir_1 = np.array(vtx.coord)
                dir_1 /= np.linalg.norm(dir_1)
        dir_1 = self._mean_shift_direction(dir_1, ntype=ntype,
                search_angle=search_angle_1)

        # search secondary direction
        max_weight=0
        for vtx in self.vertices:
            if len(vtx.child)!=0:
                continue
            if ntype is not None and vtx.type not in ntype:
                continue
            tmp_dir = self._vector_decouple(vtx.coord, dir_1)
            tmp_weight = self._angle_range_sample_searcher(tmp_dir, search_angle_2, ntype, dir_1)
            if tmp_weight > max_weight:
                max_weight = tmp_weight
                dir_2 = tmp_dir
        dir_2 = self._mean_shift_direction(dir_2, ntype=ntype, constrain_dir=dir_1,
                search_angle=search_angle_2)
        dir_3 = np.cross(dir_1, dir_2)
        def transform_coord(x):
            x = np.array(x)
            c1 = np.dot(x, dir_1)
            c2 = np.dot(x, dir_2)
            c3 = np.dot(x, dir_3)
            if dir_order == 'xyz':
                c_new = [c1, c2, c3]
            else:
                c_new = [c3, c2, c1]
            return np.array(c_new)

        self._rotate_neuron_by_transform(transform_coord)

        if attempt > 0:
            self.rotate_neuron_by_meanshift(ntype,dir_order,attempt-1)

    def rotate_neuron_by_max_dir(self,ntype=None,dir_order='zyx',dir_method='tmap'):
        # mean coord is set to the first dir
        # second dir is set based on PCA
        if ntype is None:
            ntype = list(range(0,8))
        
        coord = np.array([vtx.coord for vtx in self.vertices if vtx.type in ntype])
        if dir_method == 'mean':
            dir_1 = self._compute_max_mean_dir(ntype)
        else:
            dir_1 = self._compute_max_tmap_dir(ntype)
        coord_1 = np.dot(coord, dir_1)
        coord_23 = coord - (dir_1[np.newaxis,:]*coord_1[:,np.newaxis]) 

        # fit pca for the rest 2 directions
        pca = PCA()
        pca.fit(coord_23)

        def transform_coord(x):
            x = np.array(x)
            c1 = np.dot(x, dir_1)
            c23 = x-(dir_1*c1)
            c23_new = pca.transform(x[np.newaxis,:])[0,:]
            if dir_order == 'xyz':
                c_new = [c1, c23_new[0], c23_new[1]]
            else:
                c_new = [c23_new[1], c23_new[0], c1]
            return np.array(c_new)

        for vidx, vtx in enumerate(self.vertices):
            self.vertices[vidx].coord = transform_coord(vtx.coord)
        for idx in range(len(self.roots_dir)):
            self.roots_dir[idx] = transform_coord(self.roots_dir[idx])

        # make positive direction more dense
        for order in range(3):
            count = np.sum([vtx.coord[order] for vtx in self.vertices if vtx.type in ntype]) 
            if count < 0:
                for vidx, vtx in enumerate(self.vertices):
                    self.vertices[vidx].coord[order] *= -1

    def rotate_neuron_by_pca(self,ntype=None,pca_order='zyx'):
        if ntype is None:
            ntype = list(range(0,8))
        pca = PCA()
        coord = np.array([vtx.coord for vtx in self.vertices if vtx.type in ntype])
        if len(coord) < 3:
            return
        # fit pca for the branch
        pca.fit(coord)
        for vidx, vtx in enumerate(self.vertices):
            self.vertices[vidx].coord = pca.transform(vtx.coord[np.newaxis,:])[0,:]
        for idx in range(len(self.roots_dir)):
            tmp_dir = pca.transform(np.array(self.roots_dir[idx])[np.newaxis,:])
            self.roots_dir[idx] = tmp_dir[0,:]
        ## set center to the root
        center = np.array(self.vertices[self.roots[0]].coord)
        for vidx, vtx in enumerate(self.vertices):
            self.vertices[vidx].coord -= center
        # make positive direction more dense
        for order in range(3):
            count = np.sum([vtx.coord[order] for vtx in self.vertices if vtx.type in ntype]) 
            if count < 0:
                for vidx, vtx in enumerate(self.vertices):
                    self.vertices[vidx].coord[order] *= -1
        # reorder if necessary
        if pca_order == 'zyx':
            for vidx, vtx in enumerate(self.vertices):
                self.vertices[vidx].coord = self.vertices[vidx].coord[::-1]


    def rotate_neuron_by_axon(self):
        if len(self.roots_dir) == 0:
            self.compute_root_axon_dir()
        vector_z = self.roots_dir[0]
        vector_y = np.array((0,-vector_z[2]/vector_z[1],1))
        vector_y /= np.linalg.norm(vector_y)
        vector_x = np.cross(vector_y, vector_z)
        center = np.array(self.vertices[self.roots[0]].coord)
        for vidx, vtx in enumerate(self.vertices):
            tmp_coord = self.vertices[vidx].coord
            tmp_coord = (np.dot(tmp_coord,vector_x),
                    np.dot(tmp_coord,vector_y),
                    np.dot(tmp_coord,vector_z))
            self.vertices[vidx].coord = np.array(tmp_coord)
        for idx in range(len(self.roots_dir)):
            tmp_dir = (np.dot(self.roots_dir[idx],vector_x),
                    np.dot(self.roots_dir[idx],vector_y),
                    np.dot(self.roots_dir[idx],vector_z))
            self.roots_dir[idx] = tmp_dir

    def check_branch_distance(self, center=None, ntype=None, threshold=50):
        if center is None:
            center = self.roots[0]
        if ntype is None:
            ntype = [0,1,2,3,4]
        vtx_list = self._collect_childs([center])
        max_dis = 0
        for vid in vtx_list:
            vtx = self.vertices[vid]
            if vtx.type not in ntype:
                continue
            vec = vtx.coord - self.vertices[center].coord
            dis = np.linalg.norm(vec)
            max_dis = max(dis, max_dis)
            if max_dis > threshold: # outlier?
                self.warning('distance to soma larger than %d'%threshold)
                return False
        return True
    
    def remove_fragements(self, flag_del_frag=False, flag_check_root_type=False):
        if len(self.roots) <= 1:
            return
        # find max connected component
        max_root_ridx = 0
        max_count = 0
        for ridx, rid in enumerate(self.roots):
            vtx_count = 0
            child_list = [rid]
            vtx_list = []
            while len(child_list) > 0:
                pid = child_list.pop()
                vtx_list.append(pid)
                vtx_count += 1
                for cid in self.vertices[pid].child:
                    child_list.append(cid)
            if vtx_count > max_count:
                max_count = vtx_count
                max_root_ridx = ridx
                max_tree = vtx_list

        max_root_idx = self.roots[max_root_ridx]
        if flag_check_root_type:
            if self.vertices[max_root_idx].type != 1:
                self.error('Warning, node type of maximum root is not 1. Found %d'%\
                        self.vertices[max_root_idx].type)
        if not flag_del_frag:
            self.roots = [max_root_idx]
            if len(self.roots_dir) > 0:
                self.roots_dir = [self.roots_dir[max_root_ridx]]
            return

        # filter fragments 
        new_vertices = [None]
        self.roots = [0]
        # create vertex
        dict_oldidx_to_newidx = {}
        for oldidx,vtx in enumerate(self.vertices):
            if oldidx not in max_tree:
                continue
            if oldidx == max_root_idx:
                newidx = 0
            else:
                newidx = len(new_vertices)
            dict_oldidx_to_newidx[oldidx] = newidx
            newvtx = copy.deepcopy(vtx)
            newvtx.vid = newidx+1
            if newidx == 0:
                new_vertices[newidx] = newvtx
            else:
                new_vertices.append(newvtx)
        # relink parent and child
        for vtx in new_vertices:
            if vtx.parent >= 0:
                vtx.parent = dict_oldidx_to_newidx[vtx.parent]
            for cid in range(len(vtx.child)):
                vtx.child[cid] = dict_oldidx_to_newidx[vtx.child[cid]]
        
        # update tree
        self.vertices = new_vertices
        self.roots = [0]
        self.dict_vid_to_index = {}
        for idx, vtx in enumerate(new_vertices):
            self.dict_vid_to_index[vtx.vid] = idx
        if len(self.roots_dir) > 0:
            self.roots_dir = [self.roots_dir[max_root_ridx]]
 
    def remove_branch_by_type(self, ntype):
        """
            to remove axon: ntype=[2]
        """
        # filter by type
        new_vertices = []
        self.roots = []
        # create vertex
        dict_oldidx_to_newidx = {}
        for oldidx,vtx in enumerate(self.vertices):
            if vtx.type in ntype:
                continue
            newidx = len(new_vertices)
            if vtx.parent<0:
                self.roots.append(newidx)
            dict_oldidx_to_newidx[oldidx] = newidx
            newvtx = copy.deepcopy(vtx)
            newvtx.vid = newidx+1
            new_vertices.append(newvtx)
        # relink parent and child
        for vtx in new_vertices:
            if vtx.parent >= 0:
                vtx.parent = dict_oldidx_to_newidx[vtx.parent]
            new_child = []
            for cid in range(len(vtx.child)):
                if vtx.child[cid] in dict_oldidx_to_newidx:
                    new_child.append(dict_oldidx_to_newidx[vtx.child[cid]])
            vtx.child = new_child
        
        # update tree
        self.vertices = new_vertices
        self.dict_vid_to_index = {}
        for idx, vtx in enumerate(new_vertices):
            self.dict_vid_to_index[vtx.vid] = idx
        if len(self.roots_dir) > 0:
            self.roots_dir = [self.roots_dir[max_root_ridx]]

    def _search_axons_subtree(self):
        # search axons root
        list_axon_roots = {}
        list_parent_labels = []
        for idx, vtx in enumerate(self.vertices):
            flag_is_root = False
            if vtx.type == 2:
                if vtx.parent < 0 \
                        or self.vertices[vtx.parent].type != 2:
                    flag_is_root = True
                elif self.vertices[vtx.parent].labels['label_0'][0] != vtx.labels['label_0'][0]:
                    flag_is_root = True
                    list_parent_labels.append(self.vertices[vtx.parent].labels['label_0'][0])
            if flag_is_root:
                if vtx.labels['label_0'][0] in list_axon_roots:
                    print('Warning: identify label %d record for multiple sub-tree, check the data!'%\
                            vtx.labels['label_0'][0])
                    continue
                list_axon_roots[vtx.labels['label_0'][0]]={
                        'idx':idx,
                        'fork':0,
                        'tip':0,
                        'label':vtx.labels['label_0'][0],
                        'connect_to_others':False,
                        'list_vtx':[]}
        # check if axon part is connected to a subtree
        for lid in list_axon_roots:
            if lid in list_parent_labels:
                list_axon_roots[lid]['connect_to_others'] = True
        return list_axon_roots

    def _remove_middle_axons(self,list_axon_roots, thr_minimum_num_nodes=15,thr_minimum_num_tips=13):
        list_lid_to_del = []
        for lid in list_axon_roots.keys(): 
            if len(list_axon_roots[lid]['list_vtx'])<thr_minimum_num_nodes:
                list_lid_to_del.append(lid) 
            elif list_axon_roots[lid]['connect_to_others'] and \
                    list_axon_roots[lid]['tip']<thr_minimum_num_tips:
                list_lid_to_del.append(lid) 
        for lid in list_lid_to_del:
            del(list_axon_roots[lid])

    def _mean_shift_vtx_center_and_radius(self, list_vtx, interval_scale=1,
            max_shift_times=10, converge_shift=0.1):
        coord = np.array([vtx.coord for vtx in list_vtx])
        weight = np.array([vtx.len for vtx in list_vtx])
        center_pre = np.sum(coord*weight[:,np.newaxis], axis=0)/np.sum(weight)
        for i in range(max_shift_times):
            dis = np.linalg.norm(coord-center_pre[np.newaxis,:], axis=1)
            radius = np.sum(dis*weight)/(np.sum(weight)+1e-16)
            radius = radius+np.std(dis)*interval_scale
            coord_new = coord[dis<radius,:]
            weight_new = weight[dis<radius] 
            center_new = np.sum(coord_new*weight_new[:,np.newaxis], axis=0)/np.sum(weight_new)
            shift = np.linalg.norm(center_pre-center_new)
            center_pre = center_new
            if shift < converge_shift:
                break
        percent_coverage = np.sum(dis<radius)/float(len(dis))
        return {'radius':radius, 'center':center_pre, 'radius_coverage':percent_coverage}

    def get_axon_subtree_by_label(self):
        self.roots_subtree = []
        # search subtrees
        list_axon_roots = self._search_axons_subtree()
        # count forks and assign vtx
        for idx, vtx in enumerate(self.vertices):
            if vtx.type != 2:
                continue
            list_axon_roots[vtx.labels['label_0'][0]]['list_vtx'].append(vtx)
            if len(vtx.child) == 0:
                list_axon_roots[vtx.labels['label_0'][0]]['tip'] += 1
            if len(vtx.child) > 1:
                list_axon_roots[vtx.labels['label_0'][0]]['fork'] += 1
        self._remove_middle_axons(list_axon_roots)
        # compute radius and center of prunned subtree
        for lid in list_axon_roots: 
            res = self._mean_shift_vtx_center_and_radius(list_axon_roots[lid]['list_vtx'])
            list_axon_roots[lid].update(res)
        self.subtree = list_axon_roots
        return list_axon_roots

    def _collect_childs(self,parents):
        child_list = [x for x in parents]
        res_list = []
        while len(child_list) > 0:
            pid = child_list.pop()
            res_list.append(pid)
            for cid in self.vertices[pid].child:
                child_list.append(cid)
        return res_list

    def check_abnormality(self, dis_thr=2, event_thr=10):
        event_count = 0
        dis_max = 0
        for ridx, rid in enumerate(self.roots):
            child_list = [rid]
            while len(child_list) > 0:
                pid = child_list.pop()
                for cid in self.vertices[pid].child:
                    tmp = self.vertices[pid].coord-self.vertices[cid].coord
                    tmp = np.linalg.norm(tmp)
                    dis_max = max(tmp, dis_max)
                    if tmp > dis_thr:
                        event_count += 1
                        if event_count > event_thr:
                            return True
                    child_list.append(cid)
        print(dis_max)
        return False

class NeuronBundle():
    def __init__(self):
        self.reset()

    def reset(self):
        self.neurons = []

    def load_eswc_from_list_file(self, fname_list):
        with open(fname_list) as fp:
            list_fnames = fp.read().splitlines()
        self.load_eswc_from_files(list_fnames)

    def check_and_flip(self, axis=2, plane=228, ntype=[3,4]):
        for neuron in self.neurons:
            neuron.check_and_flip(axis, plane, ntype)
    
    def load_eswc_from_files(self, list_fnames):
        for idx,fname in enumerate(list_fnames):
            print('\r%d|%d:%s'%(idx,len(list_fnames),fname),end=''),
            self.add_eswc(fname)

    def add_eswc(self, fname):
        tmp_neuron = Neuron()
        tmp_neuron.load_eswc(fname)
        tmp_neuron.compute_distance_from_root()
        tmp_neuron.labels['idx'] = [len(self.neurons)]
        self.neurons.append(tmp_neuron)

    def save_vtk(self, fname_vtk, type_filter=None):
        ug = self.get_vtk_unstructured_data_grid(type_filter=type_filter)
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fname_vtk)
        writer.SetInputData(ug)
        writer.Write()

    def get_vtk_unstructured_data_grid(self, type_filter=None):
        ug = vtk.vtkAppendFilter()
        for neuron in self.neurons:
            ug.AddInputData(neuron.get_vtk_unstructured_data_grid(type_filter=type_filter))
        ug.Update()
        return ug.GetOutput()

    def save_root_vtk(self, fname_vtk):
        ug = vtk.vtkAppendFilter()
        for neuron in self.neurons:
            ug.AddInputData(neuron.get_root_vtk_unstructured_data_grid())
        ug.Update()
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(fname_vtk)
        writer.SetInputData(ug.GetOutput())
        writer.Write()

