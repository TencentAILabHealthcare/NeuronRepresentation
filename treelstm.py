import torch, dgl
import torch.nn as nn


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, mode="sum"):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)
        self.mode = mode

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        if self.mode == "sum":
            h_sum = nodes.mailbox["h"].sum(-2)
            c_sum = nodes.mailbox["c"].sum(-2)
            # equation (2)
            f = torch.sigmoid(self.U_f(h_sum)).view(*c_sum.size())
            # second term of equation (5)
            c = f * c_sum
            iou = self.U_iou(h_sum)
        elif self.mode == "mean":
            h_avg = nodes.mailbox["h"].mean(-2)
            c_avg = nodes.mailbox["c"].mean(-2)
            # equation (2)
            f = torch.sigmoid(self.U_f(h_avg)).view(*c_avg.size())
            # second term of equation (5)
            c = f * c_avg
            iou = self.U_iou(h_avg)
        return {"iou": iou, "c": c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data["c"]
        # equation (6)
        h = o * torch.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False):
        super(TreeLSTM, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.BatchNorm1d(2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        self.cell = TreeLSTMCell(h_size, h_size, mode=mode)
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward_backbone(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        feats = self.mlp1(batch.feats.cuda())
        g.ndata["iou"] = self.cell.W_iou(feats)
        g.ndata["h"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c"] = torch.zeros((n, self.h_size)).cuda()
        # propagate
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        logits = g.ndata.pop("c")[batch.offset.long()]
        return logits

    def forward(self, batch):
        logits = self.forward_backbone(batch)
        if self.fc:
            logits = self.linear(logits)
            return logits
        else:
            return logits


class TreeLSTM_wo_MLP(nn.Module):
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False):
        super(TreeLSTM_wo_MLP, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        self.cell = TreeLSTMCell(x_size, h_size, mode=mode)
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward_backbone(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        # feats = self.mlp1(batch.feats.cuda())
        feats = batch.feats.cuda()
        g.ndata["iou"] = self.cell.W_iou(feats)
        g.ndata["h"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c"] = torch.zeros((n, self.h_size)).cuda()
        # propagate
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        logits = g.ndata.pop("c")[batch.offset.long()]
        return logits

    def forward(self, batch):
        logits = self.forward_backbone(batch)
        if self.fc:
            logits = self.linear(logits)
            return logits
        else:
            return logits


class TreeLSTMCellv2(nn.Module):
    def __init__(self, x_size, h_size, mode="sum"):
        super(TreeLSTMCellv2, self).__init__()
        self.W_iouf = nn.Linear(x_size, 4 * h_size)
        self.U_iouf = nn.Linear(h_size, 4 * h_size)
        self.mode = mode
        self.init_state = True
        self.h_size = h_size

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        # equation (2)
        h_in, c_in = nodes.mailbox["h"], nodes.mailbox["c"]
        if self.mode == "sum":
            h_in, c_in = h_in.sum(-2), c_in.sum(-2)
        elif self.mode == "mean":
            h_in, c_in = h_in.mean(-2), c_in.mean(-2)
        else:
            raise ValueError("must in [sum, mean]")
        x_iouf = nodes.data["iouf"]
        xi, xo, xu, xf = torch.chunk(x_iouf, 4, 1)
        h_iouf = self.U_iouf(h_in)
        hi, ho, hu, hf = torch.chunk(h_iouf, 4, 1)
        i = torch.sigmoid(xi + hi)
        f = torch.sigmoid(xf + hf)
        o = torch.sigmoid(xo + ho)
        u = torch.tanh(xu + hu)
        c = i * u + f * c_in
        h = o * torch.tanh(c)
        return {"h": h, "c": c}

    def apply_node_func(self, nodes):
        if self.init_state:
            # equation (1), (3), (4)
            iouf = nodes.data["iouf"]
            i, o, u, f = torch.chunk(iouf, 4, 1)
            i, o, u, f = (
                torch.sigmoid(i),
                torch.sigmoid(o),
                torch.tanh(u),
                torch.sigmoid(f),
            )
            # equation (5)
            c = i * u + f * nodes.data["c"]
            h = o * torch.tanh(c)
            self.init_state = False
            return {"h": h, "c": c}
        else:
            return {"h": nodes.data["h"], "c": nodes.data["c"]}


class TreeLSTMDoubleCell(nn.Module):
    def __init__(self, x_size, h_size, mode="sum"):
        super(TreeLSTMDoubleCell, self).__init__()
        self.W1_iouf = nn.Linear(x_size, 4 * h_size)
        self.U1_iouf = nn.Linear(h_size, 4 * h_size)
        self.W2_iouf = nn.Linear(h_size, 4 * h_size)
        self.U2_iouf = nn.Linear(h_size, 4 * h_size)
        self.mode = mode
        self.init_state = True
        self.h_size = h_size

    def message_func(self, edges):
        return {
            "h1": edges.src["h1"],
            "c1": edges.src["c1"],
            "h2": edges.src["h2"],
            "c2": edges.src["c2"],
        }

    def reduce_func(self, nodes):
        # equation (2)
        h1, c1 = nodes.mailbox["h1"], nodes.mailbox["c1"]
        h2, c2 = nodes.mailbox["h2"], nodes.mailbox["c2"]
        if self.mode == "sum":
            h1, c1, h2, c2 = h1.sum(-2), c1.sum(-2), h2.sum(-2), c2.sum(-2)
        elif self.mode == "mean":
            h1, c1, h2, c2 = h1.mean(-2), c1.mean(-2), h2.mean(-2), c2.mean(-2)
        else:
            raise ValueError("must in [sum, mean]")
        x_iouf = nodes.data["iouf"]
        xi, xo, xu, xf = torch.chunk(x_iouf, 4, 1)
        h_iouf1 = self.U1_iouf(h1)
        hi1, ho1, hu1, hf1 = torch.chunk(h_iouf1, 4, 1)
        i = torch.sigmoid(xi + hi1)
        f = torch.sigmoid(xf + hf1)
        o = torch.sigmoid(xo + ho1)
        u = torch.tanh(xu + hu1)
        c1 = i * u + f * c1
        h1 = o * torch.tanh(c1)

        x_iouf2 = self.W2_iouf(c1)
        xi, xo, xu, xf = torch.chunk(x_iouf2, 4, 1)
        h_iouf2 = self.U2_iouf(h2)
        hi2, ho2, hu2, hf2 = torch.chunk(h_iouf2, 4, 1)
        i = torch.sigmoid(xi + hi2)
        f = torch.sigmoid(xf + hf2)
        o = torch.sigmoid(xo + ho2)
        u = torch.tanh(xu + hu2)
        c2 = i * u + f * c2
        h2 = o * torch.tanh(c2)
        return {"h1": h1, "c1": c1, "h2": h2, "c2": c2}

    def apply_node_func(self, nodes):
        if self.init_state:
            # equation (1), (3), (4)
            iouf = nodes.data["iouf"]
            i, o, u, f = torch.chunk(iouf, 4, 1)
            i, o, u, f = (
                torch.sigmoid(i),
                torch.sigmoid(o),
                torch.tanh(u),
                torch.sigmoid(f),
            )
            # equation (5)
            c1 = i * u + f * nodes.data["c1"]
            h1 = o * torch.tanh(c1)

            iouf2 = self.W2_iouf(c1)
            i, o, u, f = torch.chunk(iouf2, 4, 1)
            i, o, u, f = (
                torch.sigmoid(i),
                torch.sigmoid(o),
                torch.tanh(u),
                torch.sigmoid(f),
            )
            c2 = i * u + f * nodes.data["c2"]
            h2 = o * torch.tanh(c2)
            self.init_state = False
            return {"h1": h1, "c1": c1, "h2": h2, "c2": h2}
        else:
            return {
                "h1": nodes.data["h1"],
                "c1": nodes.data["c1"],
                "h2": nodes.data["h2"],
                "c2": nodes.data["c2"],
            }


class TreeLSTMDouble(nn.Module):
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False):
        super(TreeLSTMDouble, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.BatchNorm1d(2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        self.cell = TreeLSTMDoubleCell(h_size, h_size, mode=mode)
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward_backbone(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        # feats = batch.feats.cuda()
        feats = self.mlp1(batch.feats.cuda())
        # feats = self.mlp1(torch.randn_like(batch.feats).cuda())
        g.ndata["iouf"] = self.cell.W1_iouf(feats)
        g.ndata["h1"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c1"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["h2"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c2"] = torch.zeros((n, self.h_size)).cuda()
        # propagate
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        logits = g.ndata.pop("c2")[batch.offset.long()]
        return logits

    def forward(self, batch):
        logits = self.forward_backbone(batch)
        if self.fc:
            logits = self.linear(logits)
            return logits
        else:
            return logits


class TreeLSTMv2(nn.Module):
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False):
        super(TreeLSTMv2, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.BatchNorm1d(2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        self.cell = TreeLSTMCellv2(h_size, h_size, mode=mode)
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward_backbone(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        # feats = batch.feats.cuda()
        feats = self.mlp1(batch.feats.cuda())
        g.ndata["iouf"] = self.cell.W_iouf(feats)
        g.ndata["h"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c"] = torch.zeros((n, self.h_size)).cuda()
        # propagate
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        logits = g.ndata.pop("c")[batch.offset.long()]
        return logits

    def forward(self, batch):
        logits = self.forward_backbone(batch)
        if self.fc:
            logits = self.linear(logits)
            return logits
        else:
            return logits
