import math
from dig.xgraph.method import GNN_LRP
import torch


class GNNWalk(GNN_LRP): 
    def __init__(self, model: torch.nn.Module, explain_graph=False):
        super().__init__(model, explain_graph)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device

        labels = tuple(i for i in range(kwargs.get('num_classes')))
        self.model.eval()

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True)

        edge_index_with_loop, _ = self.add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)
        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            node_idx = node_idx.reshape([1]).to(self.device)
            assert node_idx is not None
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]

        if kwargs.get('walks'):
            walks = kwargs.pop('walks')
            
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                edge_masks = []
                hard_edge_masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    edge_mask = edge_attr.detach()
                    valid_mask = (edge_mask != -math.inf)
                    edge_mask[edge_mask == - math.inf] = edge_mask[valid_mask].min() - 1  # replace the negative inf

                    edge_masks.append(edge_mask)
                    hard_edge_masks.append(self.control_sparsity(edge_attr, kwargs.get('sparsity')).sigmoid())

                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        return walks, edge_masks, related_preds
    
    def compute_walk_score(self, walk_steps, edge_indecies, fully_connected_steps): 
         # hyper-parameter gamma
        epsilon = 1e-30   # prevent from zero division
        gamma = [2, 1, 1]

        # --- record original weights of GNN ---
        ori_gnn_weights = []
        gnn_gamma_modules = []
        clear_probe = x
        for i, walk_step in enumerate(walk_steps):
            modules = walk_step['module']
            gamma_ = gamma[i] if i <= 1 else 1
            if hasattr(modules[0], 'nn'):
                clear_probe = modules[0](clear_probe, edge_indecies, probe=False)
                # clear nodes that are not created by user
            gamma_module = copy.deepcopy(modules[0])
            if hasattr(modules[0], 'nn'):
                for j, fc_step in enumerate(gamma_module.fc_steps):
                    fc_modules = fc_step['module']
                    if hasattr(fc_modules[0], 'weight'):
                        ori_fc_weight = fc_modules[0].weight.data
                        fc_modules[0].weight.data = ori_fc_weight + gamma_ * ori_fc_weight
            else:
                ori_gnn_weights.append(modules[0].weight.data)
                gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
            gnn_gamma_modules.append(gamma_module)

        # --- record original weights of fc layer ---
        ori_fc_weights = []
        fc_gamma_modules = []
        for i, fc_step in enumerate(fully_connected_steps):
            modules = fc_step['module']
            gamma_module = copy.deepcopy(modules[0])
            if hasattr(modules[0], 'weight'):
                ori_fc_weights.append(modules[0].weight.data)
                gamma_ = 1
                gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
            else:
                ori_fc_weights.append(None)
            fc_gamma_modules.append(gamma_module)

        # --- GNN_LRP implementation ---
        for walk_indices in walk_indices_list:
            walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
            for walk_idx in walk_indices:
                walk_node_indices.append(edge_index_with_loop[1, walk_idx])

            h = x.requires_grad_(True)
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                if hasattr(modules[0], 'nn'):
                    # for the specific 2-layer nn GINs.
                    gin = modules[0]
                    run1 = gin(h, edge_index, probe=True)
                    std_h1 = gin.fc_steps[0]['output']
                    gamma_run1 = gnn_gamma_modules[i](h, edge_index, probe=True)
                    p1 = gnn_gamma_modules[i].fc_steps[0]['output']
                    q1 = (p1 + epsilon) * (std_h1 / (p1 + epsilon)).detach()

                    std_h2 = GraphSequential(*gin.fc_steps[1]['module'])(q1)
                    p2 = GraphSequential(*gnn_gamma_modules[i].fc_steps[1]['module'])(q1)
                    q2 = (p2 + epsilon) * (std_h2 / (p2 + epsilon)).detach()
                    q = q2
                else:

                    std_h = GraphSequential(*modules)(h, edge_index)

                    # --- LRP-gamma ---
                    p = gnn_gamma_modules[i](h, edge_index)
                    q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                # --- pick a path ---
                mk = torch.zeros((h.shape[0], 1), device=self.device)
                k = walk_node_indices[i + 1]
                mk[k] = 1
                ht = q * mk + q.detach() * (1 - mk)
                h = ht

            # --- FC LRP_gamma ---
            # debug that torch.zeros(h.shape[0], dtype=torch.long, device=self.device)
            # should be an edge_index with [num_edge, 2]
            for i, fc_step in enumerate(fc_steps):
                modules = fc_step['module']
                std_h = nn.Sequential(*modules)(h) if i != 0 \
                    else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))

                # --- gamma ---
                s = fc_gamma_modules[i](h) if i != 0 \
                    else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                h = ht

            if not self.explain_graph:
                f = h[node_idx, label]
            else:
                f = h[0, label]
            x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
            I = walk_node_indices[0]
            r = x_grads[I, :] @ x[I].T
            walk_scores.append(r) 

