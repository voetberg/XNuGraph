import torch


def encoder_forward(model, planes, x):
    x, _, _, _, _ = model.unpack_batch(x)
    x = {plane: x[plane][:, : model.in_features] for plane in planes}
    return model.encoder(x)


def message_forward(model, planes, x, message_passing_steps):
    x, edge_index_plane, edge_index_nexus, nexus, _ = model.unpack_batch(x)
    x = {plane: x[plane][:, : model.in_features] for plane in planes}
    m = model.encoder(x)

    for _ in range(message_passing_steps):
        # shortcut connect features
        for p in planes:
            s = x[p].detach().unsqueeze(1).expand(-1, m[p].size(1), -1)
            m[p] = torch.cat((m[p], s), dim=-1)

        model.plane_net(m, edge_index_plane)
        model.nexus_net(m, edge_index_nexus, nexus)

    return m


def decoder_forward(model, planes, x):
    m = message_forward(model, planes, x, message_passing_steps=5)
    _, _, _, _, batch = model.unpack_batch(x)

    decoder_out = model.semantic_decoder(m, batch)["x_semantic"]
    return decoder_out
