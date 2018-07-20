import os

def write_scalars(writer, scalars, names, n_iter, tag=None):
    for i, scalar, in enumerate(scalars):
        if tag is not None:
            name = os.path.join(tag, names[i])
        else:
            name = names[i]
        writer.add_scalar(name, scalar, n_iter)

def write_hist_parameters(writer, net, n_iter):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

