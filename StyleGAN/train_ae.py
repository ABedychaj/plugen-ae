import pickle
import dnnlib
import dnnlib.tflib as tflib

from sklearn.utils import shuffle
from datetime import datetime

import torch
import torch.optim as optim

from AutoEncoder import AE_single_layer
from utils import load_dataset, save_model

# only ffhq
gdrive_urls = {
    'gdrive:networks/stylegan2-ffhq-config-a.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-a.pkl',
    'gdrive:networks/stylegan2-ffhq-config-b.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-b.pkl',
    'gdrive:networks/stylegan2-ffhq-config-c.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-c.pkl',
    'gdrive:networks/stylegan2-ffhq-config-d.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-d.pkl',
    'gdrive:networks/stylegan2-ffhq-config-e.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'gdrive:networks/stylegan2-ffhq-config-f.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl': 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}


def get_path_or_url(path_or_gdrive_path):
    return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)


# ----------------------------------------------------------------------------


_cached_networks = dict()


def load_networks(path_or_gdrive_path):
    path_or_url = get_path_or_url(path_or_gdrive_path)
    if path_or_url in _cached_networks:
        return _cached_networks[path_or_url]

    if dnnlib.util.is_url(path_or_url):
        stream = dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
    else:
        stream = open(path_or_url, 'rb')

    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path_or_url] = G, D, Gs
    return G, D, Gs


def generate_images_in_w_space(dlatents, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    imgs = []
    for row, dlatent in enumerate(dlatents):
        row_images = Gs.components.synthesis.run(dlatent, **Gs_kwargs)
        imgs.append(row_images[0])
    return imgs


def iterate_batches(all_w, batch_size):
    num_batches = (len(all_w) + batch_size - 1) // batch_size
    for batch in range(num_batches):
        w = all_w[batch * batch_size: (batch + 1) * batch_size]
        yield w


network_pkl = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
G, D, Gs = load_networks(network_pkl)

Gs_syn_kwargs = dnnlib.EasyDict()

Gs_syn_kwargs.output_transform = dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
)
Gs_syn_kwargs.randomize_noise = False
Gs_syn_kwargs.minibatch_size = 1

noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

all_w, all_a = load_dataset(keep=False, values=[0] * 17)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AE Model
model = AE_single_layer(input_shape=512, hidden_dim=100).to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

start_epoch = 0
end_epoch = 10
batch_size = 32

for epoch in range(start_epoch, end_epoch):
    all_w = shuffle(all_w)
    model.train()
    running_loss = 0.0
    count_batches = 0
    for w in iterate_batches(all_w, batch_size):
        w = w.to(device)
        optimizer.zero_grad()

        input_images = generate_images_in_w_space(w.cpu(), 1.0)
        encoded, decoded = model(w)

        output_images = generate_images_in_w_space(encoded.cpu().detach().numpy(), 1.0)
        loss = criterion(torch.FloatTensor(input_images), torch.FloatTensor(output_images))
        loss = torch.autograd.Variable(loss, requires_grad=True)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count_batches += 1
        if (count_batches + 1) % 100 == 0:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print(
                f"{dt_string} | Epoch {epoch + 1} | Batch {count_batches} | Loss {running_loss / count_batches:.4f}"
            )
    save_model(f"ae_model/model_e{epoch + 1}.pch", model, optimizer)
