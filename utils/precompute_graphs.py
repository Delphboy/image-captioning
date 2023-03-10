import torch
from tqdm import tqdm

import factories.data_factory as dfact
from constants import Constants as const
from graphs.spatial_graph_generator import SpatialGraphGenerator
from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes


def precompute_flickr_spatial(embed_size):
    spatial_generator = SpatialGraphGenerator()
    cnn = FasterRcnnResNet101BoundingBoxes(embed_size)

    loader, _, _ = dfact.get_flickr8k_data(
        root_folder=const.FLICKR_ROOT,
        annotation_file=const.FLICKR_ANN,
        transform=const.STANDARD_TRANSFORM,
        train_ratio=1.0,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )

    cnn.eval()

    graphs = {}
    for idx, (images, _, _) in enumerate(tqdm(loader)):
        prediction = cnn(images)[0]
        graphs[loader.dataset.indices[idx]] = spatial_generator._generate_spatial_graph(prediction)

    torch.save(graphs, 'saved_models/flickr_spatial_graphs.pt')

