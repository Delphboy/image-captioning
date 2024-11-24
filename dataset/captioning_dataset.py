import json
import os
from itertools import chain

import numpy as np
import torch
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset
from torch_geometric.data import Data
from dataset.vocabulary import Vocab


class CocoImageGraphDataset(Dataset):
    def __init__(
        self,
        captions_file: str,
        butd_root: str,
        sgae_root: str,
        vsua_root: str,
        is_semantic: bool = True,
        freq_threshold: int = 5,
        split: str = "train",
        graph_mode: str = "basic",
    ):
        self.captions_file = captions_file
        self.butd_root = butd_root
        self.sgae_root = sgae_root
        self.vsua_root = vsua_root
        self.is_semantic = is_semantic
        self.freq_threshold = freq_threshold

        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split

        self.graph_mode = graph_mode

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.object_node_feat_locations = []
        self.attribute_node_feat_locations = []
        self.edge_locations = []
        self.captions = []
        self.training_captions = []

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                for i in range(5):
                    obj_node_feat_path = os.path.join(
                        self.butd_root,
                        f"{image_data['cocoid']}.npz",
                    )
                    attr_node_feat_path = os.path.join(
                        self.sgae_root,
                        f"{image_data['cocoid']}.npy",
                    )

                    if self.is_semantic:
                        edge_path = os.path.join(
                            self.sgae_root,
                            f"{image_data['cocoid']}.npy",
                        )
                    else:
                        edge_path = os.path.join(
                            self.vsua_root,
                            f"{image_data['cocoid']}.npy",
                        )

                    caps = [
                        " ".join(sentence["tokens"])
                        for sentence in image_data["sentences"]
                    ]

                    self.object_node_feat_locations.append(obj_node_feat_path)
                    self.attribute_node_feat_locations.append(attr_node_feat_path)
                    self.edge_locations.append(edge_path)
                    self.captions.append(caps)
                    self.training_captions.append(caps[i])

        assert (
            len(self.object_node_feat_locations)
            == len(self.attribute_node_feat_locations)
            == len(self.edge_locations)
            == len(self.captions)
            == len(self.training_captions)
        )

        self.vocab = Vocab(self.captions_file, self.freq_threshold)

    def _get_object_nodes(self, index):
        butd_data = np.load(self.object_node_feat_locations[index], allow_pickle=True)[
            "feat"
        ]
        objects = torch.from_numpy(butd_data).type(torch.float32)
        assert objects.shape[0] > 1, f"Bad object loaded: shape {objects.shape}"
        return objects

    def _get_attribute_nodes(self, index):
        coco_img_sg = np.load(
            self.attribute_node_feat_locations[index],
            allow_pickle=True,
            encoding="latin1",
        ).item()
        attributes = torch.from_numpy(coco_img_sg["obj_attr"]).type(torch.long)
        assert (
            attributes.shape[0] > 1
        ), f"Bad attributes loaded: shape {attributes.shape}"
        return attributes

    def _get_semantic_edges_and_relationships(self, index):
        # Scene Graph Data (EDGES)
        coco_img_sg = np.load(
            self.edge_locations[index], allow_pickle=True, encoding="latin1"
        ).item()
        edges = torch.from_numpy(coco_img_sg["rela_matrix"]).type(torch.long)
        relationships = edges[:, -1].reshape([-1, 1])
        edges = edges[:, :-1]
        assert (
            edges.shape[0] == relationships.shape[0]
        ), "Edges and relationships do not match"
        return edges, relationships

    def _get_geometric_edges_and_relationships(self, index):
        coco_img_sg = np.load(
            self.edge_locations[index], allow_pickle=True, encoding="latin1"
        ).item()
        edges = torch.from_numpy(coco_img_sg["edges"]).type(torch.long)
        relationships = torch.from_numpy(coco_img_sg["feats"]).type(torch.float32)
        return edges, relationships

    def _build_graph(self, index):
        if self.graph_mode == "semantic_basic":
            return self._build_basic_semantic_graph(index)
        elif self.graph_mode == "spatial_basic":
            raise NotImplementedError(f"{self.graph_mode} is not yet implemented")
        elif self.graph_mode == "5nn":
            raise NotImplementedError(f"{self.graph_mode} is not yet implemented")
        elif self.graph_mode == "butd":
            return self._get_object_nodes(index)
        else:
            raise Exception("Unsupported graph type")

    def _build_basic_semantic_graph(self, index) -> Data:
        # Build a semantic graph in the style of Yao et al
        data = Data()
        objects = self._get_object_nodes(index)
        data.x = objects

        edges, relationships = self._get_semantic_edges_and_relationships(index)
        edges, _ = add_self_loops(edges.t())  # , num_nodes=objects.shape[0])
        data.edge_index = edges.contiguous()

        data.edge_attr = relationships

        return data

    def __getitem__(self, index):
        graph = self._build_graph(index)

        captions = self.captions[index]

        # randomly select a caption from the list of captions
        # caption = "<bos> " + np.random.choice(captions) + " <eos>"
        caption = "<bos> " + captions[0] + " <eos>"
        seq = self.vocab.numericalize(caption)

        return graph, seq, captions

    def __len__(self):
        return len(self.captions)

    @property
    def text(self):
        return list(chain.from_iterable(caption_list for caption_list in self.captions))

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[str(wi.item())]
                if word == "<eos>":
                    break
                if word == "<bos>":
                    continue
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions
