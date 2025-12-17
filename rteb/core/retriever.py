import json
from collections import OrderedDict, defaultdict

from beir.retrieval.search.dense.util import cos_sim, dot_score
from pytorch_lightning import LightningModule
import torch
import torch.distributed as dist


class Retriever(LightningModule):

    def __init__(
        self,
        topk: int = 100,
        similarity: str = "cosine",
        save_prediction: bool = False,
        apply_compression: bool = False,
    ):
        super().__init__()
        self.topk = topk
        if similarity == "cosine":
            self.similarity_fn = cos_sim
            self.largest = True
        elif similarity == "dot":
            self.similarity_fn = dot_score
            self.largest = True
        elif similarity == "euclidean":
            self.similarity_fn = torch.cdist
            self.largest = False
        else:
            raise ValueError(f"similarity {similarity} is invalid.")
        self.in_memory = True
        self.save_file = None
        self.save_prediction = save_prediction
        self.compressions = ["full", "float8", "binary"] if apply_compression else ["full"]

    @property
    def local_prediction_file_name(self):
        assert self.save_file is not None
        num_shards = self.trainer.num_devices
        return f"{self.save_file}-{self.local_rank}-of-{num_shards}"

    def get_local_prediction_files(self, num_shards=None):
        assert self.save_file is not None
        if num_shards is None:
            num_shards = self.trainer.num_devices
        return [f"{self.save_file}-{i}-of-{num_shards}" for i in range(num_shards)]

    def on_predict_epoch_start(self):
        self.local_prediction = defaultdict(dict)

    def calculate_similarity(self, query_embds, corpus_embds, batch_scores):
        for compression in self.compressions:
            if compression == "full":
                batch_scores[compression].append(
                    self.similarity_fn(query_embds,corpus_embds.to(query_embds.device)).cpu()
                )
            elif compression == "float8":
                # Cast to float8, then back to float16 for similarity calculation
                q_embeds = query_embds.type(torch.float8_e4m3fn).type(torch.float16)
                c_embds = corpus_embds.type(torch.float8_e4m3fn).type(torch.float16).to(query_embds.device)
                batch_scores[compression].append(self.similarity_fn(q_embeds, c_embds).cpu())
            elif compression == "binary":
                # Needs float type for easy similarity calculation
                q_embeds = torch.where(query_embds > 0, 1, 0).float()
                c_embeds = torch.where(corpus_embds > 0, 1, 0).float().to(query_embds.device)
                batch_scores[compression].append(self.similarity_fn(q_embeds, c_embeds).cpu())
            else:
                raise ValueError(f"Unsupported compression type {compression}!")

    def predict_step(self, batch, batch_idx):
        query_ids, query_embds = batch["id"], batch["embd"].float()
        if isinstance(query_ids, torch.Tensor):
            # TODO: change dataloader to support int id
            raise NotImplementedError("id must be a string.")
        corpus_ids = []
        batch_scores = defaultdict(list)
        # Compute the similarity in batches
        for corpus_batch in self.corpus_embd_dataloader:
            corpus_ids += corpus_batch["id"]
            corpus_embds = corpus_batch["embd"].float()
            self.calculate_similarity(query_embds, corpus_embds, batch_scores)
        # Concat the scores and compute top-k
        for compression in batch_scores:
            scores = torch.cat(batch_scores[compression], dim=1)
            if not self.largest:
                scores = scores * -1
            topk = min(self.topk, len(corpus_ids))
            topk_scores, topk_ids = torch.topk(scores, topk, dim=1, largest=True)
            topk_scores, topk_ids = topk_scores.tolist(), topk_ids.tolist()
            for i, qid in enumerate(query_ids):
                result = OrderedDict()
                for j in range(topk):
                    cid = corpus_ids[topk_ids[i][j]]
                    result[cid] = topk_scores[i][j]
                self.local_prediction[compression][qid] = result

    def on_predict_epoch_end(self):
        if self.trainer.num_devices > 1:
            if self.in_memory:
                gathered_prediction = [None] * self.trainer.num_devices
                dist.all_gather_object(gathered_prediction, self.local_prediction)
                self.prediction = {k: v for preds in gathered_prediction for k, v in preds.items()}
            else:
                with open(self.local_prediction_file_name, "w") as f:
                    json.dump(self.local_prediction, f)
                self.trainer.strategy.barrier()
                self.prediction = {}
                if self.trainer.is_global_zero:
                    for file in self.get_local_prediction_files():
                        with open(file) as f:
                            self.prediction.update(json.load(f))
        else:
            self.prediction = self.local_prediction

        if self.save_prediction and self.trainer.is_global_zero:
            assert self.save_file is not None
            with open(self.save_file, "w") as f:
                json.dump(self.prediction, f)
