from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


class CLIPReward:
    """Lazy, frozen CLIP image/text similarity scorer for terminal rewards."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            model_id, local_files_only=local_files_only
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def score(self, image_paths: Sequence[str | Path], prompts: Sequence[str]) -> torch.Tensor:
        if len(image_paths) != len(prompts) or not image_paths:
            raise ValueError("Reward scoring requires equally sized, non-empty image/prompt lists")
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(
            text=list(prompts),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
        ).to(self.device)
        outputs = self.model(**inputs)
        image = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return (image * text).sum(dim=-1).float().cpu()

    def relative(
        self,
        candidate_path: str | Path,
        reference_path: str | Path,
        prompt: str,
    ) -> float:
        scores = self.score([candidate_path, reference_path], [prompt, prompt])
        return float(scores[0] - scores[1])

    def relative_many(
        self,
        candidate_paths: Sequence[str | Path],
        reference_path: str | Path,
        prompt: str,
    ) -> list[float]:
        if not candidate_paths:
            raise ValueError("At least one candidate image is required")
        paths = [*candidate_paths, reference_path]
        scores = self.score(paths, [prompt] * len(paths))
        reference_score = scores[-1]
        return [float(value - reference_score) for value in scores[:-1]]


def create_reward(settings):
    scorer = settings.scorer.strip().lower()
    if scorer != "clip":
        raise ValueError("Controller training currently supports reward.scorer='clip'")
    return CLIPReward(
        model_id=settings.model_id,
        device=settings.device,
        local_files_only=settings.local_files_only,
    )
