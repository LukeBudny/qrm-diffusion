from __future__ import annotations

import os
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

    def relative_many_components(
        self,
        candidate_paths: Sequence[str | Path],
        reference_path: str | Path,
        prompt: str,
    ) -> list[dict[str, float]]:
        return [
            {"alignment_delta": value, "reward": value}
            for value in self.relative_many(candidate_paths, reference_path, prompt)
        ]


class ImageRewardScorer:
    """Frozen ImageReward preference scorer using an explicit local checkpoint."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        med_config: str | Path,
        device: str = "cpu",
    ) -> None:
        import ImageReward as image_reward

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        config_path = Path(med_config).expanduser().resolve()
        if not checkpoint_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                "ImageReward requires local checkpoint and med_config files: "
                f"{checkpoint_path}, {config_path}"
            )
        self.model = image_reward.load(
            str(checkpoint_path), device=device, med_config=str(config_path)
        )

    @torch.no_grad()
    def relative_many(
        self,
        candidate_paths: Sequence[str | Path],
        reference_path: str | Path,
        prompt: str,
    ) -> list[float]:
        if not candidate_paths:
            raise ValueError("At least one candidate image is required")
        paths = [str(Path(path)) for path in [*candidate_paths, reference_path]]
        _, scores = self.model.inference_rank(prompt, paths)
        if not isinstance(scores, list):
            scores = [float(scores)]
        reference_score = float(scores[-1])
        return [float(value) - reference_score for value in scores[:-1]]


class CompositeReward:
    """Scale-normalized prompt alignment plus learned human preference reward."""

    def __init__(
        self,
        alignment,
        preference,
        *,
        alignment_weight: float,
        preference_weight: float,
        alignment_scale: float,
        preference_scale: float,
        clip_value: float = 3.0,
    ) -> None:
        if alignment_weight < 0 or preference_weight < 0:
            raise ValueError("Composite reward weights must be non-negative")
        if alignment_weight + preference_weight <= 0:
            raise ValueError("At least one composite reward weight must be positive")
        if alignment_scale <= 0 or preference_scale <= 0:
            raise ValueError("Composite reward scales must be positive")
        if clip_value <= 0:
            raise ValueError("Composite reward clip must be positive")
        weight_total = alignment_weight + preference_weight
        self.alignment = alignment
        self.preference = preference
        self.alignment_weight = alignment_weight / weight_total
        self.preference_weight = preference_weight / weight_total
        self.alignment_scale = float(alignment_scale)
        self.preference_scale = float(preference_scale)
        self.clip_value = float(clip_value)

    def relative_many_components(
        self,
        candidate_paths: Sequence[str | Path],
        reference_path: str | Path,
        prompt: str,
    ) -> list[dict[str, float]]:
        alignment = self.alignment.relative_many(
            candidate_paths, reference_path, prompt
        )
        preference = self.preference.relative_many(
            candidate_paths, reference_path, prompt
        )
        results = []
        for alignment_delta, preference_delta in zip(alignment, preference):
            unclipped = (
                self.alignment_weight
                * alignment_delta
                / self.alignment_scale
                + self.preference_weight
                * preference_delta
                / self.preference_scale
            )
            results.append(
                {
                    "alignment_delta": alignment_delta,
                    "preference_delta": preference_delta,
                    "unclipped_reward": unclipped,
                    "reward": max(-self.clip_value, min(self.clip_value, unclipped)),
                }
            )
        return results

    def relative_many(
        self,
        candidate_paths: Sequence[str | Path],
        reference_path: str | Path,
        prompt: str,
    ) -> list[float]:
        return [
            item["reward"]
            for item in self.relative_many_components(
                candidate_paths, reference_path, prompt
            )
        ]

    def relative(
        self,
        candidate_path: str | Path,
        reference_path: str | Path,
        prompt: str,
    ) -> float:
        return self.relative_many([candidate_path], reference_path, prompt)[0]


def create_reward(settings):
    if settings.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    scorer = settings.scorer.strip().lower()
    alignment = CLIPReward(
        model_id=settings.model_id,
        device=settings.device,
        local_files_only=settings.local_files_only,
    )
    if scorer == "clip":
        return alignment
    if scorer != "composite":
        raise ValueError("Controller reward.scorer must be 'clip' or 'composite'")
    if settings.preference_scorer.strip().lower() != "image_reward":
        raise ValueError("Composite reward currently supports ImageReward preference")
    preference = ImageRewardScorer(
        checkpoint=settings.preference_checkpoint,
        med_config=settings.preference_config,
        device=settings.preference_device,
    )
    return CompositeReward(
        alignment,
        preference,
        alignment_weight=settings.alignment_weight,
        preference_weight=settings.preference_weight,
        alignment_scale=settings.alignment_scale,
        preference_scale=settings.preference_scale,
        clip_value=settings.composite_clip,
    )
