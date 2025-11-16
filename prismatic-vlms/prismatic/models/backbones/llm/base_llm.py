"""
base_llm.py

Abstract class definition of a large (autoregressive) language model backbone (LLM), with full annotations of class
methods, utility functions, and initialization logic.

We also define the generic HFLLMBackbone class here, providing a default interface for loading any HF
AutoModelForCausalLM (e.g., LLamaForCausalLM). In general, we make the assumption that any given LLM backbone implements
the AutoModelForCausalLM API (though we may add Seq2Seq models in the future).

We make this assumption to keep the LLM handling in this codebase relatively lightweight, and to inherit all the nice HF
utilities around different types of decoding/generation strategies.
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type
import types
import warnings

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch
from prismatic.models.backbones.llm.rl_actor import LLMActorConfig, PruningActorRegistry, TokenPruningActor

# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary HF LLM Backbones ===
class LLMBackbone(nn.Module, ABC):
    def __init__(self, llm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = llm_backbone_id

        # Instance attributes for an LLM Backbone
        self.llm: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerBase = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None: ...

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the LLM given targets (labels), returning the scalar Cross-Entropy Loss"""
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> Type[PromptBuilder]: ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# === Abstract Base Class for Arbitrary HF Causal LLMs ===
class HFCausalLLMBackbone(LLMBackbone, ABC):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_family: str,
        llm_cls: Type[PreTrainedModel],
        hf_hub_path: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
        llm_pruning_actor_layers: Optional[Sequence[int]] = None,
        llm_pruning_actor_hidden_dim: Optional[int] = None,
        llm_pruning_actor_num_samples: Optional[int] = None,
        llm_pruning_reward_alpha: Optional[float] = None,
        llm_pruning_reward_beta: Optional[float] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode
        # Initialize LLM (downloading from HF Hub if necessary) --> `llm_cls` is the actual {Model}ForCausalLM class!
        #   => Note: We're eschewing use of the AutoModel API so that we can be more explicit about LLM-specific details
        if not self.inference_mode:
            overwatch.info(f"Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
            self.llm = llm_cls.from_pretrained(
                hf_hub_path,
                token=hf_token,
                use_flash_attention_2=use_flash_attention_2 if not self.inference_mode else False,
                # Quantization parameters
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # [Contract] `inference_mode` means we're loading from a pretrained checkpoint; no need to load base weights!
        else:
            overwatch.info(f"Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
            llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token)
            self.llm = llm_cls._from_config(llm_config)
        # Lightweight Handling (with extended explanation) for setting some LLM Parameters
        #   => Set `decoder.use_cache = False` --> incompatible with gradient checkpointing (+ training in general)
        #
        #      Reference: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = False if not self.inference_mode else True

        #   => Turns out that when gradient checkpointing is on and the underlying LLM has no "trainable" parameters
        #      (requires_grad is False), backprop will fail; setting `enable_input_requires_grad()` registers a new
        #      forward hook that fixes this =>> also totally safe for the "full finetuning" setting!
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # === RL Token Pruning Actors ===
        self.pruning_actor_layers: Tuple[int, ...] = (
            tuple(sorted(set(llm_pruning_actor_layers))) if llm_pruning_actor_layers else tuple()
        )
        self.pruning_actor_hidden_dim = llm_pruning_actor_hidden_dim
        self.pruning_actor_num_samples = max(int(llm_pruning_actor_num_samples or 1), 1)
        self.pruning_reward_alpha = float(llm_pruning_reward_alpha) if llm_pruning_reward_alpha is not None else 1.0
        self.pruning_reward_beta = float(llm_pruning_reward_beta) if llm_pruning_reward_beta is not None else 1.0
        self.pruning_actors = PruningActorRegistry()
        self.latest_pruning_logits: Dict[int, torch.Tensor] = {}
        self.latest_pruning_keep_prob: Dict[int, torch.Tensor] = {}
        self.latest_pruning_masks: Dict[int, torch.Tensor] = {}
        self.latest_pruning_rewards: Dict[int, float] = {}
        self.latest_pruning_reward_tensors: Dict[int, torch.Tensor] = {}
        self.pruning_reward_fn: Optional[
            Callable[
                [
                    int,
                    nn.Module,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Optional[torch.Tensor],
                    Optional[torch.Tensor],
                    Optional[torch.Tensor],
                    torch.Tensor,
                ],
                torch.Tensor,
            ]
        ] = None
        self._current_image_token_mask: Optional[torch.Tensor] = None
        if self.pruning_actor_layers:
            self._install_pruning_actors()
            self.set_pruning_reward_fn(self._default_pruning_reward_fn)
        self.pruning_inference_threshold = 0.5
        self.rl_training_mode: bool = False
        self.rl_num_samples: int = 1
        self.pruning_reference_actors: nn.ModuleDict = nn.ModuleDict()
        self.latest_rl_log_probs: Dict[int, torch.Tensor] = {}
        self.latest_rl_ref_log_probs: Dict[int, torch.Tensor] = {}
        self.latest_rl_sampling_advantage: Dict[int, torch.Tensor] = {}
        self.latest_pruning_ratios: Dict[int, torch.Tensor] = {}
        self.latest_ref_keep_prob: Dict[int, torch.Tensor] = {}
        self.latest_image_masks: Dict[int, torch.Tensor] = {}
        self.latest_keep_prob_current: Dict[int, torch.Tensor] = {}

        # Load (Fast) Tokenizer
        overwatch.info(f"Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API", ctx_level=1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_hub_path,
            model_max_length=self.llm_max_length,
            token=hf_token,
            padding_side="right",
        )

        # Explicitly verify that Tokenizer padding_side is set to right for training!
        assert self.tokenizer.padding_side == "right", "Tokenizer `padding_side` is not set to `right`!"

        # Validation =>> Our VLM logic currently operates under the assumption that the tokenization of a new input
        #                starts with a <BOS> token unless `add_special_tokens = False`; for these models, we empirically
        #                find that adding image patches *after* the BOS leads to much better performance.
        #
        # As a result we explicitly validate that a tokenizer conforms to the expected behavior; if you're reading this
        # line, it's probably because you're adding a new LLM with a different tokenizer behavior. If so, feel free to
        # override the `SPECIAL_CASES` set below, but make sure to make the appropriate changes in the `datasets.py`
        # and VLM `forward()` logic!
        SPECIAL_CASES = {
            # Phi-2 Tokenizer doesn't add any BOS tokens by default, and sets BOS == EOS == "<|endoftext|>"
            #   =>> We'll prepend BOS to first input (to play nicely with image token insertion logic; verified that
            #       this works well with base LLM generation.
            #   =>> Like Llama-2 Tokenizers -- we'll add a special PAD token for training purposes.
            "phi-2-3b",
        }
        if self.identifier in SPECIAL_CASES:
            return

        # Note =>> this assert should hold for all Llama-derived tokenizers (`LlamaTokenizerFast` ==> includes Mistral!
        assert (self.tokenizer("Test 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id) and (
            self.tokenizer("Test 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id
        ), (
            f"Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\n"
            "Please read the comment in `base_llm.py` for more information!"
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a `transformer_auto_wrap_policy` where we wrap each instance of `self.transformer_layer_cls`"""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
        )

        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """Dispatch to underlying LLM instance's `gradient_checkpointing_enable`; defined for all `PretrainedModel`."""
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embedding_module = self.llm.get_input_embeddings()
        # 确保 input_ids 与嵌入权重在同一设备（FSDP + CPU offload 下尤为重要）
        target_device = embedding_module.weight.device
        if input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        return embedding_module(input_ids)

    # [Contract] Should match the `forward` call of the underlying `llm` instance!
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        image_token_mask = kwargs.pop("image_token_mask", None)
        if image_token_mask is not None:
            target_device = inputs_embeds.device if inputs_embeds is not None else self.llm.device
            image_token_mask = image_token_mask.to(device=target_device, dtype=torch.bool)
        self._current_image_token_mask = image_token_mask
        self.latest_pruning_logits.clear()
        self.latest_pruning_keep_prob.clear()
        self.latest_pruning_masks.clear()
        self.latest_pruning_rewards.clear()
        self.latest_pruning_reward_tensors.clear()
        self.latest_rl_log_probs.clear()
        self.latest_rl_ref_log_probs.clear()
        self.latest_rl_sampling_advantage.clear()
        self.latest_pruning_ratios.clear()
        self.latest_ref_keep_prob.clear()
        self.latest_image_masks.clear()
        self.latest_keep_prob_current.clear()
        try:
            output: CausalLMOutputWithPast = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        finally:
            self._current_image_token_mask = None
        return output

    # === RL Actor Utilities ===
    def _install_pruning_actors(self) -> None:
        transformer_layers = self._resolve_transformer_layers()
        num_layers = len(transformer_layers)
        hidden_dim = self.pruning_actor_hidden_dim or max(self.embed_dim // 2, 128)
        actor_config = LLMActorConfig(target_layers=self.pruning_actor_layers, hidden_dim=hidden_dim)
        self.pruning_actor_hidden_dim = hidden_dim

        for layer_id in actor_config.target_layers:
            if layer_id < 1 or layer_id > num_layers:
                warnings.warn(
                    f"Skipping invalid LLM pruning actor layer {layer_id}; valid range is [1, {num_layers}].",
                    RuntimeWarning,
                )
                continue

            layer_module = transformer_layers[layer_id - 1]
            if hasattr(layer_module, "_original_forward_with_actor"):
                warnings.warn(
                    f"Layer {layer_id} already wrapped with pruning actor. Skipping duplicate installation.",
                    RuntimeWarning,
                )
                continue

            actor = TokenPruningActor(self.embed_dim, actor_config.hidden_dim)
            self.pruning_actors.add_actor(layer_id, actor)
            self._wrap_layer_with_actor(layer_module, layer_id, actor)

    def enable_pruning_rl(self, reference_state_dict: Optional[Dict[int, Dict[str, torch.Tensor]]], num_samples: int) -> None:
        if not self.pruning_actor_layers:
            raise RuntimeError("No pruning actors installed; set `llm_pruning_actor_layers` before enabling RL.")
        self.rl_training_mode = True
        self.rl_num_samples = max(int(num_samples), 1)
        self.pruning_reference_actors = nn.ModuleDict()
        hidden_dim = self.pruning_actor_hidden_dim or max(self.embed_dim // 2, 128)
        for layer_id in self.pruning_actor_layers:
            actor = TokenPruningActor(self.embed_dim, hidden_dim)
            if reference_state_dict and layer_id in reference_state_dict:
                actor.load_state_dict(reference_state_dict[layer_id], strict=False)
            else:
                actor.load_state_dict(self.pruning_actors.get_actor(layer_id).state_dict())
            actor.eval()
            for param in actor.parameters():
                param.requires_grad_(False)
            self.pruning_reference_actors[str(layer_id)] = actor

    def disable_pruning_rl(self) -> None:
        self.rl_training_mode = False
        self.pruning_reference_actors = nn.ModuleDict()

    def _resolve_transformer_layers(self) -> Sequence[nn.Module]:
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            return self.llm.model.layers
        if hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "layers"):
            return self.llm.transformer.layers
        if hasattr(self.llm, "layers"):
            return self.llm.layers
        raise AttributeError(
            f"Unable to locate transformer layers for LLM `{self.identifier}`; please verify architecture support."
        )

    def _wrap_layer_with_actor(
        self,
        layer_module: nn.Module,
        layer_id: int,
        actor: TokenPruningActor,
    ) -> None:
        original_forward = layer_module.forward

        def forward_with_actor(module_self: nn.Module, *args, **kwargs):
            hidden_states = self._extract_hidden_states(args, kwargs)
            if self.rl_training_mode:
                mask, actor_logits, keep_prob = self._rl_select_pruning_mask(layer_id, hidden_states, actor)
            else:
                mask, actor_logits, keep_prob = self._select_pruning_mask(layer_id, hidden_states, actor)
            pruned_hidden_states = hidden_states * mask.unsqueeze(-1)

            new_args, new_kwargs = self._replace_hidden_states(args, kwargs, pruned_hidden_states)
            outputs = original_forward(*new_args, **new_kwargs)

            module_self.token_pruning_actor_logits = actor_logits.detach()
            module_self.token_pruning_keep_prob = keep_prob.detach()
            module_self.token_pruning_mask = mask.detach()
            return outputs

        layer_module._original_forward_with_actor = original_forward  # type: ignore[attr-defined]
        layer_module.forward = types.MethodType(forward_with_actor, layer_module)

    def get_latest_pruning_logits(self) -> Dict[int, torch.Tensor]:
        """Return the most recent actor logits keyed by transformer layer id."""
        return self.latest_pruning_logits

    def get_latest_pruning_masks(self) -> Dict[int, torch.Tensor]:
        return self.latest_pruning_masks

    def get_latest_pruning_rewards(self) -> Dict[int, float]:
        return self.latest_pruning_rewards

    def get_latest_pruning_reward_tensors(self) -> Dict[int, torch.Tensor]:
        return self.latest_pruning_reward_tensors

    def get_total_pruning_reward(self) -> torch.Tensor:
        if not self.latest_pruning_reward_tensors:
            if len(self.pruning_actors.actors):
                device = next(self.pruning_actors.parameters()).device
            else:
                device = next(self.llm.parameters()).device
            return torch.tensor(0.0, device=device, dtype=next(self.llm.parameters()).dtype)
        rewards = [tensor.mean() for tensor in self.latest_pruning_reward_tensors.values()]
        return torch.stack(rewards).sum()

    def get_actor_state_dict(self) -> Dict[int, Dict[str, torch.Tensor]]:
        return {layer_id: actor.state_dict() for layer_id, actor in self.pruning_actors.actors.items()}

    def set_pruning_reward_fn(
        self,
        reward_fn: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Register a reward function used to score sampled pruning masks."""
        self.pruning_reward_fn = reward_fn

    def _extract_hidden_states(self, args: Tuple, kwargs: Dict) -> torch.Tensor:
        if args:
            return args[0]
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        raise ValueError("Unable to locate `hidden_states` argument for transformer layer forward pass.")

    def _replace_hidden_states(
        self,
        args: Tuple,
        kwargs: Dict,
        hidden_states: torch.Tensor,
    ) -> Tuple[Tuple, Dict]:
        if args:
            mutable_args = list(args)
            mutable_args[0] = hidden_states
            return tuple(mutable_args), kwargs
        kwargs["hidden_states"] = hidden_states
        return args, kwargs

    def _select_pruning_mask(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        actor: TokenPruningActor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_logits = actor(hidden_states)
        keep_prob = torch.softmax(actor_logits, dim=-1)[..., 1]

        if self.training:
            mask = self._build_continuous_mask(keep_prob)
            reward_tensor = self._compute_pruning_reward(
                layer_id,
                hidden_states.detach(),
                mask,
                actor_logits,
                keep_prob,
            )
            self.latest_pruning_reward_tensors[layer_id] = reward_tensor
            self.latest_pruning_rewards[layer_id] = float(reward_tensor.detach().mean().item())
        else:
            mask = self._build_discrete_mask(keep_prob)
            reward_tensor = self._compute_pruning_reward(
                layer_id,
                hidden_states.detach(),
                mask.detach(),
                actor_logits.detach(),
                keep_prob.detach(),
            )
            self.latest_pruning_reward_tensors[layer_id] = reward_tensor.detach()
            self.latest_pruning_rewards[layer_id] = float(reward_tensor.detach().mean().item())

        self.latest_pruning_logits[layer_id] = actor_logits.detach()
        self.latest_pruning_keep_prob[layer_id] = keep_prob.detach()
        self.latest_pruning_masks[layer_id] = mask.detach()

        return mask.to(hidden_states.dtype), actor_logits, keep_prob

    def _rl_select_pruning_mask(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        actor: TokenPruningActor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_logits = actor(hidden_states)
        keep_prob = torch.softmax(actor_logits, dim=-1)[..., 1].clamp(1e-6, 1 - 1e-6)
        self.latest_keep_prob_current[layer_id] = keep_prob.detach()

        image_mask = self._current_image_token_mask
        if image_mask is None:
            image_mask = torch.ones_like(keep_prob, dtype=torch.bool, device=keep_prob.device)
        else:
            image_mask = image_mask.to(device=keep_prob.device, dtype=torch.bool)
        self.latest_image_masks[layer_id] = image_mask.detach()

        ref_actor = self.pruning_reference_actors.get(str(layer_id))
        if ref_actor is not None:
            with torch.no_grad():
                ref_logits = ref_actor(hidden_states)
                ref_keep_prob = torch.softmax(ref_logits, dim=-1)[..., 1].clamp(1e-6, 1 - 1e-6)
        else:
            ref_keep_prob = keep_prob.detach()
        self.latest_ref_keep_prob[layer_id] = ref_keep_prob.detach()

        bernoulli = torch.distributions.Bernoulli(probs=keep_prob)
        bernoulli_ref = torch.distributions.Bernoulli(probs=ref_keep_prob)

        sample_masks: List[torch.Tensor] = []
        sample_log_probs: List[torch.Tensor] = []
        sample_rewards: List[torch.Tensor] = []

        ones_mask = torch.ones_like(keep_prob, device=keep_prob.device)

        for _ in range(self.rl_num_samples):
            sample = bernoulli.sample()
            sample_mask = torch.where(image_mask, sample, ones_mask)
            sample_mask[:, 0] = 1.0
            sample_masks.append(sample_mask)

            log_prob = bernoulli.log_prob(sample)
            log_prob = (log_prob * image_mask).sum(dim=-1)
            sample_log_probs.append(log_prob)

            reward = self._compute_rl_reward(hidden_states, sample_mask, image_mask)
            sample_rewards.append(reward)

        rewards_tensor = torch.stack(sample_rewards, dim=0)  # [num_samples, batch]
        mean_reward = rewards_tensor.mean(dim=0, keepdim=True)
        std_reward = rewards_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
        advantages = (rewards_tensor - mean_reward) / std_reward

        best_indices = advantages.argmax(dim=0)
        batch_indices = torch.arange(advantages.shape[1], device=advantages.device)

        chosen_mask = torch.stack(sample_masks, dim=0)[best_indices, batch_indices]
        chosen_log_prob = torch.stack(sample_log_probs, dim=0)[best_indices, batch_indices]
        chosen_reward = rewards_tensor[best_indices, batch_indices]
        chosen_advantage = advantages[best_indices, batch_indices]

        ref_log_prob = (bernoulli_ref.log_prob(chosen_mask) * image_mask).sum(dim=-1)

        # Compute pruning ratio for monitoring (fraction pruned)
        image_token_counts = image_mask.sum(dim=-1).clamp_min(1)
        kept_tokens = (chosen_mask * image_mask).sum(dim=-1)
        prune_ratio = 1.0 - kept_tokens / image_token_counts.to(chosen_mask.dtype)

        self.latest_pruning_logits[layer_id] = actor_logits.detach()
        self.latest_pruning_keep_prob[layer_id] = keep_prob.detach()
        self.latest_pruning_masks[layer_id] = chosen_mask.detach()
        self.latest_pruning_reward_tensors[layer_id] = chosen_reward.detach()
        self.latest_pruning_rewards[layer_id] = float(chosen_reward.detach().mean().item())

        self.latest_rl_log_probs[layer_id] = chosen_log_prob
        self.latest_rl_ref_log_probs[layer_id] = ref_log_prob
        self.latest_rl_sampling_advantage[layer_id] = chosen_advantage
        self.latest_pruning_ratios[layer_id] = prune_ratio

        return chosen_mask.to(hidden_states.dtype), actor_logits, keep_prob

    def _build_continuous_mask(self, keep_prob: torch.Tensor) -> torch.Tensor:
        image_mask = self._current_image_token_mask
        if image_mask is None:
            image_mask = torch.ones_like(keep_prob, dtype=torch.bool, device=keep_prob.device)
        else:
            image_mask = image_mask.to(device=keep_prob.device, dtype=torch.bool)

        continuous_mask = torch.ones_like(keep_prob, dtype=keep_prob.dtype, device=keep_prob.device)
        continuous_mask = torch.where(image_mask, keep_prob, continuous_mask)
        continuous_mask[:, 0] = 1.0

        for batch_idx in range(continuous_mask.shape[0]):
            img_mask_row = image_mask[batch_idx]
            if not img_mask_row.any():
                continue
            if continuous_mask[batch_idx, img_mask_row].max() <= 0.0:
                image_keep_prob = keep_prob[batch_idx, img_mask_row]
                image_indices = torch.nonzero(img_mask_row, as_tuple=False).flatten()
                selected_index = int(image_indices[image_keep_prob.argmax()].item())
                continuous_mask[batch_idx, selected_index] = 1.0

        continuous_mask = torch.clamp(continuous_mask, 0.0, 1.0)
        return continuous_mask

    def _build_discrete_mask(self, keep_prob: torch.Tensor) -> torch.Tensor:
        image_mask = self._current_image_token_mask
        if image_mask is None:
            image_mask = torch.ones_like(keep_prob, dtype=torch.bool, device=keep_prob.device)
        else:
            image_mask = image_mask.to(device=keep_prob.device, dtype=torch.bool)

        discrete = torch.ones_like(keep_prob, dtype=torch.bool, device=keep_prob.device)
        candidate = keep_prob >= self.pruning_inference_threshold
        discrete = torch.where(image_mask, candidate, discrete)
        discrete[:, 0] = True

        for batch_idx in range(discrete.shape[0]):
            img_mask_row = image_mask[batch_idx]
            if not img_mask_row.any():
                continue
            if not discrete[batch_idx, img_mask_row].any():
                image_keep_prob = keep_prob[batch_idx, img_mask_row]
                image_indices = torch.nonzero(img_mask_row, as_tuple=False).flatten()
                selected_index = int(image_indices[image_keep_prob.argmax()].item())
                discrete[batch_idx, selected_index] = True

        return discrete

    def _compute_pruning_reward(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        actor_logits: torch.Tensor,
        keep_prob: torch.Tensor,
    ) -> torch.Tensor:
        if self.pruning_reward_fn is None:
            return self._default_pruning_reward_fn(layer_id, hidden_states, mask, actor_logits, keep_prob)
        reward = self.pruning_reward_fn(layer_id, hidden_states, mask, actor_logits, keep_prob)
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, device=hidden_states.device, dtype=hidden_states.dtype)
        return reward

    def _compute_rl_reward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_values = mask.to(hidden_states.dtype)
        batch_size = hidden_states.shape[0]
        rewards = []
        eps = 1e-12

        for batch_idx in range(batch_size):
            original_mask = image_mask[batch_idx]
            if not original_mask.any():
                rewards.append(torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32))
                continue

            gating_row = torch.where(
                original_mask,
                mask_values[batch_idx],
                torch.ones_like(mask_values[batch_idx]),
            ).to(hidden_states.dtype)

            hidden_before = hidden_states[batch_idx]
            hidden_after = hidden_before * gating_row.unsqueeze(-1)

            pre_matrix = hidden_before[original_mask].to(torch.float32)
            post_matrix = (hidden_before[original_mask] * gating_row[original_mask].unsqueeze(-1)).to(torch.float32)

            pre_log = self._log_singular_product(pre_matrix)
            post_log = self._log_singular_product(post_matrix)
            sv_ratio = torch.exp(torch.clamp(post_log - pre_log, min=-20.0, max=20.0))

            pre_attn_sum = self._attention_sum(hidden_before, original_mask)
            post_attn_sum = self._attention_sum(hidden_after, original_mask)
            attn_ratio = pre_attn_sum / (post_attn_sum + eps)

            post_count = gating_row[original_mask].sum()
            original_count = original_mask.sum()
            r2 = (post_count + eps) / (original_count.float().clamp_min(1.0))

            reward_total = sv_ratio + attn_ratio + r2
            rewards.append(reward_total)

        return torch.stack(rewards).to(hidden_states.dtype)

    def set_pruning_reward_hyperparams(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        if alpha is not None:
            self.pruning_reward_alpha = float(alpha)
        if beta is not None:
            self.pruning_reward_beta = float(beta)

    def _default_pruning_reward_fn(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        actor_logits: torch.Tensor,
        keep_prob: torch.Tensor,
    ) -> torch.Tensor:
        image_mask = self._current_image_token_mask
        if image_mask is None:
            return torch.zeros(hidden_states.shape[0], device=hidden_states.device, dtype=hidden_states.dtype)

        image_mask = image_mask.to(device=hidden_states.device, dtype=torch.bool)
        mask_values = mask.to(hidden_states.dtype)
        batch_size = hidden_states.shape[0]
        rewards = []
        eps = 1e-12

        for batch_idx in range(batch_size):
            original_mask = image_mask[batch_idx]
            if not original_mask.any():
                rewards.append(torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32))
                continue

            gating_row = torch.where(
                original_mask,
                mask_values[batch_idx],
                torch.ones_like(mask_values[batch_idx]),
            )
            gating_row = gating_row.to(hidden_states.dtype)

            original_count = original_mask.sum()

            hidden_before = hidden_states[batch_idx]
            hidden_after = hidden_before * gating_row.unsqueeze(-1)

            pre_matrix = hidden_before[original_mask].to(torch.float32)
            post_matrix = (hidden_before[original_mask] * gating_row[original_mask].unsqueeze(-1)).to(torch.float32)

            pre_log = self._log_singular_product(pre_matrix)
            post_log = self._log_singular_product(post_matrix)
            sv_ratio = torch.exp(torch.clamp(post_log - pre_log, min=-20.0, max=20.0))

            pre_attn_sum = self._attention_sum(hidden_before, original_mask)
            post_attn_sum = self._attention_sum(hidden_after, original_mask)
            attn_ratio = pre_attn_sum / (post_attn_sum + eps)

            r1 = sv_ratio + attn_ratio

            post_count = gating_row[original_mask].sum()
            r2 = (post_count + eps) / (original_count.float().clamp_min(1.0))

            reward_value = (
                self.pruning_reward_alpha * torch.log(r1 + eps)
                - self.pruning_reward_beta * torch.log(r2 + eps)
            )
            rewards.append(reward_value)

        reward_tensor = torch.stack(rewards).to(hidden_states.dtype)
        return reward_tensor

    def _log_singular_product(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.numel() == 0:
            return torch.tensor(0.0, device=matrix.device, dtype=torch.float32)
        # 数值稳定：使用 Gram 矩阵的 slogdet 代替 SVD，且强制在 FP32、禁用 autocast 下计算
        with torch.autocast("cuda", enabled=False):
            mat = matrix.to(torch.float32)
            gram = mat.transpose(0, 1) @ mat  # (D, D), PSD
            eps = 1e-6
            dim = gram.shape[-1]
            gram = gram + eps * torch.eye(dim, device=gram.device, dtype=gram.dtype)
            sign, logdet = torch.linalg.slogdet(gram)
            # 理论上 sign 应为正；若出现非正，退化为对角线近似，确保有限值
            safe_logdet = torch.where(
                sign > 0,
                logdet,
                torch.log(torch.clamp(torch.diagonal(gram), min=eps)).sum(),
            )
            return 0.5 * safe_logdet

    def _attention_sum(self, hidden_states: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        if image_mask is None or not image_mask.any():
            return torch.tensor(1.0, device=hidden_states.device, dtype=torch.float32)

        text_mask = ~image_mask
        if not text_mask.any():
            return torch.tensor(1.0, device=hidden_states.device, dtype=torch.float32)

        hidden_fp32 = hidden_states.to(torch.float32)
        scale = hidden_fp32.shape[-1] ** -0.5
        attn_scores = torch.matmul(hidden_fp32, hidden_fp32.transpose(0, 1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_image_to_text = attn_weights[image_mask][:, text_mask].sum()
        attn_text_to_image = attn_weights[text_mask][:, image_mask].sum()

        total = attn_image_to_text + attn_text_to_image + 1e-12
        return total

    def get_rl_statistics(self) -> Dict[str, Dict[int, torch.Tensor]]:
        return {
            "log_prob": self.latest_rl_log_probs,
            "ref_log_prob": self.latest_rl_ref_log_probs,
            "sampling_advantage": self.latest_rl_sampling_advantage,
            "prune_ratio": self.latest_pruning_ratios,
            "keep_prob": self.latest_keep_prob_current,
            "ref_keep_prob": self.latest_ref_keep_prob,
            "image_mask": self.latest_image_masks,
        }
