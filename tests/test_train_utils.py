from omegaconf import OmegaConf

from moe_transformer.train import build_model_config, get_lr, resolve_device


def make_training_cfg(**overrides):
    defaults = dict(
        learning_rate=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=10,
        max_steps=100,
    )
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def test_lr_warms_up_linearly():
    cfg = make_training_cfg()
    lr_start = get_lr(0, cfg)
    lr_mid_warmup = get_lr(5, cfg)
    lr_end_warmup = get_lr(9, cfg)
    assert lr_start < lr_mid_warmup < lr_end_warmup
    assert lr_end_warmup <= cfg.learning_rate


def test_lr_peaks_near_end_of_warmup_then_decays():
    cfg = make_training_cfg()
    lr_at_warmup_end = get_lr(cfg.warmup_steps, cfg)
    lr_later = get_lr(cfg.warmup_steps + 20, cfg)
    lr_at_max_steps = get_lr(cfg.max_steps, cfg)
    assert lr_at_warmup_end > lr_later > lr_at_max_steps


def test_lr_floors_at_min_lr_ratio():
    cfg = make_training_cfg()
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    lr_at_end = get_lr(cfg.max_steps, cfg)
    lr_past_end = get_lr(cfg.max_steps + 50, cfg)
    assert abs(lr_at_end - min_lr) < 1e-8
    assert abs(lr_past_end - min_lr) < 1e-8


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu", "dense") == "cpu"
    assert resolve_device("cpu", "moe") == "cpu"


def test_resolve_device_auto_never_picks_mps_for_moe():
    # Regardless of what's actually available on this machine, "auto" must
    # never resolve to "mps" for a MoE model -- see moe.py's docstring on
    # why MPS is pathological for the capacity-dispatch gather/scatter.
    device = resolve_device("auto", "moe")
    assert device != "mps"
    assert device in ("cpu", "cuda")


def test_build_model_config_strips_kind_and_maps_fields():
    model_cfg = OmegaConf.create(
        {
            "kind": "moe",
            "vocab_size": 100,
            "n_embd": 32,
            "n_head": 2,
            "n_layer": 2,
            "block_size": 16,
            "dropout": 0.0,
            "bias": False,
            "ffn_hidden_dim": None,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "num_experts": 4,
            "top_k": 2,
            "capacity_factor": 1.25,
            "expert_ffn_hidden_dim": None,
            "aux_loss_weight": 0.01,
            "router_z_loss_weight": 0.001,
        }
    )
    config = build_model_config(model_cfg)
    assert config.vocab_size == 100
    assert config.n_embd == 32
    assert config.num_experts == 4
    assert not hasattr(config, "kind")
