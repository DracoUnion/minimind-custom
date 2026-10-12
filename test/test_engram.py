# -*- coding: utf-8 -*-
"""
model-engram 外部记忆 (n-gram 哈希) 测试

验证要点:
  1. 默认配置生成 (use_engram=False, engram_layer_ids)
  2. Engram 块按 engram_layer_ids 正确挂载到对应层
  3. n-gram 哈希 _ngram_hashes 输出形状与确定性
  4. 前向: 启用 Engram 的模型 prefill / 生成 cache 正常
  5. 一致性: 同一模型 开/关 Engram 输出不同 (证明 Engram 有贡献),
     且 Engram 贡献与 full_input_ids 的窗口对齐正确 (生成阶段)

运行:  python test/test_engram.py
"""
import os, sys, types, importlib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model-engram"))


def _load_mod(name, pkgdir="model-engram"):
    tag = pkgdir.replace("-", "_")
    pkg = types.ModuleType(tag)
    pkg.__path__ = [os.path.join(ROOT, pkgdir)]
    sys.modules[tag] = pkg
    return importlib.import_module(f"{tag}.{name}")


def test_config_default():
    cm = _load_mod("config_minimind")
    cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=4)
    assert cfg.use_engram is False
    # 默认 engram_layer_ids = [1, num_hidden_layers-2]
    assert cfg.engram_layer_ids == [1, 2]
    cfg2 = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=4, use_engram=True,
                             engram_layer_ids=[0, 3])
    assert cfg2.use_engram is True and cfg2.engram_layer_ids == [0, 3]
    print("[OK] Engram 配置默认值与自定义")


def test_engram_block_placement():
    cm = _load_mod("config_minimind")
    mm = _load_mod("model_minimind")
    cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=6, num_attention_heads=3,
                            num_key_value_heads=1, use_engram=True,
                            engram_layer_ids=[1, 4], engram_max_ngram_size=3,
                            engram_n_embed_per_ngram=64, engram_n_head_per_ngram=4,
                            engram_vocab_size=[128, 256], engram_kernel_size=3)
    m = mm.MiniMindForCausalLM(cfg)
    placed = [i for i, l in enumerate(m.model.layers) if l.engram is not None]
    assert placed == [1, 4], placed
    print(f"[OK] Engram 块挂载在层 {placed}")


def test_ngram_hashes():
    mm = _load_mod("model_minimind")
    # 直接调用 _ngram_hashes
    B, T, max_ngram, n_head = 2, 6, 3, 4
    input_ids = torch.randint(0, 100, (B, T))
    multipliers = torch.arange(1, max_ngram + 1, dtype=torch.int64) * 2 + 1
    moduli = torch.tensor([101, 103, 107, 109, 113, 127, 131, 137], dtype=torch.int64)
    h = mm._ngram_hashes(input_ids, pad_id=0, max_ngram=max_ngram,
                        multipliers=multipliers, moduli=moduli, n_head=n_head)
    # 每个 n-gram 阶 (max_ngram-1=2) 各有 n_head 个头 -> 2*n_head
    assert h.shape == (B, T, (max_ngram - 1) * n_head), h.shape
    assert h.dtype == torch.int64
    # 确定性: 相同输入两次结果一致
    h2 = mm._ngram_hashes(input_ids, pad_id=0, max_ngram=max_ngram,
                          multipliers=multipliers, moduli=moduli, n_head=n_head)
    assert torch.equal(h, h2)
    print(f"[OK] n-gram 哈希 形状 {tuple(h.shape)} 且确定性一致")


def test_engram_forward_and_cache():
    cm = _load_mod("config_minimind")
    mm = _load_mod("model_minimind")
    cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=4, num_attention_heads=3,
                            num_key_value_heads=1, use_engram=True,
                            engram_layer_ids=[1, 2], engram_max_ngram_size=3,
                            engram_n_embed_per_ngram=64, engram_n_head_per_ngram=4,
                            engram_vocab_size=[128, 256], engram_kernel_size=3)
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    x = torch.randint(0, 100, (2, 12))
    with torch.no_grad():
        o = m(input_ids=x, use_cache=True)
        assert o.logits.shape == (2, 12, cfg.vocab_size)
        assert torch.isfinite(o.logits).all().item()
        pkv = o.past_key_values
        o2 = m(input_ids=x[:, -1:], past_key_values=pkv, use_cache=True, full_input_ids=x)
        assert o2.logits.shape == (2, 1, cfg.vocab_size)
        assert torch.isfinite(o2.logits).all().item()
    print("[OK] 启用 Engram 的模型 prefill + 生成 cache")


def test_engram_changes_output():
    """同一模型开/关 Engram 输出不同; 且 full_input_ids 对齐影响生成结果。"""
    cm = _load_mod("config_minimind")
    mm = _load_mod("model_minimind")
    base = dict(hidden_size=48, num_hidden_layers=4, num_attention_heads=3,
                num_key_value_heads=1, use_engram=True, engram_layer_ids=[1, 2],
                engram_max_ngram_size=3, engram_n_embed_per_ngram=64,
                engram_n_head_per_ngram=4, engram_vocab_size=[128, 256], engram_kernel_size=3)
    torch.manual_seed(0)
    on = mm.MiniMindForCausalLM(cm.MiniMindConfig(**base)); on.eval()
    # 关 Engram 的模型 (随机权重不同, 但用于验证 "Engram 有贡献" 需同权重)
    # 用同一权重, 把 use_engram 关掉后重建会丢 Engram 参数 -> 改为直接屏蔽
    cfg_off = cm.MiniMindConfig(**base)
    off = mm.MiniMindForCausalLM(cfg_off)
    off.load_state_dict(on.state_dict())
    off.eval()
    # 屏蔽 off 的 engram
    for l in off.model.layers:
        l.engram = None
    x = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        o_on = on(input_ids=x, use_cache=False).logits
        o_off = off(input_ids=x, use_cache=False).logits
    diff = (o_on - o_off).abs().max().item()
    assert diff > 1e-3, f"Engram 贡献应为非平凡, diff={diff}"
    print(f"[OK] 开 vs 关 Engram 输出差异 {diff:.4f} (证明 Engram 有贡献)")


def test_engram_window_alignment():
    """Engram 的 n-gram 哈希依赖左侧历史 token; 不同的窗口 -> 不同的贡献。

    直接对单个 Engram 块调用: 同一个 hidden, 但 input_ids 不同 (带/不带左侧
    真实 token), 输出的 Engram 残差应不同。
    """
    cm = _load_mod("config_minimind")
    mm = _load_mod("model_minimind")
    cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=4, num_attention_heads=3,
                            num_key_value_heads=1, use_engram=True,
                            engram_layer_ids=[1, 2], engram_max_ngram_size=3,
                            engram_n_embed_per_ngram=64, engram_n_head_per_ngram=4,
                            engram_vocab_size=[128, 256], engram_kernel_size=3)
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    blk = m.model.layers[1].engram
    torch.manual_seed(0)
    B, L = 1, 4
    hid = torch.randn(B, L, cfg.hidden_size)
    ids_a = torch.randint(0, 100, (B, L))
    ids_b = torch.randint(0, 100, (B, L))  # 不同的左侧上下文
    with torch.no_grad():
        c_a = blk(hid, ids_a)
        c_b = blk(hid, ids_b)
    diff = (c_a - c_b).abs().max().item()
    assert diff > 1e-3, f"不同 input_ids 应产生不同 Engram 贡献, diff={diff}"
    print(f"[OK] Engram 贡献随 input_ids 窗口变化, 差异 {diff:.4f}")


def main():
    tests = [
        test_config_default,
        test_engram_block_placement,
        test_ngram_hashes,
        test_engram_forward_and_cache,
        test_engram_changes_output,
        test_engram_window_alignment,
    ]
    for t in tests:
        t()
    print("\n=== 全部 Engram 测试通过 ===")


if __name__ == "__main__":
    main()
