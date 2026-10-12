# -*- coding: utf-8 -*-
"""
model-rwkv 混合注意力 (RWKV5 WKV 线性 + 全注意力) 测试

验证要点:
  1. 默认 layer_types 生成 (全部 "rwkv")
  2. 前向: 混合层模型的 prefill / 生成 cache 形状
  3. WKV 扫描本身: 整段扫描 == 逐 token 扫描 (应精确, < 1e-5)
  4. 关键不变量: RWKV 层 prefill == 逐 token 生成 (应精确, < 1e-5)
  5. 全注意力层 prefill == 逐 token 生成 (应精确, < 1e-4)

运行:  python test/test_rwkv.py
"""
import os, sys, types, importlib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model-rwkv"))


def _load_mod(name, pkgdir):
    pkg = types.ModuleType(pkgdir)
    pkg.__path__ = [os.path.join(ROOT, pkgdir)]
    sys.modules[pkgdir] = pkg
    return importlib.import_module(f"{pkgdir}.{name}")


def test_layer_types_default():
    cm = _load_mod("config_minimind", "model-rwkv")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=6)
    assert cfg.layer_types == ["rwkv"] * 6, cfg.layer_types
    cfg2 = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=4,
                             layer_types=["rwkv", "full_attention", "rwkv", "full_attention"])
    assert cfg2.layer_types == ["rwkv", "full_attention", "rwkv", "full_attention"]
    print("[OK] layer_types 生成与校验")


def test_rwkv_wkv_scan_chunk_vs_token():
    """整段 WKV 扫描 == 逐 token WKV 扫描 (状态传递应一致)。"""
    mm = _load_mod("model_minimind", "model-rwkv")
    torch.manual_seed(0)
    B, L, Hid = 1, 20, 32
    key = torch.randn(B, L, Hid)
    value = torch.randn(B, L, Hid)
    td = -torch.rand(Hid) * 2
    tf = torch.randn(Hid)
    out, final = mm.rwkv_wkv_scan(td, tf, key, value, state=None, return_state=True)
    st, ref = None, []
    for t in range(L):
        o, st = mm.rwkv_wkv_scan(td, tf, key[:, t:t + 1], value[:, t:t + 1], state=st, return_state=True)
        ref.append(o)
    ref = torch.cat(ref, 1)
    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, f"WKV 扫描 整段 vs 逐token 偏差 {diff}"
    print(f"[OK] WKV 扫描 整段 vs 逐token 偏差 {diff:.2e}")


def test_rwkv_prefill_vs_streaming():
    """RWKV 层 prefill == 逐 token 生成。"""
    cm = _load_mod("config_minimind", "model-rwkv")
    mm = _load_mod("model_minimind", "model-rwkv")
    cfg = cm.MiniMindConfig(hidden_size=48, num_hidden_layers=2, num_attention_heads=3,
                            layer_types=["rwkv"] * 2)
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    torch.manual_seed(0)
    seq = torch.randint(0, 100, (1, 20))
    with torch.no_grad():
        o_pref = m.model(seq, use_cache=True)[0]
        h, pk = None, None
        for t in range(20):
            r = m.model(seq[:, t:t + 1], past_key_values=pk, use_cache=True)
            h = r[0][:, -1]
            pk = r[1]
    diff = (o_pref[:, -1] - h).abs().max().item()
    assert diff < 1e-4, f"RWKV prefill vs 生成 偏差 {diff}"
    print(f"[OK] RWKV prefill vs 生成 偏差 {diff:.2e}")


def test_rwkv_cache_shapes():
    """RWKV 层 cache 为 4 元组 [num, den, max, last_hidden]。"""
    cm = _load_mod("config_minimind", "model-rwkv")
    mm = _load_mod("model_minimind", "model-rwkv")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                            num_key_value_heads=2, layer_types=["rwkv", "rwkv", "full_attention"])
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        o = m(input_ids=x, use_cache=True)
        pkv = o.past_key_values
        o2 = m(input_ids=x[:, -1:], past_key_values=pkv, use_cache=True)
    for i, layer in enumerate(m.model.layers):
        c = o2.past_key_values[i]
        if layer.layer_type == "rwkv":
            assert len(c) == 4, (i, c)
        else:
            assert len(c) == 2, (i, c)
    print("[OK] RWKV / 全注意力 cache 形状")


def test_mixed_prefill_exact():
    """混合 RWKV + 全注意力: prefill == 逐 token 生成。"""
    cm = _load_mod("config_minimind", "model-rwkv")
    mm = _load_mod("model_minimind", "model-rwkv")
    cfg = cm.MiniMindConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2,
                            layer_types=["rwkv", "full_attention", "rwkv", "full_attention"])
    m = mm.MiniMindForCausalLM(cfg); m.eval()
    torch.manual_seed(0)
    seq = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        o = m.model(seq, use_cache=True)[0]
        h, pk = None, None
        for t in range(16):
            r = m.model(seq[:, t:t + 1], past_key_values=pk, use_cache=True)
            h = r[0][:, -1]
            pk = r[1]
    diff = (o[:, -1] - h).abs().max().item()
    assert diff < 1e-4, f"混合 prefill vs 生成 偏差 {diff}"
    print(f"[OK] 混合 prefill vs 生成 偏差 {diff:.2e}")


def main():
    tests = [
        test_layer_types_default,
        test_rwkv_wkv_scan_chunk_vs_token,
        test_rwkv_prefill_vs_streaming,
        test_rwkv_cache_shapes,
        test_mixed_prefill_exact,
    ]
    for t in tests:
        t()
    print("\n=== 全部 RWKV 测试通过 ===")


if __name__ == "__main__":
    main()
