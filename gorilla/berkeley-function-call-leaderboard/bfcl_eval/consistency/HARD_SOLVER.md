# TradingBotHard 存在性求解器

`trading_solver_hard.check_trace` 检查是否存在一个合法初始状态及随机时间选择，
使整条工具调用和 observation 链成立。不要求唯一解，也不要求恢复生成时的状态。

## 明确的语义范围

- 仅 `long_context=False`，后端源码不变。
- 初始余额、持仓非负；持仓和订单数量为整数，订单数量、价格为正。
  订单类型为 Buy/Sell，状态为 Pending/Open/Completed/Cancelled。
  初始交易记录类型为 deposit/withdrawal，金额为正。
- `get_holdings` 观测中，缺失的股票键与值为整数 `0` 的股票键等价。
  SMT 约束和后端见证回放都采用此规则；非零股数仍须一致，其他工具和字段不放宽。
  后端本身不变，卖空时仍删除键；SAT 不要求零值键的呈现方式与后端逐字一致。
- 价格、金额和涨跌幅使用有限 binary64 数字，输入最多三位有效小数。
  容忍不超过 `1e-9` 的浮点尾差；真正的 `0.001` 差异不忽略。
  不符合格式的输入返回 `invalid_format`，不计为轨迹无解。
- 默认完全排除 `get_current_time`、`get_symbol_by_name`、`get_available_stocks`。
  排除包含格式预检查、状态收集、约束和回放比对；这些调用不修改可变状态。
- 随机交易时间每次生成一个存在量。不要求复现 Python 随机种子或代码固定日期。
  记录重复出现时必须一致，日期过滤必须成立。格式为 `%Y-%m-%d %H:%M:%S`。
  `timestamp_policy="backend"` 可恢复后端的一天日期范围，但仍不求随机种子。

输入：

```json
{"steps": [{"tool": "get_account_info", "args": {},
            "observation": {"account_id": 7, "balance": 100.125, "binding_card": 99}}]}
```

工具名允许 `TradingBot.` 前缀。args 按工具签名检查参数名、必需参数和类型。
observation 可以是正常对象、业务错误对象，或后端历史工具使用的登录错误数组。
数值字段不能是字符串、布尔值或 NaN/Infinity；日期参数必须可解析。
输入格式验证实现位于 `trading_trace_format.py`。

## 实现变化

1. **未知订单联合求解**：同一订单的符号、方向、限价、数量、初始状态，以及
   是否已有成交价，与余额和持仓共同求解。没有详情读取也可以有解。
2. **订单键存在性与顺序**：每个相关 ID 是否存在、初始字典顺序均为变量。
   新建 ID 的单调性和计数器跳号由约束检查，已观察到的订单列表必须精确匹配。
3. **watchlist**：初始长度、元素、顺序参与求解，保留重复元素和删除首个匹配项语义。
   不再使用旧版少量候选列表。有限长度界来自所有观测和删除次数：未进入任何
   观测、也未被删除或用于成员判断的元素可以移除而不影响可执行性。
4. **交易历史**：用记录出现次数、相对顺序、时间过滤及追加操作建模。
   不再拒绝 1000 条记录。相同记录的初始数量无需超过任一观测中的最大次数；
   从未出现的初始记录可删去，因而无需猜测任意大的初始列表。
5. **浮点**：使用 Z3 binary64 运算，消除旧版整数余额近似和 `<=` 余额不足放宽。
   `trading_float.py` 对 `round(x, 2/3)` 按浮点有效位做十进制 ties-to-even 舍入，
   再转换回 binary64；不是错误的 `round(x * 100) / 100` 近似。
6. **持仓键**：后端键存在性仍按真实操作建模，但观测只约束股数。
   零持仓键可省略或保留，见证不承诺复现输入中的零键布局；回放按同一等价规则比较。
7. **业务错误**：检查进入分支的条件和实际错误文本，包括 `.2f`、大小写、
   缺失订单错误中的 ID 列表。股票名含逗号时，通知按后端字符串拼接规则处理。

性能优化会利用已确定的初始余额、不可变价格、比较区间的代表值，以及先尝试
容易验证的具体见证。具体见证猜测失败或超时后仍求解完整约束，不能据此返回
`unsat`。完整求解没有三秒超时。初始状态搜索与后端回放均不调用 LLM。

## 结果与保证边界

- `sat`：约束可满足，且求出的见证通过真实后端逐步回放。
  `witness_state` 是完整可加载初始状态；`random_choices.transaction_timestamps`
  是需要注入的随机时间。兼容旧接口的 `initial_state` 字段仍是摘要。
- `unsat`：当前建模约束不可满足，附带冲突步骤集合。
- `invalid_format`：不属于声明的输入格式，不能作为 WM 不自洽结果。
- `unknown`：底层求解器没有完成证明，或见证回放与模型不符。这属于求解器诊断，
  不能算作 WM 无解。无默认超时不等于保证任意规模输入在有限资源内迅速完成。

**目前没有对整个 Python 后端到 SMT 模型的等价性做机器检查的形式化证明。**
下述测试通过不能证明“任何合法输入都 100% 正确”，不能把该实现宣传为已认证的
总判定器，也不能删掉诊断分支来伪装成只有 sat/unsat。特别大的输入、极端数值和
底层求解器行为仍需要审计。新发现的反例应先保存为回归，再修复模型。

## 运行与验证

从 `gorilla/berkeley-function-call-leaderboard`：

```powershell
python -m bfcl_eval.consistency.trading_solver_hard trace.json
python -m bfcl_eval.consistency.trading_solver_hard trace.json --no-check-static-tools
python -m bfcl_eval.consistency.trading_solver_hard trace.json --check-static-tools --timestamp-policy backend
python -m unittest tests.test_trading_consistency_hard tests.test_trading_consistency_adversarial -v
```

```python
from bfcl_eval.consistency.trading_solver_hard import check_trace
result = check_trace(steps, check_static_tools=False, timestamp_policy="symbolic")
```

独立批量审计会只取生成数据的 tool/args/result，不输入生成 initial_state、分支标签
或状态快照，并保存源码 SHA-256、运行环境、逐例结果和完整见证：

```powershell
python -m bfcl_eval.consistency.audit_trading_solver_hard bfcl_eval/consistency/data_v2/trading_bot_hard/grounded_20261001_124001_32k --output bfcl_eval/consistency/audit_results/trading_solver_hard_20261001.json
```

本轮验证包括：

- 原始 45 条 grounded 轨迹只输入调用与 observation，全部 sat 并通过回放。
- 384 条 watchlist 三次操作组合（包括重复元素），全部为后端生成的正例。
- 40 条随机资金/成交正例及 40 条余额修改 0.001 的反例。
- 30 条混合成功和业务错误的后端轨迹。
- 1000 条不同历史记录、1000 条重复记录及重叠日期过滤。
- 3212 次舍入表达式与 Python `round` 的比较。
- 固定工具开关、随机时间重复读取、未知订单联合约束、生命周期、ID 跳号和顺序、
  缺失订单错误文本、零持仓键、浮点分支边界等专项回归。

这是一组回归证据，不是任意 WM 输出上的统计正确率。
