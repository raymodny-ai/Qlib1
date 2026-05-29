# Qlib1 美股量化分析系统 — 前端集成 API 文档

> **版本:** 1.0.0  
> **基础地址:** `http://localhost:8000`  
> **交互式文档:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)  
> **文档生成:** 2026-05-19  
> **目标读者:** 前端 UI 开发团队

---

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [API 端点速查表](#2-api-端点速查表)
3. [用户认证与 RBAC](#3-用户认证与-rbac)
4. [数据服务端点](#4-数据服务端点)
5. [预测与评分端点](#5-预测与评分端点)
6. [策略配置与组合端点](#6-策略配置与组合端点)
7. [回测端点](#7-回测端点)
8. [报告与风险指标端点](#8-报告与风险指标端点)
9. [PM 熔断门控端点](#9-pm-熔断门控端点)
10. [合规审计端点](#10-合规审计端点)
11. [系统运维端点](#11-系统运维端点)
12. [前端 UI 集成指南](#12-前端-ui-集成指南)
13. [数据模型参考](#13-数据模型参考)
14. [错误处理规范](#14-错误处理规范)

---

## 1. 系统架构概览

### 1.1 核心分层架构

```
┌─────────────────────────────────────────────────────┐
│                  Frontend Web UI                      │
│    (React / Vue / Angular → Tableau / Power BI)      │
├─────────────────────────────────────────────────────┤
│                RESTful API 网关层                     │
│   FastAPI + RBAC + TLS 1.2+ + 请求指标追踪           │
├──────────┬──────────┬───────────┬───────────────────┤
│ 数据服务 │ 策略引擎 │ 回测引擎  │ 安全合规           │
│ DataServer│Strategies│Portfolio │ RBAC/审计/PM Gate  │
│ 缓存监控  │ ML Pipeline│绩效报告 │ TLS/AES-256-GCM   │
├──────────┴──────────┴───────────┴───────────────────┤
│       底层数据层 (Qlib .bin / PIT / 外部API)          │
└─────────────────────────────────────────────────────┘
```

### 1.2 关键设计原则

| 原则 | 说明 |
|------|------|
| **职责分离** | Analyzers 生成预测分数，Strategies 专职持仓映射 |
| **时间隔离** | 严格时间序列划分，杜绝未来数据泄露 |
| **防篡改审计** | 密码学哈希链 (HMAC-SHA256)，全链路可追溯 |
| **最小特权** | RBAC 五角色体系，PM 拥有一键熔断权 |
| **缓存优先** | 三级缓存 (Global → Expression → Dataset)，目标命中率 ≥ 80% |

---

## 2. API 端点速查表

| 方法 | 路径 | 标签 | 权限 | 描述 |
|------|------|------|------|------|
| `GET` | `/health` | System | 无 | 系统健康检查 |
| `GET` | `/api/v1/instruments` | Data | 无 | 可用证券列表 |
| `POST` | `/api/v1/factors/{dataset}` | Data | 无 | 查询因子数据 |
| `POST` | `/api/v1/predict` | Prediction | `model:read` | 模型预测 |
| `GET` | `/api/v1/scores` | Analysis | `model:read` | 横截面评分排名 |
| `GET` | `/api/v1/risk` | Analysis | `experiment:read` | 风险指标 |
| `GET` | `/api/v1/portfolio/{strategy_id}` | Portfolio | 无 | 查询组合权重 |
| `POST` | `/api/v1/backtest` | Backtest | `experiment:submit` | 提交回测任务 |
| `GET` | `/api/v1/backtest/{task_id}` | Backtest | 无 | 查询回测状态/结果 |
| `GET` | `/api/v1/report/{experiment_id}` | Report | `report:read` | 查询实验绩效报告 |
| `GET` | `/api/v1/metrics` | Operations | `experiment:read` | 系统性能指标 |
| `GET` | `/api/v1/gate/status` | PM Gate | 无 | 查询门控状态 |
| `POST` | `/api/v1/gate/emergency-stop` | PM Gate | `signal:emergency_stop` | PM 一键熔断 |
| `POST` | `/api/v1/gate/emergency-reopen` | PM Gate | `signal:emergency_stop` | PM 恢复放行 |
| `POST` | `/api/v1/gate/global-emergency-stop` | PM Gate | `signal:emergency_stop` | PM 全局紧急熔断 |
| `POST` | `/api/v1/gate/global-emergency-reopen` | PM Gate | `signal:emergency_stop` | PM 全局恢复放行 |
| `GET` | `/api/v1/gate/history` | PM Gate | 无 | 查询门控操作历史 |
| `GET` | `/api/v1/audit/logs` | Compliance | `audit:read` | 查询审计日志 |
| `GET` | `/api/v1/audit/verify-chain` | Compliance | `audit:read` | 验证审计哈希链 |
| `POST` | `/api/v1/compliance/sox-report` | Compliance | `compliance:export` | 生成 SOX 合规报告 |
| `GET` | `/api/v1/compliance/status` | Compliance | `compliance:review` | 系统合规状态 |
| `GET` | `/api/v1/audit/export` | Compliance | `audit:export` | 导出审计报告 |

---

## 3. 用户认证与 RBAC

### 3.1 认证方式

API 通过请求头传递用户身份（开发环境简化认证）：

| 优先级 | 方式 | 示例 |
|--------|------|------|
| 1 (最高) | `X-User-ID` 请求头 | `X-User-ID: admin` |
| 2 | `X-API-Key` 请求头 | `X-API-Key: sk-xxxx` |
| 3 | Query 参数 `?user=` | `?user=researcher` |
| 4 (最低) | 默认 `anonymous` | — |

**前端请求示例:**
```http
GET /api/v1/scores HTTP/1.1
Host: localhost:8000
X-User-ID: researcher
```

### 3.2 角色与权限矩阵

| 角色 | 用户 ID | 核心权限 | 适用页面 |
|------|---------|----------|----------|
| **Quant Researcher** | `researcher` | 因子编写、模型训练、数据读取、查看报告 | 因子实验室、模型训练页 |
| **Portfolio Manager** | `pm` | 信号审批、一键熔断、模型部署、风险配置 | PM 仪表盘、门控面板 |
| **Data Admin** | (未预设) | 数据管理、API 配置、用户管理 | 数据管理后台 |
| **Compliance Auditor** | `auditor` | 审计日志只读、合规报告导出 | 合规审计页 |
| **System Admin** | `admin` | 全部权限 (`*` 通配符) | 系统管理控制台 |

### 3.3 权限粒度

```python
# 关键权限常量 (前端按需控制 UI 可见性)
model:read          # 查看模型 → 显示"模型列表"页签
model:train         # 训练模型 → 显示"训练"按钮
model:deploy        # 部署模型 → 显示"部署"按钮
experiment:submit   # 提交实验 → 显示"提交回测"按钮
signal:emergency_stop  # 一键熔断 → 显示"紧急熔断"按钮（仅 PM）
report:read         # 查看报告 → 显示"绩效报告"页签
audit:read          # 审计日志 → 显示"审计日志"页签
compliance:export   # 导出合规报告 → 显示"导出"按钮
```

### 3.4 前端 UI 控制建议

```javascript
// 前端权限判断示例
function canAccess(user, permission) {
  // 调用 GET /api/v1/gate/status 获取当前用户角色
  // 根据角色预置权限表判断
  const rolePermissions = {
    'researcher': ['model:read', 'model:train', 'data:read', 'report:read'],
    'pm': ['model:read', 'model:deploy', 'signal:emergency_stop', 'risk:read'],
    'auditor': ['audit:read', 'compliance:export', 'report:read'],
    'admin': ['*']
  };
  const perms = rolePermissions[user.role] || [];
  return perms.includes('*') || perms.includes(permission);
}
```

---

## 4. 数据服务端点

### 4.1 `GET /health` — 系统健康检查

**权限:** 无  
**用途:** 前端心跳检测、负载均衡健康探针

**响应 `200 OK`:**
```json
{
  "status": "healthy",
  "service": "qlib-us-fundamental",
  "version": "1.0.0",
  "timestamp": "2026-05-19T10:30:00.000Z",
  "uptime_seconds": 12345.6
}
```

**前端集成:** 每 30s 轮询，若 `status != "healthy"` 则显示红色告警横幅。

---

### 4.2 `GET /api/v1/instruments` — 证券列表

**权限:** 无  
**用途:** 填充前端证券选择器/搜索下拉框

**Query 参数:**
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `sector` | string | `null` | 行业筛选 (如 "Technology") |
| `limit` | int | 100 (1-1000) | 返回数量上限 |

**响应 `200 OK`:**
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "industry": null,
    "market_cap": 3000000000000.0
  },
  {
    "symbol": "MSFT",
    "name": "Microsoft Corp.",
    "sector": "Technology",
    "industry": null,
    "market_cap": 2800000000000.0
  }
]
```

**前端集成建议:** 使用虚拟滚动列表渲染大型证券池，前端做本地模糊搜索缓存以减少 API 调用。

---

### 4.3 `POST /api/v1/factors/{dataset}` — 查询因子数据

**权限:** 无  
**用途:** 获取指定股票池的多维因子矩阵，用于前端表格/图表展示

**Path 参数:**
| 参数 | 说明 |
|------|------|
| `dataset` | 数据集名称 (默认 `"fundamentals"`) |

**Request Body:**
```json
{
  "instruments": ["AAPL", "MSFT", "GOOGL"],
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "fields": ["close", "volume", "pe_ratio", "roe", "market_cap"],
  "dataset": "fundamentals"
}
```

**字段约束:** `instruments` 1-500 项, `start_date`/`end_date` 必须为 `YYYY-MM-DD`

**响应 `200 OK`:**
```json
{
  "dataset": "fundamentals",
  "instruments": ["AAPL", "MSFT", "GOOGL"],
  "date_range": {"start": "2025-01-01", "end": "2025-12-31"},
  "n_rows": 750,
  "n_fields": 5,
  "data": [
    {
      "instrument": "AAPL",
      "date": "2025-01-02",
      "close": 185.5,
      "volume": 55000000.0,
      "pe_ratio": 28.3,
      "roe": 0.45,
      "market_cap": 3000000000000.0
    }
  ]
}
```

**重要:** 当后端 DataServer 不可用时，返回**降级模拟数据**（随机值），前端应通过检查 `n_rows` 或字段值分布判断数据真实性。

**前端集成:** 适用于"数据浏览器"页面，用 Ag-Grid / Handsontable 渲染多维表格，日期列作为 X 轴筛选器。

---

## 5. 预测与评分端点

### 5.1 `POST /api/v1/predict` — 模型预测

**权限:** `model:read`  
**用途:** 提交股票池和日期，返回模型预测得分和排名

**Request Body:**
```json
{
  "model_name": "lightgbm",
  "instruments": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
  "date": "2025-06-15",
  "factors": {
    "AAPL": {"close": 185.5, "volume": 55000000, "pe_ratio": 28.3},
    "MSFT": {"close": 420.1, "volume": 23000000, "pe_ratio": 35.1}
  }
}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model_name` | string | ✅ | 模型名称 (lightgbm / xgboost) |
| `instruments` | string[] | ✅ | 1-500 支股票代码 |
| `date` | string | ✅ | 预测日期 `YYYY-MM-DD` |
| `factors` | object | ❌ | 因子数据 (不传时后端从 DataServer 自动加载) |

**响应 `200 OK`:**
```json
{
  "model_name": "lightgbm",
  "date": "2025-06-15",
  "timestamp": "2026-05-19T10:30:00.000Z",
  "predictions": [
    {"instrument": "NVDA",  "score": 0.0523, "rank": 1},
    {"instrument": "MSFT",  "score": 0.0381, "rank": 2},
    {"instrument": "AAPL",  "score": 0.0214, "rank": 3},
    {"instrument": "GOOGL", "score": 0.0156, "rank": 4},
    {"instrument": "AMZN",  "score": -0.0042, "rank": 5}
  ]
}
```

**前端集成:** 适合"实时预测"面板。以柱状图展示 Top/Bottom N 得分，绿色正分、红色负分。排名用 `<table>` 渲染。

**权限拒绝响应 `403`:**
```json
{
  "detail": "权限拒绝: user=anonymous, role=unknown, required=model:read"
}
```

---

### 5.2 `GET /api/v1/scores` — 横截面评分排名

**权限:** `model:read`  
**用途:** 获取全市场预测得分排序结果，供 BI 工具 (Tableau/Power BI) 可视化

**Query 参数:**
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `date` | string | 最新 | 评分日期 `YYYY-MM-DD` |
| `model_name` | string | `"lightgbm"` | 模型名称 |
| `limit` | int | 50 (1-500) | 返回数量上限 |

**响应 `200 OK`:**
```json
{
  "date": "2025-06-15",
  "model_name": "lightgbm",
  "total_instruments": 500,
  "scores": [
    {"instrument": "NVDA",  "score": 0.0523, "rank": 1,  "percentile": 99.8},
    {"instrument": "META",  "score": 0.0481, "rank": 2,  "percentile": 99.6},
    {"instrument": "MSFT",  "score": 0.0381, "rank": 3,  "percentile": 99.4}
  ]
}
```

`percentile` 计算: `(total - rank) / total * 100`，值越大越优秀。

**前端集成:** 这是**外部 BI 集成的核心端点**。Tableau 可直接将此 JSON 作为 Web Data Connector 数据源。前端热力图: X 轴=行业分组，Y 轴=得分排名，颜色=百分位 (渐变绿→红)。

---

## 6. 策略配置与组合端点

### 6.1 策略类型说明

| 策略 ID | 类名 | 描述 | 适用场景 |
|---------|------|------|----------|
| `topk_dropout` | TopkDropoutStrategy | 动量淘汰策略：持 TopK，跌出阈值强制卖出 | 主动量化选股 |
| `equal_weight` | EqualWeightStrategy | 等权配置策略 | 基准对比 (Benchmark) |
| `score_weight` | ScoreWeightStrategy | 预测得分加权 | 高信心集中配置 |

### 6.2 `GET /api/v1/portfolio/{strategy_id}` — 查询组合权重

**权限:** 无  
**用途:** 获取指定日期策略的目标持仓权重

**Path 参数:**
| 参数 | 说明 |
|------|------|
| `strategy_id` | 策略 ID (topk_dropout / equal_weight / score_weight) |

**Query 参数:**
| 参数 | 类型 | 必需 | 格式 |
|------|------|------|------|
| `date` | string | ✅ | `YYYY-MM-DD` |

**响应 `200 OK`:**
```json
{
  "strategy_id": "topk_dropout",
  "date": "2025-06-15",
  "n_holdings": 30,
  "total_weight": 0.983,
  "holdings": [
    {"instrument": "AAPL", "weight": 0.08, "score": 0.045},
    {"instrument": "MSFT", "weight": 0.07, "score": 0.042},
    {"instrument": "GOOGL", "weight": 0.06, "score": 0.038}
  ]
}
```

**前端集成:** 用饼图/环形图展示持仓分布，表格展示权重明细。可用 `<Treemap>` 展示市值区间。

### 6.3 策略配置参数 (前端配置表单)

当用户在前端创建/编辑策略时，需映射以下参数:

```typescript
interface StrategyConfig {
  top_k: number;            // 持仓股票数 (默认 30)
  min_k: number;            // 最少持仓数 (默认 10)
  dropout_threshold: number; // 淘汰阈值 (默认 0.2, 20%)
  rebalance_freq: number;   // 调仓频率/交易日 (默认 1)
  weight_method: 'equal' | 'score' | 'score_sqrt' | 'rank' | 'inv_vol';
  turnover_limit: number;   // 单日换手率上限 (默认 0.5)
  max_weight_per_stock: number; // 单票最大权重 (默认 0.1)
  commission_rate: number;  // 佣金费率 (默认 0.001)
  slippage_bps: number;     // 滑点/基点 (默认 1.0)
  max_drawdown_limit: number; // 最大回撤熔断线 (默认 0.15)
  stop_loss: number;        // 单票止损线 (默认 0.08)
}
```

**前端表单建议:** 使用滑块 (Slider) 控制 `top_k` (5-200)、`dropout_threshold` (0.05-0.5)；下拉菜单选择 `weight_method`；数字输入框控制费率。

---

## 7. 回测端点

### 7.1 `POST /api/v1/backtest` — 提交回测任务

**权限:** `experiment:submit`  
**用途:** 提交完整回测任务，返回 task_id 供轮询

**Request Body:**
```json
{
  "strategy_type": "topk_dropout",
  "model_name": "lightgbm",
  "start_date": "2024-01-01",
  "end_date": "2025-12-31",
  "initial_capital": 1000000.0,
  "top_k": 30,
  "rebalance_freq": 1,
  "commission_rate": 0.001
}
```

| 字段 | 类型 | 必需 | 约束 | 说明 |
|------|------|------|------|------|
| `strategy_type` | string | ❌ | topk_dropout/equal_weight/score_weight | 策略类型 |
| `model_name` | string | ✅ | — | 预测模型名称 |
| `start_date` | string | ✅ | `YYYY-MM-DD` | 回测起始日 |
| `end_date` | string | ✅ | `YYYY-MM-DD` | 回测结束日 |
| `initial_capital` | float | ❌ | ≥ 10000 | 初始资金 (默认 1,000,000) |
| `top_k` | int | ❌ | 5-200 | 持仓数 (默认 30) |
| `rebalance_freq` | int | ❌ | 1-30 | 调仓频率(交易日) |
| `commission_rate` | float | ❌ | 0.0-0.05 | 佣金费率 |

**响应 `200 OK`:**
```json
{
  "task_id": "a1b2c3d4",
  "status": "running",
  "progress": 0.0,
  "result": null,
  "error": null
}
```

**前端集成:** 提交后立即跳转到任务监控页，开始轮询 `GET /api/v1/backtest/{task_id}`。

---

### 7.2 `GET /api/v1/backtest/{task_id}` — 查询回测状态/结果

**权限:** 无  
**用途:** 轮询回测任务进度与最终结果

**状态机:**
```
pending → running → completed  (成功)
                 → failed      (失败，含 error 信息)
```

**轮询中 (running):**
```json
{
  "task_id": "a1b2c3d4",
  "status": "running",
  "progress": 0.45,
  "result": null,
  "error": null
}
```

**已完成 (completed):**
```json
{
  "task_id": "a1b2c3d4",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "total_return": 0.234,
    "annual_return": 0.112,
    "sharpe_ratio": 1.42,
    "max_drawdown": -0.15,
    "win_rate": 0.58,
    "total_trades": 342,
    "turnover": 0.35
  },
  "error": null
}
```

**前端轮询建议:**
```javascript
async function pollBacktest(taskId, intervalMs = 2000) {
  while (true) {
    const res = await fetch(`/api/v1/backtest/${taskId}`);
    const data = await res.json();
    if (data.status === 'completed' || data.status === 'failed') {
      return data;
    }
    updateProgressBar(data.progress); // 更新进度条 UI
    await sleep(intervalMs);
  }
}
```

**错误状态:**
```json
{
  "task_id": "a1b2c3d4",
  "status": "failed",
  "progress": 0.0,
  "result": null,
  "error": "DataServer 连接超时"
}
```

---

## 8. 报告与风险指标端点

### 8.1 `GET /api/v1/report/{experiment_id}` — 实验绩效报告

**权限:** `report:read`  
**用途:** 获取指定实验的完整绩效指标

**响应 `200 OK`:**
```json
{
  "experiment_id": "lightgbm_baseline_20260515_103000_abc123",
  "model_name": "lightgbm",
  "generated_at": "2026-05-15T10:30:00.000Z",
  "metrics": {
    "ic_mean": 0.045,
    "icir": 0.52,
    "rank_ic_mean": 0.048,
    "rank_icir": 0.55,
    "total_return": 0.234,
    "annualized_return": 0.112,
    "sharpe_ratio": 1.42,
    "max_drawdown": -0.15,
    "win_rate": 0.58
  }
}
```

**指标解读 (前端展示用):**

| 指标 | 优秀 | 一般 | 需改进 | 展示组件 |
|------|------|------|--------|----------|
| Rank IC Mean | ≥ 0.05 | 0.03-0.05 | < 0.03 | 仪表盘 (Gauge) |
| Rank ICIR | ≥ 0.50 | 0.30-0.50 | < 0.30 | 仪表盘 |
| Sharpe Ratio | ≥ 1.5 | 0.5-1.5 | < 0.5 | 数值+颜色标签 |
| Max Drawdown | > -10% | -10%~-20% | < -20% | 红/黄/绿标签 |
| Win Rate | ≥ 55% | 50-55% | < 50% | 进度条 |

---

### 8.2 `GET /api/v1/risk` — 风险指标

**权限:** `experiment:read`  
**用途:** 获取指定策略的风险度量（VaR/CVaR/Beta/Alpha 等）

**Query 参数:**
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `strategy_id` | string | `"topk_dropout"` | 策略 ID |
| `start_date` | string | 1年前 | 起始日期 |
| `end_date` | string | 今天 | 结束日期 |

**响应 `200 OK`:**
```json
{
  "strategy_id": "topk_dropout",
  "start_date": "2024-05-19",
  "end_date": "2025-05-19",
  "metrics": {
    "sharpe_ratio": 1.42,
    "max_drawdown": -0.15,
    "annual_volatility": 0.19,
    "var_95": -0.022,
    "cvar_95": -0.03,
    "beta": 0.88,
    "alpha": 0.08,
    "information_ratio": 0.95
  }
}
```

**前端可视化建议:**

| 指标 | 图表类型 | 说明 |
|------|----------|------|
| `var_95` / `cvar_95` | 水平条形图 | 尾部风险对比 |
| `beta` | 单值仪表盘 | vs 市场敏感度 |
| `alpha` + `information_ratio` | 散点图 | 多策略对比气泡图 |
| `max_drawdown` + `annual_volatility` | 风险收益象限图 | X=波动率 Y=最大回撤 |

---

### 8.3 回测结果数据结构 (完整)

**BacktestResult — 前端展示的全部字段:**

```typescript
interface BacktestResult {
  initial_capital: number;        // 初始资金
  final_capital: number;          // 最终资金
  total_return: number;          // 总收益率 (如 0.234 = 23.4%)
  annualized_return: number;     // 年化收益率
  annualized_volatility: number; // 年化波动率
  sharpe_ratio: number;          // 夏普比率
  max_drawdown: number;          // 最大回撤 (负数)
  max_drawdown_duration: number; // 最大回撤持续天数
  win_rate: number;              // 胜率
  profit_loss_ratio: number;     // 盈亏比
  total_trades: number;          // 总交易次数
  turnover_rate: number;         // 日均换手率
  total_commission: number;      // 总佣金
  total_slippage: number;        // 总滑点成本
  daily_returns: number[];       // 日收益率序列
  nav_curve: number[];           // 净值曲线 (NAV)
  benchmark_nav: number[];      // 基准净值曲线 (可选)
  positions_history: Array<{
    date: string;
    nav: number;
    n_positions: number;
    instruments: string[];
  }>;
}
```

**前端可视化完整面板:**
1. **NAV 曲线** — 双线图（策略 vs 基准），下方叠加回撤面积图
2. **日收益分布** — 直方图 + KDE 曲线，标注均值/偏度/峰度
3. **滚动指标** — 60/120 日滚动 Sharpe、滚动波动率
4. **持仓热力图** — X=日期 Y=股票 Z=权重
5. **交易记录表** — 日期/标的/方向/数量/价格/成本

---

## 9. PM 熔断门控端点

### 9.1 `GET /api/v1/gate/status` — 门控状态

**权限:** 全部角色可查看 (透明原则)  
**用途:** PM 仪表盘核心 — 三门控实时状态

**响应 `200 OK`:**
```json
{
  "gates": {
    "signal": "open",
    "train": "open",
    "deploy": "closed"
  },
  "can_push_signal": true,
  "can_train_model": true,
  "can_deploy_model": false,
  "is_any_closed": true,
  "stats": {
    "current_states": {"signal": "open", "train": "open", "deploy": "closed"},
    "total_actions": 15,
    "total_stops": 3,
    "total_reopens": 2,
    "total_auto_trips": 1,
    "is_any_closed": true,
    "recent_24h_stops": 0
  }
}
```

**前端 UI 设计:** 三色交通灯 (🟢/🔴) + 熔断计数器 + 24h 内操作时间线

---

### 9.2 `POST /api/v1/gate/emergency-stop` — PM 一键熔断

**权限:** `signal:emergency_stop` (仅 PM/Admin)  
**用途:** PM 仪表盘"紧急熔断"红色大按钮

**Request Body:**
```json
{
  "dimension": "signal",
  "reason": "市场剧烈波动，紧急暂停信号推送"
}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dimension` | string | 默认 `"signal"` | signal / train / deploy |
| `reason` | string | ✅ | 熔断原因 (1-500 字符, 必填审计) |

**响应 `200 OK`:**
```json
{
  "success": true,
  "action_id": "f1e2d3c4",
  "dimension": "signal",
  "action": "emergency_stop",
  "from_state": "open",
  "to_state": "closed",
  "triggered_by": "pm",
  "reason": "市场剧烈波动，紧急暂停信号推送",
  "timestamp": "2026-05-19T14:30:00.000Z",
  "message": "门控 signal 已熔断: 市场剧烈波动，紧急暂停信号推送"
}
```

**冲突/权限错误:**
- `409 Conflict` — 门控已处于熔断状态，无需重复操作
- `403 Forbidden` — 非 PM 无权操作
- `409 Conflict` (限流) — 操作过于频繁，需等待 5 秒

**全局熔断:** `POST /api/v1/gate/global-emergency-stop` 同时关闭三门控。Request body 仅含 `reason`。

---

### 9.3 `GET /api/v1/gate/history` — 门控操作历史

**Query 参数:**
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `dimension` | string | `null` | 筛选维度 |
| `limit` | int | 50 (1-500) | 返回数量 |

**响应:**
```json
{
  "total": 15,
  "history": [
    {
      "action_id": "f1e2d3c4",
      "dimension": "signal",
      "action": "emergency_stop",
      "from_state": "open",
      "to_state": "closed",
      "triggered_by": "pm",
      "triggered_by_role": "portfolio_manager",
      "reason": "市场剧烈波动",
      "timestamp": "2026-05-19T14:30:00.000Z"
    }
  ]
}
```

**前端使用:** 时间线组件展示门控操作历史，用颜色区分 `emergency_stop`(红)/`emergency_reopen`(绿)/`auto_trip`(橙)。

---

## 10. 合规审计端点

### 10.1 `GET /api/v1/audit/logs` — 审计日志查询

**权限:** `audit:read` (仅 Auditor/Admin)  
**用途:** 合规审计页面 — 多条件过滤的审计事件列表

**Query 参数:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `event_type` | string | 事件类型 (model_deploy / api_key_rotate / pm_emergency_stop ...) |
| `user` | string | 操作者过滤 |
| `start_time` | string | 起始时间 ISO 格式 |
| `end_time` | string | 结束时间 |
| `limit` | int | 100 (1-1000) |

**响应:**
```json
{
  "total": 42,
  "filters": {
    "event_type": "pm_emergency_stop",
    "user": null,
    "start_time": null,
    "end_time": null
  },
  "entries": [
    {
      "event_id": "evt_abc123",
      "event_type": "pm_emergency_stop",
      "user": "pm",
      "role": "portfolio_manager",
      "action": "emergency_stop",
      "resource": "gate/signal",
      "detail": {
        "action_id": "f1e2d3c4",
        "dimension": "signal",
        "reason": "市场剧烈波动"
      },
      "ip_address": "",
      "timestamp": "2026-05-19T14:30:00.000Z",
      "hash_chain": "a1b2c3..."
    }
  ]
}
```

**前端集成:** 专业审计表格 (时间/用户/事件/资源/详情)，支持点击展开 `detail` JSON。导出按钮调用 `GET /api/v1/audit/export`。

---

### 10.2 `GET /api/v1/audit/verify-chain` — 验证审计哈希链

**权限:** `audit:read`  
**用途:** 合规审计页面的"验证完整性"按钮

**Query 参数:** `date` (可选, `YYYYMMDD`)

**响应:**
```json
{
  "valid": true,
  "violations": [],
  "chain_intact": true,
  "total_lines": 1523
}
```

若 `valid: false`，显示红色警告:"审计日志可能被篡改!"，并列出 `violations` 详情。

---

### 10.3 `POST /api/v1/compliance/sox-report` — SOX 合规报告

**权限:** `compliance:export`  
**用途:** 生成季度合规报告，覆盖 7 大 SOX 控制点

**Request Body (可选):**
```json
{
  "quarter": "2026-Q2"
}
```

**前端集成:** "生成报告"按钮 + 季度选择器 → 下载 PDF/JSON 格式。

---

## 11. 系统运维端点

### 11.1 `GET /api/v1/metrics` — 系统性能指标 (Prometheus 兼容)

**权限:** `experiment:read`  
**用途:** Grafana 仪表盘数据源 / Prometheus scrape target / 运维监控页

**响应 `200 OK`:**
```json
{
  "service": "qlib-api",
  "version": "1.0.0",
  "uptime_seconds": 12345.6,
  "cache_hit_rate": 0.8721,
  "cache_size": 2456,
  "cache_evictions": 12,
  "active_models": 3,
  "requests_total": 4521,
  "avg_latency_ms": 23.5
}
```

**前端运维仪表盘:**
| 指标 | 展示 | 告警线 |
|------|------|--------|
| `uptime_seconds` | 运行时间计时器 | — |
| `cache_hit_rate` | 仪表盘 (0-100%) | < 80% 黄色, < 50% 红色 |
| `active_models` | 数字徽章 | — |
| `requests_total` | 累计计数器 | — |
| `avg_latency_ms` | 时序折线图 | > 500ms 黄色, > 1000ms 红色 |

### 11.2 缓存健康三层架构

系统使用三级缓存体系，`cache_hit_rate` 是三层综合命中率:

| 层级 | 名称 | 范围 | 健康阈值 |
|------|------|------|----------|
| L1 | Global Cache | 特征文件级 LRU | 命中率 ≥ 80% |
| L2 | Expression Cache | 表达式计算结果 | 命中率 ≥ 80% |
| L3 | Dataset Cache | 数据集查询结果 | 命中率 ≥ 80% |

综合健康状态: `healthy` (≥80%) / `degraded` (50-80%) / `critical` (<50%)

---

## 12. 前端 UI 集成指南

### 12.1 推荐前端页面结构

```
Qlib1 量化分析平台
├── 📊 仪表盘 (Dashboard)
│   ├── 系统健康状态卡片
│   ├── 缓存命中率仪表盘
│   ├── 活跃模型列表
│   └── 最近回测摘要
├── 📈 回测实验室 (Backtest Lab)
│   ├── 策略配置表单 → POST /api/v1/backtest
│   ├── 任务监控面板 → GET /api/v1/backtest/{task_id}
│   └── 结果对比视图
├── 🎯 信号中心 (Signal Center)
│   ├── 预测得分排名 → GET /api/v1/scores
│   ├── Top/Bottom 榜单
│   └── 单票因子详情 → POST /api/v1/factors/{dataset}
├── 💼 组合管理 (Portfolio)
│   ├── 持仓权重饼图 → GET /api/v1/portfolio/{strategy_id}
│   ├── 组合风险指标 → GET /api/v1/risk
│   └── 交易记录表格
├── 📋 绩效报告 (Performance)
│   ├── IC 时序曲线
│   ├── NAV 累积收益图
│   ├── 风险分析面板
│   └── 报告导出 → GET /api/v1/report/{experiment_id}
├── 🛡️ PM 门控 (PM Gate)
│   ├── 三门控交通灯
│   ├── 一键熔断红色按钮 → POST /api/v1/gate/emergency-stop
│   ├── 恢复放行按钮 → POST /api/v1/gate/emergency-reopen
│   └── 操作历史时间线 → GET /api/v1/gate/history
├── 🔍 合规审计 (Compliance)
│   ├── 审计日志表格 → GET /api/v1/audit/logs
│   ├── 哈希链验证 → GET /api/v1/audit/verify-chain
│   └── SOX 合规报告 → POST /api/v1/compliance/sox-report
├── 🔧 系统管理 (Admin)
│   ├── 证券浏览器 → GET /api/v1/instruments
│   ├── 因子数据浏览器 → POST /api/v1/factors/{dataset}
│   └── 运维指标 → GET /api/v1/metrics
└── 📖 API 文档
    ├── Swagger UI → /docs
    └── ReDoc → /redoc
```

### 12.2 请求封装示例 (TypeScript)

```typescript
// api/client.ts
const BASE_URL = 'http://localhost:8000';

interface ApiOptions {
  user?: string;
  params?: Record<string, string>;
}

async function apiGet<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const url = new URL(path, BASE_URL);
  if (options.params) {
    Object.entries(options.params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  
  const res = await fetch(url.toString(), {
    headers: {
      'X-User-ID': options.user || 'researcher',
      'Content-Type': 'application/json',
    },
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new ApiError(res.status, err.detail);
  }
  
  return res.json();
}

async function apiPost<T>(path: string, body: object, user?: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'X-User-ID': user || 'researcher',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new ApiError(res.status, err.detail);
  }
  
  return res.json();
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// 使用示例
const scores = await apiGet<ScoreResponse>('/api/v1/scores', {
  params: { model_name: 'lightgbm', limit: '20' },
});

const backtest = await apiPost<BacktestStatus>('/api/v1/backtest', {
  strategy_type: 'topk_dropout',
  model_name: 'lightgbm',
  start_date: '2024-01-01',
  end_date: '2025-12-31',
  initial_capital: 1_000_000,
});
```

### 12.3 认证流程

```
1. 用户在前端登录页选择角色 (开发环境简化)
   ├── Quant Researcher (researcher)
   ├── Portfolio Manager (pm)
   ├── Compliance Auditor (auditor)
   └── System Admin (admin)

2. 将角色 ID 存入 sessionStorage / cookie

3. 每次 API 请求自动附加 X-User-ID 头

4. 后端根据角色返回对应权限的数据
   - 无权限的 UI 元素直接隐藏
   - 403 响应 → 弹出"权限不足"提示
```

### 12.4 BI 工具集成

**Tableau 集成:**
1. 添加 Web Data Connector
2. URL: `http://server:8000/api/v1/scores?limit=500`
3. Tableau 自动解析 JSON 为表结构，直接拖拽生成热力图

**Power BI 集成:**
1. Get Data → Web → `http://server:8000/api/v1/risk?strategy_id=topk_dropout`
2. 或使用 Python Script 数据源调用 API

**Grafana 集成:**
1. 添加 Prometheus 数据源
2. Dashboard import JSON (指标通过 `/api/v1/metrics` 暴露)
3. 实时监控缓存、延迟、请求量

---

## 13. 数据模型参考

### 13.1 预测得分 (PredictionResult)

```typescript
interface PredictionResult {
  predictions: number[];     // 预测值数组
  rank_ic?: number;          // Rank IC (仅当 predict 传入 y_true 时)
  rank_icir?: number;        // Rank ICIR (同上)
}
```

### 13.2 模型训练结果 (TrainingResult)

```typescript
interface TrainingResult {
  model_name: string;
  train_loss: number[];
  valid_loss: number[];
  best_iteration: number;
  best_score: number;
  feature_importance: Record<string, number>;  // { feature_name: gain_ratio }
  train_time_ms: number;
  n_features: number;
  n_samples: number;
}
```

### 13.3 门控操作记录 (GateAction)

```typescript
interface GateAction {
  action_id: string;
  dimension: 'signal' | 'train' | 'deploy';
  action: 'emergency_stop' | 'emergency_reopen' | 'auto_trip';
  from_state: 'open' | 'closed';
  to_state: 'open' | 'closed';
  triggered_by: string;
  triggered_by_role: string;
  reason: string;
  timestamp: string;         // ISO 8601
  metadata?: Record<string, any>;
}
```

### 13.4 审计日志条目 (AuditEntry)

```typescript
interface AuditEntry {
  event_id: string;
  event_type: string;        // model_deploy | pm_emergency_stop | ...
  user: string;
  role: string;
  action: string;
  resource: string;
  detail: Record<string, any>;
  ip_address: string;
  timestamp: string;
  hash_chain: string;        // HMAC-SHA256 防篡改链
}
```

---

## 14. 错误处理规范

### 14.1 HTTP 状态码

| 状态码 | 含义 | 前端处理 |
|--------|------|----------|
| `200` | 成功 | 正常渲染 |
| `400` | 请求参数错误 (如日期格式错误) | Toast 红色提示校验失败 |
| `403` | 权限拒绝 | 弹出"当前角色无此操作权限" + 显示所需权限名 |
| `404` | 资源不存在 (如 experiment_id 无效) | 显示"未找到"插画 + 返回按钮 |
| `409` | 冲突 (如重复熔断、限流) | 显示冲突原因，按钮置灰 |
| `422` | 参数校验失败 (Pydantic) | 逐字段红色边框提示 |
| `500` | 服务器内部错误 | 显示"系统异常" + 重试按钮 |

### 14.2 错误响应格式

```json
{
  "detail": "权限拒绝: user=researcher, role=quant_researcher, required=signal:emergency_stop"
}
```

### 14.3 前端错误处理封装

```typescript
async function handleApiError(error: ApiError) {
  switch (error.status) {
    case 403:
      showNotification('权限不足', `需要权限: ${extractPermission(error.message)}`, 'warning');
      break;
    case 404:
      navigate('/not-found');
      break;
    case 409:
      showNotification('操作冲突', error.message, 'info');
      break;
    case 422:
      // 表单字段级错误展示
      parseValidationErrors(error.message).forEach(e => {
        form.setFieldError(e.field, e.message);
      });
      break;
    default:
      showNotification('系统错误', '请稍后重试', 'error');
  }
}
```

---

## 附录 A: 启动命令速查

```bash
# 启动 API 服务 (开发环境)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 运行实验流水线
python -m src.workflow.runner --config experiments/lgb_baseline.yaml

# 运行基准测试
make benchmark

# 运行全部测试
make test
```

## 附录 B: 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ENCRYPTION_KEY` | AES-256-GCM 密钥 (Base64) | (未设=开发临时密钥) |
| `QLIB_PROVIDER_URI` | Qlib 数据目录 | `./data/qlib_data/us_data` |
| `API_CACHE_TTL` | API 数据缓存 TTL (秒) | 3600 |

---

> 📧 **文档维护:** Qlib1 项目组  
> 📅 **最后更新:** 2026-05-19  
> ⚠️ **注意:** 本系统运行中所有操作均记录防篡改审计日志，请合规使用。
