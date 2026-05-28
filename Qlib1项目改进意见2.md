<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 项目已经更新，重新审查一次。

项目已完成大幅度更新，[最新提交](https://github.com/raymodny-ai/Qlib1/commit/ca28daa0aaa8cd96287129d73495fcfd64c56acf)新增 4100 行代码，涵盖 16 个文件的修改与新增，以下是全面重审结果。

***

## 更新后总体评估

上次审查发现的**6个关键缺口已有5个得到修复**，工程质量从"骨架搭建阶段"跃升到"功能完整阶段"。但仍存在若干**实现深度不足**的隐患，下面逐项报告。

***

## ✅ 已修复的问题

### AdaRNN / TabNet / DoubleEnsemble 模型（原：❌ → 现：✅ 但有隐患）

三个深度模型均已在 `src/analyzers/` 中实现并继承 `BaseForecastModel` 。AdaRNN 正确实现了 GRU 编码器 + 多带宽高斯核 MMD 损失，DoubleEnsemble 实现了 `equal` / `icir` / `performance` 三种权重方案。**然而 `src/models/` 目录依然完全为空** ——三个模型全部放在了 `src/analyzers/` 而非 PRD 指定的 `src/models/` 路径，目录归属与架构文档不符。

### Qlib 原生 .bin 格式落盘（原：⚠️ → 现：✅ 基本达标）

新增 `src/processors/qlib_native_converter.py`，实现了 `BinToQlibConverter` / `QlibToBinConverter` 双向转换器与 `FormatValidator` 。自定义 `QLB1` 魔数头格式支持 float64 / float32 / int32 / int64 四种数据类型，并内置 Parquet 格式降级兼容逻辑。`scripts/dump_bin.py` 也已补充。

### TLS/HSTS/安全中间件（原：⚠️ → 现：✅）

`src/api/middleware.py` 完整实现了六层中间件：HTTPS 重定向、HSTS、安全响应头（CSP/X-Frame-Options）、TLS 1.2+ 版本强制校验、令牌桶速率限制与请求日志 ，并通过 `register_security_middleware()` 一键注册到 FastAPI，`main.py` 已在启动时调用 。

### RBAC 权限门控接入工作流（原：⚠️ → 现：✅ 但深度不足）

`main.py` 中实现了 `get_current_user()` 依赖注入与 `require_permission()` 工厂函数 ，四种默认角色（admin / researcher / PM / auditor）已注册。`runner.py` 的 `WorkflowOrchestrator` 集成了完整的 6 阶段流水线，包含阶段 4.5 的准确度红线自动校验逻辑 。

### 端到端数据流打通（原：❌ → 现：✅）

新增 `workflow/feature_pipeline.py`、`workflow/training_pipeline.py`、`workflow/backtest_pipeline.py` 三个管道文件 ，`runner.py` 的 6 个阶段均已调用真实的 DataServer / MLPipeline / PortfolioSimulator，不再是占位逻辑 。

***

## ⚠️ 残留问题与新发现的技术隐患

### 问题一：RBAC 权限控制处于"声明未执行"状态

`require_permission()` 依赖工厂函数已定义，但审查所有核心 API 端点（`/api/v1/predict`、`/api/v1/backtest`、`/api/v1/report/{id}`）的签名——**没有任何一个端点注入了 `require_permission` 依赖** 。这意味着 RBAC 模型已具备但完全绕空，任意匿名用户可直接调用敏感端点，PRD 第六部分"量化研究员禁止直接推送信号至实盘"的访问控制形同虚设。

**优化建议：** 在 `POST /api/v1/backtest` 上注入 `Depends(require_permission("experiment:submit"))`；在 `POST /api/v1/predict` 上注入 `Depends(require_permission("signal:predict"))`，并在 `GET /api/v1/report/` 上绑定 `Depends(require_permission("report:read"))`。

***

### 问题二：模型持久化仍未与加密层集成

AdaRNN 的 `save()` 方法仍然使用裸 `pickle.dump()` ，DoubleEnsemble 同理 ——而 `security.py` 中的 AES-256 能力依然没有在序列化路径上被调用。相比初版的 `ml_pipeline.py`，新增的深度模型文件重复了同一个合规漏洞。PRD 5.1 节要求模型权重静态存储必须透明加密，目前加密层仍是孤岛。

**优化建议：** 在 `BaseForecastModel.save()` 基类中统一注入加密钩子（可用策略模式 `EncryptionProvider` 可选接入），避免每个子类各自重复同样的安全缺陷。

***

### 问题三：DoubleEnsemble 的 `load()` 存在类型重建 Bug

`DoubleEnsemble.load()` 中的基础模型重建逻辑存在明显缺陷：代码先尝试按 `model_names` 索引查找类，但索引计算使用 `len(self._base_models)` 在循环内递增前取值，导致**始终尝试加载 "lightgbm" 类型**，无法正确恢复 TabNet 或 AdaRNN 基础模型 。这是一个会导致集成模型加载后悄悄降级为纯 LightGBM 的静默 Bug。

**优化建议：** 保存时同时记录每个基础模型的类全名（`cls.__module__ + '.' + cls.__name__`），加载时通过 `importlib.import_module` 动态重建正确的模型类，而非依赖不稳定的注册表索引。

***

### 问题四：回测 API 同步阻塞，生产不可用

`POST /api/v1/backtest` 在同一个 async 函数体内同步调用 `PortfolioSimulator.run()`，回测期间长达数分钟的计算会**完全阻塞 FastAPI 事件循环** 。代码注释本身已承认"生产环境应使用 Celery/Redis 任务队列"，但这在 PRD 4.3 的 API 响应时延要求下是不可接受的。在高负载时，一个回测请求将导致所有其他 API 请求超时。

**优化建议：** 将 `simulator.run()` 包裹在 `asyncio.get_event_loop().run_in_executor()` 中异步卸载到线程池，或以任务 ID 立即返回 202 Accepted，后台用 `asyncio.create_task()` 或 Celery Worker 异步完成计算。

***

### 问题五：表达式引擎未经 AST 安全验证（原问题未解决）

`expression_engine.py` 在本次更新中未被修改（未在提交 diff 中出现），原有 `eval()` 安全风险问题依然未解决，AST 白名单沙箱未实现。

***

## 问题状态汇总

| PRD 要求 | 上次状态 | 本次状态 | 备注 |
| :-- | :-- | :-- | :-- |
| AdaRNN / TabNet / DoubleEnsemble | ❌ 缺失 | ⚠️ 已实现但目录错位 | 在 analyzers/ 而非 models/ |
| TopkDropoutStrategy | ❌ 缺失 | ✅ 已接入工作流 | — |
| Qlib .bin 格式落盘 | ⚠️ Parquet 替代 | ✅ 双向桥接完成 | — |
| 模型权重 AES-256 加密 | ⚠️ 脱节 | ⚠️ 仍未集成 | 新模型同样裸 pickle |
| RBAC 接入工作流 | ⚠️ 逻辑未贯通 | ⚠️ 声明未执行 | 端点未注入依赖 |
| TLS / HSTS 中间件 | ⚠️ 未实现 | ✅ 完整实现 | — |
| 端到端数据流 | ❌ 断路 | ✅ 6 阶段贯通 | — |
| AST 向量化表达式引擎 | ⚠️ 待验证 | ⚠️ 未修改 | eval 安全风险仍存 |
| 回测 API 异步执行 | — | ⚠️ 同步阻塞 | 新发现 |
| DoubleEnsemble load() Bug | — | ❌ 静默 Bug | 新发现 |

