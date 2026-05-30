/**
 * Qlib1 API Type Definitions
 * Generated from API documentation
 */

// ============ Common Types ============

export type Role = 'admin' | 'researcher' | 'pm' | 'auditor';

export interface User {
  user_id: string;
  name: string;
  role: Role;
  active: boolean;
}

// ============ Health ============

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  version: string;
  timestamp: string;
  uptime_seconds: number;
}

// ============ Instruments ============

export interface InstrumentInfo {
  symbol: string;
  name?: string;
  sector?: string;
  industry?: string;
  market_cap?: number;
}

// ============ Factors ============

export interface FactorQuery {
  instruments: string[];
  start_date: string;
  end_date: string;
  fields?: string[];
  dataset?: string;
}

export interface FactorResponse {
  dataset: string;
  instruments: string[];
  date_range: {
    start: string;
    end: string;
  };
  n_rows: number;
  n_fields: number;
  data: FactorDataItem[];
}

export interface FactorDataItem {
  instrument: string;
  date: string;
  [key: string]: string | number | null;
}

// ============ Prediction ============

export interface PredictRequest {
  model_name: string;
  instruments: string[];
  date: string;
  factors?: Record<string, Record<string, number>>;
}

export interface Prediction {
  instrument: string;
  score: number;
  rank: number;
}

export interface PredictResponse {
  model_name: string;
  date: string;
  timestamp: string;
  predictions: Prediction[];
}

// ============ Portfolio ============

export interface PortfolioWeight {
  instrument: string;
  weight: number;
  score?: number;
}

export interface PortfolioResponse {
  strategy_id: string;
  date: string;
  n_holdings: number;
  total_weight: number;
  holdings: PortfolioWeight[];
}

// ============ Backtest ============

export type BacktestStatusType = 'pending' | 'running' | 'completed' | 'failed';

export interface BacktestRequest {
  strategy_type: 'topk_dropout' | 'equal_weight' | 'score_weight';
  model_name: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  top_k?: number;
  rebalance_freq?: number;
  commission_rate?: number;
}

export interface BacktestResult {
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  turnover: number;
}

export interface BacktestStatus {
  task_id: string;
  status: BacktestStatusType;
  progress: number;
  result?: BacktestResult | null;
  error?: string | null;
}

// ============ Report ============

export interface ReportMetrics {
  ic_mean?: number;
  icir?: number;
  rank_ic_mean?: number;
  rank_icir?: number;
  total_return?: number;
  annualized_return?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  win_rate?: number;
}

export interface ReportResponse {
  experiment_id: string;
  model_name: string;
  generated_at: string;
  metrics: ReportMetrics;
}

// ============ Scores ============

export interface ScoreItem {
  instrument: string;
  score: number;
  rank: number;
  percentile: number;
}

export interface ScoreResponse {
  date: string;
  model_name: string;
  total_instruments: number;
  scores: ScoreItem[];
}

// ============ Risk ============

export interface RiskMetrics {
  sharpe_ratio: number;
  max_drawdown: number;
  annual_volatility: number;
  var_95: number;
  cvar_95: number;
  beta: number;
  alpha: number;
  information_ratio: number;
}

export interface RiskResponse {
  strategy_id: string;
  start_date: string;
  end_date: string;
  metrics: RiskMetrics;
}

// ============ System Metrics ============

export interface MetricsResponse {
  service: string;
  version: string;
  uptime_seconds: number;
  cache_hit_rate: number;
  cache_size: number;
  cache_evictions: number;
  active_models: number;
  requests_total: number;
  avg_latency_ms: number;
}

// ============ PM Gate ============

export type GateDimension = 'signal' | 'train' | 'deploy';
export type GateState = 'open' | 'closed';
export type GateAction = 'emergency_stop' | 'emergency_reopen';

export interface GateStatus {
  gates: Record<GateDimension, GateState>;
  can_push_signal: boolean;
  can_train_model: boolean;
  can_deploy_model: boolean;
  is_any_closed: boolean;
  stats: {
    total_actions: number;
    last_action_at?: string;
  };
}

export interface GateActionRequest {
  dimension?: GateDimension;
  reason: string;
}

export interface GlobalGateActionRequest {
  reason: string;
}

export interface GateActionResponse {
  success: boolean;
  action_id?: string;
  dimension?: string;
  action?: string;
  from_state?: string;
  to_state?: string;
  triggered_by?: string;
  reason?: string;
  timestamp?: string;
  message?: string;
}

export interface GateHistoryEntry {
  action_id: string;
  dimension: GateDimension;
  action: GateAction;
  triggered_by: string;
  reason: string;
  timestamp: string;
}

export interface GateHistoryResponse {
  total: number;
  history: GateHistoryEntry[];
}

// ============ Audit ============

export interface AuditLogEntry {
  event_id: string;
  event_type: string;
  user_id: string;
  timestamp: string;
  details: Record<string, unknown>;
  hash?: string;
  prev_hash?: string;
}

export interface AuditQueryParams {
  event_type?: string;
  user?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
}

export interface AuditLogsResponse {
  total: number;
  filters: {
    event_type: string | null;
    user: string | null;
    start_time: string | null;
    end_time: string | null;
  };
  entries: AuditLogEntry[];
}

export interface AuditChainVerification {
  verified: boolean;
  date: string;
  total_entries: number;
  broken_at: number | null;
}

// ============ Compliance ============

export interface ControlStatus {
  control_id: string;
  status: 'passed' | 'warning' | 'failed';
  details?: string;
}

export interface ComplianceStatus {
  overall_status: 'compliant' | 'warning' | 'non_compliant';
  audit_chain_verified: boolean;
  period: string;
  controls: ControlStatus[];
}

export interface SOXReport {
  quarter: string;
  generated_at: string;
  controls: ControlStatus[];
  audit_chain_verified: boolean;
  summary: {
    total_controls: number;
    passed: number;
    warnings: number;
    failed: number;
  };
}

// ============ Data Management (Sprint 3: F-060–F-063) ============

export type DataSourceStatusType = 'connected' | 'disconnected' | 'error' | 'degraded';

export interface DataSourceInfo {
  source_id: string;
  name: string;
  provider: string;
  status: DataSourceStatusType;
  last_sync: string | null;
  coverage_start: string | null;
  coverage_end: string | null;
  record_count: number;
  quality_score: number;
  description: string;
}

export interface DatasetInfo {
  name: string;
  description: string;
  n_instruments: number;
  n_fields: number;
  date_range: { start: string; end: string };
  size_mb: number;
  last_updated: string | null;
}

export type IngestMode = 'full' | 'incremental';

export interface DataIngestRequest {
  dataset: string;
  mode: IngestMode;
  sources?: string[];
  force?: boolean;
}

export interface DataIngestResponse {
  task_id: string;
  dataset: string;
  mode: IngestMode;
  status: string;
  message: string;
}

export interface DataPreviewResponse {
  dataset: string;
  total_rows: number;
  preview_rows: number;
  columns: string[];
  rows: Record<string, unknown>[];
}

// ============ API Error ============

export interface ApiError {
  detail: string;
}

// ============ Permission Map ============

export const PERMISSIONS = {
  MODEL_READ: 'model:read',
  EXPERIMENT_READ: 'experiment:read',
  EXPERIMENT_SUBMIT: 'experiment:submit',
  REPORT_READ: 'report:read',
  SIGNAL_EMERGENCY_STOP: 'signal:emergency_stop',
  AUDIT_READ: 'audit:read',
  AUDIT_EXPORT: 'audit:export',
  COMPLIANCE_EXPORT: 'compliance:export',
  COMPLIANCE_REVIEW: 'compliance:review',
  LOGS_READ: 'logs:read',
  LOGS_EXPORT: 'logs:export',
} as const;

export type Permission = (typeof PERMISSIONS)[keyof typeof PERMISSIONS];

export const ROLE_PERMISSIONS: Record<Role, Permission[]> = {
  admin: Object.values(PERMISSIONS),
  researcher: [
    PERMISSIONS.MODEL_READ,
    PERMISSIONS.EXPERIMENT_READ,
    PERMISSIONS.EXPERIMENT_SUBMIT,
    PERMISSIONS.REPORT_READ,
    PERMISSIONS.LOGS_READ,
  ],
  pm: [
    PERMISSIONS.EXPERIMENT_READ,
    PERMISSIONS.REPORT_READ,
    PERMISSIONS.SIGNAL_EMERGENCY_STOP,
    PERMISSIONS.LOGS_READ,
  ],
  auditor: [
    PERMISSIONS.AUDIT_READ,
    PERMISSIONS.AUDIT_EXPORT,
    PERMISSIONS.COMPLIANCE_EXPORT,
    PERMISSIONS.COMPLIANCE_REVIEW,
    PERMISSIONS.LOGS_READ,
    PERMISSIONS.LOGS_EXPORT,
  ],
};

// ============ Strategy Types ============

export type StrategyType = 'topk_dropout' | 'equal_weight' | 'score_weight';

export const STRATEGY_LABELS: Record<StrategyType, string> = {
  topk_dropout: 'Top-K 保留策略',
  equal_weight: '等权重策略',
  score_weight: '得分加权策略',
};

// ============ Logs ============

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  logger: string;
  message: string;
  module?: string;
  function?: string;
  line_number?: number;
  user_id?: string;
  request_id?: string;
  extra?: Record<string, unknown>;
}

export interface LogQueryParams {
  level?: LogLevel;
  logger?: string;
  search?: string;
  start_time?: string;
  end_time?: string;
  module?: string;
  limit?: number;
  offset?: number;
}

export interface LogsResponse {
  total: number;
  logs: LogEntry[];
  filters: {
    level: LogLevel | null;
    logger: string | null;
    start_time: string | null;
    end_time: string | null;
  };
}

export interface LogStats {
  total_logs: number;
  by_level: Record<LogLevel, number>;
  last_error_at?: string;
  error_rate_24h: number;
}