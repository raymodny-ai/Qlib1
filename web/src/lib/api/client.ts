/**
 * Qlib1 API Client
 * Axios-based HTTP client with RBAC authentication support
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  HealthResponse,
  InstrumentInfo,
  FactorQuery,
  FactorResponse,
  PredictRequest,
  PredictResponse,
  PortfolioResponse,
  BacktestRequest,
  BacktestStatus,
  ReportResponse,
  ScoreResponse,
  RiskResponse,
  MetricsResponse,
  GateStatus,
  GateActionRequest,
  GlobalGateActionRequest,
  GateActionResponse,
  GateHistoryResponse,
  AuditQueryParams,
  AuditLogsResponse,
  AuditChainVerification,
  ComplianceStatus,
  SOXReport,
  ApiError,
  LogQueryParams,
  LogsResponse,
  LogStats,
} from '@/types/api';

// API Base URL from environment
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false,
});

// ============ Request Interceptors ============

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Inject X-User-ID header for RBAC authentication
    // In a real app, this would come from auth context/session
    if (typeof window !== 'undefined') {
      const userId = localStorage.getItem('qlib1_user_id') || 'anonymous';
      config.headers.set('X-User-ID', userId);
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ============ Response Interceptors ============

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiError>) => {
    const status = error.response?.status;
    const detail = error.response?.data?.detail;

    switch (status) {
      case 401:
        console.error('[API] Unauthorized:', detail);
        break;
      case 403:
        console.error('[API] Forbidden:', detail);
        break;
      case 404:
        console.error('[API] Not Found:', detail);
        break;
      case 409:
        console.error('[API] Conflict:', detail);
        break;
      case 422:
        console.error('[API] Validation Error:', detail);
        break;
      case 426:
        console.error('[API] Upgrade Required (TLS):', detail);
        break;
      case 429:
        console.error('[API] Rate Limited:', detail);
        break;
      case 500:
        console.error('[API] Internal Server Error:', detail);
        break;
    }

    return Promise.reject(error);
  }
);

// ============ API Endpoints ============

/**
 * System Health Check
 * No authentication required
 */
export const getHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

/**
 * List Instruments
 * No authentication required
 */
export const getInstruments = async (params?: {
  sector?: string;
  limit?: number;
}): Promise<InstrumentInfo[]> => {
  const response = await apiClient.get<InstrumentInfo[]>('/api/v1/instruments', { params });
  return response.data;
};

/**
 * Query Factor Data
 * No authentication required
 */
export const queryFactors = async (
  dataset: string,
  query: FactorQuery
): Promise<FactorResponse> => {
  const response = await apiClient.post<FactorResponse>(
    `/api/v1/factors/${dataset}`,
    query
  );
  return response.data;
};

/**
 * Model Prediction
 * Requires permission: model:read
 */
export const predict = async (request: PredictRequest): Promise<PredictResponse> => {
  const response = await apiClient.post<PredictResponse>('/api/v1/predict', request);
  return response.data;
};

/**
 * Get Portfolio Weights
 * No authentication required
 */
export const getPortfolio = async (
  strategyId: string,
  params: { date: string }
): Promise<PortfolioResponse> => {
  const response = await apiClient.get<PortfolioResponse>(
    `/api/v1/portfolio/${strategyId}`,
    { params }
  );
  return response.data;
};

/**
 * Submit Backtest Task
 * Requires permission: experiment:submit
 */
export const submitBacktest = async (
  request: BacktestRequest
): Promise<BacktestStatus> => {
  const response = await apiClient.post<BacktestStatus>('/api/v1/backtest', request);
  return response.data;
};

/**
 * Get Backtest Task Status
 * No authentication required
 */
export const getBacktestStatus = async (
  taskId: string
): Promise<BacktestStatus> => {
  const response = await apiClient.get<BacktestStatus>(
    `/api/v1/backtest/${taskId}`
  );
  return response.data;
};

/**
 * Get Performance Report
 * Requires permission: report:read
 */
export const getReport = async (
  experimentId: string
): Promise<ReportResponse> => {
  const response = await apiClient.get<ReportResponse>(
    `/api/v1/report/${experimentId}`
  );
  return response.data;
};

/**
 * Get Cross-Section Scores
 * Requires permission: model:read
 */
export const getScores = async (params?: {
  date?: string;
  model_name?: string;
  limit?: number;
}): Promise<ScoreResponse> => {
  const response = await apiClient.get<ScoreResponse>('/api/v1/scores', { params });
  return response.data;
};

/**
 * Get Risk Metrics
 * Requires permission: experiment:read
 */
export const getRiskMetrics = async (params?: {
  strategy_id?: string;
  start_date?: string;
  end_date?: string;
}): Promise<RiskResponse> => {
  const response = await apiClient.get<RiskResponse>('/api/v1/risk', { params });
  return response.data;
};

/**
 * Get System Metrics
 * Requires permission: experiment:read
 */
export const getSystemMetrics = async (): Promise<MetricsResponse> => {
  const response = await apiClient.get<MetricsResponse>('/api/v1/metrics');
  return response.data;
};

// ============ PM Gate Endpoints ============

/**
 * Get Gate Status
 * No authentication required
 */
export const getGateStatus = async (): Promise<GateStatus> => {
  const response = await apiClient.get<GateStatus>('/api/v1/gate/status');
  return response.data;
};

/**
 * Emergency Stop Gate
 * Requires permission: signal:emergency_stop
 */
export const emergencyStopGate = async (
  request: GateActionRequest
): Promise<GateActionResponse> => {
  const response = await apiClient.post<GateActionResponse>(
    '/api/v1/gate/emergency-stop',
    request
  );
  return response.data;
};

/**
 * Emergency Reopen Gate
 * Requires permission: signal:emergency_stop
 */
export const emergencyReopenGate = async (
  request: GateActionRequest
): Promise<GateActionResponse> => {
  const response = await apiClient.post<GateActionResponse>(
    '/api/v1/gate/emergency-reopen',
    request
  );
  return response.data;
};

/**
 * Global Emergency Stop
 * Requires permission: signal:emergency_stop
 */
export const globalEmergencyStop = async (
  request: GlobalGateActionRequest
): Promise<GateActionResponse[]> => {
  const response = await apiClient.post<GateActionResponse[]>(
    '/api/v1/gate/global-emergency-stop',
    request
  );
  return response.data;
};

/**
 * Global Emergency Reopen
 * Requires permission: signal:emergency_stop
 */
export const globalEmergencyReopen = async (
  request: GlobalGateActionRequest
): Promise<GateActionResponse[]> => {
  const response = await apiClient.post<GateActionResponse[]>(
    '/api/v1/gate/global-emergency-reopen',
    request
  );
  return response.data;
};

/**
 * Get Gate History
 * No authentication required
 */
export const getGateHistory = async (params?: {
  dimension?: string;
  limit?: number;
}): Promise<GateHistoryResponse> => {
  const response = await apiClient.get<GateHistoryResponse>(
    '/api/v1/gate/history',
    { params }
  );
  return response.data;
};

// ============ Audit Endpoints ============

/**
 * Query Audit Logs
 * Requires permission: audit:read
 */
export const getAuditLogs = async (
  params?: AuditQueryParams
): Promise<AuditLogsResponse> => {
  const response = await apiClient.get<AuditLogsResponse>('/api/v1/audit/logs', {
    params,
  });
  return response.data;
};

/**
 * Verify Audit Chain
 * Requires permission: audit:read
 */
export const verifyAuditChain = async (
  params?: { date?: string }
): Promise<AuditChainVerification> => {
  const response = await apiClient.get<AuditChainVerification>(
    '/api/v1/audit/verify-chain',
    { params }
  );
  return response.data;
};

/**
 * Export Audit Report
 * Requires permission: audit:export
 */
export const exportAuditReport = async (
  params?: AuditQueryParams
): Promise<{ message: string; path: string }> => {
  const response = await apiClient.get<{ message: string; path: string }>(
    '/api/v1/audit/export',
    { params }
  );
  return response.data;
};

// ============ Compliance Endpoints ============

/**
 * Get Compliance Status
 * Requires permission: compliance:review
 */
export const getComplianceStatus = async (): Promise<ComplianceStatus> => {
  const response = await apiClient.get<ComplianceStatus>('/api/v1/compliance/status');
  return response.data;
};

/**
 * Generate SOX Report
 * Requires permission: compliance:export
 */
export const generateSOXReport = async (
  request?: { quarter?: string }
): Promise<SOXReport> => {
  const response = await apiClient.post<SOXReport>(
    '/api/v1/compliance/sox-report',
    request || {}
  );
  return response.data;
};

// ============ Export ============

export default apiClient;
export { apiClient };

// ============ Logs Endpoints ============

/**
 * Get Logs
 * Requires permission: logs:read
 */
export const getLogs = async (params?: LogQueryParams): Promise<LogsResponse> => {
  const response = await apiClient.get<LogsResponse>('/api/v1/logs', { params });
  return response.data;
};

/**
 * Get Log Stats
 * Requires permission: logs:read
 */
export const getLogStats = async (): Promise<LogStats> => {
  const response = await apiClient.get<LogStats>('/api/v1/logs/stats');
  return response.data;
};

/**
 * Export Logs
 * Requires permission: logs:export
 */
export const exportLogs = async (
  params?: LogQueryParams
): Promise<{ message: string; path: string }> => {
  const response = await apiClient.get<{ message: string; path: string }>(
    '/api/v1/logs/export',
    { params }
  );
  return response.data;
};

/**
 * Stream Logs (Server-Sent Events)
 * Requires permission: logs:read
 */
export const streamLogs = async (params?: {
  level?: string;
  since?: string;
}): Promise<ReadableStream> => {
  const url = new URL(`${API_BASE_URL}/api/v1/logs/stream`);
  if (params?.level) url.searchParams.set('level', params.level);
  if (params?.since) url.searchParams.set('since', params.since);


  const response = await fetch(url.toString(), {
    headers: {
      'X-User-ID': localStorage.getItem('qlib1_user_id') || 'anonymous',
    },
  });

  if (!response.body) {
    throw new Error('No response body');
  }

  return response.body;
};