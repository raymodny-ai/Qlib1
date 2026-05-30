import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import { EquityCurveChart, generateMockEquityCurve, type EquityPoint } from '@/components/charts/EquityCurveChart';
import { DrawdownChart, generateMockDrawdown } from '@/components/charts/DrawdownChart';
import { MetricsCard } from '@/components/backtest/MetricsCard';
import type { BacktestStatus } from '@/types/api';

interface ResultsPanelProps {
  result: BacktestStatus;
  startDate?: string;
  endDate?: string;
  initialCapital?: number;
}

function evaluateThreshold(value: number, good: number, bad: number): 'good' | 'warn' | 'bad' {
  if (value >= good) return 'good';
  if (value >= bad) return 'warn';
  return 'bad';
}

export function ResultsPanel({ result, startDate, endDate, initialCapital = 1_000_000 }: ResultsPanelProps) {
  const metrics = result.result;
  if (!metrics) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
        No result data available
      </Typography>
    );
  }

  // Generate mock equity curve and drawdown from dates + capital
  const eqData: EquityPoint[] = generateMockEquityCurve(
    startDate || '2023-01-01',
    endDate || '2024-01-01',
    initialCapital
  );
  const ddData = generateMockDrawdown(eqData.map((d) => ({ date: d.date, strategy: d.strategy })));

  // Extract metrics (use type-safe access with fallback)
  const sharpe = metrics.sharpe_ratio;
  const maxDrawdown = metrics.max_drawdown;
  const annReturn = metrics.annual_return;
  const winRate = metrics.win_rate;
  const calmar = maxDrawdown !== 0 ? annReturn / Math.abs(maxDrawdown) : 0;

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
        Backtest Results
      </Typography>

      {/* Metrics Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={4} md={2.4}>
          <MetricsCard
            label="Sharpe Ratio"
            value={sharpe.toFixed(3)}
            threshold={evaluateThreshold(sharpe, 1.5, 0.8)}
            tooltip="Risk-adjusted return. >1.5 is excellent, <0.8 needs attention."
          />
        </Grid>
        <Grid item xs={6} sm={4} md={2.4}>
          <MetricsCard
            label="Max Drawdown"
            value={`${(Math.abs(maxDrawdown) * 100).toFixed(2)}%`}
            threshold={evaluateThreshold(-Math.abs(maxDrawdown), -0.1, -0.25)}
            tooltip="Maximum peak-to-trough decline. Lower is better."
          />
        </Grid>
        <Grid item xs={6} sm={4} md={2.4}>
          <MetricsCard
            label="Ann. Return"
            value={`${(annReturn * 100).toFixed(2)}%`}
            threshold={evaluateThreshold(annReturn, 0.15, 0.05)}
            tooltip="Annualized return. >15% is strong, <5% is weak."
          />
        </Grid>
        <Grid item xs={6} sm={4} md={2.4}>
          <MetricsCard
            label="Win Rate"
            value={`${(winRate * 100).toFixed(1)}%`}
            threshold={evaluateThreshold(winRate, 0.55, 0.45)}
            tooltip="Percentage of winning trades."
          />
        </Grid>
        <Grid item xs={6} sm={4} md={2.4}>
          <MetricsCard
            label="Calmar Ratio"
            value={calmar.toFixed(3)}
            threshold={evaluateThreshold(calmar, 1.0, 0.5)}
            tooltip="Annualized return / max drawdown."
          />
        </Grid>
      </Grid>

      <Divider sx={{ mb: 3 }} />

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <EquityCurveChart data={eqData} />
        </Grid>
        <Grid item xs={12}>
          <DrawdownChart data={ddData} />
        </Grid>
      </Grid>
    </Box>
  );
}
