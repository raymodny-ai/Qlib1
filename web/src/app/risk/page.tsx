'use client';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getRiskMetrics } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';

interface RiskMetricCardProps {
  label: string;
  value: number;
  unit: string;
  threshold?: { warning: number; danger: number };
  higherIsBetter?: boolean;
}

function RiskMetricCard({ label, value, unit, threshold, higherIsBetter = false }: RiskMetricCardProps) {
  let color = 'text.primary';
  let status = 'normal';

  if (threshold) {
    const absVal = Math.abs(value);
    if (higherIsBetter) {
      if (absVal < threshold.danger) color = 'error.main';
      else if (absVal < threshold.warning) color = 'warning.main';
      else color = 'success.main';
    } else {
      if (absVal > threshold.danger) color = 'error.main';
      else if (absVal > threshold.warning) color = 'warning.main';
      else color = 'success.main';
    }
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="h4" sx={{ fontWeight: 700, color }}>
          {value.toFixed(4)}
          <Typography component="span" variant="body2" sx={{ ml: 0.5 }}>
            {unit}
          </Typography>
        </Typography>
      </CardContent>
    </Card>
  );
}

export default function RiskPage() {
  const [strategyId, setStrategyId] = useState('topk_dropout');

  const { data: riskData, isLoading } = useQuery({
    queryKey: ['risk', strategyId],
    queryFn: () => getRiskMetrics({ strategy_id: strategyId }),
  });

  return (
    <AppShell>
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" sx={{ fontWeight: 600 }}>
            Risk Dashboard
          </Typography>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>Strategy</InputLabel>
            <Select
              value={strategyId}
              onChange={(e) => setStrategyId(e.target.value)}
              label="Strategy"
            >
              <MenuItem value="topk_dropout">Top-K Dropout</MenuItem>
              <MenuItem value="equal_weight">Equal Weight</MenuItem>
              <MenuItem value="score_weight">Score Weight</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {riskData && (
          <>
            {/* Return Metrics */}
            <Typography variant="h6" sx={{ mb: 2 }}>
              Return Metrics
            </Typography>
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Sharpe Ratio"
                  value={riskData.metrics.sharpe_ratio}
                  unit=""
                  threshold={{ warning: 1.0, danger: 0.5 }}
                  higherIsBetter
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Annual Return"
                  value={riskData.metrics.alpha * 100}
                  unit="%"
                  higherIsBetter
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Information Ratio"
                  value={riskData.metrics.information_ratio}
                  unit=""
                  higherIsBetter
                />
              </Grid>
            </Grid>

            {/* Risk Metrics */}
            <Typography variant="h6" sx={{ mb: 2 }}>
              Risk Metrics
            </Typography>
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Max Drawdown"
                  value={riskData.metrics.max_drawdown * 100}
                  unit="%"
                  threshold={{ warning: 15, danger: 20 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Annual Volatility"
                  value={riskData.metrics.annual_volatility * 100}
                  unit="%"
                  threshold={{ warning: 25, danger: 30 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="VaR (95%)"
                  value={riskData.metrics.var_95 * 100}
                  unit="%"
                  threshold={{ warning: 3, danger: 5 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="CVaR (95%)"
                  value={riskData.metrics.cvar_95 * 100}
                  unit="%"
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Beta"
                  value={riskData.metrics.beta}
                  unit=""
                  threshold={{ warning: 1.0, danger: 1.2 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <RiskMetricCard
                  label="Alpha (Annualized)"
                  value={riskData.metrics.alpha * 100}
                  unit="%"
                  higherIsBetter
                />
              </Grid>
            </Grid>
          </>
        )}
      </Box>
    </AppShell>
  );
}