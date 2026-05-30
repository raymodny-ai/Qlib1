'use client';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import { useQuery } from '@tanstack/react-query';
import { getReport } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';
import { useState } from 'react';

export default function ReportsPage() {
  const [experimentId, setExperimentId] = useState('');

  const { data: report, isLoading } = useQuery({
    queryKey: ['report', experimentId],
    queryFn: () => getReport(experimentId),
    enabled: !!experimentId,
  });

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Performance Reports
        </Typography>

        {/* Query Form */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
              <TextField
                label="Experiment ID"
                value={experimentId}
                onChange={(e) => setExperimentId(e.target.value)}
                size="small"
                sx={{ minWidth: 300 }}
                placeholder="e.g., exp_20260101_lgb_v1"
              />
              <Button variant="contained" disabled={!experimentId}>
                Query
              </Button>
            </Box>
          </CardContent>
        </Card>

        {/* Report Results */}
        {report && (
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Report: {report.experiment_id}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Generated: {new Date(report.generated_at).toLocaleString()}
              </Typography>

              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  IC Metrics
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">IC Mean</Typography>
                    <Typography variant="h6">{report.metrics.ic_mean?.toFixed(4) || '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">ICIR</Typography>
                    <Typography variant="h6">{report.metrics.icir?.toFixed(4) || '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Rank IC Mean</Typography>
                    <Typography variant="h6">{report.metrics.rank_ic_mean?.toFixed(4) || '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Rank ICIR</Typography>
                    <Typography variant="h6">{report.metrics.rank_icir?.toFixed(4) || '-'}</Typography>
                  </Box>
                </Box>
              </Box>

              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Performance Metrics
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Total Return</Typography>
                    <Typography variant="h6">{report.metrics.total_return ? `${(report.metrics.total_return * 100).toFixed(2)}%` : '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Annualized Return</Typography>
                    <Typography variant="h6">{report.metrics.annualized_return ? `${(report.metrics.annualized_return * 100).toFixed(2)}%` : '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Sharpe Ratio</Typography>
                    <Typography variant="h6">{report.metrics.sharpe_ratio?.toFixed(2) || '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Max Drawdown</Typography>
                    <Typography variant="h6">{report.metrics.max_drawdown ? `${(report.metrics.max_drawdown * 100).toFixed(2)}%` : '-'}</Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        )}
      </Box>
    </AppShell>
  );
}