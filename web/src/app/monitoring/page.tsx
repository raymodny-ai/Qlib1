'use client';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import LinearProgress from '@mui/material/LinearProgress';
import Alert from '@mui/material/Alert';
import { useQuery } from '@tanstack/react-query';
import { getSystemMetrics } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';

export default function MonitoringPage() {
  const { data: metrics, isLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: getSystemMetrics,
    refetchInterval: 10000,
  });

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          System Monitoring
        </Typography>

        {/* Latency Alert */}
        {metrics && metrics.avg_latency_ms > 200 && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Average latency exceeds 200ms threshold. Current: {metrics.avg_latency_ms.toFixed(1)}ms
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Uptime */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Uptime
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 700 }}>
                  {metrics ? `${Math.floor(metrics.uptime_seconds / 86400)}d ${Math.floor((metrics.uptime_seconds % 86400) / 3600)}h` : '-'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Cache Hit Rate */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Cache Hit Rate
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={(metrics?.cache_hit_rate || 0) * 100}
                      color={(metrics?.cache_hit_rate || 0) > 0.8 ? 'success' : (metrics?.cache_hit_rate || 0) > 0.6 ? 'warning' : 'error'}
                      sx={{ height: 10, borderRadius: 5 }}
                    />
                  </Box>
                  <Typography variant="h6" sx={{ fontWeight: 700 }}>
                    {((metrics?.cache_hit_rate || 0) * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Performance Metrics */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Performance
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Avg Latency
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700, color: (metrics?.avg_latency_ms || 0) > 200 ? 'error.main' : 'success.main' }}>
                      {metrics?.avg_latency_ms.toFixed(1)}ms
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Total Requests
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      {metrics?.requests_total.toLocaleString()}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Cache Status */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Cache Status
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Cached Items
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      {metrics?.cache_size}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Evictions
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 700 }}>
                      {metrics?.cache_evictions}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Model Status */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Models
                </Typography>
                <Typography variant="body2">
                  Active Models: <strong>{metrics?.active_models || 0}</strong>
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </AppShell>
  );
}