import { AppShell } from '@/components/layout/AppShell';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Skeleton from '@mui/material/Skeleton';
import { useQuery } from '@tanstack/react-query';
import { getHealth, getSystemMetrics, getGateStatus } from '@/lib/api/client';
import { StatusCard } from '@/components/common/StatusCard';
import { QuickStatsCard } from '@/components/dashboard/QuickStatsCard';

export function DashboardPage() {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
  });

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: getSystemMetrics,
    enabled: health?.status === 'healthy',
  });

  const { data: gateStatus } = useQuery({
    queryKey: ['gate-status'],
    queryFn: getGateStatus,
  });

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <StatusCard
            title="System Health"
            status={healthLoading ? 'loading' : health?.status === 'healthy' ? 'healthy' : 'unhealthy'}
            subtitle={healthLoading ? 'Checking...' : `Version ${health?.version}`}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          {metrics ? (
            <QuickStatsCard
              title="Cache Hit Rate"
              value={`${(metrics.cache_hit_rate * 100).toFixed(1)}%`}
              subtitle={`${metrics.cache_size} cached items`}
              status={metrics.cache_hit_rate > 0.8 ? 'success' : metrics.cache_hit_rate > 0.6 ? 'warning' : 'error'}
            />
          ) : (
            <Card>
              <CardContent>
                <Skeleton variant="text" width="40%" />
                <Skeleton variant="text" width="60%" height={48} />
              </CardContent>
            </Card>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          {metrics ? (
            <QuickStatsCard
              title="Active Models"
              value={metrics.active_models.toString()}
              subtitle={`${metrics.requests_total} total requests`}
              status="info"
            />
          ) : (
            <Card>
              <CardContent>
                <Skeleton variant="text" width="40%" />
                <Skeleton variant="text" width="60%" height={48} />
              </CardContent>
            </Card>
          )}
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                PM Gate Status
              </Typography>
              {gateStatus ? (
                <Box sx={{ display: 'flex', gap: 2 }}>
                  {(['signal', 'train', 'deploy'] as const).map((dim) => (
                    <Box
                      key={dim}
                      sx={{
                        flex: 1,
                        p: 2,
                        borderRadius: 1,
                        bgcolor: gateStatus.gates[dim] === 'open' ? 'success.light' : 'error.light',
                        color: gateStatus.gates[dim] === 'open' ? 'success.dark' : 'error.dark',
                        textAlign: 'center',
                      }}
                    >
                      <Typography variant="caption" sx={{ textTransform: 'uppercase' }}>
                        {dim}
                      </Typography>
                      <Typography variant="h5" sx={{ fontWeight: 700 }}>
                        {gateStatus.gates[dim].toUpperCase()}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Skeleton variant="rectangular" height={80} />
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Performance
              </Typography>
              {metrics ? (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Avg Latency</Typography>
                    <Typography
                      variant="body2"
                      sx={{ color: metrics.avg_latency_ms > 200 ? 'error.main' : 'success.main' }}
                    >
                      {metrics.avg_latency_ms.toFixed(1)}ms
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Uptime</Typography>
                    <Typography variant="body2">
                      {Math.floor(metrics.uptime_seconds / 3600)}h {Math.floor((metrics.uptime_seconds % 3600) / 60)}m
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Cache Evictions</Typography>
                    <Typography variant="body2">{metrics.cache_evictions}</Typography>
                  </Box>
                </Box>
              ) : (
                <Skeleton variant="text" />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
